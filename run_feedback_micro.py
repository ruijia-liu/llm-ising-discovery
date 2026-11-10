"""Bounded development-only feedback probe; no training or new measurements."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoTokenizer

from build_publication_curriculum import atomic_json, sha
from discovery import student_v3 as task
from discovery.student_v2 import load_views
from discovery.teaching import expression_from_dict
from run_structure_discovery import evaluate
from train_student_pilot import load_base
from train_student_v3 import evaluation_features

OUT = Path('results/feedback_micro_v1')
MODEL = Path('models/Qwen2.5-0.5B-Instruct')
ADAPTER = Path('results/fast_development_v2/raw/adapter')
IDS = [f'w{i:04d}-sigma0' for i in (24, 26, 28, 30, 32, 34)]


def feedback(evidence, text):
    support = task.parse_structures(text)
    fitted, fit = task.fit_coefficients(evidence, support)
    queries, observed = task.observed_arrays(evidence)
    predicted = np.array([fitted.delta(*q) for q in queries])
    residual = observed - predicted
    worst = np.argsort(-np.abs(residual), kind='stable')[:4]
    return {'observed_data_rmse': fit['observation_residual_rmse'],
            'fitted_coefficients': fit['fitted_coefficients_before_zero_removal'],
            'largest_residuals': [{'measurement_index_1based': int(i)+1,
                'observed_minus_predicted': round(float(residual[i]), 6)} for i in worst]}


def main():
    if OUT.exists():
        raise ValueError('Preserve existing attempts.')
    OUT.mkdir()
    start = time.perf_counter()
    original = {r['episode_id']: r for r in map(json.loads,
        Path('results/fast_development_v2/raw/episodes.jsonl').read_text(encoding='utf-8').splitlines())
        if r['split'] == 'validation'}
    views = {v['episode_id']: v for v in load_views('validation')}
    hashes = {str(p): sha(p) for p in (ADAPTER/'adapter_model.safetensors', Path(__file__))}
    plan = {'ids': IDS, 'selection': 'every other development world in index order, including one empty control',
        'arms': ['generic_revision', 'residual_feedback'], 'calls': 12, 'max_new_tokens_per_call': 128,
        'training_updates': 0, 'new_measurements': 0, 'formal_tests_used': False,
        'baseline': 'preserved raw terminal answers', 'hashes': hashes,
        'selection_rule': 'keep valid revision only if observed residual RMSE improves by more than 1e-9; otherwise retain original',
        'note': 'Previously inspected small development subset; added inference budget, not a teaching-effect test.'}
    atomic_json(OUT/'plan.json', plan)
    # Build all prompts before loading labels for scoring.
    requests = []
    for eid in IDS:
        evidence = task.evidence_only(views[eid])
        fb = feedback(evidence, original[eid]['text'])
        for arm in plan['arms']:
            instruction = '请重新检查上述结构与已有32条观测是否一致，可以保留或修正结构。不得申请新测量。只输出相同格式的structures JSON，不解释。'
            if arm == 'residual_feedback':
                instruction = '数值程序仅使用上述观测拟合你提出的结构，得到：'+json.dumps(fb, ensure_ascii=False)+'。残差为观测减预测；非零残差提示当前结构不能解释全部观测。'+instruction
            messages = task.terminal_prompt(evidence)+[
                {'role': 'assistant', 'content': original[eid]['text']},
                {'role': 'user', 'content': instruction}]
            requests.append({'episode_id': eid, 'arm': arm, 'messages': messages, 'feedback': fb})
    atomic_json(OUT/'requests.json', requests)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = PeftModel.from_pretrained(load_base(MODEL), ADAPTER).eval()
    rows = []
    with torch.inference_mode():
        for req in requests:
            ids = tokenizer.apply_chat_template(req['messages'], tokenize=True,
                add_generation_prompt=True, return_tensors='pt').to('cuda')
            output = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=128,
                do_sample=False, repetition_penalty=1.0, pad_token_id=tokenizer.eos_token_id, use_cache=True)
            text = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
            row = {k: req[k] for k in ('episode_id', 'arm')}
            row.update(text=text, input_tokens=ids.shape[1], output_tokens=output.shape[1]-ids.shape[1])
            rows.append(row)
            atomic_json(OUT/'generations.json', rows)
            print(row['episode_id'], row['arm'], text, flush=True)
    # Labels and independent evaluation features are used only after all generation.
    grammar = task.Grammar()
    for row in rows:
        eid = row['episode_id']
        evidence = task.evidence_only(views[eid])
        initial_support = task.parse_structures(original[eid]['text'])
        chosen, initial_fit = task.fit_coefficients(evidence, initial_support)
        row.update(valid=False, accepted=False, revised_recovery=False)
        truth = expression_from_dict(views[eid]['final_answer']['expression'])
        features = evaluation_features(grammar, views[eid])
        try:
            support = task.parse_structures(row['text'])
            revised, fit = task.fit_coefficients(evidence, support)
            row.update(valid=True, residual_rmse=fit['observation_residual_rmse'],
                revised_recovery=bool(evaluate(grammar, truth, revised, features)['full_recovery'] and fit['identifiable_given_proposal']))
            if fit['identifiable_given_proposal'] and fit['observation_residual_rmse'] < initial_fit['observation_residual_rmse']-1e-9:
                chosen = revised
                row['accepted'] = True
        except (ValueError, TypeError, KeyError, IndexError) as error:
            row['error'] = str(error)
        row['selected_recovery'] = bool(evaluate(grammar, truth, chosen, features)['full_recovery'])
        row['baseline_recovery'] = original[eid]['full_recovery']
        row['nonempty'] = bool(truth.terms)
    assert all(sha(p) == h for p, h in hashes.items())
    summary = {arm: {key: sum(bool(r[key]) for r in rows if r['arm'] == arm)
        for key in ('valid', 'accepted', 'revised_recovery', 'selected_recovery', 'baseline_recovery')}
        for arm in plan['arms']}
    result = {'summary': summary, 'rows': rows, 'elapsed_seconds': time.perf_counter()-start,
        'input_tokens': sum(r['input_tokens'] for r in rows), 'output_tokens': sum(r['output_tokens'] for r in rows)}
    atomic_json(OUT/'result.json', result)
    (OUT/'REPORT_ZH.md').write_text('# 单次结构反馈小测验\n\n'+json.dumps(summary, ensure_ascii=False, indent=2)+
        '\n\n6个旧开发世界（5个非空、1个空），两支各修正一次；复用初答，无训练、无新增测量。'
        '反馈只来自已观测数据；选择规则在推理前固定。额外推理预算不等于训练收益。\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
