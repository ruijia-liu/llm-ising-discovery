"""Read-only independent data/budget/answer audit; never invokes model generation."""
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from build_publication_curriculum import atomic_json, read, sha
from discovery import student_v3
from discovery.physics import Expression, Grammar, Term
from discovery.student_fastdev import terminal_messages, terminal_prompt
from discovery.student_v2 import encode, load_views, packed
from discovery.teaching import decode_query, expression_from_dict
from run_structure_discovery import evaluate
from train_student_v3 import evaluation_features

ROOT = Path('results/parameter_diversity_v1')
CONTROL = Path('results/fast_development_v2')


def lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines()]


def audit_inputs():
    config = read(ROOT/'config.json')
    for key in ('protected_sha256', 'source_sha256'):
        assert all(sha(path) == digest for path, digest in config[key].items()), key
    assert all(sha(ROOT/'sources'/path) == digest for path, digest in config['source_sha256'].items())
    assert {d.metadata['Name']: d.version for d in importlib.metadata.distributions()} == config['environment']
    model = Path('models/Qwen2.5-0.5B-Instruct')
    assert all(sha(model/name) == digest for name, digest in config['base_files_sha256'].items())
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    train = load_views('train')
    validation = [v for v in load_views('validation') if v['final_answer']['expression']['terms']]
    assert [v['episode_id'] for v in validation] == config['validation_ids']
    all_old = [expression_from_dict(v['final_answer']['expression']) for v in train+validation]
    originals = {x['episode_id']: x['messages'] for x in lines(CONTROL/'raw_messages.jsonl') if x['split'] == 'train'}
    records = lines(ROOT/'training_messages.jsonl')
    assert len(records) == 84
    new_signatures = set()
    original_measurements = new_measurements = supervised_tokens = input_tokens = 0
    index_map = {}
    for index, record in enumerate(records):
        entry = config['examples'][index]
        assert all(record[k] == v for k, v in entry.items())
        index_map[entry['original_index'], entry['variant']] = index
        view = train[entry['original_index']]
        original_truth = expression_from_dict(view['final_answer']['expression'])
        truth = expression_from_dict(entry['expression'])
        assert truth.support == original_truth.support
        if entry['variant']:
            rng = np.random.default_rng([97027, int(view['episode_id'][1:5]), entry['variant']])
            coefficients = rng.uniform(.3, .8, len(original_truth.terms))*rng.choice([-1, 1], len(original_truth.terms))
            expected_truth = Expression(tuple(Term(term.points, float(c)) for term, c in zip(original_truth.terms, coefficients)))
            assert truth == expected_truth and all(truth != old for old in all_old)
            signature = packed(truth.to_dict())
            assert signature not in new_signatures
            new_signatures.add(signature)
        else:
            assert truth == original_truth
            assert record['messages'] == originals[view['episode_id']]
        original_evidence = student_v3.evidence_only(view)
        queries = view['initial']['queries']+[e['query'] for e in view['experiments']]
        regenerated = dict(original_evidence, measurements=[])
        for query, observation in zip(queries, original_evidence['measurements']):
            state, pos = decode_query(query)
            flipped = state.copy()
            flipped[pos] *= -1
            energy_difference = truth.energy(flipped)-truth.energy(state)
            local_delta = truth.delta(state, pos)
            assert np.isclose(energy_difference, local_delta, atol=1e-9, rtol=0)
            if entry['variant']:
                value = local_delta
                new_measurements += 1
            else:
                value = observation['delta']
                assert np.isclose(value, energy_difference, atol=1e-9, rtol=0)
                original_measurements += 1
            regenerated['measurements'].append({'query': observation['query'], 'delta': value})
        expected_messages = terminal_prompt(regenerated, 'raw')+[{'role': 'assistant', 'content': packed({'structures': truth.support})}]
        assert expected_messages == record['messages']
        # Verify coefficients can still be identified on the unchanged query bank.
        fitted, diagnostics = student_v3.fit_coefficients(regenerated, truth.support)
        assert diagnostics['identifiable_given_proposal'] and fitted.support == truth.support
        assert all(abs(a.coefficient-b.coefficient) < 1e-10 for a, b in zip(fitted.terms, truth.terms))
        encoded = encode(tokenizer, expected_messages)
        assert len(encoded['ids']) == entry['input_tokens']
        assert np.count_nonzero(encoded['labels'][1:] != -100) == entry['supervised_tokens']
    old_schedule = read(CONTROL/'config.json')['schedule']
    old_training = lines(CONTROL/'raw/training.jsonl')
    assert old_schedule == config['original_schedule'] == [r['indices'] for r in old_training]
    visits, frequencies = Counter(), Counter()
    for old_indices, new_indices in zip(old_schedule, config['schedule']):
        assert len(old_indices) == len(new_indices) == 2
        for old, new in zip(old_indices, new_indices):
            expected_variant = visits[old] % 4 if train[old]['final_answer']['expression']['terms'] else 0
            assert new == index_map[old, expected_variant]
            visits[old] += 1
            frequencies[new] += 1
            input_tokens += records[new]['input_tokens']
            supervised_tokens += records[new]['supervised_tokens']
    assert visits == Counter({i: 16 for i in range(24)})
    assert all(frequencies[i] == (4 if r['expression']['terms'] else 16) for i, r in enumerate(records))
    assert input_tokens == config['input_tokens'] and supervised_tokens == config['supervised_tokens'] == 5568
    assert abs(input_tokens-config['control_input_tokens'])/config['control_input_tokens'] <= .01
    assert len(new_signatures) == 60 and new_measurements == 1920 and original_measurements == 768
    return {'passed': True, 'all_84_tasks_reconstructed_and_retokenized': True,
        'same_384_ordered_targets': True, 'same_5568_supervised_tokens': True,
        'new_parameters_absent_from_original_train_and_development': True,
        'original_observations_verified': original_measurements, 'new_observations_verified': new_measurements,
        'all_training_true_proposals_identifiable': True,
        'protected_files_unchanged': len(config['protected_sha256']),
        'environment_package_versions_unchanged': len(config['environment'])}


def audit_results(input_audit):
    config, result = read(ROOT/'config.json'), read(ROOT/'result.json')
    assert result['completed'] and result['config_sha256'] == sha(ROOT/'config.json')
    assert result['episodes_sha256'] == sha(ROOT/'episodes.jsonl')
    training = lines(ROOT/'training.jsonl')
    assert len(training) == 192 and [r['indices'] for r in training] == config['schedule']
    assert training[-1]['input_tokens'] == config['input_tokens']
    assert training[-1]['supervised_tokens'] == 5568
    # First epoch contains only original examples. Equal losses/gradients are a
    # useful direct check that reuse did not silently change initialization or training.
    old_training = lines(CONTROL/'raw/training.jsonl')
    identical_first_epoch = all(a['loss'] == b['loss'] and a['gradient_norm'] == b['gradient_norm'] for a, b in zip(training[:12], old_training[:12]))
    assert identical_first_epoch
    checkpoint = torch.load(ROOT/'step_checkpoint.pt', map_location='cpu', weights_only=False)
    adapter = load_file(ROOT/'adapter/adapter_model.safetensors')
    assert checkpoint['step'] == 192 and set(checkpoint['adapter']) == set(adapter)
    assert all(torch.equal(adapter[k], checkpoint['adapter'][k]) for k in adapter)
    assert sha(ROOT/'adapter/adapter_model.safetensors') == result['training']['adapter_sha256']
    views = {v['episode_id']: v for v in load_views('validation')}
    rows = lines(ROOT/'episodes.jsonl')
    controls = read(ROOT/'control_rows.json')
    archived = [r for r in lines(CONTROL/'raw/episodes.jsonl') if r['split'] == 'validation' and not r['null_world']]
    assert controls == archived
    assert [r['episode_id'] for r in rows] == [r['episode_id'] for r in controls] == config['validation_ids']
    grammar = Grammar()
    feature_cache = {eid: evaluation_features(grammar, views[eid]) for eid in config['validation_ids']}
    for row in controls+rows:
        view = views[row['episode_id']]
        truth = expression_from_dict(view['final_answer']['expression'])
        evidence = student_v3.evidence_only(view)
        prompt = terminal_prompt(evidence, 'raw')
        assert hashlib.sha256(json.dumps(prompt, ensure_ascii=False).encode()).hexdigest() == row['prompt_sha256']
        try:
            proposal = student_v3.parse_structures(row['text'], grammar)
            fitted, fit = student_v3.fit_coefficients(evidence, proposal)
        except (ValueError, TypeError, KeyError, IndexError):
            assert not row['valid_expression'] and not row['full_recovery']
        else:
            expected = evaluate(grammar, truth, fitted, feature_cache[row['episode_id']])
            expected['full_recovery'] &= fit['identifiable_given_proposal']
            expected['raw_support_success'] = proposal == truth.support
            assert row['valid_expression'] and all(row[k] == v for k, v in expected.items())
    old_count, new_count = (sum(r['full_recovery'] for r in rr) for rr in (controls, rows))
    assert old_count == result['control_nonempty_recovered'] == 4
    assert new_count == result['diverse_nonempty_recovered']
    assert result['improved'] == (new_count > old_count)
    paired = Counter((a['full_recovery'], b['full_recovery']) for a, b in zip(controls, rows))
    verification = {**input_audit, 'answers_rescored': 20, 'same_validation_prompts': True,
        'first_12_update_losses_and_gradient_norms_identical_to_control': identical_first_epoch,
        'final_adapter_tensors_equal_checkpoint': len(adapter),
        'paired_gained': paired[False, True], 'paired_lost': paired[True, False],
        'both_recovered': paired[True, True], 'neither_recovered': paired[False, False],
        'verifier_sha256': sha(__file__), 'original_main_still_paused': True, 'formal_tests_used': False}
    atomic_json(ROOT/'verification.json', verification)
    (ROOT/'verification_source.py').write_bytes(Path(__file__).read_bytes())
    return verification, result, controls, rows


def report(verified, result, controls, rows):
    original = read(CONTROL/'raw/result.json')
    config = read(ROOT/'config.json')
    improved = result['improved']
    content = ['# 单次参数多样性对照：结果', '',
        f"非空新参数题完整恢复：原数据 **4/10**，增加参数例子 **{result['diverse_nonempty_recovered']}/10**。", '',
        ('达到事前约定的扩大信号；本轮就此结束，没有自动扩大训练。' if improved else
         '未达到事前约定的改善条件。本配置停止重复，不增加步数、种子或重新调提示。'), '',
        '保留原始5×5自旋及能量差输入。24个原训练世界中，20个非空世界各增加3个参数例，',
        '共60个新增参数世界；构型、位置、作用结构和逐次答案标签不变。每个原非空例原来重复16次，',
        '现在原例与3个变体各出现4次；4个空世界仍各16次。新增1,920个观测通过完整能量差核验。', '',
        '对照复用已完成且核验的原数据训练；新增分支从同一基础模型与seed301重新训练，',
        '不是继续训练旧适配器。第一轮12个更新均为原例，损失及梯度范数与历史对照逐步完全相同。', '',
        '|训练预算|原数据（复用）|参数多样性|', '|---|---:|---:|',
        '|更新次数|192|192|', '|答案呈现次数|384|384|', '|监督token|5,568|5,568|',
        f"|输入token|{original['input_tokens']:,}|{result['training']['input_tokens']:,}|",
        f"|训练墙钟秒|{original['training_wall_seconds_including_checkpoints']:.1f}|{result['training']['training_wall_seconds_including_checkpoints']:.1f}|", '',
        f"输入token相差{100*config['relative_input_difference']:.3f}%，低于事前1%上限；不宣称计算量或墙钟精确相等。",
        '训练非同期，不根据墙钟差宣称提速。公共拟合器只拟合学生所提结构，并不增加结构候选。', '',
        f"配对变化：新增恢复{verified['paired_gained']}题，失去原来恢复的{verified['paired_lost']}题；两边都恢复{verified['both_recovered']}题。",
        f"原始结构直接正确：{sum(r['raw_support_success'] for r in controls)}/10 → {result['raw_support_recovered']}/10。",
        '扩大/停止决策仅使用完整恢复数，没有事后换指标。', '',
        '|世界|原数据恢复|参数多样性恢复|', '|---|---:|---:|']
    for a, b in zip(controls, rows):
        content.append(f"|{a['episode_id']}|{int(a['full_recovery'])}|{int(b['full_recovery'])}|")
    content += ['', '这些10题是训练未见参数、此前已经用于开发的题；不是全新确认测试。',
        '单种子小样本只是本配置的开发信号，不能证明稳定提升、新结构发现或过程教学有效。',
        '没有自主实验、额外训练题评估、噪声扩展、多种子或B/C训练；正式保留世界未打开。',
        '原长训练保持B/401第215步暂停。两边20份原始回答、全部训练任务、预算与最终权重均已复核。', '']
    (ROOT/'REPORT_ZH.md').write_text('\n'.join(content), encoding='utf-8')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs-only', action='store_true')
    args = parser.parse_args()
    inputs = audit_inputs()
    if args.inputs_only:
        atomic_json(ROOT/'input_verification.json', inputs)
        print(json.dumps(inputs, indent=2))
    else:
        verified, result, controls, rows = audit_results(inputs)
        report(verified, result, controls, rows)
        print(json.dumps(verified, indent=2))
