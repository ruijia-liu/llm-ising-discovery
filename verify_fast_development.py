"""Audit the completed paired screen and write its measured, qualified report."""
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoTokenizer

from build_publication_curriculum import atomic_json, read, sha
from discovery import student_v3
from discovery.physics import Grammar, base_flip
from discovery.student_fastdev import resident_loss, residual_evidence, terminal_prompt
from discovery.student_v2 import encode, load_views
from discovery.teaching import expression_from_dict
from run_structure_discovery import evaluate
from train_student_pilot import bounded_loss, make_student, tensors
from train_student_v3 import evaluation_features


ROOT = Path('results/fast_development_v2')


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def verify():
    config, result = read(ROOT/'config.json'), read(ROOT/'result.json')
    environment = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    if environment != config['environment']:
        raise ValueError('Runtime package versions changed.')
    if not result['completed'] or result['config_sha256'] != sha(ROOT/'config.json'):
        raise ValueError('Completed result provenance failed.')
    for key in ('protected_sha256', 'source_sha256', 'data_sha256'):
        if any(sha(p) != h for p, h in config[key].items()):
            raise ValueError('A frozen source, old result, or input changed.')
    if any(sha(ROOT/'sources'/p) != h for p, h in config['source_sha256'].items()):
        raise ValueError('Source snapshot mismatch.')
    base_config = read('results/student_mainline_v1/A_seed401/config.json')
    if any(sha(Path('models/Qwen2.5-0.5B-Instruct')/p) != h for p, h in base_config['base_files_sha256'].items()):
        raise ValueError('Pinned base files changed.')
    views = {s: load_views(s) for s in ('train', 'validation')}
    by_id = {v['episode_id']: v for entries in views.values() for v in entries}
    grammar = Grammar()
    all_rows = {}
    max_rounding_error = 0.0
    for variant in ('raw', 'residual'):
        records = lines(ROOT/variant/'training.jsonl')
        if len(records) != 192 or [r['indices'] for r in records] != config['schedule']:
            raise ValueError('Incomplete or changed training schedule.')
        counts = Counter(i for row in records for i in row['indices'])
        if counts != Counter({i: 16 for i in range(24)}):
            raise ValueError('Unequal world exposure.')
        rows = lines(ROOT/variant/'episodes.jsonl')
        if [r['episode_id'] for r in rows] != [v['episode_id'] for entries in views.values() for v in entries]:
            raise ValueError('Missing or reordered score rows.')
        arm = result['results'][variant]
        if sha(ROOT/variant/'episodes.jsonl') != arm['episodes_sha256']:
            raise ValueError('Answer archive changed.')
        if sha(ROOT/variant/'adapter/adapter_model.safetensors') != arm['adapter_sha256']:
            raise ValueError('Adapter changed.')
        if records[-1]['input_tokens'] != arm['input_tokens'] or records[-1]['supervised_tokens'] != arm['supervised_tokens']:
            raise ValueError('Resource total differs from actual committed updates.')
        for row in rows:
            view = by_id[row['episode_id']]
            evidence = student_v3.evidence_only(view)
            prompt = terminal_prompt(evidence, variant)
            if hashlib.sha256(json.dumps(prompt, ensure_ascii=False).encode()).hexdigest() != row['prompt_sha256']:
                raise ValueError('Actual prompt identity changed.')
            truth = expression_from_dict(view['final_answer']['expression'])
            try:
                support = student_v3.parse_structures(row['text'], grammar)
                fitted, fitting = student_v3.fit_coefficients(evidence, support)
            except (ValueError, TypeError, KeyError, IndexError):
                if row['valid_expression'] or row['full_recovery']:
                    raise ValueError('Invalid answer scored as valid.')
            else:
                expected = evaluate(grammar, truth, fitted, evaluation_features(grammar, view))
                expected['full_recovery'] &= fitting['identifiable_given_proposal']
                expected['raw_support_success'] = support == truth.support
                if not row['valid_expression'] or any(row[k] != v for k, v in expected.items()):
                    raise ValueError('Re-scored prediction differs from saved result.')
            transformed = residual_evidence(evidence)
            queries, observed = student_v3.observed_arrays(evidence)
            for (state, pos), observed_delta, converted in zip(queries, observed, transformed['measurements']):
                restored = base_flip(state, pos)+2*int(state[pos])*transformed['scale']*converted[1]
                scaled_error = abs(restored-observed_delta)/transformed['scale']
                max_rounding_error = max(max_rounding_error, scaled_error)
                if scaled_error > 1.0001e-8:
                    raise ValueError('Input rounding exceeds the declared bound.')
        for split in views:
            selected = [r for r in rows if r['split'] == split]
            for key in ('valid_expression', 'raw_support_success', 'raw_false_proposal', 'full_recovery', 'false_discovery', 'prediction_success'):
                if sum(bool(r[key]) for r in selected) != arm['summary'][split][key]:
                    raise ValueError('A summary count differs from underlying answers.')
        all_rows[variant] = rows
    paired = {}
    for split in views:
        paired[split] = {}
        for metric in ('raw_support_success', 'full_recovery'):
            counts = Counter((bool(a[metric]), bool(b[metric])) for a, b in zip(all_rows['raw'], all_rows['residual']) if a['split'] == split)
            paired[split][metric] = {f'raw_{a}_residual_{b}': counts[a, b] for a, b in ((False, False), (False, True), (True, False), (True, True))}
    verification = {'passed': True, 'answers_rescored': 72, 'all_prompts_reconstructed': True,
                    'protected_original_files_unchanged': len(config['protected_sha256']),
                    'same_384_presentations_per_arm': True, 'all_24_worlds_seen_16_times': True,
                    'original_main_still_paused': True, 'formal_tests_used': False,
                    'runtime_package_versions_unchanged': len(environment),
                    'maximum_delta_reconstruction_error_divided_by_scale': max_rounding_error,
                    'paired_counts': paired, 'verifier_sha256': sha(__file__)}
    atomic_json(ROOT/'verification.json', verification)
    (ROOT/'verification_sources').mkdir(exist_ok=True)
    (ROOT/'verification_sources'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    return result, verification


def trained_gradient_check():
    """Nonzero adapters exercise LoRA-A gradients, absent at initialization."""
    target = ROOT/'trained_gradient_verification.json'
    if target.exists():
        return read(target)
    if torch.cuda.is_initialized():
        raise ValueError('CPU audit unexpectedly initialized CUDA.')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained('models/Qwen2.5-0.5B-Instruct', local_files_only=True)
    model = make_student('models/Qwen2.5-0.5B-Instruct', 301)
    saved = load_file(ROOT/'raw/adapter/adapter_model.safetensors')
    set_peft_model_state_dict(model, saved)
    loaded = get_peft_model_state_dict(model)
    if set(loaded) != set(saved) or any(not torch.equal(loaded[k].detach().cpu(), saved[k]) for k in saved):
        raise ValueError('Trained adapter was not loaded exactly.')
    model.train()
    examples = [encode(tokenizer, student_v3.terminal_messages(v)) for v in load_views('train')]
    example = max(examples, key=lambda e: len(e['ids']))
    reference = None
    for fn in (bounded_loss, resident_loss):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(955)
        torch.cuda.manual_seed_all(955)
        total, count = fn(model, *tensors(example))
        loss = total/count
        loss.backward()
        gradients = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        value = float(loss.detach())
        if reference is None:
            reference = value, gradients
        else:
            check = {'gradient_tensor_count': len(gradients),
                     'nonzero_gradient_tensors': sum(bool(g.count_nonzero()) for g in gradients.values()),
                     'loss_absolute_difference': abs(value-reference[0]),
                     'gradient_max_absolute_difference': max(float((g-reference[1][n]).abs().max()) for n, g in gradients.items()),
                     'passed': bool(np.isclose(value, reference[0], rtol=1e-5, atol=1e-7)
                         and all(torch.allclose(g, reference[1][n], rtol=1e-5, atol=1e-7) for n, g in gradients.items())),
                     'adapter_sha256': sha(ROOT/'raw/adapter/adapter_model.safetensors'),
                     'no_optimizer_updates': True, 'source_sha256': sha(__file__)}
            atomic_json(target, check)
            if not check['passed'] or check['nonzero_gradient_tensors'] != 192:
                raise ValueError('Trained-adapter gradient check failed.')
        del loss, total
    return check


def report(result, verification, gradients):
    raw, residual = (result['results'][v] for v in ('raw', 'residual'))
    a_dev, b_dev = raw['summary']['validation'], residual['summary']['validation']
    content = ['# 快速开发对照：实测结果', '',
        f"开发题原始结构正确：{a_dev['raw_support_success']}/12 → {b_dev['raw_support_success']}/12；"
        f"拟合后严格恢复：{a_dev['full_recovery']}/12 → {b_dev['full_recovery']}/12。", '',
        '主训练仍暂停在B/401第215步。本次是单种子、小样本的共同终答任务筛选，不是A/B/C教学效果实验。',
        '两分支重新训练，基础模型、seed301、192更新、384答案呈现、样本顺序、终答标签与拟合工具相同。',
        '使用24道旧训练题、12道此前已用于开发的参数变化题；后者未进入本轮梯度训练，但不属于全新确认测试。', '',
        '| 指标 | 原始输入 raw | 扣除已知规律、缩放和精简输入 residual |', '|---|---:|---:|']
    for split, name in (('train', '训练'), ('validation', '开发验证')):
        a, b = raw['summary'][split], residual['summary'][split]
        for key, label, denominator in (('raw_support_success', '原始结构正确', 'n'), ('full_recovery', '拟合后严格恢复', 'n'),
                                       ('nonempty_recovered', '非空机制恢复', 'nonempty_n'), ('valid_expression', '输出合法', 'n'),
                                       ('raw_false_proposal', '空世界原始误报', 'null_n'), ('false_discovery', '空世界拟合后误报', 'null_n')):
            content.append(f'|{name}：{label}|{a[key]}/{a[denominator]}|{b[key]}/{b[denominator]}|')
    content += ['', '| 实际消耗 | raw | residual |', '|---|---:|---:|']
    for key, label in (('training_wall_seconds_including_checkpoints', '训练墙钟秒（含检查点）'), ('generation_seconds', '36题生成秒'),
                       ('input_tokens', '训练输入token'), ('supervised_tokens', '监督token'), ('peak_training_allocated_gib', '训练峰值分配显存GiB')):
        content.append(f'|{label}|{raw[key]:.3f}|{residual[key]:.3f}|')
    profile = result['profile']
    content += ['', f"精简方案相对本轮raw训练耗时减少 {1-residual['training_wall_seconds_including_checkpoints']/raw['training_wall_seconds_including_checkpoints']:.1%}。",
        f"本轮运行总墙钟 {result['elapsed_seconds']:.1f}秒（不包含此前失败预检、诊断及本报告复核）。",
        f"单纯GPU常驻激活在同一确定性设置下的短任务计时：{profile['speedup']:.3f}倍；不能外推长C任务。",
        '更少清理/保存、避免中途全题评估等也是新运行的工程变化，历史旧v3耗时不作为随机配对速度试验。', '',
        '原始结构分数直接判学生提出的结构；拟合后严格恢复要求结构、系数及两个独立构型分布的预测均通过。',
        '拟合器只拟合模型提议，不搜索、不新增项。报告原始误报，避免数值拟合把错误项消为零后掩盖误报。',
        '残差、按观测缩放、字段精简一起改变，不能单独归因。输入舍入8位，原始测量仍用于共同系数拟合。', '',
        '本轮只训练及评估无噪声条件。将观测自身用于缩放时，带噪声条件下的归一化噪声不能简单视为独立高斯；',
        '现有normalized_noise_sigma字段只是原始噪声尺度除以缩放量。扩展到噪声实验前需要修订说明并检验，不能直接外推。', '',
        '## 数值检查与保留记录', '',
        'v1预检失败，尚未训练。原实现自身重复计算同样出现梯度差异，关闭dropout仍存在；',
        '启用确定性算法及cuBLAS工作区设置后，初始化权重和已训练非零适配器均通过严格梯度比较。',
        f"已训练检查覆盖 {gradients['nonzero_gradient_tensors']} 个非零梯度张量，最大绝对差 {gradients['gradient_max_absolute_difference']:.3g}。",
        '该设置与PyTorch的[确定性计算说明](https://docs.pytorch.org/docs/2.9/notes/randomness.html)一致；未进一步隔离具体底层内核。',
        'v1失败、两次梯度诊断、v2全部逐题回答和训练记录完整保留，未放宽梯度容差、另换种子或加训。',
        f"复核72份回答和提示，保护的{verification['protected_original_files_unchanged']}项原文件摘要不变；正式保留测试未打开。", '',
        '## 解释边界', '',
        '本次只能判断一个便宜输入方案是否值得扩大验证。12题、一个种子的差异不能证明普遍增益或过程教学效果。',
        '既有开发题曾被反复检查，不能用于新主方案的最终效果声明；需要重新冻结方案并保留正式测试。',
        '误报是结果，不作为项目继续的硬门槛。原九组长训练仍保持暂停。', '']
    (ROOT/'REPORT_ZH.md').write_text('\n'.join(content), encoding='utf-8')


if __name__ == '__main__':
    result, verification = verify()
    gradients = trained_gradient_check()
    report(result, verification, gradients)
    print(json.dumps({'summary': {k: v['summary'] for k, v in result['results'].items()},
                      'verification': verification, 'trained_gradients': gradients}, ensure_ascii=False, indent=2))
