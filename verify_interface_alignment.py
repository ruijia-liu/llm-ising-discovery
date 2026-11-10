"""Re-score saved replies, replay charged measurements, and report the alignment pilot."""
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
from discovery.active_student import parse_hypothesis
from discovery.interface_alignment import neutral_session, practice_cases, prepare_plan
from discovery.lab import Laboratory
from discovery.mechanism_scoring import independent_features, score_answer
from discovery.student_laboratory import StudentLaboratory
from discovery.student_v2 import encode, load_views, packed
from discovery.teaching import decode_query, expression_from_dict
from run_interface_alignment import MODEL, check_frozen

ROOT = Path('results/interface_alignment_v1')


def lines(path):
    path = Path(path)
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []


def replay(session, primary, text, maximum, row):
    plan, commitment = prepare_plan(session, primary, text, maximum)
    if commitment is None:
        if row['experiment_executed']:
            raise ValueError('Finish was counted as a measurement.')
        return
    # Runtime support tuples become lists on the public JSON boundary.
    if session.evidence()['used'] != 8 or json.loads(packed(commitment)) != row['commitment']:
        raise ValueError('Premeasurement commitment cannot be reproduced from the visible evidence.')
    observation = session.reveal(commitment['ticket'])
    if observation != row['observation'] or row['used'] != 8+len(plan['queries']):
        raise ValueError('Measured data, commitment hash or accounting differs.')


def verify():
    config, result = read(ROOT/'config.json'), read(ROOT/'result.json')
    if not result['completed'] or result['config_sha256'] != sha(ROOT/'config.json'):
        raise ValueError('Incomplete alignment result.')
    check_frozen(ROOT, config)
    if {d.metadata['Name']: d.version for d in importlib.metadata.distributions()} != config['environment']:
        raise ValueError('Runtime package versions changed.')
    if any(sha(ROOT/'sources'/p) != h for p, h in config['source_sha256'].items()):
        raise ValueError('Saved source snapshot differs.')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    entries = lines(ROOT/'messages.jsonl')
    with np.load(ROOT/'training.npz', allow_pickle=False) as archive:
        for index, entry in enumerate(entries):
            encoded = encode(tokenizer, entry['messages'])
            if not np.array_equal(encoded['ids'], archive[f'{index}_ids']) or not np.array_equal(encoded['labels'], archive[f'{index}_labels']):
                raise ValueError('Cached input tokens or assistant supervision changed.')
    updates = lines(ROOT/'training/updates.jsonl')
    if len(updates) != 48 or [r['indices'] for r in updates] != config['schedule']:
        raise ValueError('The fixed schedule was not completed.')
    counts = Counter(i for r in updates for i in r['indices'])
    if counts != Counter({**{i: 1 for i in range(96)}, **{i: 3 for i in range(96, 128)}}):
        raise ValueError('Training exposures differ from the fixed plan.')
    for metric in ('input_tokens', 'supervised_tokens'):
        if updates[-1][metric] != config['actual_training_budget'][metric] or updates[-1][metric] != result['training'][metric]:
            raise ValueError('Token accounting mismatch.')
    checkpoint = torch.load(ROOT/'training/checkpoint.pt', map_location='cpu', weights_only=True)
    adapter = load_file(ROOT/'training/adapter/adapter_model.safetensors')
    if checkpoint['step'] != 48 or set(adapter) != set(checkpoint['adapter']) or any(not torch.equal(adapter[k], checkpoint['adapter'][k]) for k in adapter):
        raise ValueError('Evaluation adapter is not the final fixed checkpoint.')
    if sha(ROOT/'training/adapter/adapter_model.safetensors') != result['after']['adapter_sha256']:
        raise ValueError('Evaluated adapter hash mismatch.')
    views = {v['episode_id']: v for v in load_views('validation')}
    cases = {c['id']: c for c in practice_cases('validation')}
    verified_answers = replayed_experiments = replayed_measurements = 0
    phase_rows = {}
    for phase in ('before', 'after'):
        folder = ROOT/phase
        rows = {kind: lines(folder/(kind+'.jsonl')) for kind in ('neutral', 'fixed', 'probes')}
        if ([r['id'] for r in rows['neutral']] != list(cases)
                or [r['episode'] for r in rows['fixed']] != config['fixed_validation_ids']
                or [r['episode'] for r in rows['probes']] != config['one_plan_validation_ids']):
            raise ValueError('Incomplete or changed diagnostic cohorts.')
        ledger = lines(folder/'calls.jsonl')
        if len(ledger) != result[phase]['inference']['calls']*2:
            raise ValueError('Unmatched model request or reply.')
        replies = []
        for i in range(0, len(ledger), 2):
            request, reply = ledger[i:i+2]
            if request['event'] != 'request_started' or reply['event'] != 'reply_received' or request['number'] != reply['number']:
                raise ValueError('Broken generation ledger.')
            if not 0 <= reply['reply']['output_tokens'] <= request['max_new_tokens']:
                raise ValueError('Inference budget exceeded.')
            replies.append(reply['reply'])
        for metric in ('input_tokens', 'output_tokens', 'generation_seconds'):
            if sum(r[metric] for r in replies) != result[phase]['inference'][metric]:
                raise ValueError('Inference cost mismatch.')
        for row in rows['neutral']:
            case = cases[row['id']]
            try:
                replay(neutral_session(case), case['current_structures'], row['reply']['text'], 2, row)
            except (ValueError, KeyError, TypeError, IndexError):
                if row['interface_accepted']:
                    raise
            else:
                if not row['interface_accepted']:
                    raise ValueError('A valid neutral action was labelled invalid.')
                if json.loads(row['reply']['text']) == case['answer'] and not row['exact_plan']:
                    raise ValueError('Plan-copy score mismatch.')
            verified_answers += 1
        for row in rows['fixed']:
            view = views[row['episode']]
            truth = expression_from_dict(view['final_answer']['expression'])
            expected = score_answer(truth, row['prediction']['answer'], independent_features(98621, int(row['episode'][1:5])), 0, {})
            if expected != row['score']:
                raise ValueError('Fixed-evidence independent score mismatch.')
            if row['prediction']['answer'] is not None:
                if parse_hypothesis(row['prediction']['reply']['text']) != row['prediction']['answer']['raw_structures']:
                    raise ValueError('Scored structures differ from the actual answer.')
            verified_answers += 1
        for row in rows['probes']:
            view = views[row['episode']]
            truth = expression_from_dict(view['final_answer']['expression'])
            if row['initial']['answer'] is not None:
                session = StudentLaboratory(Laboratory(truth.delta, noise_sigma=0, noise_seed=0),
                            [decode_query(q) for q in view['initial']['queries']])
                try:
                    replay(session, row['initial']['answer']['raw_structures'], row['plan_reply']['text'], 4, row)
                except (ValueError, KeyError, TypeError, IndexError):
                    if row['status'] != 'invalid_plan':
                        raise
                else:
                    if row['status'] == 'invalid_plan':
                        raise ValueError('Valid autonomous plan marked invalid.')
            expected = score_answer(truth, row['after']['answer'] if row['after'] else None,
                                   independent_features(98621, int(row['episode'][1:5])), 0, {})
            if expected != row['score']:
                raise ValueError('Post-probe hypothesis score mismatch.')
            verified_answers += 1
        measured = [r for kind in ('neutral', 'probes') for r in rows[kind] if r['experiment_executed']]
        commitments, reveals = lines(folder/'commitments.jsonl'), lines(folder/'reveals.jsonl')
        if len(measured) != len(commitments) or len(measured) != len(reveals):
            raise ValueError('Missing measurement audit records.')
        for r in measured:
            if not any(c['commitment'] == r['commitment'] for c in commitments) or not any(o['observation'] == r['observation'] for o in reveals):
                raise ValueError('In-memory results differ from the locked/revealed records.')
        replayed_experiments += len(measured)
        replayed_measurements += sum(r['used']-8 for r in measured)
        phase_rows[phase] = rows
    verification = {'passed': True, 'diagnostic_rows_checked': verified_answers, 'training_tasks_retokenized': len(entries),
                    'all_48_updates_and_task_exposures_matched': True, 'adapter_tensors_match_final_checkpoint': len(adapter),
                    'experiments_replayed': replayed_experiments, 'measured_flips_replayed': replayed_measurements,
                    'protected_original_files_unchanged': len(config['protected_sha256']),
                    'runtime_packages_unchanged': len(config['environment']), 'main_training_still_paused': True,
                    'formal_tests_used': False, 'verifier_sha256': sha(__file__)}
    atomic_json(ROOT/'verification.json', verification)
    (ROOT/'verification_source.py').write_bytes(Path(__file__).read_bytes())
    return result, verification, phase_rows


def report(result, verification, phase_rows):
    before, after = result['before'], result['after']
    rows = ['# 共用接口对齐：小规模前后检查', '',
        '本轮是未来B/C共有的接口准备，不是两组教学效果比较。原九组主训练仍暂停，正式测试未打开。',
        '起点为上轮raw小模型；新增48步训练，混合真实部署提示的96份机制前缀任务与96次接口练习。',
        '接口练习只转换已经给定的计划，不能把复制成功当作自主选实验或发现规律。', '',
        '|指标|练习前|练习后|', '|---|---:|---:|',
        f"|给定计划完全正确|{before['neutral']['exact_plan']}/16|{after['neutral']['exact_plan']}/16|",
        f"|给定计划通过实验接口|{before['neutral']['interface_accepted']}/16|{after['neutral']['interface_accepted']}/16|",
        f"|明确要求测量且实际执行|{before['neutral']['requested_experiments_executed']}/12|{after['neutral']['requested_experiments_executed']}/12|",
        f"|开发题固定证据：原始结构正确|{before['fixed']['raw_support_success']}/12|{after['fixed']['raw_support_success']}/12|",
        f"|开发题固定证据：拟合后严格恢复|{before['fixed']['full_recovery']}/12|{after['fixed']['full_recovery']}/12|",
        f"|开发题固定证据：非空机制恢复|{before['fixed']['nonempty_recovered']}/10|{after['fixed']['nonempty_recovered']}/10|",
        f"|自主一轮探针：实际追加实验|{before['probes']['experiments_executed']}/6|{after['probes']['experiments_executed']}/6|",
        f"|自主一轮探针：新增计费测量数|{before['probes']['additional_measurements']}|{after['probes']['additional_measurements']}|",
        f"|自主一轮探针：主动结束|{before['probes']['student_finished']}/6|{after['probes']['student_finished']}/6|",
        f"|自主一轮探针：无效假设或计划|{before['probes']['invalid']}/6|{after['probes']['invalid']}/6|", '',
        '自主探针只允许一次计划，不能与正式32预算的自由探索成功率混用。其最终更新假设也不是学生完整探索后的终答。',
        '两边固定证据提示均为真实部署hypothesis_prompt，重新测量基准；不能与前一轮“实验结束”措辞的分数直接相减。', '',
        '|资源|练习前检查|练习后检查|', '|---|---:|---:|',
        f"|模型调用|{before['inference']['calls']}|{after['inference']['calls']}|",
        f"|生成token|{before['inference']['output_tokens']}|{after['inference']['output_tokens']}|",
        f"|检查墙钟秒|{before['elapsed_seconds']:.1f}|{after['elapsed_seconds']:.1f}|", '',
        f"48步训练墙钟{result['training']['training_wall_seconds']:.1f}秒；输入{result['training']['input_tokens']:,} token，"
        f"监督{result['training']['supervised_tokens']:,} token；本轮运行总墙钟{result['elapsed_seconds']:.1f}秒（不含复核）。", '',
        '## 误报与限制', '',
        f"固定证据的空世界原始误报：{before['fixed']['raw_false_proposal']}/2 → {after['fixed']['raw_false_proposal']}/2；"
        f"拟合后误报：{before['fixed']['false_discovery']}/2 → {after['fixed']['false_discovery']}/2。",
        '保留错误和无效回答，不要求零误报。全部例题、模型原始输出、计划、事前预测和计费结果均在目录中。',
        '前后比较同时增加前缀提示监督、接口练习和训练量，没有同量额外训练的消融对照，不能把变化单独归因于某一因素。',
        '只有一个开发选定的起点和训练种子、12个已被查看的参数变化世界；没有新结构或噪声检验。',
        '中立接口练习只展示已有的空机制与单点场，可能影响自由探针的候选偏好，需保留输出作分析。',
        '若进入B/C实验，两组必须共享完全相同的准备，仍需单独比较过程教学的增益。', '',
        f"已复核{verification['diagnostic_rows_checked']}条诊断记录，重建{verification['training_tasks_retokenized']}份训练token/监督掩码，"
        f"重放{verification['experiments_replayed']}次实验、{verification['measured_flips_replayed']}次计费测量。",
        f"最后48步检查点与适配器的{verification['adapter_tensors_match_final_checkpoint']}个张量一致；"
        f"{verification['protected_original_files_unchanged']}项原文件、{verification['runtime_packages_unchanged']}个包版本不变。", '']
    (ROOT/'REPORT_ZH.md').write_text('\n'.join(rows), encoding='utf-8')


if __name__ == '__main__':
    result, verification, rows = verify()
    report(result, verification, rows)
    print(json.dumps({'before': result['before'], 'after': result['after'], 'training': result['training'],
                      'verification': verification}, ensure_ascii=False, indent=2))
