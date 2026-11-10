"""Verify the no-training base-model probe and summarize the four observed states."""
import json
from pathlib import Path

from build_publication_curriculum import atomic_json, read, sha
from discovery.interface_alignment import neutral_session, practice_cases
from verify_interface_alignment import lines, replay

ROOT = Path('results/base_interface_probe_v1')


def run():
    config, result = read(ROOT/'config.json'), read(ROOT/'result.json')
    if not result['completed'] or result['config_sha256'] != sha(ROOT/'config.json'):
        raise ValueError('Incomplete base probe.')
    for field in ('protected_files', 'sources'):
        if any(sha(p) != h for p, h in config[field].items()):
            raise ValueError('A protected input or source changed.')
    if any(sha(ROOT/'sources'/p) != h for p, h in config['sources'].items()):
        raise ValueError('Saved source snapshot changed.')
    cases = practice_cases('validation')
    original_prompts = {x['tag'].removeprefix('neutral_'): x['messages']
                        for x in lines('results/interface_alignment_v1/before/calls.jsonl')
                        if x['event'] == 'request_started' and x['tag'].startswith('neutral_')}
    diagnostics = {}
    total_replayed = 0
    for name in config['models']:
        rows = lines(ROOT/name/'answers.jsonl')
        calls = lines(ROOT/name/'calls.jsonl')
        if len(rows) != 16 or len(calls) != 32 or [r['id'] for r in rows] != config['cases']:
            raise ValueError('Incomplete base-model rows.')
        core_match = empty_batches = duplicated_alternatives = 0
        for index, (case, row) in enumerate(zip(cases, rows)):
            request, reply = calls[2*index:2*index+2]
            if request['messages'] != case['prompt'] or request['messages'] != original_prompts[case['id']]:
                raise ValueError('Prompts are not identical across the four measured states.')
            if reply['reply'] != row['reply'] or request['max_new_tokens'] != 768 or row['reply']['output_tokens'] > 768:
                raise ValueError('Actual reply or inference budget mismatch.')
            try:
                replay(neutral_session(case), case['current_structures'], row['reply']['text'], 2, row)
            except (ValueError, KeyError, TypeError, IndexError):
                if row['interface_accepted']:
                    raise
            else:
                if not row['interface_accepted']:
                    raise ValueError('Valid plan was labelled invalid.')
            if row['experiment_executed']:
                total_replayed += row['used']-8
            try:
                plan = json.loads(row['reply']['text'])
                expected = case['answer']
                keys = ('action',) if expected['action'] == 'finish' else ('action', 'alternative_structures', 'queries')
                core_match += all(plan.get(k) == expected[k] for k in keys)
                empty_batches += plan.get('action') == 'experiment' and plan.get('queries') == []
            except (ValueError, TypeError, AttributeError):
                pass
            duplicated_alternatives += row.get('error') == 'Alternatives must differ after structural canonicalization.'
        summary = result['results'][name]
        for key in ('schema_valid', 'exact_plan', 'requested_action_correct', 'interface_accepted', 'experiment_executed'):
            if sum(bool(r[key]) for r in rows) != summary[key]:
                raise ValueError('A summary count differs from the recorded answers.')
        for key in ('input_tokens', 'output_tokens', 'generation_seconds'):
            if sum(r['reply'][key] for r in rows) != summary[key]:
                raise ValueError('Inference cost differs from actual outputs.')
        events = lines(ROOT/name/'events.jsonl')
        measured = [r for r in rows if r['experiment_executed']]
        if len(events) != len(measured)*2:
            raise ValueError('Missing committed/revealed event pair.')
        for index, row in enumerate(measured):
            commitment, observation = events[2*index:2*index+2]
            if (commitment['event'] != 'predictions_committed' or observation['event'] != 'observations_revealed'
                    or commitment['commitment'] != row['commitment'] or observation['observation'] != row['observation']):
                raise ValueError('Measurement was not recorded in commitment/reveal order.')
        diagnostics[name] = {'core_plan_match_excluding_prose': int(core_match), 'empty_query_arrays': int(empty_batches),
                             'duplicate_primary_and_alternative': int(duplicated_alternatives)}
    verification = {'passed': True, 'answers_checked': 32, 'all_prompts_identical_to_prior_05_checks': True,
                    'additional_measurements_replayed': total_replayed, 'training_performed': False,
                    'original_main_still_paused': True, 'formal_tests_used': False,
                    'diagnostics': diagnostics, 'verifier_sha256': sha(__file__)}
    atomic_json(ROOT/'verification.json', verification)
    (ROOT/'verification_source.py').write_bytes(Path(__file__).read_bytes())
    prior = read('results/interface_alignment_v1/result.json')
    table = [('0.5B基础模型，无项目训练', result['results']['base_05']),
             ('0.5B：24题终答训练后', prior['before']['neutral']),
             ('0.5B：再加48步共用练习', prior['after']['neutral']),
             ('1.5B基础模型，无项目训练', result['results']['base_15'])]
    text = ['# 基础模型接口探针：没有训练新模型', '',
        '两种基础模型在本机完成同样16道中立接口题，使用相同提示、768输出token与零重试。',
        '这里只要求按给定计划转换字段，并不要求发现隐藏机制或设计新实验。', '',
        '|模型状态|计划逐项完全正确|通过接口（含结束）|要求测量且实际执行|', '|---|---:|---:|---:|']
    for name, row in table:
        text.append(f"|{name}|{row['exact_plan']}/16|{row['interface_accepted']}/16|{row['requested_experiments_executed']}/12|")
    text += ['', '0.5B基础模型在全部16题输出了空queries，未通过测量批次检查。',
             '1.5B的结束/实验动作与要求一致，但8题给出的备选结构与当前结构重复，实验室正确拒绝。',
             '其输出经常沿用格式示例的空备选、理由与更新条件；一些双测量任务也只提交一个构型。',
             f"若只比较动作、备选结构与构型而不要求理由逐字相同，1.5B为{diagnostics['base_15']['core_plan_match_excluding_prose']}/16。",
             '这个附加诊断不能取代事前约定的完整复制分数，也不是科学发现成功率。', '',
             '## 对下一步的含义', '',
             '目前没有证据支持直接改用1.5B后就能解决问题。先检查冗长提示及具体格式示例是否干扰模型执行实际任务。',
             '这些接口题在“按给定计划转换”之后附加了允许自主结束的通用计划提示，需要在下一开发版本中消除这种任务表述歧义。',
             '本次所有旧分数保持原样，不能把改写提示后的成绩当作同一检查。', '',
             f"0.5B基础模型推理{result['results']['base_05']['generation_seconds']:.1f}秒，"
             f"1.5B基础模型推理{result['results']['base_15']['generation_seconds']:.1f}秒；未执行梯度训练。",
             '1.5B短上下文推理已能在本机运行；22,982-token训练资源探针与正式多模型稳健性研究仍未执行。',
             '原主训练暂停状态、模型与环境保持不变。正式保留测试未打开，全部32个回答与计费记录已复核。', '']
    (ROOT/'REPORT_ZH.md').write_text('\n'.join(text), encoding='utf-8')
    print(json.dumps({'results': result['results'], 'verification': verification}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
