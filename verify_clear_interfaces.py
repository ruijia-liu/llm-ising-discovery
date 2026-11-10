import json
from pathlib import Path

from build_publication_curriculum import atomic_json, read, sha
from discovery.action_prompt_v2 import practice_cases_v2
from discovery.interface_alignment import neutral_session
from verify_interface_alignment import lines, replay

ROOT = Path('results/clear_interface_probe_v1')


def run():
    result, config = read(ROOT/'result.json'), read(ROOT/'config.json')
    if not result['completed'] or result['config_sha256'] != sha(ROOT/'config.json'):
        raise ValueError('Incomplete wording probe.')
    for field in ('sources', 'protected_files'):
        if any(sha(p) != h for p, h in config[field].items()):
            raise ValueError('Original sources or results changed.')
    cases = practice_cases_v2('validation')
    if cases != read(ROOT/'cases.json'):
        raise ValueError('Revised case generation changed.')
    charged = 0
    for name, summary in result['results'].items():
        rows, calls = lines(ROOT/name/'answers.jsonl'), lines(ROOT/name/'calls.jsonl')
        if len(rows) != 16 or len(calls) != 32 or [r['id'] for r in rows] != config['case_ids']:
            raise ValueError('Missing paired cases or inference responses.')
        measured = []
        for index, (case, row) in enumerate(zip(cases, rows)):
            request, reply = calls[2*index:2*index+2]
            if request['messages'] != case['prompt'] or reply['reply'] != row['reply'] or request['max_new_tokens'] != 768:
                raise ValueError('Prompt, reply or budget mismatch.')
            try:
                replay(neutral_session(case), case['current_structures'], row['reply']['text'], 2, row)
            except (ValueError, TypeError, KeyError, IndexError):
                if row['interface_accepted']:
                    raise
            else:
                if not row['interface_accepted']:
                    raise ValueError('Valid plan marked invalid.')
            try:
                exact = json.loads(row['reply']['text']) == case['answer']
            except ValueError:
                exact = False
            if exact != row['exact_plan']:
                raise ValueError('Plan-copy score differs.')
            if row['experiment_executed']:
                charged += row['used']-8
                measured.append(row)
        for key in ('schema_valid', 'exact_plan', 'requested_action_correct', 'interface_accepted', 'experiment_executed'):
            if sum(bool(r[key]) for r in rows) != summary[key]:
                raise ValueError('Aggregate score mismatch.')
        for key in ('input_tokens', 'output_tokens', 'generation_seconds'):
            if sum(r['reply'][key] for r in rows) != summary[key]:
                raise ValueError('Aggregate cost mismatch.')
        events = lines(ROOT/name/'events.jsonl')
        if len(events) != 2*len(measured):
            raise ValueError('Missing measurement event pair.')
        for index, row in enumerate(measured):
            commit, reveal = events[2*index:2*index+2]
            if (commit['event'] != 'predictions_committed' or reveal['event'] != 'observations_revealed'
                    or commit['commitment'] != row['commitment'] or reveal['observation'] != row['observation']):
                raise ValueError('Prediction-before-observation log mismatch.')
    verification = {'passed': True, 'answers_checked': 32, 'measurements_replayed': charged,
                    'old_results_unchanged': True, 'formal_tests_used': False, 'training_performed': False,
                    'source_sha256': sha(__file__)}
    atomic_json(ROOT/'verification.json', verification)
    (ROOT/'verification_source.py').write_bytes(Path(__file__).read_bytes())
    old = read('results/base_interface_probe_v1/result.json')['results']
    text = ['# 去掉具体例子、明确任务后的接口检查', '',
        '没有训练模型。两种基础模型分别重做同样16道给定计划题，只修改提示的措辞和组织，目标与构型值均不变。',
        '新提示明确要求按给定计划执行，删除具体机制、构型和理由的格式示例；接口解析与计费标准不变。', '',
        '|模型|旧提示：完全正确|新提示：完全正确|旧：要求测量且执行|新：要求测量且执行|', '|---|---:|---:|---:|---:|']
    for name, label in (('base_05', '0.5B基础模型'), ('base_15', '1.5B基础模型')):
        before, after = old[name], result['results'][name]
        text.append(f"|{label}|{before['exact_plan']}/16|{after['exact_plan']}/16|{before['requested_experiments_executed']}/12|{after['requested_experiments_executed']}/12|")
    text += ['', '## 解释', '',
             '这些题已经提供全部计划，成功只说明能按要求提交，不代表自主选实验、恢复机制或过程教学有效。',
             '本次同时删除具体示例、缩短提示、明确复制任务的优先级，不能单独归因于其中一项。',
             '题目已用于开发，不是新确认集；旧结果与失败全部保留。没有因为分数再换措辞或追加训练。',
             '较短的真实自主规划提示函数也已实现，并验证它保留原公开数据负载与允许结束的权利；',
             '但本轮未测试它是否提高自主取证或机制正确率，不能将本表外推到真实研究行为。', '',
             '|模型|旧输入token|新输入token|旧生成秒|新生成秒|', '|---|---:|---:|---:|---:|']
    for name, label in (('base_05', '0.5B'), ('base_15', '1.5B')):
        a, b = old[name], result['results'][name]
        text.append(f"|{label}|{a['input_tokens']}|{b['input_tokens']}|{a['generation_seconds']:.1f}|{b['generation_seconds']:.1f}|")
    text += ['', '正确执行需要输出完整构型，不能把原先空计划生成得快当作有效实验提速。',
             f'32个回答全部复核，{charged}次计费测量重放一致，事前承诺与揭示日志完整。',
             '原九组长训练仍暂停；原主提示、冻结协议和模型均未替换。正式保留测试没有打开。', '']
    (ROOT/'REPORT_ZH.md').write_text('\n'.join(text), encoding='utf-8')
    print(json.dumps({'result': result, 'verification': verification}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
