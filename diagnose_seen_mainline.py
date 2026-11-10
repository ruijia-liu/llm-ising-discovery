"""Read-only current-model check on exact, demonstrably seen training tasks."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import torch
from safetensors.torch import load_file

from build_publication_curriculum import atomic_json, main_worlds, read, sha
from discovery.mechanism_scoring import independent_features, score_answer
from discovery.model_transport import ModelWorker
from discovery.student_mainline import evidence_prefix, terminal_messages
from discovery.teaching import expression_from_dict
from evaluate_mainline import fixed_prediction


MAIN = Path('results/student_mainline_v1')
PRIOR = Path('results/current_student_diagnostic_v2')
SOURCE = Path('data/research_curriculum_publication_v1')
MODEL = Path('models/Qwen2.5-0.5B-Instruct')


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Preserve prior diagnostic outputs, including failures.')
    if read(MAIN/'queue_state.json')['status'] != 'stopped' or not (MAIN/'B_seed401/STOP_AFTER_CHECKPOINT').exists():
        raise ValueError('This diagnostic requires the preserved user-requested training stop.')
    checkpoint = MAIN/'B_seed401/step_checkpoint.pt'
    prior = read(PRIOR/'config.json')
    if sha(checkpoint) != prior['checkpoint_sha256']:
        raise ValueError('The checkpoint differs from the stopped snapshot.')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    step = saved['step']
    if step != 215 or prior['B_step'] != step:
        raise ValueError('This named diagnostic is for the observed step-215 stop only.')
    adapter_b = PRIOR/'B_adapter'
    tensors = load_file(adapter_b/'adapter_model.safetensors', device='cpu')
    if set(tensors) != set(saved['adapter']) or not all(torch.equal(tensors[k], saved['adapter'][k]) for k in tensors):
        raise ValueError('B adapter does not match every checkpoint tensor.')
    config_b = read(MAIN/'B_seed401/config.json')
    records = [json.loads(line) for line in (MAIN/'B_seed401/training.jsonl').read_text(encoding='utf-8').splitlines()]
    if records != saved['records'] or [r['step'] for r in records] != list(range(1, step+1)):
        raise ValueError('Checkpoint and committed learning history disagree.')
    if [r['episode_id'] for r in records] != [r['episode_id'] for r in config_b['schedule'][:step]]:
        raise ValueError('Observed learning history differs from the frozen task plan.')
    seen = {r['episode_id']: r['step'] for r in records}
    worlds = [w for w in main_worlds() if w['split']=='train']
    selected = []
    for family in ('field', 'direction', 'distance_two', 'three_body', 'mixed', 'null'):
        candidates = sorted((w for w in worlds if w['family']==family and
                             all(f"{w['world_id']}-sigma{s}" in seen for s in (0, 1))), key=lambda w:w['world_id'])
        if not candidates:
            raise ValueError('No world with both noise episodes demonstrably seen in family '+family)
        w = candidates[0]
        selected.append({**w, 'B_committed_steps': {str(s): seen[f"{w['world_id']}-sigma{s}"] for s in (0, 1)},
                         'eligible_worlds_in_family': len(candidates)})
    wanted = {f"{w['world_id']}-sigma{s}" for w in selected for s in (0, 1)}
    views = {}
    with (SOURCE/'views/train/B.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            view=json.loads(line)
            if view['episode_id'] in wanted:
                views[view['episode_id']]=view
    if set(views)!=wanted:
        raise ValueError('A selected original teaching view is missing.')
    messages = {f'{episode}_terminal{budget:02d}':terminal_messages(view,budget)
                for episode,view in views.items() for budget in (8,32)}
    found=set()
    with Path('data/student_mainline_v1/messages.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            entry=json.loads(line)
            if entry['key'] in messages:
                if entry['messages']!=messages[entry['key']]:
                    raise ValueError('An input or target differs from its actual cached training task.')
                found.add(entry['key'])
    if found!=set(messages):
        raise ValueError('Selected tasks are missing from the original cache.')
    result_a=read(MAIN/'A_seed401/result.json')
    adapter_a=MAIN/'A_seed401/adapter'
    if not result_a['completed'] or result_a['completed_updates']!=1200 or sha(adapter_a/'adapter_model.safetensors')!=result_a['adapter_sha256']:
        raise ValueError('A final adapter provenance failed.')
    if torch.cuda.is_initialized():
        raise ValueError('Preparation unexpectedly initialized CUDA in the controller.')
    config={'scope':'extra_seen_training_diagnostic_only','A_step':1200,'B_step':step,
            'selection':'lexicographically first world per family with both noise episodes committed by B step215',
            'worlds':[{k:v for k,v in w.items() if k!='truth'} for w in selected],
            'sigmas':[0,1],'budgets':{'A':8,'B':32},'evaluation_seed':99131,'features_per_distribution':256,
            'B_adapter_tensors_identical_to_checkpoint':len(tensors),'all_24_cached_task_messages_identical':True,
            'input_sha256':{str(p):sha(p) for p in (checkpoint,MAIN/'B_seed401/config.json',MAIN/'B_seed401/training.jsonl',
                           adapter_a/'adapter_model.safetensors',adapter_b/'adapter_model.safetensors',
                           SOURCE/'views/train/B.jsonl',Path('data/student_mainline_v1/manifest.json'))},
            'source_sha256':{p:sha(p) for p in ('diagnose_seen_mainline.py','CURRENT_SEEN_DIAGNOSTIC_PROTOCOL.md',
                            'discovery/student_mainline.py','discovery/student_v3.py','discovery/mechanism_scoring.py',
                            'student_inference_worker.py','discovery/model_transport.py','evaluate_mainline.py')},
            'formal_tests_used':False,'training_resumed':False,'model_selection':False}
    atomic_json(output/'config.json',config)
    (output/'sources').mkdir()
    for name in ('diagnose_seen_mainline.py','CURRENT_SEEN_DIAGNOSTIC_PROTOCOL.md'):
        (output/'sources'/name).write_bytes(Path(name).read_bytes())
    del saved,tensors
    thresholds=read('results/structure_discovery_noise/calibration.json')['thresholds']
    rows=[]; started=time.perf_counter()
    try:
        for group,adapter in (('A',adapter_a),('B',adapter_b)):
            budget=config['budgets'][group]
            with ModelWorker(MODEL,adapter,log_path=output/f'{group}_worker.log') as worker:
                for w in selected:
                    features=independent_features(99131,w['world_index'])
                    for sigma in (0,1):
                        episode=f"{w['world_id']}-sigma{sigma}"
                        view=views[episode]
                        truth=expression_from_dict(view['final_answer']['expression'])
                        evidence=evidence_prefix(view,budget)
                        prediction=fixed_prediction(evidence,budget,worker)
                        score=score_answer(truth,prediction['answer'],features,sigma,thresholds)
                        row={'group':group,'world':w['world_id'],'family':w['family'],'sigma':sigma,'budget':budget,
                             'nonempty':bool(truth.terms),'B_committed_step_for_this_episode':w['B_committed_steps'][str(sigma)],
                             'prompt':messages[f'{episode}_terminal{budget:02d}'][:-1],
                             'prediction':prediction,'score':score}
                        rows.append(row)
                        with (output/'episodes.jsonl').open('a',encoding='utf-8') as handle:
                            handle.write(json.dumps(row,ensure_ascii=False)+'\n')
                        print(f"{group} {episode}: raw={score['raw_support_success']} full={score['full_recovery']} valid={score['valid_expression']}",flush=True)
        if any(sha(p)!=h for p,h in config['input_sha256'].items()) or any(sha(p)!=h for p,h in config['source_sha256'].items()):
            raise ValueError('An input or source changed during diagnosis.')
    except BaseException as error:
        atomic_json(output/'failure.json',{'completed':False,'type':type(error).__name__,'error':str(error),'completed_answers':len(rows)})
        raise
    summary=[]
    for group in ('A','B'):
        for sigma in (0,1):
            selected_rows=[r for r in rows if r['group']==group and r['sigma']==sigma]
            summary.append({'group':group,'sigma':sigma,'n':len(selected_rows),
                **{k:sum(bool(r['score'][k]) for r in selected_rows) for k in ('valid_expression','raw_support_success','full_recovery','prediction_success','false_discovery')},
                'nonempty_n':sum(r['nonempty'] for r in selected_rows),
                'nonempty_recovered':sum(r['nonempty'] and r['score']['full_recovery'] for r in selected_rows)})
    repeated={}
    for group in ('A','B'):
        counts=Counter(json.dumps(r['prediction']['answer']['raw_structures'],sort_keys=True) for r in rows if r['group']==group and r['prediction']['answer'] is not None)
        repeated[group]={'unique_raw_proposals':len(counts),'most_common':counts.most_common(1),'valid_answers':sum(counts.values())}
    result={'completed':True,'scope':config['scope'],'summary':summary,'raw_proposal_counts':repeated,
            'answers':len(rows),'elapsed_seconds':time.perf_counter()-started,'formal_tests_used':False,
            'training_resumed':False,'config_sha256':sha(output/'config.json'),'episodes_sha256':sha(output/'episodes.jsonl')}
    atomic_json(output/'result.json',result)
    lines=['# 已实际训练过的原题抽查','','A为1200步，B为暂停的215步；没有恢复训练或使用正式测试。',
           '按家族和已提交的训练记录选题，全部选中世界的两种噪声轨迹都确实出现过。提示与原缓存逐项相同。','',
           '|模型|噪声|合法|原始结构正确|非空完整恢复|全部完整恢复|空世界误报|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"|{r['group']}|{r['sigma']}|{r['valid_expression']}/6|{r['raw_support_success']}/6|{r['nonempty_recovered']}/5|{r['full_recovery']}/6|{r['false_discovery']}/1|")
    lines+=['','仅六个物理世界、每世界两种噪声。训练进度和可见预算不同，不用此表推断A/B优劣。',
            '原题失败提示当前学习不足；原题与开发成绩差异还混合了参数及教师主动/随机取证分布差异。',
            'B仅训练约18%，C尚未训练；不能预言1200步结果或宣称过程教学无效。',
            '原始回答、拟合、独立预测分数、实际出现步数和提示均保留；没有挑最佳checkpoint。','']
    (output/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    print(output/'REPORT_ZH.md',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
