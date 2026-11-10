"""Small descriptive development check; no formal tests or model selection."""
import json
from pathlib import Path
import torch
from peft import set_peft_model_state_dict
from train_student_pilot import make_student
from build_publication_curriculum import atomic_json, read, sha
from discovery.evaluation_worlds import load_worlds, streams, fixed_evidence
from discovery.mechanism_scoring import independent_features, score_answer
from discovery.teaching import expression_from_dict
from discovery.model_transport import ModelWorker
from evaluate_mainline import fixed_prediction

ROOT = Path('results/current_student_diagnostic_v2')

def run():
    ROOT.mkdir(exist_ok=False)
    source = Path('results/student_mainline_v1/B_seed401/step_checkpoint.pt')
    snapshot = ROOT/'B_checkpoint.pt'
    snapshot.write_bytes(source.read_bytes())
    saved = torch.load(snapshot, map_location='cpu', weights_only=True)
    step = saved['step']
    model = make_student('models/Qwen2.5-0.5B-Instruct', 401)
    set_peft_model_state_dict(model, saved['adapter'])
    model.save_pretrained(ROOT/'B_adapter', safe_serialization=True)
    del model, saved
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    worlds = [w for i,w in enumerate(load_worlds('development')) if i%10 == 0]
    atomic_json(ROOT/'config.json', {'scope':'development_diagnostic_only', 'B_step':step,
        'A_step':1200,'worlds':[w['world_id'] for w in worlds], 'sigmas':[0,1],
        'checkpoint_sha256':sha(snapshot),'formal_tests_used':False,
        'note':'Unequal training progress; descriptive check, not an A/B comparison or checkpoint selection.'})
    thresholds = read('results/structure_discovery_noise/calibration.json')['thresholds']
    rows=[]
    for group,adapter in [('A',Path('results/student_mainline_v1/A_seed401/adapter')),('B',ROOT/'B_adapter')]:
        with ModelWorker('models/Qwen2.5-0.5B-Instruct',adapter,log_path=ROOT/f'{group}_worker.log') as worker:
            for world in worlds:
                truth=expression_from_dict(world['truth'])
                features=independent_features(98711,world['world_index'])
                for sigma in (0,1):
                    evidence,_=fixed_evidence(truth.delta,sigma,streams('development',world['world_index']))
                    budget=8 if group=='A' else 32
                    prediction=fixed_prediction(evidence,budget,worker)
                    score=score_answer(truth,prediction['answer'],features,sigma,thresholds)
                    row={'group':group,'world':world['world_id'],'family':world['family'],
                         'sigma':sigma,'nonempty':bool(truth.terms),'budget':budget,
                         'prediction':prediction,'score':score}
                    rows.append(row)
                    with (ROOT/'episodes.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps(row,ensure_ascii=False)+'\n')
                    print(json.dumps({'group':group,'world':world['world_id'],'sigma':sigma,**score}),flush=True)
    summary=[]
    for group in ('A','B'):
        for sigma in (0,1):
            subset=[r for r in rows if r['group']==group and r['sigma']==sigma]
            summary.append({'group':group,'sigma':sigma,'n':len(subset),
                **{key:sum(bool(r['score'][key]) for r in subset) for key in
                   ('valid_expression','raw_support_success','prediction_success','full_recovery','false_discovery')},
                'nonempty_n':sum(r['nonempty'] for r in subset),
                'nonempty_recovered':sum(r['nonempty'] and r['score']['full_recovery'] for r in subset)})
    atomic_json(ROOT/'summary.json',summary)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    run()
