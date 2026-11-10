"""Paired evaluation driver. Formal worlds remain gated until all main arms finish."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np

from build_publication_curriculum import atomic_json,read,sha
from discovery.active_student import run_active,parse_hypothesis
from discovery.evaluation_worlds import fixed_evidence,load_worlds,prefix,streams
from discovery.evaluation_profiles import registered_student
from discovery.episode_journal import EpisodeJournal
from discovery.answer_diagnostics import episode_diagnostics
from discovery.lab import Laboratory
from discovery.mechanism_scoring import independent_features,score_active,score_answer
from discovery.model_transport import ModelWorker
from discovery.noise_evaluation import validate_calibration
from discovery.physics import Grammar,base_flip
from discovery.search import SparseSearch,run_search
from discovery.student_laboratory import StudentLaboratory,fit_proposal,visible_queries
from discovery.student_mainline import hypothesis_prompt
from discovery.teaching import expression_from_dict


def fixed_prediction(evidence,budget,generate):
    visible=prefix(evidence,budget)
    reply=generate(hypothesis_prompt(visible),128)
    answer=None; error=None
    try:
        structures=parse_hypothesis(reply['text'])
        model,fit=fit_proposal(visible,structures)
        answer={'used':budget,'raw_structures':structures,'expression':model.to_dict(),'fit':fit}
    except (ValueError,KeyError,TypeError,IndexError) as problem:
        error=str(problem)
    return {'reply':reply,'answer':answer,'error':error}


def fixed_sparse(evidence,budget):
    visible=prefix(evidence,budget)
    queries,observations=visible_queries(visible)
    grammar=Grammar(); search=SparseSearch(grammar,noise_sigma=evidence['measurement_noise_sigma'])
    features=grammar.features([q[0] for q in queries],[q[1] for q in queries])
    residual=observations-np.asarray([base_flip(*q) for q in queries])
    fitted=search.expression(search.fit(features,residual)[0])
    return {'used':budget,'raw_structures':fitted.support,'expression':fitted.to_dict()}


def score_sparse_active(truth,snapshots,events,features,sigma,thresholds):
    final=snapshots[32]
    score=score_answer(truth,{'expression':final.to_dict(),'raw_structures':final.support},features,sigma,thresholds)
    committed=any(expression_from_dict(h).support==final.support for e in events if e['kind']=='experiment' for h in e['hypotheses'])
    score.update(final_support_committed_before_measurement=committed,
                 tested_idea_recovery=bool(truth.terms and score['full_recovery'] and committed),
                 budget_curve=[{'budget':n,**score_answer(truth,{'expression':snapshots[n].to_dict(),'raw_structures':snapshots[n].support},features,sigma,thresholds)}
                               for n in (8,16,24,32)])
    return score


def run(args):
    study=getattr(args,'study','mainline')
    is_llm=args.method.startswith('llm_')
    if not is_llm and study!='mainline':
        raise ValueError('The same mainline sparse results are reused; do not duplicate them as ablation students.')
    profile=registered_student(study,args.group,args.seed) if is_llm else None
    worlds=load_worlds(args.scope)
    if args.limit:
        if args.scope!='development':
            raise ValueError('Do not select a formal test subset.')
        # Across the six development families, not just the first/easiest family.
        worlds=[w for j,w in enumerate(worlds) if j%10<args.limit]
    output=Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError('Preserve existing evaluation; explicitly resume complete episodes.')
    calibration=read('results/structure_discovery_noise/calibration.json')
    validate_calibration(calibration,98731)
    source_files=('evaluate_mainline.py','discovery/evaluation_worlds.py','discovery/mechanism_scoring.py',
                  'discovery/active_student.py','student_inference_worker.py','discovery/model_transport.py',
                  'discovery/student_laboratory.py','discovery/student_mainline.py',
                  'discovery/lab.py','discovery/physics.py','discovery/search.py','discovery/noise_evaluation.py',
                  'discovery/student_v2.py','discovery/student_v3.py','discovery/student_spins.py','run_structure_discovery.py',
                  'discovery/evaluation_profiles.py','discovery/episode_journal.py','EVALUATION_RECOVERY_PROTOCOL.md',
                  'discovery/answer_diagnostics.py','SECONDARY_ANALYSIS_PROTOCOL.md',
                  'MAINLINE_STUDY_PROTOCOL.md','ACTIVE_STUDENT_PROTOCOL.md')
    if study=='prediction_ablation':
        source_files+=('PROCESS_ABLATION_PROTOCOL.md',)
    config={'scope':args.scope,'method':args.method,'study':study,'group':args.group if is_llm else 'sparse',
            'seed':args.seed if is_llm else None,'worlds':[w['world_id'] for w in worlds],
            'sigmas':[0,1],'calibration_sha256':sha('results/structure_discovery_noise/calibration.json'),
            'sources':{p:sha(p) for p in source_files},
            'adapter_sha256':profile['adapter_sha256'] if is_llm else None,
            'training_config_sha256':profile['training_config_sha256'] if is_llm else None,
            'training_result_sha256':profile['training_result_sha256'] if is_llm else None,
            'model_revision':profile['model_revision'] if is_llm else None}
    if args.resume:
        if config!=read(output/'config.json'):
            raise ValueError('Evaluation configuration/source changed.')
    else:
        atomic_json(output/'config.json',config)
        for source in source_files:
            target=output/'sources'/source; target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(Path(source).read_bytes())
    context=ModelWorker(profile['model'],profile['adapter'],log_path=output/'worker_stderr.log') if is_llm else nullcontext(None)
    configuration_sha256=sha(output/'config.json')
    rows=[]
    with context as worker:
        for world in worlds:
            truth=expression_from_dict(world['truth'])
            inputs=streams(args.scope,world['world_index'])
            features=independent_features(inputs['evaluation_seed'],world['world_index'])
            for sigma in (0,1):
                key=world['world_id']+f'-sigma{sigma}'
                path=output/'episodes'/(key+'.json')
                events_path=output/'episodes'/(key+'.events.jsonl')
                if path.exists():
                    rows.append(read(path)['result']); continue
                if events_path.exists():
                    raise RuntimeError('Legacy event log has no durable call receipts; explicitly audit it instead of silently repeating measurements.')
                path.parent.mkdir(parents=True,exist_ok=True)
                identity={'configuration_sha256':configuration_sha256,'world_id':world['world_id'],
                          'world_index':world['world_index'],'sigma':sigma}
                with EpisodeJournal(output/'private_journals'/key,identity) as journal:
                    oracle=journal.oracle(truth.delta)
                    generate=journal.generator(worker) if is_llm else None
                    if args.method in ('llm_fixed','sparse_fixed'):
                        evidence,events=fixed_evidence(oracle,sigma,inputs)
                        journal.event({'kind':'fixed_evidence','evidence':evidence,'acquisition_events':events})
                        answers={}; curve=[]
                        for budget in (8,16,24,32):
                            raw=fixed_prediction(evidence,budget,generate) if is_llm else {'answer':fixed_sparse(evidence,budget)}
                            answers[str(budget)]=raw
                            curve.append({'budget':budget,**score_answer(truth,raw['answer'],features,sigma,calibration['thresholds'])})
                        score={**curve[-1],'budget_curve':curve}
                        episode={'evidence':evidence,'acquisition_events':events,'answers':answers}
                    elif args.method=='llm_active':
                        lab=Laboratory(oracle,noise_sigma=sigma,noise_seed=inputs['noise_seed'])
                        session=StudentLaboratory(lab,inputs['initial'])
                        episode=run_active(session,generate,on_event=journal.event)
                        score=score_active(truth,episode,features,sigma,calibration['thresholds'])
                    else:
                        lab=Laboratory(oracle,noise_sigma=sigma,noise_seed=inputs['noise_seed'])
                        snapshots,events=run_search(lab,Grammar(),inputs['initial'],strategy='active',seed=inputs['search_seed'])
                        score=score_sparse_active(truth,snapshots,events,features,sigma,calibration['thresholds'])
                        episode={'snapshots':{str(n):m.to_dict() for n,m in snapshots.items()},'events':events}
                    execution=journal.complete(episode)
                diagnostics=episode_diagnostics(truth,episode,score,args.method,features,sigma,calibration['thresholds'])
                result={'world_id':world['world_id'],'level':world['level'],'sigma':sigma,'method':args.method,'study':study,
                        'group':config['group'],'seed':config['seed'],
                        'structure_id':hashlib.sha256(json.dumps(truth.support).encode()).hexdigest(),
                        'score':score,'grader_diagnostics':diagnostics}
                atomic_json(path,{'result':result,'episode':episode,'execution':execution})
                rows.append(result)
                print(f'{args.scope} {args.method} {key}: recovery={score["full_recovery"]}',flush=True)
    atomic_json(output/'result.json',{'completed':True,'scope':args.scope,'method':args.method,'rows':rows,
                'worlds':len(worlds),'episodes':len(rows),'formal_tests_used':args.scope=='formal',
                'full_recoveries':sum(r['score']['full_recovery'] for r in rows)})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',choices=('development','formal'),default='development')
    parser.add_argument('--method',choices=('llm_fixed','llm_active','sparse_fixed','sparse_active'),required=True)
    parser.add_argument('--study',choices=('mainline','prediction_ablation'),default='mainline')
    parser.add_argument('--group',choices=['A','B','C','C_no_predictions'],default='A')
    parser.add_argument('--seed',type=int,choices=(401,402,403),default=401)
    parser.add_argument('--limit',type=int,help='Development worlds per family only; never filters formal worlds.')
    parser.add_argument('--output',required=True)
    parser.add_argument('--resume',action='store_true')
    run(parser.parse_args())
