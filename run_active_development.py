"""Six prespecified OLD training worlds: one integration run, no new training."""
import argparse
import json
import os
from pathlib import Path

from build_publication_curriculum import atomic_json, read, sha
from discovery.active_student import LIMITS, run_active
from discovery.lab import Laboratory
from discovery.mechanism_scoring import independent_features, score_active
from discovery.model_transport import ModelWorker
from discovery.student_laboratory import StudentLaboratory
from discovery.student_v2 import load_views
from discovery.teaching import decode_query, expression_from_dict


def run(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserve this bounded integration run; no silent retry or overwrite.')
    output.mkdir(parents=True)
    chosen = ('w0000-sigma0','w0004-sigma0','w0008-sigma0','w0012-sigma0','w0016-sigma0','w0020-sigma0')
    views = {v['episode_id']: v for v in load_views('train')}
    model, adapter = 'models/Qwen2.5-0.5B-Instruct', 'results/student_training_v3/adapter'
    sources = ('run_active_development.py','student_inference_worker.py','discovery/active_student.py',
               'discovery/model_transport.py','discovery/mechanism_scoring.py','discovery/student_laboratory.py',
               'discovery/student_mainline.py')
    config = {'stage': 'development_integration_not_teaching_comparison', 'model': model, 'adapter': adapter,
              'adapter_sha256': sha(Path(adapter)/'adapter_model.safetensors'), 'episodes': chosen,
              'limits': LIMITS, 'formal_tests_used': False, 'training_performed': False,
              'learning_gate': False, 'sources': {s:sha(s) for s in sources}}
    atomic_json(output/'config.json', config)
    for source in sources:
        path = output/'sources'/source; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(source).read_bytes())
    thresholds = read('results/structure_discovery_noise/calibration.json')['thresholds']
    results = []
    with ModelWorker(model, adapter, log_path=output/'worker_stderr.log') as worker:
        atomic_json(output/'worker.json', worker.ready)
        for episode_id in chosen:
            view = views[episode_id]
            truth = expression_from_dict(view['final_answer']['expression'])
            lab = Laboratory(truth.delta, noise_sigma=0, noise_seed=0)
            session = StudentLaboratory(lab, [decode_query(q) for q in view['initial']['queries']])
            # Only serialized prompts from run_active cross into the worker.
            with (output/(episode_id+'.events.jsonl')).open('w', encoding='utf-8') as handle:
                def durable(event):
                    handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+'\n')
                    handle.flush(); os.fsync(handle.fileno())
                    if event['kind'] == 'model_reply':
                        print(f'{episode_id} {event["stage"]} used={event["used"]} generated={event["reply"]["output_tokens"]}', flush=True)
                episode = run_active(session, worker, on_event=durable)
            atomic_json(output/(episode_id+'.json'), episode)
            score = score_active(truth, episode, independent_features(98701, int(episode_id[1:5])), 0, thresholds)
            row = {'episode_id': episode_id, 'status': episode['status'], 'used': episode['used'],
                   'inference': episode['inference'], 'score': score}
            results.append(row)
            atomic_json(output/'partial_scores.json', results)
            print(f'{episode_id}: {episode["status"]}, used={episode["used"]}, recovery={score["full_recovery"]}', flush=True)
    if config['adapter_sha256'] != sha(Path(adapter)/'adapter_model.safetensors'):
        raise AssertionError('Frozen development model changed.')
    atomic_json(output/'result.json', {'completed':True, 'formal_tests_used':False, 'training_performed':False,
                'worlds': len(results), 'results':results, 'valid_final_answers':sum(r['score']['valid_expression'] for r in results),
                'full_recoveries':sum(r['score']['full_recovery'] for r in results),
                'tested_idea_recoveries':sum(r['score']['tested_idea_recovery'] for r in results),
                'measured_flips':sum(r['used'] for r in results)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/active_student_development_v2')
    run(parser.parse_args().output)
