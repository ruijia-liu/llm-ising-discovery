"""No-training comparison of pinned base models on the same neutral plan-copy tasks."""
import json
from pathlib import Path
import time

from build_publication_curriculum import atomic_json, read, sha
from discovery.active_student import parse_plan
from discovery.interface_alignment import neutral_session, practice_cases, prepare_plan
from discovery.model_transport import ModelWorker
from run_fast_development import protected_files
from run_interface_alignment import append


ROOT = Path('results/base_interface_probe_v1')
MODELS = {'base_05': Path('models/Qwen2.5-0.5B-Instruct'), 'base_15': Path('models/Qwen2.5-1.5B-Instruct')}


def run():
    if ROOT.exists():
        raise ValueError('Preserve prior or partial base-model probes.')
    before = protected_files()
    before.update({p: sha(p) for p in ('results/interface_alignment_v1/result.json',
                                      'results/interface_alignment_v1/verification.json',
                                      'results/interface_alignment_v1/training/adapter/adapter_model.safetensors')})
    sources = ('probe_base_interfaces.py', 'BASE_INTERFACE_PROBE_PROTOCOL.md', 'discovery/interface_alignment.py',
               'discovery/model_transport.py', 'student_inference_worker.py', 'discovery/student_laboratory.py',
               'discovery/active_student.py', 'discovery/lab.py', 'discovery/physics.py', 'run_interface_alignment.py')
    models = {}
    for name, path in MODELS.items():
        manifest = read(path/'download_manifest.json')
        if any(sha(path/file) != metadata['sha256'] for file, metadata in manifest['files'].items()):
            raise ValueError('Pinned base-model file changed.')
        models[name] = {'path': str(path), 'manifest': manifest}
    config = {'scope': 'base_instruction_probe_no_training', 'models': models,
              'cases': [c['id'] for c in practice_cases('validation')], 'output_tokens': 768,
              'formal_tests_used': False, 'training_performed': False,
              'sources': {p: sha(p) for p in sources}, 'protected_files': before}
    atomic_json(ROOT/'config.json', config)
    for name in sources:
        path = ROOT/'sources'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(name).read_bytes())
    results = {}
    started = time.perf_counter()
    try:
        for name, model in MODELS.items():
            folder = ROOT/name
            folder.mkdir()
            rows, elapsed = [], time.perf_counter()
            with ModelWorker(model, log_path=folder/'worker.log') as worker:
                for case in practice_cases('validation'):
                    append(folder/'calls.jsonl', {'id': case['id'], 'event': 'request_started',
                                                 'messages': case['prompt'], 'max_new_tokens': 768})
                    reply = worker(case['prompt'], 768)
                    append(folder/'calls.jsonl', {'id': case['id'], 'event': 'reply_received', 'reply': reply})
                    session = neutral_session(case)
                    row = {'id': case['id'], 'expected_action': case['answer']['action'], 'reply': reply,
                           'schema_valid': False, 'exact_plan': False, 'requested_action_correct': False,
                           'interface_accepted': False, 'experiment_executed': False, 'used': 8}
                    try:
                        parsed = parse_plan(reply['text'], 2)
                        row.update(schema_valid=True, exact_plan=parsed == case['answer'],
                                   requested_action_correct=parsed['action'] == case['answer']['action'])
                        _, commitment = prepare_plan(session, case['current_structures'], reply['text'], 2)
                        if commitment:
                            append(folder/'events.jsonl', {'id': case['id'], 'event': 'predictions_committed', 'commitment': commitment})
                            observation = session.reveal(commitment['ticket'])
                            append(folder/'events.jsonl', {'id': case['id'], 'event': 'observations_revealed', 'observation': observation})
                            row.update(commitment=commitment, observation=observation, experiment_executed=True)
                        else:
                            session.finish(case['current_structures'])
                        row.update(interface_accepted=True, used=session.evidence()['used'])
                    except (ValueError, TypeError, KeyError, IndexError) as error:
                        row['error'] = str(error)
                    rows.append(row)
                    append(folder/'answers.jsonl', row)
                    print(f'{name} {case["id"]}: exact={row["exact_plan"]} action={row["requested_action_correct"]} measured={row["experiment_executed"]}', flush=True)
            summary = {'n': len(rows), **{k: sum(bool(r[k]) for r in rows) for k in
                ('schema_valid', 'exact_plan', 'requested_action_correct', 'interface_accepted', 'experiment_executed')},
                'requested_experiments_executed': sum(r['expected_action'] == 'experiment' and r['experiment_executed'] for r in rows),
                'additional_measurements': sum(r['used']-8 for r in rows),
                'output_tokens': sum(r['reply']['output_tokens'] for r in rows),
                'input_tokens': sum(r['reply']['input_tokens'] for r in rows),
                'generation_seconds': sum(r['reply']['generation_seconds'] for r in rows),
                'elapsed_seconds': time.perf_counter()-elapsed}
            atomic_json(folder/'result.json', summary)
            results[name] = summary
        if any(sha(p) != h for p, h in before.items()) or any(sha(p) != h for p, h in config['sources'].items()):
            raise ValueError('Protected original or running source changed.')
        result = {'completed': True, 'results': results, 'elapsed_seconds': time.perf_counter()-started,
                  'training_performed': False, 'formal_tests_used': False, 'original_training_still_paused': True,
                  'config_sha256': sha(ROOT/'config.json')}
        atomic_json(ROOT/'result.json', result)
    except BaseException as error:
        atomic_json(ROOT/'failure.json', {'completed': False, 'type': type(error).__name__, 'error': str(error),
                                         'elapsed_seconds': time.perf_counter()-started})
        raise
    print('Completed '+str(ROOT), flush=True)


if __name__ == '__main__':
    run()
