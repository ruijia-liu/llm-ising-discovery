"""Paired no-training wording probe. Preserve original scores and strict validators."""
import json
from pathlib import Path
import time

from build_publication_curriculum import atomic_json, read, sha
from discovery.action_prompt_v2 import practice_cases_v2
from discovery.active_student import parse_plan
from discovery.interface_alignment import neutral_session, practice_cases, prepare_plan
from discovery.model_transport import ModelWorker
from probe_base_interfaces import MODELS
from run_fast_development import protected_files
from run_interface_alignment import append

ROOT = Path('results/clear_interface_probe_v1')


def run():
    if ROOT.exists():
        raise ValueError('Preserve existing prompt probe.')
    original, revised = practice_cases('validation'), practice_cases_v2('validation')
    for old, new in zip(original, revised):
        if any(old[k] != new[k] for k in ('id', 'answer', 'current_structures', 'initial_query_seed')):
            raise ValueError('Wording change altered a target or laboratory condition.')
    protected = protected_files()
    protected.update({p: sha(p) for p in ('results/base_interface_probe_v1/result.json',
                                        'results/base_interface_probe_v1/verification.json')})
    sources = ('probe_clear_interfaces.py', 'discovery/action_prompt_v2.py', 'CLEAR_INTERFACE_PROMPT_PROTOCOL.md',
               'discovery/interface_alignment.py', 'discovery/active_student.py', 'student_inference_worker.py',
               'discovery/model_transport.py', 'discovery/student_laboratory.py', 'discovery/lab.py',
               'run_interface_alignment.py', 'probe_base_interfaces.py')
    config = {'scope': 'clear_interface_wording_no_training', 'model_metadata': read('results/base_interface_probe_v1/config.json')['models'],
              'case_ids': [c['id'] for c in revised], 'target_and_lab_conditions_unchanged': True,
              'output_tokens': 768, 'repairs': 0, 'training_performed': False, 'formal_tests_used': False,
              'protected_files': protected, 'sources': {p: sha(p) for p in sources}}
    atomic_json(ROOT/'config.json', config)
    atomic_json(ROOT/'cases.json', revised)
    for source in sources:
        target = ROOT/'sources'/source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(source).read_bytes())
    started = time.perf_counter()
    summaries = {}
    try:
        for name, model in MODELS.items():
            folder = ROOT/name
            folder.mkdir()
            rows = []
            with ModelWorker(model, log_path=folder/'worker.log') as worker:
                for case in revised:
                    append(folder/'calls.jsonl', {'event': 'request_started', 'id': case['id'], 'messages': case['prompt'], 'max_new_tokens': 768})
                    reply = worker(case['prompt'], 768)
                    append(folder/'calls.jsonl', {'event': 'reply_received', 'id': case['id'], 'reply': reply})
                    session = neutral_session(case)
                    row = {'id': case['id'], 'expected_action': case['answer']['action'], 'reply': reply,
                           'schema_valid': False, 'exact_plan': False, 'requested_action_correct': False,
                           'interface_accepted': False, 'experiment_executed': False, 'used': 8}
                    try:
                        plan = parse_plan(reply['text'], 2)
                        row.update(schema_valid=True, exact_plan=plan == case['answer'],
                                   requested_action_correct=plan['action'] == case['answer']['action'])
                        _, committed = prepare_plan(session, case['current_structures'], reply['text'], 2)
                        if committed:
                            append(folder/'events.jsonl', {'event': 'predictions_committed', 'id': case['id'], 'commitment': committed})
                            observation = session.reveal(committed['ticket'])
                            append(folder/'events.jsonl', {'event': 'observations_revealed', 'id': case['id'], 'observation': observation})
                            row.update(commitment=committed, observation=observation, experiment_executed=True)
                        else:
                            session.finish(case['current_structures'])
                        row.update(interface_accepted=True, used=session.evidence()['used'])
                    except (ValueError, TypeError, KeyError, IndexError) as error:
                        row['error'] = str(error)
                    rows.append(row)
                    append(folder/'answers.jsonl', row)
                    print(f'clear {name} {case["id"]}: exact={row["exact_plan"]} measured={row["experiment_executed"]}', flush=True)
            summaries[name] = {'n': len(rows), **{k: sum(bool(r[k]) for r in rows) for k in
                ('schema_valid', 'exact_plan', 'requested_action_correct', 'interface_accepted', 'experiment_executed')},
                'requested_experiments_executed': sum(r['expected_action'] == 'experiment' and r['experiment_executed'] for r in rows),
                'additional_measurements': sum(r['used']-8 for r in rows),
                'input_tokens': sum(r['reply']['input_tokens'] for r in rows),
                'output_tokens': sum(r['reply']['output_tokens'] for r in rows),
                'generation_seconds': sum(r['reply']['generation_seconds'] for r in rows)}
            atomic_json(folder/'result.json', summaries[name])
        if any(sha(p) != h for p, h in protected.items()) or any(sha(p) != h for p, h in config['sources'].items()):
            raise ValueError('Original result or source changed.')
        atomic_json(ROOT/'result.json', {'completed': True, 'results': summaries, 'training_performed': False,
                    'formal_tests_used': False, 'original_main_still_paused': True,
                    'elapsed_seconds': time.perf_counter()-started, 'config_sha256': sha(ROOT/'config.json')})
    except BaseException as error:
        atomic_json(ROOT/'failure.json', {'completed': False, 'type': type(error).__name__, 'error': str(error)})
        raise
    print('Completed '+str(ROOT), flush=True)


if __name__ == '__main__':
    run()
