"""Plan or execute all registered formal evaluations, sequentially and without retries."""
import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

from build_publication_curriculum import atomic_json, read, sha
from discovery.episode_journal import durable_new
from discovery.evaluation_profiles import registered_student
from discovery.evaluation_results import load_completed_evaluation
from discovery.evaluation_worlds import require_completed_main_training
from discovery.run_lock import exclusive_run
from evaluate_mainline import run as evaluate


LEASE = Path('results/.formal_evaluation_controller.lock')
CALIBRATION = 'results/structure_discovery_noise/calibration.json'
RUNTIME_FILES = (
    'run_evaluation_queue.py', 'discovery/run_lock.py', 'evaluate_mainline.py',
    'build_publication_curriculum.py', 'discovery/evaluation_worlds.py', 'discovery/evaluation_profiles.py',
    'discovery/evaluation_results.py', 'discovery/mechanism_scoring.py', 'discovery/active_student.py',
    'student_inference_worker.py', 'discovery/model_transport.py', 'discovery/student_laboratory.py',
    'discovery/student_mainline.py', 'discovery/lab.py', 'discovery/physics.py', 'discovery/search.py',
    'discovery/noise_evaluation.py', 'discovery/student_v2.py', 'discovery/student_v3.py',
    'discovery/student_spins.py', 'discovery/student_format.py', 'discovery/teaching.py',
    'run_structure_discovery.py', 'discovery/episode_journal.py', 'discovery/answer_diagnostics.py',
    'EVALUATION_RECOVERY_PROTOCOL.md', 'MAINLINE_STUDY_PROTOCOL.md', 'ACTIVE_STUDENT_PROTOCOL.md',
    'SECONDARY_ANALYSIS_PROTOCOL.md')


def plan(study, output_root=None):
    if study not in ('mainline', 'prediction_ablation'):
        raise ValueError('Unknown registered study.')
    root = Path(output_root or ('results/mainline_formal' if study == 'mainline' else 'results/process_ablation_formal'))
    groups = ('A', 'B', 'C') if study == 'mainline' else ('C_no_predictions',)
    jobs = []
    for method in ('llm_fixed', 'llm_active'):
        for seed in (401, 402, 403):
            for group in groups:
                jobs.append({'method': method, 'group': group, 'seed': seed,
                             'output': str(root/method/f'{group}_seed{seed}')})
    if study == 'mainline':
        jobs += [{'method': m, 'group': 'sparse', 'seed': None, 'output': str(root/m)}
                 for m in ('sparse_fixed', 'sparse_active')]
    return {'schema': 'formal-evaluation-queue-v1', 'study': study, 'scope': 'formal',
            'output_root': str(root), 'worlds_per_job': 200, 'noise_conditions': [0, 1],
            'conditions_per_job': 400, 'jobs': jobs,
            'execution_requires_all_main_training_complete': True,
            'planning_reads_no_worlds_and_loads_no_models': True}


def fingerprint(study):
    paths = RUNTIME_FILES+(('PROCESS_ABLATION_PROTOCOL.md',) if study == 'prediction_ablation' else ())
    return {name: sha(name) for name in paths+(CALIBRATION,)}


def verified_job(job, study, sources, profile):
    loaded = load_completed_evaluation(job['output'], scope='formal', study=study,
                                      method=job['method'], group=job['group'], seed=job['seed'])
    for name, digest in loaded['config']['sources'].items():
        if name not in sources or sources[name] != digest or sha(name) != digest:
            raise ValueError('Completed evaluation has a different source version; preserve and audit it.')
    if loaded['config']['calibration_sha256'] != sources[CALIBRATION]:
        raise ValueError('Completed evaluation used a different noise calibration.')
    if profile is not None and any(loaded['config'].get(name) != value for name, value in profile.items()):
        raise ValueError('Completed evaluation used another final adapter or training record.')
    return loaded['provenance']


def execute(specification, *, resume=False):
    # Do this before any output initialization, student model, or heldout-world read.
    require_completed_main_training()
    study = specification['study']
    if specification != plan(study, specification['output_root']):
        raise ValueError('Execute the complete registered queue without a selected test subset.')
    profiles = {(j['group'], j['seed']) for j in specification['jobs'] if j['method'].startswith('llm_')}
    training = {}
    for group, seed in sorted(profiles):
        profile = registered_student(study, group, seed)
        training[f'{group}_seed{seed}'] = {k: profile[k] for k in
            ('adapter_sha256', 'training_config_sha256', 'training_result_sha256', 'model_revision')}
    root = Path(specification['output_root'])
    with exclusive_run(LEASE):
        config = {'plan': specification, 'sources': fingerprint(study), 'training_profiles': training}
        path = root/'queue_config.json'
        if path.exists():
            if not resume or read(path) != config:
                raise ValueError('Explicit resume with the unchanged queue plan and source is required.')
        else:
            if root.exists() and any(root.iterdir()):
                raise ValueError('Existing outputs have no queue registration; audit before adopting them.')
            atomic_json(path, config)
        invocation = uuid.uuid4().hex
        receipts = root/'queue_invocations'/invocation
        started = time.perf_counter()
        durable_new(receipts/'start.json', {'pid': os.getpid(), 'started_unix': time.time(), 'resume': resume,
                                          'queue_config_sha256': sha(path)})
        provenances = []
        try:
            for index, job in enumerate(specification['jobs']):
                if fingerprint(study) != config['sources']:
                    raise ValueError('Evaluation source changed while the queue was running.')
                profile = training.get(f"{job['group']}_seed{job['seed']}")
                if (Path(job['output'])/'result.json').exists():
                    provenances.append(verified_job(job, study, config['sources'], profile))
                    durable_new(receipts/f'{index:03d}-preserved.json', {'job': job, 'new_evaluation_calls': 0})
                    continue
                existing = (Path(job['output'])/'config.json').exists()
                if existing and not resume:
                    raise ValueError('Partial evaluation requires an explicit queue resume.')
                job_started = time.perf_counter()
                durable_new(receipts/f'{index:03d}-start.json', {'job': job, 'started_unix': time.time(),
                                                               'resuming_saved_receipts': existing})
                atomic_json(root/'queue_state.json', {'status': 'evaluating', 'queue_pid': os.getpid(),
                    'invocation': invocation, 'job_index': index, 'job': job,
                    'note': 'Informational only; confirm a live process before reporting that it is running.'})
                arguments = SimpleNamespace(scope='formal', method=job['method'], study=study,
                    group=job['group'], seed=job['seed'] if job['seed'] is not None else 401,
                    limit=None, output=job['output'], resume=existing)
                try:
                    evaluate(arguments)
                    provenance = verified_job(job, study, config['sources'], profile)
                except BaseException as error:
                    durable_new(receipts/f'{index:03d}-end.json', {'status': 'failed_or_interrupted',
                        'job': job, 'elapsed_wall_seconds': time.perf_counter()-job_started,
                        'error_type': type(error).__name__, 'error': str(error), 'automatic_retries': 0})
                    raise
                durable_new(receipts/f'{index:03d}-end.json', {'status': 'completed', 'job': job,
                    'elapsed_wall_seconds': time.perf_counter()-job_started, 'provenance': provenance})
                provenances.append(provenance)
            completion = {'completed': True, 'study': study, 'jobs': len(provenances),
                          'queue_config_sha256': sha(path), 'evaluations': provenances}
            completion_path = root/'queue_completion.json'
            if completion_path.exists() and read(completion_path) != completion:
                raise ValueError('Completed queue provenance changed.')
            if not completion_path.exists():
                atomic_json(completion_path, completion)
        except BaseException as error:
            durable_new(receipts/'end.json', {'status': 'failed_or_interrupted',
                'elapsed_wall_seconds': time.perf_counter()-started, 'error_type': type(error).__name__,
                'error': str(error), 'automatic_retries': 0})
            atomic_json(root/'queue_state.json', {'status': 'stopped', 'invocation': invocation,
                'error_type': type(error).__name__, 'error': str(error), 'automatic_retries': 0})
            raise
        durable_new(receipts/'end.json', {'status': 'completed', 'elapsed_wall_seconds': time.perf_counter()-started})
        atomic_json(root/'queue_state.json', {'status': 'all_registered_evaluations_completed',
                                             'invocation': invocation, 'jobs': len(provenances)})
        return completion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=('mainline', 'prediction_ablation'), default='mainline')
    parser.add_argument('--output-root')
    parser.add_argument('--execute', action='store_true', help='Without this flag, only display the plan.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan-output', help='Save a new plan file without executing it.')
    args = parser.parse_args()
    specification = plan(args.study, args.output_root)
    if args.plan_output:
        if args.execute or args.resume or Path(args.plan_output).exists():
            raise ValueError('Use a new plan-only output file.')
        atomic_json(args.plan_output, specification)
    if args.execute:
        execute(specification, resume=args.resume)
    elif args.resume:
        raise ValueError('Resume is an execution option.')
    else:
        print(json.dumps(specification, ensure_ascii=False, indent=2))
