"""Small common-interface warmup and paired pre/post diagnostics, not a B/C trial."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoTokenizer

from build_publication_curriculum import atomic_json, read, sha
from discovery.active_student import parse_hypothesis, parse_plan, planning_prompt
from discovery.interface_alignment import neutral_session, practice_cases, prepare_plan
from discovery.lab import Laboratory
from discovery.mechanism_scoring import independent_features, score_answer
from discovery.model_transport import ModelWorker
from discovery.student_fastdev import resident_loss
from discovery.student_laboratory import StudentLaboratory, fit_proposal
from discovery.student_mainline import evidence_prefix, hypothesis_prompt, terminal_messages
from discovery.student_v2 import encode, load_views, packed
from discovery.teaching import decode_query, expression_from_dict
from run_fast_development import protected_files
from train_student_pilot import capture_training_state, make_student, tensors

MODEL = Path('models/Qwen2.5-0.5B-Instruct')
START_ADAPTER = Path('results/fast_development_v2/raw/adapter')
OUTPUT = Path('results/interface_alignment_v1')
SOURCES = ('run_interface_alignment.py', 'discovery/interface_alignment.py', 'test_interface_alignment.py',
           'INTERFACE_ALIGNMENT_PROTOCOL.md', 'discovery/student_fastdev.py', 'train_student_pilot.py',
           'discovery/active_student.py', 'discovery/student_mainline.py', 'discovery/student_laboratory.py',
           'discovery/student_v2.py', 'discovery/student_v3.py', 'discovery/student_spins.py',
           'discovery/student_format.py', 'discovery/physics.py', 'discovery/teaching.py',
           'discovery/lab.py', 'discovery/mechanism_scoring.py', 'discovery/model_transport.py',
           'student_inference_worker.py', 'run_structure_discovery.py', 'run_fast_development.py')


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
        handle.flush()
        os.fsync(handle.fileno())


def prepare(output):
    if output.exists():
        raise ValueError('Preserve completed and partial alignment runs; no silent retry.')
    if not read('results/fast_development_v2/verification.json')['passed']:
        raise ValueError('Previous common-task result was not verified.')
    protected = protected_files()
    protected.update({str(START_ADAPTER/'adapter_model.safetensors'): sha(START_ADAPTER/'adapter_model.safetensors'),
                      'results/fast_development_v2/result.json': sha('results/fast_development_v2/result.json')})
    if protected[str(START_ADAPTER/'adapter_model.safetensors')] != read('results/fast_development_v2/raw/result.json')['adapter_sha256']:
        raise ValueError('Starting adapter identity failed.')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    views = load_views('train')
    cases = practice_cases('train')
    entries = []
    for v in views:
        for budget in (8, 16, 24, 32):
            entries.append({'kind': 'hypothesis', 'id': f"{v['episode_id']}_{budget}",
                            'messages': terminal_messages(v, budget)})
    entries += [{'kind': 'interface', 'id': c['id'], 'messages': c['messages']} for c in cases]
    if len(entries) != 128 or len(views) != 24:
        raise ValueError('Unexpected alignment training cohort.')
    encoded = [encode(tokenizer, entry['messages']) for entry in entries]
    arrays = {}
    for i, (entry, data) in enumerate(zip(entries, encoded)):
        entry.update(input_tokens=len(data['ids']), supervised_tokens=int(np.count_nonzero(data['labels'][1:] != -100)))
        arrays[f'{i}_ids'] = data['ids']
        arrays[f'{i}_labels'] = data['labels']
    rng = np.random.default_rng(501)
    hypotheses = list(map(int, rng.permutation(96)))
    formats = [96+int(i) for _ in range(3) for i in rng.permutation(32)]
    schedule = [hypotheses[i:i+2]+formats[i:i+2] for i in range(0, 96, 2)]
    config = {'scope': 'common_interface_alignment_not_BC', 'seed': 501, 'updates': 48,
              'train_cap_seconds': 420, 'learning_rate': 1e-4, 'task_weights': [.5]*4,
              'start_adapter': str(START_ADAPTER), 'schedule': schedule,
              'formal_tests_used': False, 'main_training_resumed': False,
              'fixed_validation_ids': [v['episode_id'] for v in load_views('validation')],
              'one_plan_validation_ids': [f'w{i:04d}-sigma0' for i in (24, 26, 28, 30, 32, 34)],
              'neutral_train_cases': 32, 'neutral_validation_cases': 16,
              'inference': {'hypothesis_tokens': 128, 'plan_tokens': 768, 'repairs': 0, 'maximum_plans_per_world': 1},
              'protected_sha256': protected, 'source_sha256': {p: sha(p) for p in SOURCES},
              'data_sha256': {f'data/research_curriculum_v1/views/{s}/B.jsonl': sha(f'data/research_curriculum_v1/views/{s}/B.jsonl')
                              for s in ('train', 'validation')},
              'environment': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
              'actual_training_budget': {'input_tokens': sum(entries[i]['input_tokens'] for batch in schedule for i in batch),
                    'supervised_tokens': sum(entries[i]['supervised_tokens'] for batch in schedule for i in batch),
                    'hypothesis_presentations': 96, 'interface_presentations': 96},
              'largest_context': max(e['input_tokens'] for e in entries)}
    atomic_json(output/'config.json', config)
    np.savez_compressed(output/'training.npz', **arrays)
    for entry in entries:
        append(output/'messages.jsonl', entry)
    atomic_json(output/'neutral_validation.json', practice_cases('validation'))
    for name in SOURCES:
        destination = output/'sources'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(name).read_bytes())
    atomic_json(output/'input_manifest.json', {name: sha(output/name) for name in ('training.npz', 'messages.jsonl', 'neutral_validation.json')})
    return config


def check_frozen(output, config):
    for group in ('protected_sha256', 'source_sha256', 'data_sha256'):
        if any(sha(p) != h for p, h in config[group].items()):
            raise ValueError('A protected source or input changed: '+group)
    if any(sha(output/p) != h for p, h in read(output/'input_manifest.json').items()):
        raise ValueError('Alignment input cache changed.')


def train(output):
    config = read(output/'config.json')
    check_frozen(output, config)
    folder = output/'training'
    folder.mkdir()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model = make_student(MODEL, config['seed'])
    saved = load_file(START_ADAPTER/'adapter_model.safetensors')
    set_peft_model_state_dict(model, saved)
    loaded = get_peft_model_state_dict(model)
    if any(not torch.equal(loaded[k].detach().cpu(), v) for k, v in saved.items()):
        raise ValueError('Starting adapter did not load exactly.')
    del saved, loaded
    with np.load(output/'training.npz', allow_pickle=False) as archive:
        data = [{'ids': archive[f'{i}_ids'], 'labels': archive[f'{i}_labels']} for i in range(128)]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['learning_rate'], weight_decay=0)
    records = []
    seconds = 0.0
    input_tokens = supervised_tokens = 0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    def save(step):
        state = capture_training_state(model, optimizer)
        state.update(step=step, records=records, training_seconds=seconds)
        temp = folder/'checkpoint.tmp'
        torch.save(state, temp)
        temp.replace(folder/'checkpoint.pt')

    for step, batch in enumerate(config['schedule'], 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        losses = {'hypothesis': 0.0, 'interface': 0.0}
        for index, weight in zip(batch, config['task_weights']):
            total, count = resident_loss(model, *tensors(data[index]))
            loss = total/count*weight
            loss.backward()
            losses['hypothesis' if index < 96 else 'interface'] += float(loss.detach())
            input_tokens += len(data[index]['ids'])
            supervised_tokens += count
            del total, loss
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        if not torch.isfinite(norm) or not all(np.isfinite(v) for v in losses.values()):
            raise ValueError('Non-finite alignment training.')
        optimizer.step()
        torch.cuda.synchronize()
        seconds += time.perf_counter()-start
        records.append({'step': step, 'indices': batch, 'loss': losses, 'gradient_norm': float(norm),
                        'training_seconds': seconds, 'input_tokens': input_tokens, 'supervised_tokens': supervised_tokens})
        append(folder/'updates.jsonl', records[-1])
        if step % 8 == 0:
            print(f'alignment {step}/48 seconds={seconds:.1f} losses={losses}', flush=True)
        if step % 24 == 0:
            save(step)
        if time.perf_counter()-started > config['train_cap_seconds'] and step < 48:
            save(step)
            raise TimeoutError('Fixed alignment training cap reached; no checkpoint selection.')
    wall = time.perf_counter()-started
    model.save_pretrained(folder/'adapter', safe_serialization=True)
    atomic_json(folder/'result.json', {'completed': True, 'updates': len(records), 'training_seconds': seconds,
              'training_wall_seconds': wall, 'input_tokens': input_tokens, 'supervised_tokens': supervised_tokens,
              'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
              'adapter_sha256': sha(folder/'adapter/adapter_model.safetensors')})


def summarize_scores(rows):
    return {'n': len(rows), **{k: sum(bool(r['score'][k]) for r in rows) for k in
        ('valid_expression', 'raw_support_success', 'full_recovery', 'raw_false_proposal', 'false_discovery')},
        'nonempty_n': sum(r['nonempty'] for r in rows),
        'nonempty_recovered': sum(r['nonempty'] and r['score']['full_recovery'] for r in rows)}


def evaluate_checkpoint(output, phase, adapter, config):
    folder = output/phase
    folder.mkdir()
    started = time.perf_counter()
    neutral, fixed, probes = [], [], []
    views = {v['episode_id']: v for v in load_views('validation')}
    features = {key: independent_features(98621, int(key[1:5])) for key in views}
    counters = {'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'generation_seconds': 0.0}
    with ModelWorker(MODEL, adapter, log_path=folder/'worker.log') as worker:
        def generate(messages, cap, tag):
            counters['calls'] += 1
            number = counters['calls']
            append(folder/'calls.jsonl', {'event': 'request_started', 'number': number,
                                         'tag': tag, 'messages': messages, 'max_new_tokens': cap})
            reply = worker(messages, cap)
            append(folder/'calls.jsonl', {'event': 'reply_received', 'number': number, 'reply': reply})
            for key in ('input_tokens', 'output_tokens', 'generation_seconds'):
                counters[key] += reply[key]
            return reply

        def hypothesis(evidence, tag):
            reply = generate(hypothesis_prompt(evidence), 128, tag)
            try:
                structures = parse_hypothesis(reply['text'])
                expression, fit = fit_proposal(evidence, structures)
                answer = {'used': evidence['used'], 'raw_structures': structures, 'expression': expression.to_dict(), 'fit': fit}
                error = None
            except (ValueError, KeyError, TypeError, IndexError) as problem:
                answer, error = None, str(problem)
            return {'reply': reply, 'answer': answer, 'error': error}

        for case in practice_cases('validation'):
            session = neutral_session(case)
            reply = generate(case['prompt'], 768, 'neutral_'+case['id'])
            row = {'id': case['id'], 'expected_action': case['answer']['action'], 'reply': reply,
                   'schema_valid': False, 'exact_plan': False, 'requested_action_correct': False,
                   'interface_accepted': False, 'experiment_executed': False, 'used': 8}
            try:
                parsed = parse_plan(reply['text'], 2)
                row.update(schema_valid=True, exact_plan=parsed == case['answer'],
                           requested_action_correct=parsed['action'] == case['answer']['action'])
                plan, commitment = prepare_plan(session, case['current_structures'], reply['text'], 2)
                if commitment:
                    append(folder/'commitments.jsonl', {'id': case['id'], 'kind': 'neutral', 'commitment': commitment})
                    observation = session.reveal(commitment['ticket'])
                    append(folder/'reveals.jsonl', {'id': case['id'], 'kind': 'neutral', 'observation': observation})
                    row.update(commitment=commitment, observation=observation, experiment_executed=True)
                else:
                    session.finish(case['current_structures'])
                row.update(interface_accepted=True, used=session.evidence()['used'])
            except (ValueError, KeyError, TypeError, IndexError) as error:
                row['error'] = str(error)
            neutral.append(row)
            append(folder/'neutral.jsonl', row)
            print(f'{phase} format {case["id"]}: exact={row["exact_plan"]} measured={row["experiment_executed"]}', flush=True)

        for episode in config['fixed_validation_ids']:
            view = views[episode]
            prediction = hypothesis(evidence_prefix(view, 32), 'fixed_'+episode)
            truth = expression_from_dict(view['final_answer']['expression'])
            score = score_answer(truth, prediction['answer'], features[episode], 0, {})
            row = {'episode': episode, 'nonempty': bool(truth.terms), 'prediction': prediction, 'score': score}
            fixed.append(row)
            append(folder/'fixed.jsonl', row)
            print(f'{phase} fixed {episode}: raw={score["raw_support_success"]} full={score["full_recovery"]}', flush=True)

        for episode in config['one_plan_validation_ids']:
            view = views[episode]
            truth = expression_from_dict(view['final_answer']['expression'])
            session = StudentLaboratory(Laboratory(truth.delta, noise_sigma=0, noise_seed=0),
                        [decode_query(q) for q in view['initial']['queries']])
            if session.evidence() != evidence_prefix(view, 8):
                raise ValueError('Probe initial evidence differs from the saved common eight measurements.')
            first = hypothesis(session.evidence(), 'probe_initial_'+episode)
            row = {'episode': episode, 'initial': first, 'status': 'invalid_hypothesis', 'used': 8,
                   'experiment_executed': False, 'nonempty': bool(truth.terms), 'after': None}
            if first['answer'] is not None:
                primary = first['answer']['raw_structures']
                reply = generate(planning_prompt(session.evidence(), primary, first['answer']['fit'], None, 4),
                                 768, 'probe_plan_'+episode)
                row['plan_reply'] = reply
                try:
                    plan, commitment = prepare_plan(session, primary, reply['text'], 4)
                    row['plan'] = plan
                    if commitment is None:
                        session.finish(primary)
                        row.update(status='student_finished', after=first)
                    else:
                        append(folder/'commitments.jsonl', {'id': episode, 'kind': 'probe', 'commitment': commitment})
                        observation = session.reveal(commitment['ticket'])
                        append(folder/'reveals.jsonl', {'id': episode, 'kind': 'probe', 'observation': observation})
                        row.update(status='one_experiment_executed', experiment_executed=True,
                                   commitment=commitment, observation=observation,
                                   after=hypothesis(session.evidence(), 'probe_update_'+episode))
                except (ValueError, KeyError, TypeError, IndexError) as error:
                    row.update(status='invalid_plan', error=str(error))
            row['used'] = session.evidence()['used']
            row['initial_score'] = score_answer(truth, first['answer'], features[episode], 0, {})
            row['score'] = score_answer(truth, row['after']['answer'] if row['after'] else None, features[episode], 0, {})
            probes.append(row)
            append(folder/'probes.jsonl', row)
            print(f'{phase} probe {episode}: {row["status"]} used={row["used"]}', flush=True)
    result = {'completed': True, 'adapter_sha256': sha(Path(adapter)/'adapter_model.safetensors'),
              'neutral': {'n': len(neutral), **{k: sum(bool(r[k]) for r in neutral) for k in
                  ('schema_valid', 'exact_plan', 'requested_action_correct', 'interface_accepted', 'experiment_executed')},
                  'requested_experiment_cases': sum(r['expected_action'] == 'experiment' for r in neutral),
                  'requested_experiments_executed': sum(r['expected_action'] == 'experiment' and r['experiment_executed'] for r in neutral),
                  'additional_measurements': sum(r['used']-8 for r in neutral)},
              'fixed': summarize_scores(fixed), 'probe_scores': summarize_scores(probes),
              'probes': {'n': len(probes), 'experiments_executed': sum(r['experiment_executed'] for r in probes),
                         'additional_measurements': sum(r['used']-8 for r in probes),
                         'student_finished': sum(r['status'] == 'student_finished' for r in probes),
                         'invalid': sum(r['status'].startswith('invalid') for r in probes)},
              'inference': counters, 'elapsed_seconds': time.perf_counter()-started}
    atomic_json(folder/'result.json', result)
    return result


def run(output):
    started = time.perf_counter()
    config = prepare(output)
    try:
        before = evaluate_checkpoint(output, 'before', START_ADAPTER, config)
        if torch.cuda.is_initialized():
            raise ValueError('Controller must not initialize CUDA or hold model weights.')
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--output', str(output.resolve()), '--train']
        completed = subprocess.run(command, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), check=False)
        if completed.returncode != 0:
            raise RuntimeError('Alignment training child failed; preserve its exact checkpoint and logs.')
        after = evaluate_checkpoint(output, 'after', output/'training/adapter', config)
        check_frozen(output, config)
        result = {'completed': True, 'scope': config['scope'], 'before': before, 'after': after,
                  'training': read(output/'training/result.json'), 'elapsed_seconds': time.perf_counter()-started,
                  'protected_files_unchanged': True, 'main_training_still_paused': True, 'formal_tests_used': False,
                  'config_sha256': sha(output/'config.json')}
        atomic_json(output/'result.json', result)
    except BaseException as error:
        atomic_json(output/'failure.json', {'completed': False, 'type': type(error).__name__, 'error': str(error),
                                           'elapsed_seconds': time.perf_counter()-started})
        raise
    print('Completed '+str(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    train(args.output) if args.train else run(args.output)
