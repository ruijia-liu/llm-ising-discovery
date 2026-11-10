"""One bounded coefficient-diversity screen, reusing the audited original-data control."""
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch
from transformers import AutoTokenizer

from build_publication_curriculum import atomic_json, read, sha
from discovery import student_v3
from discovery.physics import Expression, Grammar, Term
from discovery.student_fastdev import resident_loss, terminal_messages, terminal_prompt
from discovery.student_v2 import encode, load_views, packed
from discovery.teaching import decode_query, expression_from_dict
from run_fast_development import protected_files
from run_structure_discovery import evaluate
from train_student_pilot import capture_training_state, make_student, tensors
from train_student_v3 import evaluation_features

ROOT = Path('results/parameter_diversity_v1')
CONTROL = Path('results/fast_development_v2')
MODEL = Path('models/Qwen2.5-0.5B-Instruct')
SOURCES = ('run_parameter_diversity.py', 'PARAMETER_DIVERSITY_PROTOCOL.md')


def lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines()]


def append(path, value):
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
        handle.flush()


def variants(view):
    truth = expression_from_dict(view['final_answer']['expression'])
    result = [view]
    if not truth.terms:
        return result
    world = int(view['episode_id'][1:5])
    for variant in range(1, 4):
        rng = np.random.default_rng([97027, world, variant])
        coefficients = rng.uniform(.3, .8, len(truth.terms))*rng.choice([-1, 1], len(truth.terms))
        expression = Expression(tuple(Term(t.points, float(k)) for t, k in zip(truth.terms, coefficients)))
        # Only the terminal evidence is exported; original teacher prose is not reused.
        changed = copy.deepcopy({k: view[k] for k in ('episode_id', 'noise_sigma', 'initial', 'experiments', 'final_answer')})
        changed['episode_id'] += f'-parameter{variant}'
        changed['final_answer']['expression'] = expression.to_dict()
        pairs = [(q, None) for q in changed['initial']['queries']]
        pairs += [(e['query'], e) for e in changed['experiments']]
        initial = []
        for query, experiment in pairs:
            state, position = decode_query(query)
            observed = expression.delta(state, position)
            flipped = state.copy()
            flipped[position] *= -1
            if not np.isclose(observed, expression.energy(flipped)-expression.energy(state), atol=1e-9, rtol=0):
                raise ValueError('A new coefficient observation failed full-energy verification.')
            if experiment is None:
                initial.append(observed)
            else:
                experiment['observed'] = observed
        changed['initial']['observed'] = initial
        result.append(changed)
    return result


def prepare():
    if ROOT.exists():
        raise ValueError('Keep any completed or partial attempt; no overwrite or score-driven retry.')
    control_config = read(CONTROL/'config.json')
    control_result = read(CONTROL/'raw/result.json')
    if not read(CONTROL/'verification.json')['passed'] or control_result['updates'] != 192:
        raise ValueError('Original-data control is not verified and complete.')
    environment = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    if environment != control_config['environment']:
        raise ValueError('Package versions differ from the control.')
    protected = protected_files()
    for key in ('source_sha256', 'data_sha256', 'protected_sha256'):
        for path, digest in control_config[key].items():
            if sha(path) != digest:
                raise ValueError('Original control provenance changed: '+path)
            protected[path] = digest
    for relative in ('config.json', 'raw/result.json', 'raw/episodes.jsonl', 'raw/training.jsonl',
                     'raw/adapter/adapter_model.safetensors', 'verification.json'):
        path = CONTROL/relative
        protected[str(path)] = sha(path)
    base_hashes = read('results/student_mainline_v1/A_seed401/config.json')['base_files_sha256']
    for name, digest in base_hashes.items():
        if sha(MODEL/name) != digest:
            raise ValueError('Pinned model bytes changed.')
    if sha(CONTROL/'raw/adapter/adapter_model.safetensors') != control_result['adapter_sha256']:
        raise ValueError('Control adapter changed.')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    train, validation = load_views('train'), load_views('validation')
    if len(train) != 24 or len(validation) != 12:
        raise ValueError('Changed cohort sizes.')
    validation = [v for v in validation if expression_from_dict(v['final_answer']['expression']).terms]
    if len(validation) != 10:
        raise ValueError('Expected all ten pre-existing nonempty parameter worlds.')
    old_vectors = {(expression_from_dict(v['final_answer']['expression']).support,
                    tuple(t.coefficient for t in expression_from_dict(v['final_answer']['expression']).terms))
                   for v in train+validation}
    seen_new = set()
    data, metadata, sets = [], [], []
    for index, view in enumerate(train):
        indices = []
        original = encode(tokenizer, terminal_messages(view, 'raw'))
        for variant, v in enumerate(variants(view)):
            example = encode(tokenizer, terminal_messages(v, 'raw'))
            truth = expression_from_dict(v['final_answer']['expression'])
            signature = (truth.support, tuple(t.coefficient for t in truth.terms))
            if variant:
                if signature in old_vectors or signature in seen_new:
                    raise ValueError('New parameters duplicate a train/development parameter vector.')
                seen_new.add(signature)
            if not np.array_equal(example['labels'][example['labels'] != -100], original['labels'][original['labels'] != -100]):
                raise ValueError('Parameter replacement changed answer supervision.')
            if [m['query'] for m in student_v3.evidence_only(v)['measurements']] != [m['query'] for m in student_v3.evidence_only(view)['measurements']]:
                raise ValueError('Parameter replacement changed the query design.')
            indices.append(len(data))
            data.append(example)
            metadata.append({'key': v['episode_id'], 'original_index': index, 'variant': variant,
                             'expression': truth.to_dict(), 'input_tokens': len(example['ids']),
                             'supervised_tokens': int(np.count_nonzero(example['labels'][1:] != -100))})
        sets.append(indices)
    visits = [0]*24
    schedule = []
    for original_indices in control_config['schedule']:
        step = []
        for index in original_indices:
            step.append(sets[index][visits[index] % len(sets[index])])
            visits[index] += 1
        schedule.append(step)
    if visits != [16]*24 or len(schedule) != 192:
        raise ValueError('Changed update/presentation count.')
    inputs = sum(metadata[i]['input_tokens'] for batch in schedule for i in batch)
    supervision = sum(metadata[i]['supervised_tokens'] for batch in schedule for i in batch)
    input_difference = abs(inputs-control_result['input_tokens'])/control_result['input_tokens']
    if supervision != control_result['supervised_tokens'] or input_difference > .01:
        raise ValueError('Training token budgets do not match the declared criterion.')
    control_rows = [r for r in lines(CONTROL/'raw/episodes.jsonl') if r['split'] == 'validation' and not r['null_world']]
    if [r['episode_id'] for r in control_rows] != [v['episode_id'] for v in validation] or sum(r['full_recovery'] for r in control_rows) != 4:
        raise ValueError('Unexpected original primary comparator.')
    config = {'scope': 'one_parameter_diversity_screen_only', 'seed': 301, 'parameter_seed': 97027,
        'updates': 192, 'presentations': 384, 'training_cap_seconds': 600,
        'original_worlds': 24, 'new_parameter_worlds': len(seen_new), 'total_distinct_examples': len(data),
        'new_measurements_verified_by_full_energy': len(seen_new)*32,
        'control': str(CONTROL/'raw'), 'control_reused_no_new_training': True,
        'original_schedule': control_config['schedule'], 'schedule': schedule, 'examples': metadata,
        'input_tokens': inputs, 'supervised_tokens': supervision,
        'control_input_tokens': control_result['input_tokens'], 'relative_input_difference': input_difference,
        'validation_ids': [v['episode_id'] for v in validation], 'control_nonempty_recovered': 4,
        'decision': 'expand_only_if_nonempty_full_recovery_strictly_exceeds_4_of_10_else_stop_configuration',
        'protected_sha256': protected, 'source_sha256': {p: sha(p) for p in SOURCES},
        'environment': environment, 'base_files_sha256': base_hashes,
        'formal_tests_used': False, 'autonomous_experiments': False, 'multiple_seeds': False,
        'main_training_resumed': False, 'development_questions_previously_inspected': True}
    atomic_json(ROOT/'config.json', config)
    for path in SOURCES:
        target = ROOT/'sources'/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(path).read_bytes())
    for entry, example in zip(metadata, data):
        append(ROOT/'training_messages.jsonl', {**entry, 'messages': example['messages']})
    atomic_json(ROOT/'control_rows.json', control_rows)
    print('PREPARED '+json.dumps({k: config[k] for k in ('new_parameter_worlds', 'input_tokens', 'supervised_tokens', 'relative_input_difference')}), flush=True)
    return tokenizer, data, validation, config


def train(data, config):
    model = make_student(MODEL, 301)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.0)
    records = []
    inputs = supervised = 0
    compute_seconds = 0.0
    torch.cuda.reset_peak_memory_stats()

    def save(step):
        state = capture_training_state(model, optimizer)
        state.update(step=step, records=records, variant='parameter_diversity')
        path = ROOT/'step_checkpoint.tmp'
        torch.save(state, path)
        path.replace(ROOT/'step_checkpoint.pt')

    started = time.perf_counter()
    for step, indices in enumerate(config['schedule'], 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        tick = time.perf_counter()
        mean_loss = 0.0
        for index in indices:
            example = data[index]
            total, count = resident_loss(model, *tensors(example))
            loss = total/count/len(indices)
            loss.backward()
            mean_loss += float(loss.detach())
            inputs += len(example['ids'])
            supervised += count
            del total, loss
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not np.isfinite(mean_loss) or not torch.isfinite(norm):
            raise ValueError('Nonfinite update; keep last completed checkpoint.')
        optimizer.step()
        torch.cuda.synchronize()
        compute_seconds += time.perf_counter()-tick
        row = {'step': step, 'indices': indices, 'loss': mean_loss, 'gradient_norm': float(norm),
               'training_seconds': compute_seconds, 'input_tokens': inputs, 'supervised_tokens': supervised,
               'terminal_presentations': 2*step}
        records.append(row)
        append(ROOT/'training.jsonl', row)
        if step % 24 == 0:
            save(step)
            print(f'TRAIN {step}/192, seconds={compute_seconds:.1f}, loss={mean_loss:.5f}', flush=True)
        if time.perf_counter()-started > 600 and step < 192:
            save(step)
            raise TimeoutError('Fixed training cap reached; no checkpoint score substitution or extension.')
    wall = time.perf_counter()-started
    model.save_pretrained(ROOT/'adapter', safe_serialization=True)
    training = {'updates': 192, 'terminal_presentations': 384, 'input_tokens': inputs,
        'supervised_tokens': supervised, 'training_seconds': compute_seconds,
        'training_wall_seconds_including_checkpoints': wall,
        'peak_training_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
        'adapter_sha256': sha(ROOT/'adapter/adapter_model.safetensors')}
    atomic_json(ROOT/'training_result.json', training)
    return model, training


def score(model, tokenizer, validation):
    grammar = Grammar()
    rows = []
    model.eval()
    with torch.inference_mode():
        for view in validation:
            evidence = student_v3.evidence_only(view)
            prompt = terminal_prompt(evidence, 'raw')
            ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors='pt').to('cuda')
            torch.cuda.synchronize()
            started = time.perf_counter()
            generated = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=128,
                do_sample=False, repetition_penalty=1.0, pad_token_id=tokenizer.eos_token_id, use_cache=True)
            torch.cuda.synchronize()
            text = tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True)
            row = {'episode_id': view['episode_id'], 'text': text,
                'prompt_sha256': hashlib.sha256(json.dumps(prompt, ensure_ascii=False).encode()).hexdigest(),
                'input_tokens': ids.shape[1], 'output_tokens': generated.shape[1]-ids.shape[1],
                'generation_seconds': time.perf_counter()-started, 'valid_expression': False,
                'raw_support_success': False, 'full_recovery': False}
            truth = expression_from_dict(view['final_answer']['expression'])
            try:
                support = student_v3.parse_structures(text, grammar)
                fitted, fitting = student_v3.fit_coefficients(evidence, support)
            except (ValueError, TypeError, KeyError, IndexError) as error:
                row['error'] = str(error)
            else:
                row.update(valid_expression=True, raw_support_success=support == truth.support,
                           fitted_expression=fitted.to_dict(), fitting=fitting,
                           **evaluate(grammar, truth, fitted, evaluation_features(grammar, view)))
                row['full_recovery'] &= fitting['identifiable_given_proposal']
            rows.append(row)
            append(ROOT/'episodes.jsonl', row)
            print(f"SCORE {row['episode_id']} recovered={row['full_recovery']}", flush=True)
    return rows


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    tokenizer, data, validation, config = prepare()
    try:
        model, training = train(data, config)
        rows = score(model, tokenizer, validation)
        count = sum(r['full_recovery'] for r in rows)
        result = {'completed': True, 'training': training, 'nonempty_worlds': 10,
            'control_nonempty_recovered': 4, 'diverse_nonempty_recovered': count,
            'improved': count > 4, 'decision': 'eligible_for_later_expansion' if count > 4 else 'stop_repeating_this_configuration',
            'raw_support_recovered': sum(r['raw_support_success'] for r in rows),
            'valid_answers': sum(r['valid_expression'] for r in rows),
            'generation_seconds': sum(r['generation_seconds'] for r in rows),
            'elapsed_seconds': time.perf_counter()-started, 'config_sha256': sha(ROOT/'config.json'),
            'episodes_sha256': sha(ROOT/'episodes.jsonl'), 'formal_tests_used': False}
        for hashes in (config['protected_sha256'], config['source_sha256']):
            if any(sha(path) != digest for path, digest in hashes.items()):
                raise ValueError('A protected result/input/source changed while running.')
        atomic_json(ROOT/'result.json', result)
        print('RESULT '+json.dumps(result), flush=True)
    except BaseException as error:
        atomic_json(ROOT/'failure.json', {'type': type(error).__name__, 'message': str(error),
            'elapsed_seconds': time.perf_counter()-started, 'completed': False})
        raise


if __name__ == '__main__':
    main()
