"""Bounded paired development screen. Never resumes the paused main study."""
import argparse
import gc
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
from discovery.physics import Grammar
from discovery.student_fastdev import resident_loss, terminal_messages, terminal_prompt
from discovery.student_v2 import encode, load_views
from discovery.teaching import expression_from_dict
from run_structure_discovery import evaluate
from train_student_pilot import bounded_loss, capture_training_state, make_student, tensors
from train_student_v3 import evaluation_features

MODEL = Path('models/Qwen2.5-0.5B-Instruct')
MAIN = Path('results/student_mainline_v1')
VARIANTS = ('raw', 'residual')
SEED = 301
STEPS = 192
CAP_SECONDS = 600
SOURCES = ('run_fast_development.py', 'discovery/student_fastdev.py', 'test_student_fastdev.py',
           'FAST_DEVELOPMENT_PROTOCOL.md', 'train_student_pilot.py', 'train_student_v3.py',
           'discovery/student_v3.py', 'discovery/student_spins.py', 'discovery/student_v2.py',
           'discovery/student_format.py', 'discovery/physics.py', 'discovery/teaching.py',
           'run_structure_discovery.py', 'build_publication_curriculum.py')


def protected_files():
    config = read(MAIN/'A_seed401/config.json')
    manifest = Path('data/student_mainline_v1/manifest.json')
    frozen = {**config['sources'], **read(manifest)['config']['sources']}
    if any(sha(p) != h for p, h in frozen.items()):
        raise ValueError('An original frozen source changed.')
    if sha(manifest) != config['manifest_sha256']:
        raise ValueError('Original cache manifest changed.')
    if read(MAIN/'queue_state.json')['status'] != 'stopped':
        raise ValueError('Original main queue must remain stopped.')
    paths = [manifest, MAIN/'queue_state.json', MAIN/'B_seed401/STOP_AFTER_CHECKPOINT',
             MAIN/'B_seed401/step_checkpoint.pt', MAIN/'B_seed401/training.jsonl',
             MAIN/'A_seed401/adapter/adapter_model.safetensors', MAIN/'A_seed401/result.json',
             Path('results/student_training_v3/result.json'),
             Path('results/student_training_coverage/result.json')]
    return {**frozen, **{str(p): sha(p) for p in paths}}


def profile(example, output):
    model = make_student(MODEL, SEED)
    model.train()
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    records = []
    for repetition in range(3):
        reference = None
        for name, loss_fn in (('original_offload', bounded_loss), ('resident', resident_loss)):
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.manual_seed(912+repetition)
            torch.cuda.manual_seed_all(912+repetition)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            total, count = loss_fn(model, *tensors(example))
            loss = total/count
            loss.backward()
            torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            value = float(loss.detach())
            gradients = {key: p.grad.detach().cpu().clone() for key, p in parameters.items()}
            row = {'path': name, 'repetition': repetition, 'warmup': repetition == 0,
                   'context_tokens': len(example['ids']), 'supervised_tokens': count,
                   'seconds': seconds, 'loss': value,
                   'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                   'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30}
            if not np.isfinite(value) or not all(torch.isfinite(g).all() for g in gradients.values()):
                raise ValueError('Non-finite profiling loss or gradient.')
            if reference is None:
                reference = value, gradients
            else:
                row['loss_absolute_difference'] = abs(value-reference[0])
                row['gradient_max_absolute_difference'] = max(float((g-reference[1][k]).abs().max())
                                                             for k, g in gradients.items())
                row['gradient_tensor_count'] = len(gradients)
                row['loss_and_gradients_match'] = bool(np.isclose(value, reference[0], rtol=1e-5, atol=1e-7)
                    and all(torch.allclose(g, reference[1][k], rtol=1e-5, atol=1e-7) for k, g in gradients.items()))
            records.append(row)
            atomic_json(output/'profile_partial.json', records)
            print('profile '+json.dumps(row), flush=True)
            if name == 'resident' and not row['loss_and_gradients_match']:
                raise ValueError('Resident path failed loss/gradient equivalence.')
            del total, loss, gradients
    means = {name: float(np.mean([r['seconds'] for r in records if r['path'] == name and not r['warmup']]))
             for name in ('original_offload', 'resident')}
    result = {'completed': True, 'records': records, 'mean_seconds_excluding_warmup': means,
              'speedup': means['original_offload']/means['resident'],
              'scope': 'one longest legacy short terminal task; no long C auxiliary benchmark'}
    atomic_json(output/'profile.json', result)
    del model, parameters, reference
    gc.collect()
    torch.cuda.empty_cache()
    return result


def score(model, tokenizer, views, variant, output):
    model.eval()
    grammar = Grammar()
    rows = []
    with torch.inference_mode():
        for split in ('train', 'validation'):
            for view in views[split]:
                evidence = student_v3.evidence_only(view)
                prompt = terminal_prompt(evidence, variant)
                assert prompt == terminal_messages(view, variant)[:-1]
                ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True,
                                                     return_tensors='pt').to('cuda')
                torch.cuda.synchronize()
                start = time.perf_counter()
                generated = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=128,
                    do_sample=False, repetition_penalty=1.0, pad_token_id=tokenizer.eos_token_id, use_cache=True)
                torch.cuda.synchronize()
                seconds = time.perf_counter()-start
                answer = tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True)
                # Only after generation: read the gold label and independent predictions.
                truth = expression_from_dict(view['final_answer']['expression'])
                row = {'episode_id': view['episode_id'], 'split': split, 'variant': variant,
                       'null_world': not bool(truth.terms), 'text': answer,
                       'prompt_sha256': hashlib.sha256(json.dumps(prompt, ensure_ascii=False).encode()).hexdigest(),
                       'input_tokens': ids.shape[1], 'output_tokens': generated.shape[1]-ids.shape[1],
                       'generation_seconds': seconds, 'valid_expression': False,
                       'raw_support_success': False, 'raw_false_proposal': False, 'full_recovery': False,
                       'false_discovery': False, 'prediction_success': False}
                try:
                    support = student_v3.parse_structures(answer, grammar)
                    fitted, fitting = student_v3.fit_coefficients(evidence, support)
                except (ValueError, TypeError, KeyError, IndexError) as error:
                    row['error'] = str(error)
                else:
                    row.update(valid_expression=True, raw_support_success=support == truth.support,
                               raw_false_proposal=not truth.terms and bool(support),
                               raw_structures=support, fitted_expression=fitted.to_dict(), fitting=fitting,
                               **evaluate(grammar, truth, fitted, evaluation_features(grammar, view)))
                    row['full_recovery'] = row['full_recovery'] and fitting['identifiable_given_proposal']
                rows.append(row)
                with (output/'episodes.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                print(f"{variant} {split} {view['episode_id']} raw={row['raw_support_success']} full={row['full_recovery']}", flush=True)
                del ids, generated
    summary = {}
    for split in views:
        selected = [r for r in rows if r['split'] == split]
        summary[split] = {'n': len(selected),
            **{key: sum(bool(r[key]) for r in selected) for key in
               ('valid_expression', 'raw_support_success', 'raw_false_proposal', 'full_recovery', 'false_discovery', 'prediction_success')},
            'nonempty_n': sum(not r['null_world'] for r in selected),
            'nonempty_recovered': sum(not r['null_world'] and r['full_recovery'] for r in selected),
            'null_n': sum(r['null_world'] for r in selected),
            'invalid_null': sum(r['null_world'] and not r['valid_expression'] for r in selected)}
    return summary, sum(r['generation_seconds'] for r in rows)


def train_arm(tokenizer, views, data, variant, schedule, output):
    output.mkdir()
    started = time.perf_counter()
    model = make_student(MODEL, SEED)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.0)
    records = []
    training_seconds = 0.0
    input_tokens = supervised_tokens = 0
    torch.cuda.reset_peak_memory_stats()

    def save(step):
        saved = capture_training_state(model, optimizer)
        saved.update(step=step, variant=variant, records=records, training_seconds=training_seconds)
        temporary = output/'step_checkpoint.tmp'
        torch.save(saved, temporary)
        temporary.replace(output/'step_checkpoint.pt')

    train_start = time.perf_counter()
    for step, indices in enumerate(schedule, 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        mean_loss = 0.0
        for index in indices:
            example = data[variant]['train'][index]
            total, count = resident_loss(model, *tensors(example))
            loss = total/count/len(indices)
            loss.backward()
            mean_loss += float(loss.detach())
            input_tokens += len(example['ids'])
            supervised_tokens += count
            del total, loss
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not np.isfinite(mean_loss) or not torch.isfinite(norm):
            save(step-1)
            raise ValueError('Non-finite training state.')
        optimizer.step()
        torch.cuda.synchronize()
        training_seconds += time.perf_counter()-start
        row = {'step': step, 'indices': indices, 'loss': mean_loss, 'gradient_norm': float(norm),
               'training_seconds': training_seconds, 'input_tokens': input_tokens,
               'supervised_tokens': supervised_tokens, 'terminal_presentations': 2*step,
               'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30}
        records.append(row)
        with (output/'training.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row)+'\n')
        if step % 24 == 0:
            save(step)
            print(f'{variant} {step}/{STEPS}, loss={mean_loss:.5f}, training_seconds={training_seconds:.1f}', flush=True)
        if time.perf_counter()-train_start > CAP_SECONDS and step < STEPS:
            save(step)
            atomic_json(output/'stopped.json', {'step': step, 'reason': 'fixed_600_second_cap', 'completed': False})
            raise TimeoutError('Development training cap reached; no checkpoint score comparison.')
    train_wall = time.perf_counter()-train_start
    model.save_pretrained(output/'adapter', safe_serialization=True)
    peak = torch.cuda.max_memory_allocated()/2**30
    summary, generation_seconds = score(model, tokenizer, views, variant, output)
    result = {'completed': True, 'variant': variant, 'updates': len(records), 'summary': summary,
              'training_seconds': training_seconds, 'training_wall_seconds_including_checkpoints': train_wall,
              'arm_wall_seconds': time.perf_counter()-started, 'generation_seconds': generation_seconds,
              'input_tokens': input_tokens, 'supervised_tokens': supervised_tokens,
              'peak_training_allocated_gib': peak, 'terminal_presentations': 2*len(records),
              'adapter_sha256': sha(output/'adapter/adapter_model.safetensors'),
              'episodes_sha256': sha(output/'episodes.jsonl')}
    atomic_json(output/'result.json', result)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Preserve any completed or partial development run; no overwrite.')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    protected = protected_files()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    views = {s: load_views(s) for s in ('train', 'validation')}
    if len(views['train']) != 24 or len(views['validation']) != 12:
        raise ValueError('Unexpected development cohort size.')
    data = {variant: {s: [encode(tokenizer, terminal_messages(v, variant)) for v in entries]
                      for s, entries in views.items()} for variant in VARIANTS}
    for split in views:
        for raw, residual in zip(data['raw'][split], data['residual'][split]):
            if raw['messages'][-1] != residual['messages'][-1] or not np.array_equal(
                    raw['labels'][raw['labels'] != -100], residual['labels'][residual['labels'] != -100]):
                raise ValueError('Answer supervision differs between variants.')
    rng = np.random.default_rng(SEED)
    schedule = []
    for _ in range(16):
        order = list(map(int, rng.permutation(24)))
        schedule.extend([order[i:i+2] for i in range(0, 24, 2)])
    if schedule != read('results/student_training_v3/config.json')['schedule']:
        raise ValueError('Schedule differs from the legacy common task.')
    config = {'scope': 'fast_development_v2_not_ABC', 'variants': VARIANTS, 'seed': SEED,
              'deterministic_algorithms': True, 'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'],
              'updates': STEPS, 'training_cap_seconds_per_arm': CAP_SECONDS, 'schedule': schedule,
              'source_sha256': {p: sha(p) for p in SOURCES}, 'protected_sha256': protected,
              'data_sha256': {str(Path('data/research_curriculum_v1/views')/s/'B.jsonl'):
                             sha(Path('data/research_curriculum_v1/views')/s/'B.jsonl') for s in views},
              'cohorts': {s: [v['episode_id'] for v in entries] for s, entries in views.items()},
              'token_audit': {variant: {s: [{'episode_id': v['episode_id'], 'tokens': len(e['ids']),
                  'supervised_tokens': int(np.count_nonzero(e['labels'][1:] != -100))}
                  for v, e in zip(views[s], examples)] for s, examples in values.items()}
                  for variant, values in data.items()},
              'environment': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
              'model_manifest_sha256': sha(MODEL/'download_manifest.json'),
              'formal_tests_used': False, 'main_training_resumed': False,
              'development_validation_previously_inspected': True}
    atomic_json(output/'config.json', config)
    for name in SOURCES:
        path = output/'sources'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(name).read_bytes())
    for variant in VARIANTS:
        with (output/f'{variant}_messages.jsonl').open('w', encoding='utf-8') as handle:
            for split, entries in data[variant].items():
                for view, entry in zip(views[split], entries):
                    handle.write(json.dumps({'split': split, 'episode_id': view['episode_id'],
                                             'messages': entry['messages']}, ensure_ascii=False)+'\n')
    try:
        timing = profile(max(data['raw']['train'], key=lambda e: len(e['ids'])), output)
        results = {variant: train_arm(tokenizer, views, data, variant, schedule, output/variant)
                   for variant in VARIANTS}
        if protected_files() != protected or any(sha(p) != h for p, h in config['source_sha256'].items()):
            raise ValueError('Protected original files or running sources changed.')
        result = {'completed': True, 'results': results, 'profile': timing,
                  'elapsed_seconds': time.perf_counter()-started, 'original_main_preserved_and_paused': True,
                  'formal_tests_used': False, 'config_sha256': sha(output/'config.json')}
        atomic_json(output/'result.json', result)
    except BaseException as error:
        atomic_json(output/'failure.json', {'type': type(error).__name__, 'error': str(error),
                    'elapsed_seconds': time.perf_counter()-started, 'completed': False,
                    'original_main_preserved_and_paused': protected_files() == protected})
        raise
    print('Completed '+str(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/fast_development_v2')
    run(parser.parse_args().output)
