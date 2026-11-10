"""Bounded common-task check: all 24 noiseless worlds, structure-only decoding."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from audit_student_validation_scope import classify
from discovery import student_v3 as formatting
from discovery.noise_evaluation import evaluate_noisy
from discovery.physics import Grammar
from discovery.student_v2 import encode, load_views, normalized_task_loss
from discovery.teaching import expression_from_dict
from evaluate_fixed_evidence_baselines import infer_sparse
from run_structure_discovery import compile_queries, evaluate as evaluate_exact, evaluation_queries
from train_student_pilot import (bounded_loss, capture_training_state, json_write, make_student,
                                restore_training_state, tensors)

MODEL = Path('models/Qwen2.5-0.5B-Instruct')
OUTPUT = Path('results/student_training_v3')
CHECKS = (0, 48, 96, 192)
TOLERANCE = {'uniform_rmse': 0.01, 'biased_rmse': 0.01, 'coefficient_max_error': 0.002}
SEED = 301
SOURCE_PATHS = ('train_student_v3.py', 'discovery/student_v3.py', 'discovery/student_spins.py',
                'discovery/student_v2.py', 'discovery/student_format.py', 'discovery/physics.py',
                'train_student_pilot.py', 'audit_student_validation_scope.py',
                'STUDENT_STRUCTURE_FIT_PROTOCOL.md')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def old_hashes():
    paths = []
    for root in ('student_training_v1', 'student_training_v2', 'student_training_v2_spins'):
        directory = Path('results')/root
        for pattern in ('**/adapter_model.safetensors', '**/step_checkpoint.pt', '**/validation.json',
                        'PAUSED.json', 'result.json', 'train_step*.json', 'validation_step*.json'):
            paths.extend(directory.glob(pattern))
    return {str(p): sha(p) for p in sorted(set(paths))}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    json_write(temporary, value)
    temporary.replace(path)


def evaluation_features(grammar, view):
    world = int(view['episode_id'][1:5])
    return [compile_queries(grammar, q) for q in
            evaluation_queries(np.random.default_rng([95301, world]), 256)]


def summarize(rows, split, step):
    keys = ('valid_expression', 'raw_support_success', 'raw_false_proposal', 'prediction_success',
            'support_success', 'full_recovery', 'false_discovery', 'rank_deficient_proposal')
    summary = {'split': split, 'step': step, 'worlds': len(rows),
               **{key: sum(bool(row.get(key, False)) for row in rows) for key in keys},
               'nonempty_worlds': sum(not r['null_world'] for r in rows),
               'nonempty_recoveries': sum(r['full_recovery'] and not r['null_world'] for r in rows),
               'null_worlds': sum(r['null_world'] for r in rows),
               'invalid_null_answers': sum(r['null_world'] and not r['valid_expression'] for r in rows),
               'exact_recoveries': sum(r.get('original_exact_benchmark', {}).get('full_recovery', False) for r in rows),
               'generation_seconds': sum(r.get('generation_seconds', 0) for r in rows),
               'results': rows}
    summary['gate_passed'] = (summary['valid_expression'] >= (22 if split == 'train' else 10)
                              and summary['nonempty_recoveries'] >= (16 if split == 'train' else 5))
    return summary


def score(model, tokenizer, examples, views, split, step, directory, feature_cache):
    path = directory/f'{split}_step{step:03d}.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    partial_path = directory/f'{split}_step{step:03d}_partial.json'
    rows = json.loads(partial_path.read_text(encoding='utf-8')) if partial_path.exists() else []
    assert [r['episode_id'] for r in rows] == [v['episode_id'] for v in views[:len(rows)]]
    grammar = Grammar()
    model.eval()
    with torch.inference_mode():
        for index in range(len(rows), len(views)):
            view, example = views[index], examples[index]
            evidence = formatting.evidence_only(view)
            prompt = formatting.terminal_prompt(evidence)
            assert prompt == example['messages'][:-1]
            ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True,
                                                return_tensors='pt').to('cuda')
            start = time.perf_counter()
            output = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=128,
                                    do_sample=False, repetition_penalty=1.0,
                                    pad_token_id=tokenizer.eos_token_id, use_cache=True)
            answer = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
            row = {'episode_id': view['episode_id'], 'split': split, 'step': step,
                   'null_world': not view['final_answer']['expression']['terms'], 'text': answer,
                   'input_tokens': ids.shape[1], 'output_tokens': output.shape[1]-ids.shape[1],
                   'generation_seconds': time.perf_counter()-start}
            try:
                support = formatting.parse_structures(answer, grammar)
                fitted, fit = formatting.fit_coefficients(evidence, support)
            except (ValueError, TypeError, IndexError, KeyError) as error:
                row.update(valid_expression=False, full_recovery=False, parsing_error=str(error))
            else:
                # Truth and independent prediction arrays enter only after fitting.
                truth = expression_from_dict(view['final_answer']['expression'])
                features = feature_cache[view['episode_id']]
                row.update(valid_expression=True, raw_support_success=support == truth.support,
                           raw_false_proposal=not truth.terms and bool(support),
                           rank_deficient_proposal=not fit['identifiable_given_proposal'],
                           fitted_expression=fitted.to_dict(), fitting=fit,
                           **evaluate_noisy(grammar, truth, fitted, features, TOLERANCE))
                row['full_recovery'] = row['full_recovery'] and fit['identifiable_given_proposal']
                row['original_exact_benchmark'] = evaluate_exact(grammar, truth, fitted, features)
            rows.append(row)
            atomic_json(partial_path, rows)
            print(f'v3 check step={step} {split} {index+1}/{len(views)} '
                  f'valid={row["valid_expression"]} recovered={row["full_recovery"]}', flush=True)
            del ids, output
            gc.collect()
            torch.cuda.empty_cache()
    summary = summarize(rows, split, step)
    atomic_json(path, summary)
    model.train()
    return summary


def baseline(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/'fixed_evidence_baseline.json'
    if path.exists():
        raise ValueError('Keep the existing baseline result.')
    grammar, rows = Grammar(), []
    for split in ('train', 'validation'):
        for view in load_views(split):
            start = time.perf_counter()
            hypothesis = infer_sparse(view['initial'], view['experiments'], 0.0, grammar)
            seconds = time.perf_counter()-start
            truth = expression_from_dict(view['final_answer']['expression'])
            features = evaluation_features(grammar, view)
            rows.append({'episode_id': view['episode_id'], 'split': split, 'null_world': not truth.terms,
                         'expression': hypothesis.to_dict(), 'inference_seconds': seconds,
                         **evaluate_noisy(grammar, truth, hypothesis, features, TOLERANCE),
                         'original_exact_benchmark': evaluate_exact(grammar, truth, hypothesis, features)})
    summary = {split: {'worlds': sum(r['split'] == split for r in rows),
                       'full_recoveries': sum(r['full_recovery'] for r in rows if r['split'] == split),
                       'exact_recoveries': sum(r['original_exact_benchmark']['full_recovery']
                                               for r in rows if r['split'] == split)}
               for split in ('train', 'validation')}
    atomic_json(path, {'queries_per_world': 32, 'extra_queries': 0, 'summary': summary, 'results': rows,
                      'source_sha256': sha(__file__), 'formal_test_worlds_used': False})
    print(summary, flush=True)


def prepare(tokenizer):
    views = {split: load_views(split) for split in ('train', 'validation')}
    ids = [v['episode_id'].split('-')[0] for v in views['train']]
    assert ids == [f'w{i:04d}' for i in range(24)] and len(views['validation']) == 12
    cohorts = classify(views['train'], views['validation'], ids)
    assert set(cohorts.values()) == {'seen_structure'}
    data = {split: [dict(encode(tokenizer, formatting.terminal_messages(v)), episode_id=v['episode_id'])
                    for v in entries] for split, entries in views.items()}
    for split in views:
        c_views = {v['episode_id']: v for v in load_views(split, 'C')}
        for example in data[split]:
            assert example['messages'] == formatting.terminal_messages(c_views[example['episode_id']])
    return views, data, cohorts


def run(output, resume=False):
    output = Path(output)
    if (output/'result.json').exists():
        raise ValueError('This v3 run is complete; preserve it.')
    if not resume and output.exists() and any(p.name != 'fixed_evidence_baseline.json' for p in output.iterdir()):
        raise ValueError('Use --resume for this exact incomplete run; never overwrite.')
    if not (output/'fixed_evidence_baseline.json').exists():
        raise ValueError('Run --baseline-only first to verify the shared evidence.')
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    views, data, cohorts = prepare(tokenizer)
    rng = np.random.default_rng(SEED)
    schedule = []
    for _ in range(16):
        order = list(map(int, rng.permutation(24)))
        schedule += [order[i:i+2] for i in range(0, 24, 2)]
    config = {'stage': 'structure_proposal_plus_shared_coefficient_fit', 'seed': SEED, 'sigma': 0.0,
              'train_world_ids': [v['episode_id'].split('-')[0] for v in views['train']],
              'validation_cohorts': cohorts, 'maximum_updates': 192, 'checkpoints': list(CHECKS),
              'schedule': schedule, 'learning_rate': 2e-4, 'weight_decay': 0.0,
              'lora': {'r': 8, 'alpha': 16, 'dropout': 0.05},
              'answer_occurrences_per_world_at_maximum': 16,
              'training_gate': {'minimum_valid': 22, 'minimum_nonempty_recovered': 16},
              'development_validation_gate': {'minimum_valid': 10, 'minimum_nonempty_recovered': 5},
              'zero_false_discoveries_required': False, 'development_tolerances': TOLERANCE,
              'full_column_rank_required': True, 'original_exact_benchmark_scored_separately': True,
              'source_sha256': {p: sha(p) for p in SOURCE_PATHS},
              'data_sha256': {f'{s}/{g}': sha(f'data/research_curriculum_v1/views/{s}/{g}.jsonl')
                              for s in ('train', 'validation') for g in ('B', 'C')},
              'baseline_sha256': sha(output/'fixed_evidence_baseline.json'),
              'protected_old_hashes': old_hashes(), 'formal_test_worlds_used': False,
              'decoder': {'max_new_tokens': 128, 'do_sample': False, 'repetition_penalty': 1.0},
              'loss_weights': {'terminal': 1.0, 'auxiliary': 0.0},
              'model_revision': json.loads((MODEL/'download_manifest.json').read_text(encoding='utf-8'))['revision']}
    if resume:
        if config != json.loads((output/'config.json').read_text(encoding='utf-8')):
            raise ValueError('Configuration, source or data changed; cannot resume.')
    else:
        atomic_json(output/'config.json', config)
        for source in SOURCE_PATHS:
            destination = output/'sources'/source
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(Path(source).read_bytes())
        for split, examples in data.items():
            with (output/f'{split}_terminal.jsonl').open('w', encoding='utf-8') as handle:
                for example in examples:
                    handle.write(json.dumps({'episode_id': example['episode_id'], 'messages': example['messages']},
                                            ensure_ascii=False)+'\n')
        atomic_json(output/'data_audit.json', {'B_C_terminal_tasks_identical': True,
                    'only_structures_in_supervised_answers': True, 'no_labels_in_inference_prompt': True,
                    'validation_classification_uses_actual_training': cohorts,
                    'train': [{'episode_id': e['episode_id'], 'input_tokens': len(e['ids']),
                               'supervised_tokens': int(np.count_nonzero(e['labels'][1:] != -100))}
                              for e in data['train']]})
    grammar = Grammar()
    feature_cache = {v['episode_id']: evaluation_features(grammar, v) for split in views for v in views[split]}
    model = make_student(MODEL, SEED)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.0)
    records, completed, training_seconds = [], 0, 0.0
    checkpoint = output/'step_checkpoint.pt'

    def save():
        saved = capture_training_state(model, optimizer)
        saved.update(step=completed, records=records, training_seconds=training_seconds)
        temporary = output/'step_checkpoint.tmp'
        torch.save(saved, temporary)
        temporary.replace(checkpoint)

    if resume:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        restore_training_state(model, optimizer, saved)
        records, completed, training_seconds = saved['records'], saved['step'], saved['training_seconds']
        del saved
    else:
        save()
    evaluations = [json.loads(path.read_text(encoding='utf-8')) for s in CHECKS
                   if (path := output/f'train_step{s:03d}.json').exists() and s <= completed]
    if completed in CHECKS and not any(e['step'] == completed for e in evaluations):
        evaluations.append(score(model, tokenizer, data['train'], views['train'], 'train', completed, output, feature_cache))
    passed = any(e['gate_passed'] for e in evaluations)
    for step, indices in enumerate(schedule, 1):
        if step <= completed:
            continue
        if passed:
            break
        start = time.perf_counter()
        model.train()
        model.config.use_cache = False
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for index in indices:
            loss_sum, count = bounded_loss(model, *tensors(data['train'][index]))
            loss = normalized_task_loss(loss_sum, count)/len(indices)
            loss.backward()
            total += float(loss.detach())
            del loss, loss_sum
            gc.collect()
            torch.cuda.empty_cache()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not np.isfinite(total) or not torch.isfinite(norm):
            raise RuntimeError('Non-finite structure training.')
        optimizer.step()
        torch.cuda.synchronize()
        training_seconds += time.perf_counter()-start
        completed = step
        records.append({'step': step, 'episode_indices': indices, 'terminal_loss': total,
                        'gradient_norm': float(norm), 'training_seconds': training_seconds,
                        'processed_tokens': sum(len(data['train'][i]['ids']) for b in schedule[:step] for i in b),
                        'supervised_tokens': sum(int(np.count_nonzero(data['train'][i]['labels'][1:] != -100))
                                                 for b in schedule[:step] for i in b),
                        'terminal_presentations': step*2})
        save()
        with (output/'training.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(records[-1])+'\n')
        print(f'v3 common {step}/192 terminal_loss={total:.4f} training_seconds={training_seconds:.1f}', flush=True)
        if step in CHECKS:
            evaluation = score(model, tokenizer, data['train'], views['train'], 'train', step, output, feature_cache)
            evaluations.append(evaluation)
            passed = evaluation['gate_passed']
    model.save_pretrained(output/'adapter', safe_serialization=True)
    validation = score(model, tokenizer, data['validation'], views['validation'], 'validation', completed, output, feature_cache)
    if old_hashes() != config['protected_old_hashes']:
        raise AssertionError('An old experiment file changed.')
    ready = passed and validation['gate_passed']
    atomic_json(output/'result.json', {'stage': config['stage'], 'completed_updates': completed,
                'training_gate_passed': passed, 'validation_gate_passed': validation['gate_passed'],
                'ready_for_small_B_C': ready, 'training_seconds': training_seconds,
                'training_checks': evaluations, 'validation': validation,
                'processed_tokens': records[-1]['processed_tokens'] if records else 0,
                'supervised_tokens': records[-1]['supervised_tokens'] if records else 0,
                'terminal_presentations': completed*2, 'v1_v2_preserved': True,
                'formal_test_worlds_used': False, 'B_C_effect_tested': False,
                'next_action': 'define_and_run_small_paired_B_C' if ready else 'record_failed_common_task_gate_before_scaling'})
    print(f'Completed v3. Train gate={passed}; validation gate={validation["gate_passed"]}; ready B/C={ready}.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(OUTPUT))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--baseline-only', action='store_true')
    args = parser.parse_args()
    baseline(args.output) if args.baseline_only else run(args.output, args.resume)
