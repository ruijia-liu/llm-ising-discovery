"""Bounded six-world, noiseless terminal-task learning check; separate v2 run.

Start from the pinned base, never from an old B or C adapter. This common-task
positive control is NOT a B/C experiment or a structure-generalization result.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from train_student_pilot import (bounded_loss, capture_training_state, json_write,
                                make_student, parse_expression, restore_training_state, tensors)
from discovery.student_v2 import (TRAIN_IDS, auxiliary_messages, encode, evidence_only, load_views,
                                  normalized_task_loss, objective_weights, terminal_messages, terminal_prompt)
from discovery.physics import Grammar
from discovery.teaching import expression_from_dict
from discovery.noise_evaluation import evaluate_noisy
from run_structure_discovery import compile_queries, evaluate as evaluate_exact, evaluation_queries

MODEL = Path('models/Qwen2.5-0.5B-Instruct')
OUTPUT = Path('results/student_training_v2')
CHECKS = (0, 12, 24, 48)
# Development gate fixed before generation/training. Original exact scores also
# retained separately; this numerical tolerance is NOT a changed benchmark score.
TOLERANCE = {'uniform_rmse': 0.01, 'biased_rmse': 0.01, 'coefficient_max_error': 0.002}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protect_v1():
    root = Path('results/student_training_v1')
    paths = [root/'PAUSED.json', root/'C_seed102/step_checkpoint.pt']
    for group in ('B', 'C'):
        paths += [root/f'{group}_seed101/adapter/adapter_model.safetensors', root/f'{group}_seed101/validation.json']
    return {str(p): sha(p) for p in paths}


def prepare(tokenizer, formatting=None):
    make_messages = formatting.terminal_messages if formatting else terminal_messages
    views = {s: load_views(s) for s in ('train', 'validation')}
    views['train'] = [v for v in views['train'] if v['episode_id'].split('-')[0] in TRAIN_IDS]
    if [v['episode_id'].split('-')[0] for v in views['train']] != list(TRAIN_IDS):
        raise AssertionError('Six-world selection changed.')
    data = {s: [dict(encode(tokenizer, make_messages(v)), episode_id=v['episode_id']) for v in values]
            for s, values in views.items()}
    # Verify all B/C terminal tasks are identical and auxiliary records cannot
    # increase terminal-answer exposures. Auxiliary tasks are prepared, not used
    # by this first common-task learning check.
    auxiliary = {'B': [], 'C': []}
    for group in ('B', 'C'):
        paired = {v['episode_id']: v for v in load_views('train', group)}
        for view, example in zip(views['train'], data['train']):
            other = paired[view['episode_id']]
            assert make_messages(other) == example['messages']
            messages = auxiliary_messages(other, group)
            task = encode(tokenizer, messages)
            auxiliary[group].append({'episode_id': view['episode_id'], 'messages': messages,
                                     'tokens': len(task['ids']),
                                     'supervised_tokens': int(np.count_nonzero(task['labels'][1:] != -100)),
                                     'terminal_answers': 0})
    return views, data, auxiliary


def score(model, tokenizer, examples, views, split, step, directory, formatting=None):
    make_prompt = formatting.terminal_prompt if formatting else terminal_prompt
    make_evidence = formatting.evidence_only if formatting else evidence_only
    model.eval()
    grammar = Grammar()
    results = []
    with torch.inference_mode():
        for index, (example, view) in enumerate(zip(examples, views)):
            prompt = make_prompt(make_evidence(view))
            assert prompt == example['messages'][:-1]
            ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True,
                                                return_tensors='pt').to('cuda')
            start = time.perf_counter()
            output = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=192,
                                    do_sample=False, repetition_penalty=1.0,
                                    pad_token_id=tokenizer.eos_token_id, use_cache=True)
            answer = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
            row = {'episode_id': view['episode_id'], 'split': split, 'step': step,
                   'null_world': not view['final_answer']['expression']['terms'], 'text': answer,
                   'input_tokens': ids.shape[1], 'output_tokens': output.shape[1]-ids.shape[1],
                   'generation_seconds': time.perf_counter()-start}
            try:
                prediction = parse_expression(answer, grammar)
                truth = expression_from_dict(view['final_answer']['expression'])
                world = int(view['episode_id'][1:5])
                queries = evaluation_queries(np.random.default_rng([95201, world]), 256)
                features = [compile_queries(grammar, q) for q in queries]
                row.update(valid_expression=True, **evaluate_noisy(grammar, truth, prediction, features, TOLERANCE))
                row['original_exact_benchmark'] = evaluate_exact(grammar, truth, prediction, features)
            except (ValueError, TypeError, IndexError, KeyError) as error:
                row.update(valid_expression=False, full_recovery=False, prediction_success=False,
                           support_success=False, parsing_error=str(error))
            results.append(row)
            print(f'check step={step} {split} {index+1}/{len(examples)} valid={row["valid_expression"]} '
                  f'physical={row["full_recovery"]}', flush=True)
            del ids, output
            gc.collect()
            torch.cuda.empty_cache()
    summary = {'step': step, 'split': split, 'worlds': len(results),
               **{k: sum(bool(r.get(k, False)) for r in results) for k in
                  ('valid_expression', 'full_recovery', 'support_success', 'prediction_success', 'false_discovery')},
               'nonempty_recoveries': sum(r['full_recovery'] and not r['null_world'] for r in results),
               'null_worlds': sum(r['null_world'] for r in results),
               'invalid_null_answers': sum(r['null_world'] and not r['valid_expression'] for r in results),
               'exact_recoveries': sum(r.get('original_exact_benchmark', {}).get('full_recovery', False) for r in results),
               'results': results}
    summary['learning_gate_passed'] = (split == 'train' and summary['valid_expression'] >= 5
                                        and summary['nonempty_recoveries'] >= 4)
    json_write(directory/f'{split}_step{step:03d}.json', summary)
    model.train()
    return summary


def run(output, resume=False, representation='hex'):
    if representation == 'local-spins':
        from discovery import student_spins as formatting
        prior = json.loads((OUTPUT/'result.json').read_text(encoding='utf-8'))
        if prior['learning_gate_passed']:
            raise ValueError('Input fallback is only planned after the common learning check fails.')
    elif representation == 'hex':
        formatting = None
    else:
        raise ValueError('Unknown input representation.')
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError('Preserve prior output; use --resume for this exact incomplete v2 run.')
    if (output/'result.json').exists():
        raise ValueError('This learning check is already complete.')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    views, data, auxiliary = prepare(tokenizer, formatting)
    rng = np.random.default_rng(201)
    schedule = []
    for _ in range(16):
        order = list(map(int, rng.permutation(6)))
        schedule += [order[i:i+2] for i in range(0, 6, 2)]
    config = {'stage': 'common_terminal_learning_check', 'seed': 201, 'sigma': 0.0,
              'input_representation': representation,
              'train_world_ids': list(TRAIN_IDS), 'validation_worlds': 12,
              'maximum_updates': 48, 'checkpoints': list(CHECKS), 'schedule': schedule,
              'learning_rate': 2e-4, 'lora': {'r': 8, 'alpha': 16, 'dropout': 0.05},
              'answer_occurrences_per_world_at_maximum': 16,
              'learning_gate': '>=5/6 executable AND >=4/5 nonempty mechanisms independently recovered',
              'zero_false_discoveries_required': False, 'development_tolerances': TOLERANCE,
              'original_exact_benchmark_scored_separately': True,
              'source_sha256': sha(__file__), 'format_source_sha256': sha('discovery/student_v2.py'),
              'representation_source_sha256': sha(formatting.__file__) if formatting else sha('discovery/student_v2.py'),
              'curriculum_sha256': {f'{split}/{g}': sha(f'data/research_curriculum_v1/views/{split}/{g}.jsonl')
                                    for split in ('train', 'validation') for g in ('B', 'C')},
              'v1_hashes': protect_v1(), 'formal_test_worlds_used': False,
              'decoder': {'max_new_tokens': 192, 'do_sample': False, 'repetition_penalty': 1.0},
              'loss_weights': {g: objective_weights(g) for g in ('common', 'B', 'C')},
              'auxiliary_tasks_used_in_this_check': False,
              'model_revision': json.loads((MODEL/'download_manifest.json').read_text(encoding='utf-8'))['revision']}
    if resume:
        original = json.loads((output/'config.json').read_text(encoding='utf-8'))
        if config != original:
            raise ValueError('Configuration/source/data changed; cannot resume this run.')
    else:
        json_write(output/'config.json', config)
        (output/'executed_trainer.py').write_bytes(Path(__file__).read_bytes())
        (output/'executed_format.py').write_bytes(Path('discovery/student_v2.py').read_bytes())
        if formatting:
            (output/'executed_representation.py').write_bytes(Path(formatting.__file__).read_bytes())
        for split in ('train', 'validation'):
            with (output/f'{split}_terminal.jsonl').open('w', encoding='utf-8') as handle:
                for example in data[split]:
                    handle.write(json.dumps({'episode_id': example['episode_id'], 'messages': example['messages']}, ensure_ascii=False)+'\n')
        for group in ('B', 'C'):
            with (output/f'{group}_auxiliary.jsonl').open('w', encoding='utf-8') as handle:
                for example in auxiliary[group]:
                    handle.write(json.dumps(example, ensure_ascii=False)+'\n')
        json_write(output/'data_audit.json', {
            'paired_terminal_tasks_identical': True, 'terminal_gold_in_auxiliary_tasks': False,
            'same_prompt_for_training_and_inference': True,
            'train': [{'episode_id': e['episode_id'], 'tokens': len(e['ids']),
                       'terminal_tokens': int(np.count_nonzero(e['labels'][1:] != -100))} for e in data['train']],
            'auxiliary_summary': {g: [{k: v for k, v in e.items() if k != 'messages'} for e in entries]
                                  for g, entries in auxiliary.items()}})
    model = make_student(MODEL, 201)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.0)
    records, completed, training_seconds = [], 0, 0.0
    checkpoint = output/'step_checkpoint.pt'
    if resume:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        restore_training_state(model, optimizer, saved)
        records, completed, training_seconds = saved['records'], saved['step'], saved['training_seconds']
        del saved
    evaluations = []
    for step in CHECKS:
        path = output/f'train_step{step:03d}.json'
        if path.exists():
            evaluations.append(json.loads(path.read_text(encoding='utf-8')))
    if not evaluations:
        evaluations.append(score(model, tokenizer, data['train'], views['train'], 'train', 0, output, formatting))
    # A process may stop after atomically saving an update but before its scheduled
    # evaluation. Recover that check before deciding whether to train further.
    if completed in CHECKS and completed > 0 and not any(e['step'] == completed for e in evaluations):
        evaluations.append(score(model, tokenizer, data['train'], views['train'], 'train', completed, output, formatting))
    passed = any(e['learning_gate_passed'] for e in evaluations)
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
            episode = data['train'][index]
            loss_sum, count = bounded_loss(model, *tensors(episode))
            # Every world/terminal answer has equal weight, independently of
            # answer length. No historical assistant messages enter this loss.
            loss = normalized_task_loss(loss_sum, count)/len(indices)
            loss.backward()
            total += float(loss.detach())
            del loss, loss_sum
            gc.collect()
            torch.cuda.empty_cache()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not np.isfinite(total) or not torch.isfinite(gradient_norm):
            raise RuntimeError('Non-finite terminal-task training.')
        optimizer.step()
        torch.cuda.synchronize()
        training_seconds += time.perf_counter()-start
        completed = step
        record = {'step': step, 'episode_indices': indices, 'terminal_loss': total,
                  'gradient_norm': float(gradient_norm), 'training_seconds': training_seconds,
                  'processed_tokens': sum(len(data['train'][i]['ids']) for batch in schedule[:step] for i in batch),
                  'terminal_presentations': step*2}
        records.append(record)
        saved = capture_training_state(model, optimizer)
        saved.update(step=step, records=records, training_seconds=training_seconds)
        temporary = output/'step_checkpoint.tmp'
        torch.save(saved, temporary)
        temporary.replace(checkpoint)
        del saved
        with (output/'training.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record)+'\n')
        print(f'v2 common {step}/48 terminal_loss={total:.4f} training_seconds={training_seconds:.1f}', flush=True)
        if step in CHECKS:
            evaluation = score(model, tokenizer, data['train'], views['train'], 'train', step, output, formatting)
            evaluations.append(evaluation)
            passed = evaluation['learning_gate_passed']
    model.save_pretrained(output/'adapter', safe_serialization=True)
    # Validation is opened once for generation, after the stopping step has been
    # determined using training-world checks only. No validation-driven tuning.
    validation = score(model, tokenizer, data['validation'], views['validation'], 'validation', completed, output, formatting)
    if protect_v1() != config['v1_hashes']:
        raise AssertionError('Old experiment files changed.')
    json_write(output/'result.json', {'stage': 'common_terminal_learning_check', 'completed_updates': completed,
               'input_representation': representation,
               'learning_gate_passed': passed, 'training_seconds': training_seconds,
               'v1_preserved': True, 'formal_test_worlds_used': False,
               'training_checks': evaluations, 'validation': validation,
               'next_action': 'paired_small_B_C_experiment' if passed else 'inspect_input_representation_and_model_capacity',
               'B_C_effect_tested': False})
    print(f'Completed v2 learning check. Gate passed: {passed}. Old batch remains paused.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(OUTPUT))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--representation', choices=('hex', 'local-spins'), default='hex')
    args = parser.parse_args()
    run(args.output, args.resume, args.representation)
