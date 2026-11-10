"""One fixed-compute coverage contrast against the completed v3 control.

Same 24 support-label slots and 384 presentations, now rotating 120 worlds.
No test worlds, early stopping, validation tuning, or extra coefficient search.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from build_publication_curriculum import COVERAGE, read
from discovery import student_v3 as formatting
from discovery.physics import Grammar
from discovery.student_v2 import encode, load_views, normalized_task_loss
from discovery.teaching import expression_from_dict
from train_student_pilot import bounded_loss, capture_training_state, make_student, restore_training_state, tensors
from train_student_v3 import (MODEL, SEED, SOURCE_PATHS, atomic_json, evaluation_features,
                              old_hashes, score, sha)

OUTPUT = Path('results/student_training_coverage')
CONTROL = Path('results/student_training_v3')


def make_schedule(control_schedule, anchor_ids, groups):
    counts = {anchor: 0 for anchor in anchor_ids}
    schedule = []
    for batch in control_schedule:
        chosen = []
        for index in batch:
            anchor = anchor_ids[index]
            variants = groups[anchor]
            chosen.append(variants[counts[anchor] % len(variants)])
            counts[anchor] += 1
        schedule.append(chosen)
    return schedule


def protected_hashes():
    result = old_hashes()
    for pattern in ('result.json', 'config.json', '**/adapter_model.safetensors', 'step_checkpoint.pt',
                    'train_step*.json', 'validation_step*.json'):
        for path in CONTROL.glob(pattern):
            result[str(path)] = sha(path)
    return result


def prepare(tokenizer):
    audit = read(COVERAGE/'audit_report.json')
    if not audit['completed']:
        raise ValueError('Coverage corpus has not completed audit.')
    for name, expected in audit['export_sha256'].items():
        if sha(COVERAGE/name) != expected:
            raise ValueError('Coverage export changed after audit.')
    training = load_views('train', source=str(COVERAGE))
    assert len(training) == 120
    data = {v['episode_id']: dict(encode(tokenizer, formatting.terminal_messages(v)), episode_id=v['episode_id'])
            for v in training}
    views = {s: load_views(s) for s in ('train', 'validation')}
    examples = {s: [dict(encode(tokenizer, formatting.terminal_messages(v)), episode_id=v['episode_id'])
                    for v in values] for s, values in views.items()}
    groups = read(COVERAGE/'coverage_groups.json')
    view_map = {v['episode_id']: v for v in training}
    for anchor, members in groups.items():
        assert len(members) == 5 and members[0] == anchor+'-sigma0'
        expected = expression_from_dict(view_map[members[0]]['final_answer']['expression']).support
        assert all(expression_from_dict(view_map[m]['final_answer']['expression']).support == expected for m in members)
    # Ensure the actual scored prompts and labels are byte-for-byte identical to
    # the previous control, not just semantically similar regenerated prompts.
    for split in examples:
        saved = [json.loads(line) for line in (CONTROL/f'{split}_terminal.jsonl').read_text(encoding='utf-8').splitlines()]
        current = [{'episode_id': e['episode_id'], 'messages': e['messages']} for e in examples[split]]
        assert current == saved
    for c in load_views('train', 'C', source=str(COVERAGE)):
        assert formatting.terminal_messages(c) == data[c['episode_id']]['messages']
    return data, views, examples, groups


def run(output, resume=False):
    output = Path(output)
    if (output/'result.json').exists():
        raise ValueError('Preserve this completed one-shot diagnostic.')
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError('Use --resume for the exact incomplete run.')
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    data, views, examples, groups = prepare(tokenizer)
    control_config, control_result = read(CONTROL/'config.json'), read(CONTROL/'result.json')
    assert control_result['completed_updates'] == 192 and control_config['seed'] == SEED
    assert control_config['learning_rate'] == 2e-4 and control_config['lora'] == {'r': 8, 'alpha': 16, 'dropout': 0.05}
    assert control_config['decoder'] == {'max_new_tokens': 128, 'do_sample': False, 'repetition_penalty': 1.0}
    anchor_ids = [v['episode_id'].split('-')[0] for v in views['train']]
    schedule = make_schedule(control_config['schedule'], anchor_ids, groups)
    sources = set(SOURCE_PATHS) | {'train_student_coverage.py', 'PUBLICATION_DATA_PROTOCOL.md'}
    config = {'stage': 'fixed_compute_training_coverage_contrast', 'seed': SEED,
              'training_worlds': 120, 'same_support_label_slots': 24, 'maximum_updates': 192,
              'terminal_presentations': 384, 'schedule': schedule,
              'scored_training_subset': 'same 24 legacy anchor worlds, not all 120 training worlds',
              'validation': 'same 12 development worlds and prompts as v3, once after update 192',
              'early_stopping': False, 'validation_driven_model_selection': False,
              'learning_rate': control_config['learning_rate'], 'lora': control_config['lora'],
              'decoder': control_config['decoder'], 'tolerances': control_config['development_tolerances'],
              'source_sha256': {s: sha(s) for s in sorted(sources)},
              'coverage_audit_sha256': sha(COVERAGE/'audit_report.json'),
              'coverage_groups_sha256': sha(COVERAGE/'coverage_groups.json'),
              'control_result_sha256': sha(CONTROL/'result.json'),
              'control_adapter_sha256': sha(CONTROL/'adapter/adapter_model.safetensors'),
              'model_revision': control_config['model_revision'],
              'old_result_hashes': protected_hashes(), 'formal_tests_used': False}
    if resume:
        if config != read(output/'config.json'):
            raise ValueError('Source/configuration/data changed; cannot resume.')
    else:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output/'config.json', config)
        for path in sources:
            destination = output/'sources'/path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(Path(path).read_bytes())
        atomic_json(output/'data_audit.json', {'matched_terminal_targets_per_anchor': True,
                    'B_C_terminal_data_identical': True, 'control_evaluation_prompts_identical': True,
                    'training_worlds': len(data), 'anchor_ids': anchor_ids,
                    'examples': [{'episode_id': key, 'tokens': len(e['ids']),
                                  'supervised_tokens': int(np.count_nonzero(e['labels'][1:] != -100))}
                                 for key, e in data.items()]})
    model = make_student(MODEL, SEED)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.0)
    completed, training_seconds, records = 0, 0.0, []
    checkpoint = output/'step_checkpoint.pt'
    if resume:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        restore_training_state(model, optimizer, saved)
        completed, training_seconds, records = saved['step'], saved['training_seconds'], saved['records']
        del saved
    for step, batch in enumerate(schedule, 1):
        if step <= completed:
            continue
        start = time.perf_counter()
        model.train()
        model.config.use_cache = False
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for episode_id in batch:
            loss_sum, count = bounded_loss(model, *tensors(data[episode_id]))
            loss = normalized_task_loss(loss_sum, count)/len(batch)
            loss.backward()
            total += float(loss.detach())
            del loss, loss_sum
            gc.collect()
            torch.cuda.empty_cache()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not np.isfinite(total) or not torch.isfinite(norm):
            raise RuntimeError('Non-finite coverage training.')
        optimizer.step()
        torch.cuda.synchronize()
        training_seconds += time.perf_counter()-start
        completed = step
        records.append({'step': step, 'episode_ids': batch, 'terminal_loss': total,
                        'gradient_norm': float(norm), 'training_seconds': training_seconds,
                        'processed_tokens': sum(len(data[i]['ids']) for b in schedule[:step] for i in b),
                        'supervised_tokens': sum(int(np.count_nonzero(data[i]['labels'][1:] != -100))
                                                 for b in schedule[:step] for i in b),
                        'terminal_presentations': step*2})
        saved = capture_training_state(model, optimizer)
        saved.update(step=step, training_seconds=training_seconds, records=records)
        temporary = output/'step_checkpoint.tmp'
        torch.save(saved, temporary)
        temporary.replace(checkpoint)
        del saved
        with (output/'training.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(records[-1])+'\n')
        print(f'coverage {step}/192 loss={total:.4f} training_seconds={training_seconds:.1f}', flush=True)
    model.save_pretrained(output/'adapter', safe_serialization=True)
    grammar = Grammar()
    features = {v['episode_id']: evaluation_features(grammar, v) for values in views.values() for v in values}
    checks = {s: score(model, tokenizer, examples[s], values, s, 192, output, features) for s, values in views.items()}
    if protected_hashes() != config['old_result_hashes']:
        raise AssertionError('A prior model, checkpoint or score changed.')
    # score() retains historical v3 gates for audit only, not a new selection rule.
    atomic_json(output/'result.json', {'stage': config['stage'], 'completed_updates': completed,
                'training_seconds': training_seconds, 'processed_tokens': records[-1]['processed_tokens'],
                'supervised_tokens': records[-1]['supervised_tokens'], 'terminal_presentations': 384,
                'anchor_training_check': checks['train'], 'validation': checks['validation'],
                'all_120_training_worlds_evaluated': False, 'v3_gate_fields_are_descriptive_only': True,
                'old_results_preserved': True, 'formal_tests_used': False,
                'next_action': 'freeze_mainline_protocol_and_implement_paired_training_and_active_evaluation'})
    print(f'Coverage diagnostic complete: validation nonempty {checks["validation"]["nonempty_recoveries"]}/10.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(OUTPUT))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.output, args.resume)
