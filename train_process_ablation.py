"""Train the three registered prediction-ablation arms after the main queue."""
import argparse
import gc
from pathlib import Path
import time
import json

import numpy as np
import torch

from build_publication_curriculum import atomic_json, read, sha
from discovery.evaluation_worlds import require_completed_main_training
from discovery.process_ablation import GROUP, task_plan
from train_student_pilot import bounded_loss, capture_training_state, make_student, restore_training_state, tensors


MODEL = Path('models/Qwen2.5-0.5B-Instruct')
CACHE = Path('data/student_process_ablation_v1')
ROOT = Path('results/student_process_ablation_v1')
SOURCES = ('train_process_ablation.py', 'discovery/process_ablation.py', 'discovery/mainline_schedule.py',
           'train_student_pilot.py', 'discovery/evaluation_worlds.py', 'PROCESS_ABLATION_PROTOCOL.md',
           'MAINLINE_STUDY_PROTOCOL.md', 'ACTIVE_STUDENT_PROTOCOL.md')


def prepare(seed):
    manifest = read(CACHE/'manifest.json')
    if not manifest['completed'] or sha(CACHE/'audit.json') != manifest['audit_sha256']:
        raise ValueError('Ablation preparation incomplete or changed.')
    config = manifest['config']
    if sha(config['parent_manifest']) != config['parent_manifest_sha256']:
        raise ValueError('Parent mainline cache changed.')
    parent = read(config['parent_manifest'])
    for name, expected in {**parent['config']['sources'], **config['sources']}.items():
        if sha(name) != expected:
            raise ValueError('Frozen protocol or format changed.')
    schedule, budget = task_plan(manifest, seed)
    keys = {task['key'] for step in schedule for task in step['tasks']}
    data = {}
    for key in sorted(keys):
        entry = manifest['entries'][key]
        path = Path(entry['cache_root'])/entry['path']
        if sha(path) != entry['sha256']:
            raise ValueError('Cached task changed.')
        with np.load(path, allow_pickle=False) as saved:
            data[key] = {'ids': saved['ids'].copy(), 'labels': saved['labels'].copy()}
        if len(data[key]['ids']) != entry['input_tokens'] or np.count_nonzero(data[key]['labels'][1:] != -100) != entry['supervised_tokens']:
            raise ValueError('Cached token or supervision count changed.')
    return schedule, budget, data


def run(seed, resume=False, stop_after=None):
    if seed not in (401, 402, 403):
        raise ValueError('Use a registered seed.')
    # A resource sequencing guard, not a reading of any reserved world.
    require_completed_main_training()
    output = ROOT/f'{GROUP}_seed{seed}'
    if (output/'result.json').exists():
        raise ValueError('Preserve a completed ablation run.')
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError('Resume the exact existing run instead of overwriting it.')
    schedule, budget, data = prepare(seed)
    base = read(MODEL/'download_manifest.json')
    for name, metadata in base['files'].items():
        if sha(MODEL/name) != metadata['sha256']:
            raise ValueError('Pinned base model changed.')
    config = {'stage': 'formal_process_ablation_v1', 'group': GROUP, 'seed': seed, 'passes': 2,
              'maximum_updates': 1200, 'model': str(MODEL), 'model_revision': base['revision'],
              'learning_rate': 2e-4, 'weight_decay': 0, 'gradient_clip': 1,
              'lora': {'r': 8, 'alpha': 16, 'dropout': .05}, 'precision': 'bfloat16',
              'budget': budget, 'schedule': schedule, 'manifest_sha256': sha(CACHE/'manifest.json'),
              'base_files_sha256': {name: meta['sha256'] for name, meta in base['files'].items()},
              'sources': {p: sha(p) for p in SOURCES}, 'early_stopping': False,
              'validation_checkpoint_selection': False, 'formal_test_access': False}
    if resume:
        if read(output/'config.json') != config:
            raise ValueError('Source, data, or configuration changed; recovery requires a versioned audit.')
    else:
        atomic_json(output/'config.json', config)
        for source in SOURCES:
            target = output/'sources'/source; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(source).read_bytes())
    torch.set_num_threads(4)
    model = make_student(MODEL, seed)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0)
    checkpoint = output/'step_checkpoint.pt'
    completed, seconds, records = 0, 0.0, []
    processed = supervised = presentations = auxiliary_presentations = 0
    if resume:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        restore_training_state(model, optimizer, saved)
        completed, seconds, records = saved['step'], saved['training_seconds'], saved['records']
        processed, supervised = saved['processed_tokens'], saved['supervised_tokens']
        presentations, auxiliary_presentations = saved['terminal_presentations'], saved['auxiliary_presentations']
        del saved
        (output/'training.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records), encoding='utf-8')
    torch.cuda.reset_peak_memory_stats()
    invocation = time.time_ns()

    def effort(row):
        with (output/'training_effort.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'invocation': invocation, **row})+'\n')

    effort({'event': 'invocation_start', 'resumed_after_step': completed, 'time_unix': time.time()})
    for step, entry in enumerate(schedule, 1):
        if step <= completed:
            continue
        started = time.perf_counter()
        model.train(); model.config.use_cache = False
        optimizer.zero_grad(set_to_none=True)
        losses = {'terminal': 0.0, 'auxiliary': 0.0}
        for index, task in enumerate(entry['tasks']):
            task_started = time.perf_counter()
            effort({'event': 'task_started', 'step': step, 'task_index': index, 'key': task['key']})
            example = data[task['key']]
            loss_sum, count = bounded_loss(model, *tensors(example))
            loss = loss_sum/count*task['weight']
            if not torch.isfinite(loss):
                atomic_json(output/'failure.json', {'step': step, 'task': task['key'], 'reason': 'non-finite loss'})
                raise RuntimeError('Non-finite loss; previous complete checkpoint retained.')
            loss.backward(); torch.cuda.synchronize()
            effort({'event': 'task_completed', 'step': step, 'task_index': index, 'key': task['key'],
                    'input_tokens': len(example['ids']), 'supervised_tokens': count,
                    'seconds': time.perf_counter()-task_started})
            losses[task['kind']] += float(loss.detach())
            processed += len(example['ids']); supervised += count
            presentations += task['kind'] == 'terminal'
            auxiliary_presentations += task['kind'] == 'auxiliary'
            del loss, loss_sum
            gc.collect(); torch.cuda.empty_cache()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            atomic_json(output/'failure.json', {'step': step, 'reason': 'non-finite gradient'})
            raise RuntimeError('Non-finite gradient; previous complete checkpoint retained.')
        optimizer.step(); torch.cuda.synchronize()
        seconds += time.perf_counter()-started; completed = step
        row = {'step': step, 'episode_id': entry['episode_id'], 'losses': losses, 'gradient_norm': float(norm),
               'training_seconds': seconds, 'processed_tokens': processed, 'supervised_tokens': supervised,
               'terminal_presentations': presentations, 'auxiliary_presentations': auxiliary_presentations,
               'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30}
        records.append(row)
        saved = capture_training_state(model, optimizer)
        saved.update(step=step, training_seconds=seconds, records=records, processed_tokens=processed,
                     supervised_tokens=supervised, terminal_presentations=presentations,
                     auxiliary_presentations=auxiliary_presentations)
        temporary = checkpoint.with_suffix('.tmp'); torch.save(saved, temporary); temporary.replace(checkpoint)
        effort({'event': 'checkpoint_committed', 'step': step})
        del saved
        with (output/'training.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row)+'\n')
        if step in (600, 1200):
            model.save_pretrained(output/f'adapter_step{step}', safe_serialization=True)
        print(f'{GROUP}/{seed} {step}/1200 terminal={losses["terminal"]:.4f} auxiliary={losses["auxiliary"]:.4f} seconds={seconds:.1f}', flush=True)
        if ((stop_after is not None and step >= stop_after) or (output/'STOP_AFTER_CHECKPOINT').exists()) and step < 1200:
            atomic_json(output/'stopped_at_checkpoint.json', {'step': step, 'resumable': True, 'reason': 'execution_stop_request'})
            return
    if (processed, supervised, presentations, auxiliary_presentations) != tuple(budget[k] for k in (
            'input_tokens', 'supervised_tokens', 'terminal_presentations', 'auxiliary_presentations')):
        raise AssertionError('Actual training budget differs from the frozen plan.')
    model.save_pretrained(output/'adapter', safe_serialization=True)
    atomic_json(output/'result.json', {'completed': True, 'group': GROUP, 'seed': seed, 'completed_updates': completed,
                'training_seconds': seconds, 'input_tokens': processed, 'supervised_tokens': supervised,
                'terminal_presentations': presentations, 'auxiliary_presentations': auxiliary_presentations,
                'formal_tests_used': False, 'adapter_sha256': sha(output/'adapter/adapter_model.safetensors'),
                'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(401, 402, 403), required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    run(args.seed, args.resume, args.stop_after)
