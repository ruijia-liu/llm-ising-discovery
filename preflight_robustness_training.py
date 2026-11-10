"""One discarded longest-context 1.5B update, only after primary training ends."""
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from build_publication_curriculum import atomic_json, read, sha
from discovery.evaluation_worlds import require_completed_main_training
from setup_robustness_model import OUTPUT as MODEL, MODEL_ID, REVISION
from train_student_pilot import bounded_loss, make_student, tensors


CACHE = Path('data/student_mainline_v1')
OUTPUT = Path('results/robustness_bf16_preflight')


def run():
    require_completed_main_training()
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise ValueError('Preserve the existing resource probe, including any failure.')
    audit = read('results/robustness_tokenizer_audit/audit.json')
    manifest = read(CACHE/'manifest.json')
    if not audit['completed'] or not audit['exact_token_and_mask_equivalence'] or audit['manifest_sha256'] != sha(CACHE/'manifest.json'):
        raise ValueError('Complete token/mask equivalence audit is required.')
    for name, expected in audit['model_files_sha256'].items():
        if sha(MODEL/name) != expected:
            raise ValueError('Audited tokenizer/config changed.')
    downloaded = read(MODEL/'download_manifest.json')
    if downloaded['revision'] != REVISION or downloaded['model_id'] != MODEL_ID or 'model.safetensors' not in downloaded['files']:
        raise ValueError('Pinned model download not complete.')
    for name, metadata in downloaded['files'].items():
        if sha(MODEL/name) != metadata['sha256']:
            raise ValueError('Pinned model file changed.')
    longest = max(manifest['episodes'], key=lambda row: manifest['entries'][row['auxiliary']['C']]['input_tokens'])
    keys = [(longest['terminal'][str(b)], .25) for b in (8, 16, 24, 32)]+[(longest['auxiliary']['C'], 1.)]
    examples = []
    for key, weight in keys:
        entry = manifest['entries'][key]; path = CACHE/entry['path']
        if sha(path) != entry['sha256']:
            raise ValueError('Longest-context cached task changed.')
        with np.load(path, allow_pickle=False) as value:
            examples.append((key, {'ids': value['ids'].copy(), 'labels': value['labels'].copy()}, weight))
    config = {'model_id': MODEL_ID, 'revision': REVISION, 'seed': 401, 'precision': 'bfloat16',
              'episode_id': longest['episode_id'], 'tasks': [key for key, _ in keys],
              'model_manifest_sha256': sha(MODEL/'download_manifest.json'),
              'mainline_manifest_sha256': sha(CACHE/'manifest.json'),
              'tokenizer_audit_sha256': sha('results/robustness_tokenizer_audit/audit.json'),
              'sources': {name: sha(name) for name in ('preflight_robustness_training.py', 'train_student_pilot.py',
                           'setup_robustness_model.py', 'ROBUSTNESS_MODEL_PLAN.md')},
              'actual_training_started': False, 'update_discarded': True, 'formal_tests_used': False}
    atomic_json(OUTPUT/'config.json', config)
    torch.set_num_threads(4)
    rows, started, optimizer_steps = [], time.perf_counter(), 0
    try:
        torch.cuda.reset_peak_memory_stats()
        model = make_student(MODEL, 401)
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0)
        for name, example, weight in examples:
            task_started = time.perf_counter()
            loss_sum, count = bounded_loss(model, *tensors(example))
            loss = weight*loss_sum/count
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite probe loss.')
            loss.backward(); torch.cuda.synchronize()
            row = {'key': name, 'context_tokens': len(example['ids']), 'supervised_tokens': count,
                   'weighted_loss': float(loss.detach()), 'seconds': time.perf_counter()-task_started}
            rows.append(row)
            atomic_json(OUTPUT/'progress.json', {'completed_tasks': rows, 'optimizer_steps': 0})
            print(json.dumps(row), flush=True)
            del loss, loss_sum
            gc.collect(); torch.cuda.empty_cache()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise RuntimeError('Non-finite probe gradient.')
        optimizer.step(); torch.cuda.synchronize(); optimizer_steps = 1
        result = {'status': 'passed', 'gradient_norm': float(norm)}
    except Exception as problem:
        result = {'status': 'failed', 'error_type': type(problem).__name__, 'error': str(problem)}
        atomic_json(OUTPUT/'preflight.json', {**config, **result, 'tasks_completed': rows, 'optimizer_steps': optimizer_steps,
                    'seconds': time.perf_counter()-started, 'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                    'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30})
        raise
    result.update(config=config, tasks=rows, optimizer_steps=optimizer_steps, update_discarded=True,
                  actual_training_started=False, formal_tests_used=False, seconds=time.perf_counter()-started,
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                  gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
    atomic_json(OUTPUT/'preflight.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    run()
