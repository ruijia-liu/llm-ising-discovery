"""One discarded maximum-context update to measure local mainline feasibility."""
import gc
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from build_publication_curriculum import atomic_json, read, sha
from discovery.student_mainline import PREFIXES, auxiliary_messages, load_group, terminal_messages
from discovery.student_v2 import encode
from train_student_pilot import bounded_loss, make_student, tensors


def run():
    output = Path('results/mainline_preflight')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserve an existing preflight; do not silently retry/replace it.')
    inventory = read('results/mainline_format_audit/audit.json')
    if inventory['format_sha256'] != sha('discovery/student_mainline.py'):
        raise AssertionError('Format changed after inventory.')
    longest = max(inventory['records'], key=lambda r: r['auxiliary']['C']['input_tokens'])
    view = next(v for v in load_group('data/research_curriculum_publication_v1', 'train', 'C', longest['sigma'])
                if v['episode_id'] == longest['episode_id'])
    model_path = Path('models/Qwen2.5-0.5B-Instruct')
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    examples = [(f'terminal_{budget}', encode(tokenizer, terminal_messages(view, budget)), 0.25) for budget in PREFIXES]
    examples += [('auxiliary_C', encode(tokenizer, auxiliary_messages(view, 'C')), 1.0)]
    torch.set_num_threads(4)
    model = make_student(model_path, 401)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0)
    torch.cuda.reset_peak_memory_stats()
    started, rows = time.perf_counter(), []
    for name, example, weight in examples:
        start = time.perf_counter()
        loss_sum, tokens = bounded_loss(model, *tensors(example))
        mean_loss = loss_sum/tokens
        if not torch.isfinite(mean_loss):
            raise AssertionError('Non-finite preflight loss.')
        (weight*mean_loss).backward()
        torch.cuda.synchronize()
        row = {'task': name, 'context_tokens': len(example['ids']), 'supervised_tokens': tokens,
               'loss': float(mean_loss.detach()), 'seconds': time.perf_counter()-start}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del loss_sum, mean_loss
        gc.collect()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(norm):
        raise AssertionError('Non-finite accumulated gradient.')
    optimizer.step()
    torch.cuda.synchronize()
    result = {'status': 'passed', 'seed': 401, 'episode_id': view['episode_id'], 'tasks': rows,
              'gradient_norm': float(norm), 'seconds': time.perf_counter()-started,
              'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
              'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
              'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'actual_training_started': False, 'update_discarded': True, 'test_worlds_used': False,
              'warning': 'One longest-context update is a resource estimate, not a full-run runtime guarantee.',
              'inventory_sha256': sha('results/mainline_format_audit/audit.json'),
              'sources': {p: sha(p) for p in ('preflight_mainline_training.py', 'discovery/student_mainline.py',
                                             'train_student_pilot.py')}}
    atomic_json(output/'preflight.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    run()
