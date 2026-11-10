"""Inventory prospective SFT tokens without training or held-out test access."""
import json
from pathlib import Path
import time

import numpy as np
from transformers import AutoTokenizer

from build_publication_curriculum import atomic_json, read, sha
from discovery.student_mainline import PREFIXES, auxiliary_messages, load_group, terminal_messages
from discovery.student_v2 import encode


def count(tokenizer, messages):
    example = encode(tokenizer, messages)
    return {'input_tokens': len(example['ids']),
            'supervised_tokens': int(np.count_nonzero(example['labels'][1:] != -100))}


def run():
    source = Path('data/research_curriculum_publication_v1')
    output = Path('results/mainline_format_audit')
    tokenizer = AutoTokenizer.from_pretrained('models/Qwen2.5-0.5B-Instruct', local_files_only=True)
    frozen = read(source/'audit_report.json')
    for name, expected in frozen['export_sha256'].items():
        if sha(source/name) != expected:
            raise AssertionError('Formal corpus was modified.')
    records = []
    started = time.perf_counter()
    for sigma in (0, 1):
        views = {g: load_group(source, 'train', g, sigma) for g in 'ABC'}
        for index in range(len(views['A'])):
            a, b, c = (views[g][index] for g in 'ABC')
            if len({v['episode_id'] for v in (a, b, c)}) != 1:
                raise AssertionError('Group ordering mismatch.')
            row = {'episode_id': a['episode_id'], 'sigma': sigma, 'terminal': {}, 'auxiliary': {}}
            for budget in PREFIXES:
                common = terminal_messages(b, budget)
                if common != terminal_messages(c, budget):
                    raise AssertionError('B/C terminal mismatch.')
                row['terminal'][str(budget)] = count(tokenizer, common)
            if terminal_messages(a, 8) != terminal_messages(b, 8):
                raise AssertionError('A initial task changed.')
            row['A_terminal_four_initial_presentations'] = {key: 4*value for key, value in row['terminal']['8'].items()}
            for group, view in (('B', b), ('C', c)):
                row['auxiliary'][group] = count(tokenizer, auxiliary_messages(view, group))
            records.append(row)
            if (index+1) % 50 == 0:
                print(f'format audit sigma={sigma}: {index+1}/300; seconds={time.perf_counter()-started:.1f}', flush=True)
    summary = {}
    for group in 'BC':
        rows = [r['auxiliary'][group] for r in records]
        summary[group] = {f'{stat}_{key}': int(fn([r[key] for r in rows]))
                          for key in ('input_tokens', 'supervised_tokens')
                          for stat, fn in (('sum', sum), ('min', min), ('max', max))}
    atomic_json(output/'audit.json', {'completed': True, 'episodes': len(records), 'records': records,
                'auxiliary_per_pass': summary, 'maximum_context': 32768,
                'truncated_messages': 0, 'B_C_terminal_equal': True, 'A_only_initial8': True,
                'formal_test_access': False, 'training_started': False,
                'curriculum_audit_sha256': sha(source/'audit_report.json'),
                'format_sha256': sha('discovery/student_mainline.py'),
                'seconds': time.perf_counter()-started})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    run()
