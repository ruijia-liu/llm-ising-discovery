"""Audit/cache the registered numerical-prediction ablation; CPU only."""
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
from transformers import AutoTokenizer

from build_publication_curriculum import atomic_json, read, sha
from discovery.process_ablation import GROUP, remove_explicit_predictions, task_plan
from discovery.student_v2 import encode
from train_mainline_student import task_plan as full_c_plan


SOURCE = Path('data/student_mainline_v1')
OUTPUT = Path('data/student_process_ablation_v1')
SOURCES = ('prepare_process_ablation.py', 'discovery/process_ablation.py',
           'discovery/mainline_schedule.py', 'discovery/student_v2.py',
           'discovery/student_format.py', 'PROCESS_ABLATION_PROTOCOL.md')


def run():
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise ValueError('Preserve the existing preparation; do not overwrite it.')
    original = read(SOURCE/'manifest.json')
    if not original['completed'] or sha(SOURCE/'messages.jsonl') != original['messages_sha256']:
        raise AssertionError('Original main cache is incomplete or changed.')
    for path, expected in original['config']['sources'].items():
        if sha(path) != expected:
            raise AssertionError('Original mainline format changed.')
    for entry in original['entries'].values():
        if sha(SOURCE/entry['path']) != entry['sha256']:
            raise AssertionError('Original mainline token file changed.')
    config = {'schema': 'student-process-ablation-cache-v1', 'group': GROUP,
              'parent_manifest': str(SOURCE/'manifest.json'), 'parent_manifest_sha256': sha(SOURCE/'manifest.json'),
              'parent_messages_sha256': original['messages_sha256'],
              'sources': {p: sha(p) for p in SOURCES}, 'formal_tests_used': False}
    atomic_json(OUTPUT/'config.json', config)
    (OUTPUT/'tokens').mkdir()
    tokenizer = AutoTokenizer.from_pretrained('models/Qwen2.5-0.5B-Instruct', local_files_only=True)
    original_messages = {}
    with (SOURCE/'messages.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            item = json.loads(line)
            if item['key'].endswith('_auxiliaryC'):
                original_messages[item['key']] = item['messages']
    if len(original_messages) != 600:
        raise AssertionError('Expected 600 original C auxiliary messages.')
    entries, episodes, counts = {}, [], Counter()
    lengths, started = [], time.perf_counter()
    with (OUTPUT/'messages.jsonl').open('w', encoding='utf-8') as messages_file, \
            (OUTPUT/'deletions.jsonl').open('w', encoding='utf-8') as deletion_file:
        for row in original['episodes']:
            parent_key = row['auxiliary']['C']
            messages, removed = remove_explicit_predictions(original_messages[parent_key])
            key = row['episode_id']+'_auxiliary'+GROUP
            example = encode(tokenizer, messages)
            path = OUTPUT/'tokens'/(key+'.npz')
            with path.open('wb') as handle:
                np.savez_compressed(handle, ids=example['ids'], labels=example['labels'])
            entries[key] = {'cache_root': str(OUTPUT), 'path': str(path.relative_to(OUTPUT)),
                            'sha256': sha(path), 'input_tokens': len(example['ids']),
                            'supervised_tokens': int(np.count_nonzero(example['labels'][1:] != -100))}
            for common_key in [*row['terminal'].values(), parent_key]:
                entries[common_key] = {'cache_root': str(SOURCE), **original['entries'][common_key]}
            episodes.append({'episode_id': row['episode_id'], 'sigma': row['sigma'],
                             'terminal': row['terminal'], 'auxiliary': {'C': parent_key, GROUP: key},
                             'candidate_expressions_per_auxiliary': sum(r['path'][-1] == 'predicted_delta' for r in removed)})
            messages_file.write(json.dumps({'key': key, 'parent_key': parent_key, 'messages': messages}, ensure_ascii=False)+'\n')
            deletion_file.write(json.dumps({'key': key, 'removed': removed}, ensure_ascii=False)+'\n')
            counts.update(r['path'][-1] for r in removed)
            lengths.append({'episode_id': row['episode_id'], 'full_C': original['entries'][parent_key]['input_tokens'],
                            'ablated_C': len(example['ids'])})
            if len(episodes)%100 == 0:
                print(f'Audited ablation {len(episodes)}/600, seconds={time.perf_counter()-started:.1f}', flush=True)
    manifest = {'completed': True, 'config': config, 'entries': entries, 'episodes': episodes,
                'messages_sha256': sha(OUTPUT/'messages.jsonl'), 'deletions_sha256': sha(OUTPUT/'deletions.jsonl')}
    budgets = {}
    for seed in (401, 402, 403):
        plan, budget = task_plan(manifest, seed)
        reference, full_budget = full_c_plan(original, 'C', seed)
        if len(plan) != 1200 or budget['terminal_presentations'] != 4800:
            raise AssertionError('Wrong registered training size.')
        for actual, expected in zip(plan, reference, strict=True):
            if actual['episode_id'] != expected['episode_id'] or actual['tasks'][:4] != expected['tasks'][:4]:
                raise AssertionError('Terminal exposure or paired episode order changed.')
            if abs(sum(t['weight'] for t in actual['tasks'][4:])-1) > 1e-12:
                raise AssertionError('Repeated auxiliary task weight changed.')
        budgets[str(seed)] = {'ablated': budget, 'full_C': full_budget}
    for path, expected in config['sources'].items():
        if sha(path) != expected:
            raise AssertionError('Source changed during preparation.')
        target = OUTPUT/'sources'/path; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(path).read_bytes())
    audit = {'completed': True, 'episodes': len(episodes), 'deleted_fields': dict(counts),
             'terminal_files_reused_without_modification': True,
             'all_non_removed_content_equal': True, 'user_and_system_messages_byte_identical': True,
             'candidate_expressions_preserved': True, 'budgets': budgets, 'auxiliary_lengths': lengths,
             'maximum_ablated_context': max(r['ablated_C'] for r in lengths),
             'seconds': time.perf_counter()-started, 'formal_tests_used': False, 'training_started': False}
    atomic_json(OUTPUT/'audit.json', audit)
    manifest['audit_sha256'] = sha(OUTPUT/'audit.json')
    atomic_json(OUTPUT/'manifest.json', manifest)
    print(json.dumps({k: audit[k] for k in ('completed', 'episodes', 'deleted_fields', 'maximum_ablated_context', 'seconds')}, indent=2), flush=True)


if __name__ == '__main__':
    run()
