"""Audit and cache the frozen A/B/C teaching inputs, without model training."""
import json
from pathlib import Path
import time

import numpy as np
from transformers import AutoTokenizer

from build_publication_curriculum import atomic_json, read, sha
from discovery.student_mainline import PREFIXES, auxiliary_messages, load_group, terminal_messages
from discovery.student_v2 import encode


SOURCE = Path('data/research_curriculum_publication_v1')
OUTPUT = Path('data/student_mainline_v1')
FORMAT_SOURCES = ('discovery/student_mainline.py', 'discovery/student_v2.py', 'discovery/student_v3.py',
                  'discovery/student_spins.py', 'discovery/student_format.py', 'discovery/teaching.py',
                  'discovery/physics.py', 'MAINLINE_STUDY_PROTOCOL.md')


def run():
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise ValueError('Preserve existing cache; do not silently overwrite an incomplete preparation.')
    audit = read(SOURCE/'audit_report.json')
    if not audit['completed']:
        raise AssertionError('Curriculum not audited.')
    for path, expected in audit['export_sha256'].items():
        if sha(SOURCE/path) != expected:
            raise AssertionError('Teaching exports changed.')
    config = {'schema': 'student-mainline-cache-v1', 'source': str(SOURCE),
              'model_revision': '7ae557604adf67be50417f59c2c2f167def9a775',
              'curriculum_audit_sha256': sha(SOURCE/'audit_report.json'),
              'sources': {p:sha(p) for p in FORMAT_SOURCES}, 'formal_tests_used':False}
    atomic_json(OUTPUT/'config.json', config)
    tokenizer = AutoTokenizer.from_pretrained('models/Qwen2.5-0.5B-Instruct', local_files_only=True)
    entries, episodes = {}, []
    started = time.perf_counter()
    (OUTPUT/'tokens').mkdir()
    with (OUTPUT/'messages.jsonl').open('w', encoding='utf-8') as messages_file:
        def save(key, messages):
            example = encode(tokenizer, messages)
            path = OUTPUT/'tokens'/(key+'.npz')
            with path.open('wb') as handle:
                np.savez_compressed(handle, ids=example['ids'], labels=example['labels'])
            entry = {'path':str(path.relative_to(OUTPUT)), 'sha256':sha(path),
                     'input_tokens':len(example['ids']),
                     'supervised_tokens':int(np.count_nonzero(example['labels'][1:] != -100))}
            entries[key] = entry
            messages_file.write(json.dumps({'key':key, 'messages':messages}, ensure_ascii=False)+'\n')
            return key
        for sigma in (0,1):
            groups = {g:load_group(SOURCE,'train',g,sigma) for g in 'ABC'}
            for i in range(300):
                a,b,c = (groups[g][i] for g in 'ABC')
                if len({v['episode_id'] for v in (a,b,c)}) != 1:
                    raise AssertionError('Unpaired worlds.')
                episode_id = a['episode_id']
                row = {'episode_id':episode_id, 'sigma':sigma, 'terminal':{}, 'auxiliary':{}}
                for budget in PREFIXES:
                    messages = terminal_messages(b,budget)
                    if messages != terminal_messages(c,budget):
                        raise AssertionError('Unpaired B/C labels or inputs.')
                    if budget == 8 and messages != terminal_messages(a,8):
                        raise AssertionError('A initial evidence or label differs.')
                    row['terminal'][str(budget)] = save(f'{episode_id}_terminal{budget:02d}', messages)
                for g,v in (('B',b),('C',c)):
                    row['auxiliary'][g] = save(f'{episode_id}_auxiliary{g}',auxiliary_messages(v,g))
                episodes.append(row)
                if (i+1)%50 == 0:
                    print(f'cache sigma={sigma} {i+1}/300 seconds={time.perf_counter()-started:.1f}', flush=True)
    for path, expected in config['sources'].items():
        if sha(path) != expected:
            raise AssertionError('Formatting changed during cache preparation.')
        snapshot = OUTPUT/'sources'/path; snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes(Path(path).read_bytes())
    atomic_json(OUTPUT/'manifest.json', {'completed':True, 'config':config, 'entries':entries, 'episodes':episodes,
                'messages_sha256':sha(OUTPUT/'messages.jsonl'), 'counts':{'episodes':600,'terminal':2400,'auxiliary_B':600,'auxiliary_C':600},
                'maximum_tokens':max(e['input_tokens'] for e in entries.values()),
                'A_references_only_initial8':True, 'B_C_terminal_identical':True, 'seconds':time.perf_counter()-started})
    print(f'Completed training cache: {len(entries)} tasks, no training or test access.', flush=True)


if __name__ == '__main__':
    run()
