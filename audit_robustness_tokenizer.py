"""Verify every cached mainline task against the pinned 1.5B tokenizer; CPU only."""
import json
import os
from pathlib import Path
import time

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np
from transformers import AutoTokenizer

from build_publication_curriculum import atomic_json, read, sha
from discovery.student_v2 import encode
from setup_robustness_model import MODEL_ID, REVISION, OUTPUT as MODEL


CACHE = Path('data/student_mainline_v1')
OUTPUT = Path('results/robustness_tokenizer_audit')
FILES = ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'config.json')


def run():
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise ValueError('Preserve an existing audit.')
    manifest = read(CACHE/'manifest.json')
    snapshot = read(MODEL/'download_manifest.json')
    if snapshot['model_id'] != MODEL_ID or snapshot['revision'] != REVISION:
        raise ValueError('Wrong pinned tokenizer.')
    for name in FILES:
        if sha(MODEL/name) != snapshot['files'][name]['sha256']:
            raise ValueError('Tokenizer or configuration changed.')
    if not manifest['completed'] or sha(CACHE/'messages.jsonl') != manifest['messages_sha256']:
        raise ValueError('Original mainline cache changed.')
    context_limit = read(MODEL/'config.json')['max_position_embeddings']
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    started, checked, mismatches = time.perf_counter(), [], []
    with (CACHE/'messages.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            expected = manifest['entries'][row['key']]
            path = CACHE/expected['path']
            if sha(path) != expected['sha256']:
                raise ValueError('Original mainline tokens changed.')
            actual = encode(tokenizer, row['messages'])
            if len(actual['ids']) > context_limit:
                raise ValueError('Task exceeds the actual model context limit.')
            with np.load(path, allow_pickle=False) as saved:
                same_ids = np.array_equal(actual['ids'], saved['ids'])
                same_labels = np.array_equal(actual['labels'], saved['labels'])
            if not same_ids or not same_labels:
                mismatches.append({'key': row['key'], 'same_input_ids': same_ids, 'same_supervision_mask': same_labels})
            checked.append(row['key'])
            if len(checked)%600 == 0:
                print(f'Tokenizer audit {len(checked)}/3600; mismatches={len(mismatches)}', flush=True)
    if len(checked) != 3600 or set(checked) != set(manifest['entries']):
        raise ValueError('Incomplete or duplicate tokenizer audit.')
    result = {'completed': True, 'model_id': MODEL_ID, 'revision': REVISION,
              'checked_tasks': len(checked), 'exact_token_and_mask_equivalence': not mismatches,
              'mismatches': mismatches, 'model_context_limit': context_limit,
              'maximum_context': manifest['maximum_tokens'], 'seconds': time.perf_counter()-started,
              'manifest_sha256': sha(CACHE/'manifest.json'),
              'model_files_sha256': {name: sha(MODEL/name) for name in FILES},
              'sources': {name: sha(name) for name in ('audit_robustness_tokenizer.py', 'setup_robustness_model.py',
                           'discovery/student_v2.py', 'discovery/student_format.py')},
              'model_weights_loaded': False, 'training_started': False, 'formal_tests_used': False}
    atomic_json(OUTPUT/'audit.json', result)
    print(json.dumps(result, indent=2), flush=True)
    if mismatches:
        raise RuntimeError('Do not reuse the mainline cache for this model; separately encode and audit.')


if __name__ == '__main__':
    run()
