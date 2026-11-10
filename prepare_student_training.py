"""Round-trip check and tokenize the full B/C curricula, without opening private/.

Saves complete conversations. No truncation, no outcome filtering, no test data.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from discovery.student_format import compact_messages, expand_messages, normalized_messages, tokenize_assistant_only


def prepare(source, model, output):
    source, model, output = map(Path, (source, model, output))
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty prepared-data directory.')
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    output.mkdir(parents=True, exist_ok=True)
    entries, arrays, counts, hashes = [], {}, {}, {}
    model_info = json.loads((model/'download_manifest.json').read_text(encoding='utf-8'))
    for split in ('train', 'validation'):
        original_by_group, compact_by_group = {}, {}
        for group in ('B', 'C'):
            path = source/'messages'/split/(group+'.jsonl')
            original_by_group[group] = [json.loads(line)['messages'] for line in path.read_text(encoding='utf-8').splitlines()]
            hashes[f'{split}/{group}'] = hashlib.sha256(path.read_bytes()).hexdigest()
            compact_by_group[group] = []
            raw_counts, token_counts, supervised_counts = [], [], []
            with (output/f'{split}_{group}_compact.jsonl').open('w', encoding='utf-8') as handle:
                for j, messages in enumerate(original_by_group[group]):
                    compact = compact_messages(messages)
                    if expand_messages(compact) != normalized_messages(messages):
                        raise AssertionError('Compaction lost or changed information.')
                    compact_by_group[group].append(compact)
                    ids, labels = tokenize_assistant_only(tokenizer, compact)
                    if len(ids) > 32768:
                        raise ValueError('Full conversation exceeds model context; do not truncate silently.')
                    key = f'{split}_{group}_{j:03d}'
                    arrays[key+'_ids'] = np.array(ids, dtype=np.int32)
                    arrays[key+'_labels'] = np.array(labels, dtype=np.int32)
                    raw_count = len(tokenizer.apply_chat_template(messages, tokenize=True))
                    raw_counts.append(raw_count)
                    token_counts.append(len(ids))
                    supervised_counts.append(sum(v != -100 for v in labels))
                    entries.append({'key': key, 'split': split, 'group': group, 'episode_index': j,
                                    'tokens': len(ids), 'supervised_tokens': supervised_counts[-1]})
                    handle.write(json.dumps({'messages': compact}, ensure_ascii=False, separators=(',', ':'))+'\n')
            counts[split+'_'+group] = {'episodes': len(token_counts), 'raw_tokens': sum(raw_counts),
                                      'raw_max_tokens': max(raw_counts), 'compact_tokens': sum(token_counts),
                                      'compact_max_tokens': max(token_counts), 'supervised_tokens': sum(supervised_counts)}
        for b, c in zip(compact_by_group['B'], compact_by_group['C']):
            if b[0] != c[0] or [m for m in b if m['role'] == 'user'] != [m for m in c if m['role'] == 'user']:
                raise AssertionError('Common B/C context differs after compaction.')
            for j in range(24):
                if json.loads(b[2+2*j]['content'])['experiment'] != json.loads(c[2+2*j]['content'])['experiment']:
                    raise AssertionError('B/C experiments changed.')
            if json.loads(b[-1]['content'])['final_answer'] != json.loads(c[-1]['content'])['final_answer']:
                raise AssertionError('B/C labels changed.')
    np.savez_compressed(output/'episodes.npz', **arrays)
    manifest = {'model_id': model_info['model_id'], 'model_revision': model_info['revision'],
                'tokenizer_sha256': model_info['files']['tokenizer.json']['sha256'],
                'source_sha256': hashes, 'loss': 'assistant content plus im_end only; system/user/header tokens masked',
                'compaction': 'lossless query hex, implicit H0, repeated-prose references; no float rounding',
                'truncated_episodes': 0, 'private_or_test_files_opened': False,
                'counts': counts, 'entries': entries}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(counts, indent=2), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/research_curriculum_v1')
    parser.add_argument('--model', default='models/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--output', default='data/student_training_v1')
    args = parser.parse_args()
    prepare(args.source, args.model, args.output)
