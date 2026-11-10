"""One Qwen3-0.6B terminal-task screen on the existing development cohort."""
import argparse
import json
from pathlib import Path
import time

import run_fast_development as fast
import numpy as np
import torch
from transformers import AutoTokenizer
from build_publication_curriculum import atomic_json, read, sha
from discovery.student_fastdev import terminal_messages
from discovery.student_v2 import load_views

ROOT = Path('results/qwen3_pilot_v1')
MODEL = Path('models/Qwen3-0.6B')


class NonThinkingTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs, enable_thinking=False)


def encode_terminal(tokenizer, messages):
    if [m['role'] for m in messages] != ['system', 'user', 'assistant']:
        raise ValueError('Only one terminal assistant answer is supported.')
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    target = messages[-1]['content'] + '<|im_end|>'
    if full != prompt + target + '\n':
        raise ValueError('Qwen3 training/inference template mismatch.')
    prefix = tokenizer.tokenizer(prompt, add_special_tokens=False)['input_ids']
    encoded = tokenizer.tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded['input_ids']
    if ids[:len(prefix)] != prefix:
        raise ValueError('Tokenized generation prefix differs from training prefix.')
    labels = [-100] * len(ids)
    start, end = len(prompt), len(prompt)+len(target)
    for i, (a, b) in enumerate(encoded['offset_mapping']):
        if start <= a < b <= end:
            labels[i] = ids[i]
    supervised = [x for x in labels if x != -100]
    if tokenizer.decode(supervised, skip_special_tokens=False) != target:
        raise ValueError('Assistant mask does not exactly cover answer plus EOS.')
    if any(x != -100 for x in labels[:len(prefix)]) or len(ids) > 32768:
        raise ValueError('Invalid mask or context length.')
    return {'messages': messages, 'ids': np.array(ids, dtype=np.int64),
            'labels': np.array(labels, dtype=np.int64)}


def main(check_only=False):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    tokenizer = NonThinkingTokenizer(AutoTokenizer.from_pretrained(MODEL, local_files_only=True))
    views = {s: load_views(s) for s in ('train', 'validation')}
    assert [len(views[s]) for s in views] == [24, 12]
    data = {'raw': {s: [encode_terminal(tokenizer, terminal_messages(v, 'raw')) for v in values]
                    for s, values in views.items()}}
    audit = {s: [{'id': v['episode_id'], 'input_tokens': len(e['ids']),
                 'supervised_tokens': int(np.count_nonzero(e['labels'] != -100))}
                for v, e in zip(views[s], data['raw'][s])] for s in views}
    if check_only:
        print(json.dumps({'template_mask_checks_passed': True, 'examples': audit}), flush=True)
        return
    if ROOT.exists():
        raise ValueError('Preserve completed or partial pilot; no overwrite.')
    protected = fast.protected_files()
    control = Path('results/fast_development_v2')
    for name in ('config.json', 'raw/result.json', 'raw/episodes.jsonl', 'raw/adapter/adapter_model.safetensors'):
        protected[str(control/name)] = sha(control/name)
    schedule = read(control/'config.json')['schedule']
    assert len(schedule) == 192
    manifest = read(MODEL/'download_manifest.json')
    for name, value in manifest['files'].items():
        if sha(MODEL/name) != value['sha256']:
            raise ValueError('Model checksum mismatch: '+name)
    config = {'model_id': manifest['model_id'], 'revision': manifest['revision'],
              'seed': 301, 'updates': 192, 'training_cap_seconds': 1800,
              'enable_thinking': False, 'generation': {'do_sample': False, 'max_new_tokens': 128},
              'schedule': schedule, 'token_audit': audit, 'template_mask_checks_passed': True,
              'formal_tests_used': False, 'development_validation_previously_inspected': True,
              'control': str(control/'raw'), 'comparison': 'historical model screen, not a controlled teaching-effect study',
              'protected_sha256': protected, 'model_manifest_sha256': sha(MODEL/'download_manifest.json'),
              'source_sha256': {p: sha(p) for p in ('run_qwen3_pilot.py', *fast.SOURCES)},
              'data_sha256': {str(Path('data/research_curriculum_v1/views')/s/'B.jsonl'):
                             sha(Path('data/research_curriculum_v1/views')/s/'B.jsonl') for s in views},
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}
    atomic_json(ROOT/'config.json', config)
    for source in config['source_sha256']:
        target = ROOT/'sources'/source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(source).read_bytes())
    fast.MODEL = MODEL
    fast.CAP_SECONDS = 1800
    start = time.perf_counter()
    try:
        result = fast.train_arm(tokenizer, views, data, 'raw', schedule, ROOT/'raw')
        checks = {p: sha(p) == h for p, h in {**protected, **config['source_sha256'], **config['data_sha256']}.items()}
        if not all(checks.values()):
            raise ValueError('Protected provenance changed.')
        rows = [json.loads(x) for x in (ROOT/'raw/episodes.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(rows) == 36 and len({(r['split'], r['episode_id']) for r in rows}) == 36
        for split, summary in result['summary'].items():
            selected = [r for r in rows if r['split'] == split]
            for metric in ('raw_support_success', 'full_recovery', 'valid_expression'):
                assert sum(r[metric] for r in selected) == summary[metric]
        atomic_json(ROOT/'verification.json', {'passed': True, 'protected_checks': checks,
                    'rows': len(rows), 'template_mask_checks_passed': True})
        atomic_json(ROOT/'result.json', {'completed': True, 'qwen3': result,
                    'historical_qwen25': read(control/'raw/result.json'),
                    'elapsed_seconds': time.perf_counter()-start, 'formal_tests_used': False,
                    'original_main_preserved_and_paused': True})
        print('Completed '+str(ROOT), flush=True)
    except BaseException as error:
        atomic_json(ROOT/'failure.json', {'completed': False, 'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    main(parser.parse_args().check_only)
