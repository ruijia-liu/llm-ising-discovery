"""Download a pinned public Qwen model snapshot into this project (no upload)."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
NAMES = ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
         'vocab.json', 'merges.txt', 'README.md', 'LICENSE', 'model.safetensors')


def download(output, tokenizer_only=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/'download_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        revision = manifest['revision']
        if manifest['model_id'] != MODEL_ID:
            raise ValueError('Unexpected model in output directory.')
    else:
        with urllib.request.urlopen(f'https://huggingface.co/api/models/{MODEL_ID}?blobs=true', timeout=60) as response:
            metadata = json.load(response)
        revision = metadata['sha']
        manifest = {'model_id': MODEL_ID, 'revision': revision,
                    'source': f'https://huggingface.co/{MODEL_ID}', 'files': {},
                    'expected_lfs_sha256': {f['rfilename']: f['lfs']['sha256'] for f in metadata.get('siblings', []) if 'lfs' in f}}
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Expected a pinned commit SHA.')
    for name in NAMES:
        if tokenizer_only and name == 'model.safetensors':
            continue
        path = output/name
        if path.exists() and name in manifest['files']:
            continue
        url = f'https://huggingface.co/{MODEL_ID}/resolve/{revision}/{name}'
        part = output/(name+'.part')
        hasher = hashlib.sha256()
        size, last = 0, time.monotonic()
        print('Downloading '+name, flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, part.open('wb') as handle:
            while block := response.read(4*1024*1024):
                handle.write(block)
                hasher.update(block)
                size += len(block)
                if time.monotonic()-last > 20:
                    print(f'{name}: {size/1024**2:.1f} MiB', flush=True)
                    last = time.monotonic()
        checksum = hasher.hexdigest()
        expected = manifest['expected_lfs_sha256'].get(name)
        if expected and checksum != expected:
            raise ValueError('Downloaded model checksum does not match published LFS hash.')
        part.replace(path)
        manifest['files'][name] = {'sha256': checksum, 'bytes': size}
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
        print(f'Completed {name}: {size/1024**2:.1f} MiB', flush=True)
    print(json.dumps({'model_id': MODEL_ID, 'revision': revision, 'files': list(manifest['files'])}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='models/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--tokenizer-only', action='store_true')
    args = parser.parse_args()
    download(args.output, args.tokenizer_only)
