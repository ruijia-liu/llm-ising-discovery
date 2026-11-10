"""Download and verify one pinned public robustness model, without loading it."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from build_publication_curriculum import atomic_json, read, sha


MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'
REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
WEIGHT_HASH = 'dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee'
NAMES = ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
         'vocab.json', 'merges.txt', 'README.md', 'LICENSE', 'model.safetensors')
OUTPUT = Path('models/Qwen2.5-1.5B-Instruct')


def run():
    path = OUTPUT/'download_manifest.json'
    if path.exists():
        manifest = read(path)
        if (manifest['model_id'], manifest['revision']) != (MODEL_ID, REVISION):
            raise ValueError('Preserve the existing different model snapshot.')
    else:
        url = f'https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true'
        with urllib.request.urlopen(url, timeout=60) as handle:
            metadata = json.load(handle)
        if metadata['sha'] != REVISION:
            raise ValueError('Official metadata did not resolve the pinned revision.')
        files = {row['rfilename']: row for row in metadata['siblings']}
        if files['model.safetensors']['lfs']['sha256'] != WEIGHT_HASH:
            raise ValueError('Unexpected official weight checksum.')
        manifest = {'model_id': MODEL_ID, 'revision': REVISION, 'source': f'https://huggingface.co/{MODEL_ID}',
                    'files': {}, 'expected_metadata': {name: files[name] for name in NAMES}}
        atomic_json(path, manifest)
    for name in NAMES:
        target = OUTPUT/name
        if name in manifest['files']:
            if not target.exists() or sha(target) != manifest['files'][name]['sha256']:
                raise ValueError('Previously verified model file changed; inspect before replacing.')
            print('Verified existing '+name, flush=True)
            continue
        print('Downloading '+name, flush=True)
        partial = OUTPUT/(name+'.part')
        hasher, size, last = hashlib.sha256(), 0, time.monotonic()
        url = f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}'
        with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as handle:
            while block := response.read(4*1024*1024):
                handle.write(block); hasher.update(block); size += len(block)
                if time.monotonic()-last > 20:
                    print(f'{name}: {size/1024**2:.1f} MiB', flush=True)
                    last = time.monotonic()
        metadata = manifest['expected_metadata'][name]
        checksum = hasher.hexdigest()
        if size != metadata['size'] or ('lfs' in metadata and checksum != metadata['lfs']['sha256']):
            raise ValueError('Downloaded bytes differ from the pinned official metadata.')
        if name == 'model.safetensors' and checksum != WEIGHT_HASH:
            raise ValueError('Downloaded weights differ from the registered checksum.')
        partial.replace(target)
        manifest['files'][name] = {'sha256': checksum, 'bytes': size}
        atomic_json(path, manifest)
        print(f'Verified {name}: {size/1024**2:.1f} MiB', flush=True)
    print(json.dumps({'completed': True, 'revision': REVISION, 'files': len(manifest['files']),
                      'model_loaded': False, 'training_started': False}), flush=True)


if __name__ == '__main__':
    run()
