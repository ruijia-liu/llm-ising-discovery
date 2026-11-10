"""Fetch a pinned public Qwen3 snapshot using the existing verified downloader."""
import json
from pathlib import Path
import urllib.request
import setup_robustness_model as download

def main():
    download.MODEL_ID = 'Qwen/Qwen3-0.6B'
    download.OUTPUT = Path('models/Qwen3-0.6B')
    manifest = download.OUTPUT/'download_manifest.json'
    if manifest.exists():
        metadata = json.loads(manifest.read_text(encoding='utf-8'))
        download.REVISION = metadata['revision']
        download.WEIGHT_HASH = metadata['expected_metadata']['model.safetensors']['lfs']['sha256']
    else:
        with urllib.request.urlopen('https://huggingface.co/api/models/Qwen/Qwen3-0.6B?blobs=true', timeout=60) as handle:
            metadata = json.load(handle)
        download.REVISION = metadata['sha']
        download.WEIGHT_HASH = next(x['lfs']['sha256'] for x in metadata['siblings'] if x['rfilename'] == 'model.safetensors')
    download.run()

if __name__ == '__main__':
    main()
