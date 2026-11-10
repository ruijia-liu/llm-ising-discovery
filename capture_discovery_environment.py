"""Capture package/hardware provenance without loading a model or GPU tensors."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Preserve the previous environment snapshot.')
    prefix = Path(sys.prefix).resolve()
    distributions = sorted(importlib.metadata.distributions(), key=lambda d: d.metadata['Name'].lower())
    packages = [{'name': d.metadata['Name'], 'version': d.version,
                 'installed_inside_environment': Path(d.locate_file('')).resolve().is_relative_to(prefix)}
                for d in distributions]
    executable = Path(sys.executable).resolve()
    try:
        executable_name = str(executable.relative_to(Path.cwd().resolve()))
    except ValueError:
        executable_name = executable.name  # Do not publish a personal absolute path.
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=10,
                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        gpu_metadata = gpu.stdout.strip() if gpu.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        gpu_metadata = None
    result = {'python': sys.version, 'executable_relative_to_workspace': executable_name,
              'platform': platform.platform(), 'machine': platform.machine(),
              'virtual_environment': sys.prefix != sys.base_prefix,
              'all_distributions_inside_environment': all(p['installed_inside_environment'] for p in packages),
              'packages': packages, 'gpu_metadata_from_read_only_query': gpu_metadata,
              'gpu_tensors_allocated': False, 'models_loaded': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    output.with_suffix('.requirements.txt').write_text(''.join(p['name']+'=='+p['version']+'\n' for p in packages), encoding='utf-8')
    print(json.dumps({'snapshot': str(output), 'packages': len(packages),
                      'virtual_environment': result['virtual_environment'],
                      'all_distributions_inside_environment': result['all_distributions_inside_environment']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
