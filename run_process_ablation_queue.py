"""Run the three prespecified ablation seeds sequentially after the main queue."""
import os
from pathlib import Path
import subprocess
import sys
import time

from build_publication_curriculum import atomic_json, read, sha
from discovery.evaluation_worlds import require_completed_main_training
from discovery.process_ablation import GROUP


ROOT = Path('results/student_process_ablation_v1')


def run():
    require_completed_main_training()
    atomic_json(ROOT/'queue_invocation.json', {'pid': os.getpid(), 'started_unix': time.time(), 'seeds': [401, 402, 403],
                'note': 'Informational only; verify actual live processes before reporting a running job.'})
    for seed in (401, 402, 403):
        output = ROOT/f'{GROUP}_seed{seed}'
        if (output/'result.json').exists():
            result = read(output/'result.json')
            if (not result['completed'] or result['completed_updates'] != 1200 or result['group'] != GROUP
                    or result['seed'] != seed or sha(output/'adapter/adapter_model.safetensors') != result['adapter_sha256']):
                raise ValueError('Existing completed ablation result failed verification.')
            print(f'Preserved completed ablation seed {seed}.', flush=True)
            continue
        args = [sys.executable, '-u', 'train_process_ablation.py', '--seed', str(seed)]
        if (output/'config.json').exists():
            if not (output/'step_checkpoint.pt').exists():
                raise RuntimeError('Incomplete initialization without a checkpoint; inspect before recovery.')
            args.append('--resume')
        child = subprocess.Popen(args, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        atomic_json(ROOT/'queue_state.json', {'active_seed': seed, 'status': 'child_started',
                                              'queue_pid': os.getpid(), 'child_pid': child.pid})
        code = child.wait()
        if code != 0 or not (output/'result.json').exists():
            atomic_json(ROOT/'queue_state.json', {'active_seed': seed, 'status': 'stopped', 'child_exit_code': code,
                                                'checkpoint_preserved': (output/'step_checkpoint.pt').exists()})
            raise RuntimeError('Ablation stopped; preserve checkpoint and inspect without automatically retrying.')
    atomic_json(ROOT/'queue_state.json', {'status': 'all_three_ablation_runs_completed', 'formal_tests_used': False})


if __name__ == '__main__':
    run()
