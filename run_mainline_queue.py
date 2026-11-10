"""Run the nine registered arms sequentially; resume only real checkpoints."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from build_publication_curriculum import atomic_json, read, sha


ROOT=Path('results/student_mainline_v1')


def run():
    ROOT.mkdir(parents=True,exist_ok=True)
    queue=[(group,seed) for seed in (401,402,403) for group in 'ABC']
    atomic_json(ROOT/'queue_invocation.json',{'pid':os.getpid(),'started_unix':time.time(),
                'queue':[{'group':g,'seed':s} for g,s in queue],
                'note':'PID and state files are informational; verify live process/tool handle before treating as running.'})
    for group,seed in queue:
        output=ROOT/f'{group}_seed{seed}'
        if (output/'result.json').exists():
            result=read(output/'result.json')
            if not result['completed'] or result['completed_updates']!=1200 or sha(output/'adapter/adapter_model.safetensors')!=result['adapter_sha256']:
                raise AssertionError('Completed result is inconsistent; do not replace it.')
            print(f'Preserved completed {group}/{seed}.',flush=True)
            continue
        args=[sys.executable,'-u','train_mainline_student.py','--group',group,'--seed',str(seed)]
        if (output/'config.json').exists():
            if not (output/'step_checkpoint.pt').exists():
                raise RuntimeError('Incomplete initialization without a checkpoint; inspect before recovery.')
            args.append('--resume')
        atomic_json(ROOT/'queue_state.json',{'active_group':group,'active_seed':seed,'status':'launching','queue_pid':os.getpid()})
        child=subprocess.Popen(args,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        atomic_json(ROOT/'queue_state.json',{'active_group':group,'active_seed':seed,'status':'child_started',
                                            'queue_pid':os.getpid(),'child_pid':child.pid})
        code=child.wait()
        if code!=0 or not (output/'result.json').exists():
            atomic_json(ROOT/'queue_state.json',{'active_group':group,'active_seed':seed,'status':'stopped',
                                                'child_exit_code':code,'checkpoint_preserved':(output/'step_checkpoint.pt').exists()})
            raise RuntimeError('Training stopped; preserve checkpoint and inspect, no automatic repeated attempts.')
    atomic_json(ROOT/'queue_state.json',{'status':'all_nine_training_runs_completed','formal_tests_used':False})
    print('All nine main training runs completed. Evaluation remains separate.',flush=True)


if __name__=='__main__':
    run()
