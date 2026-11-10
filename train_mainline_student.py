"""Frozen two-pass A/B/C SFT: 1200 checkpointed updates, no adaptive validation."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from build_publication_curriculum import atomic_json, read, sha
from discovery.mainline_schedule import paired_schedule
from train_student_pilot import bounded_loss, capture_training_state, make_student, restore_training_state, tensors


MODEL = Path('models/Qwen2.5-0.5B-Instruct')
CACHE = Path('data/student_mainline_v1')
ROOT = Path('results/student_mainline_v1')
SOURCES = ('train_mainline_student.py','discovery/mainline_schedule.py','train_student_pilot.py',
           'MAINLINE_STUDY_PROTOCOL.md','ACTIVE_STUDENT_PROTOCOL.md')


def task_plan(manifest, group, seed):
    records = []
    rows = {r['episode_id']:r for r in manifest['episodes']}
    for row in manifest['episodes']:
        records.append({'episode_id':row['episode_id'],
                        'auxiliary':{g:manifest['entries'][row['auxiliary'][g]] for g in 'BC'}})
    schedule = paired_schedule(records, seed, passes=2)
    tasks = []
    input_tokens, supervised_tokens = 0, 0
    for step in schedule['steps']:
        row = rows[step['episode_id']]
        terminal = [row['terminal'][str(b)] for b in step['A_terminal_prefixes' if group=='A' else 'B_C_terminal_prefixes']]
        current = [{'key':key,'weight':.25,'kind':'terminal'} for key in terminal]
        if group in 'BC':
            repetitions = step[group+'_auxiliary_repetitions']
            current += [{'key':row['auxiliary'][group],'weight':1/repetitions,'kind':'auxiliary'} for _ in range(repetitions)]
        input_tokens += sum(manifest['entries'][t['key']]['input_tokens'] for t in current)
        supervised_tokens += sum(manifest['entries'][t['key']]['supervised_tokens'] for t in current)
        tasks.append({'episode_id':step['episode_id'],'tasks':current})
    return tasks, {'updates':len(tasks), 'terminal_presentations':len(tasks)*4,
                   'input_tokens':input_tokens,'supervised_tokens':supervised_tokens,
                   'B_C_relative_auxiliary_input_difference':schedule['relative_auxiliary_input_difference']}


def prepare(group, seed):
    manifest = read(CACHE/'manifest.json')
    if not manifest['completed']:
        raise ValueError('Training cache incomplete.')
    for name, expected in manifest['config']['sources'].items():
        if sha(name) != expected:
            raise ValueError('Protocol/format changed after caching.')
    schedule, budget = task_plan(manifest,group,seed)
    keys = {t['key'] for step in schedule for t in step['tasks']}
    data = {}
    for key in sorted(keys):
        entry = manifest['entries'][key]
        path = CACHE/entry['path']
        if sha(path) != entry['sha256']:
            raise AssertionError('Cached training tokens changed.')
        with np.load(path,allow_pickle=False) as record:
            data[key] = {'ids':record['ids'].copy(),'labels':record['labels'].copy()}
        if len(data[key]['ids']) != entry['input_tokens'] or np.count_nonzero(data[key]['labels'][1:] != -100) != entry['supervised_tokens']:
            raise AssertionError('Token mask or length differs.')
    return schedule, budget, data


def run(group, seed, resume=False, stop_after=None):
    if group not in 'ABC' or len(group)!=1 or seed not in (401,402,403):
        raise ValueError('Use a registered arm and seed.')
    output = ROOT/f'{group}_seed{seed}'
    if (output/'result.json').exists():
        raise ValueError('Preserve a completed main study run.')
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError('Resume this exact run; do not overwrite it.')
    schedule,budget,data = prepare(group,seed)
    model_manifest=read(MODEL/'download_manifest.json')
    for name,metadata in model_manifest['files'].items():
        if sha(MODEL/name)!=metadata['sha256']:
            raise ValueError('Pinned base model or tokenizer changed.')
    config = {'stage':'formal_mainline_v1','group':group,'seed':seed,'passes':2,
              'maximum_updates':1200,'model':str(MODEL),
              'model_revision':'7ae557604adf67be50417f59c2c2f167def9a775',
              'learning_rate':2e-4,'weight_decay':0,'gradient_clip':1,
              'lora':{'r':8,'alpha':16,'dropout':.05},'budget':budget,
              'schedule':schedule,'manifest_sha256':sha(CACHE/'manifest.json'),
              'base_files_sha256':{name:meta['sha256'] for name,meta in model_manifest['files'].items()},
              'sources':{p:sha(p) for p in SOURCES},'early_stopping':False,
              'validation_checkpoint_selection':False,'formal_test_access':False}
    if resume:
        if read(output/'config.json') != config:
            raise ValueError('Source/data/configuration changed; resume forbidden without a versioned audit.')
    else:
        atomic_json(output/'config.json',config)
        for source in SOURCES:
            target=output/'sources'/source; target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(Path(source).read_bytes())
    torch.set_num_threads(4)
    model=make_student(MODEL,seed)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4,weight_decay=0)
    checkpoint=output/'step_checkpoint.pt'
    completed,seconds,records=0,0.0,[]
    processed,supervised,presentations=0,0,0
    if resume:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        restore_training_state(model,optimizer,saved)
        completed,seconds,records=saved['step'],saved['training_seconds'],saved['records']
        processed,supervised,presentations=saved['processed_tokens'],saved['supervised_tokens'],saved['terminal_presentations']
        del saved
        # The checkpoint is authoritative if interrupted between atomic save and
        # the human-readable log write. Reconstruct the log without replaying an update.
        (output/'training.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    torch.cuda.reset_peak_memory_stats()
    invocation=time.time_ns()
    def effort(record):
        with (output/'training_effort.jsonl').open('a',encoding='utf-8') as handle:
            handle.write(json.dumps({'invocation':invocation,**record})+'\n')
    effort({'event':'invocation_start','resumed_after_step':completed,'time_unix':time.time()})
    for step,entry in enumerate(schedule,1):
        if step<=completed:
            continue
        started=time.perf_counter()
        model.train(); model.config.use_cache=False
        optimizer.zero_grad(set_to_none=True)
        losses={'terminal':0.0,'auxiliary':0.0}
        for task_index,task in enumerate(entry['tasks']):
            task_started=time.perf_counter()
            effort({'event':'task_started','step':step,'task_index':task_index,'key':task['key']})
            example=data[task['key']]
            loss_sum,count=bounded_loss(model,*tensors(example))
            loss=loss_sum/count*task['weight']
            if not torch.isfinite(loss):
                atomic_json(output/'failure.json',{'step':step,'reason':'non-finite loss','task':task['key']})
                raise RuntimeError('Non-finite loss; previous complete checkpoint retained.')
            loss.backward()
            torch.cuda.synchronize()
            effort({'event':'task_completed','step':step,'task_index':task_index,'key':task['key'],
                    'input_tokens':len(example['ids']),'supervised_tokens':count,
                    'seconds':time.perf_counter()-task_started})
            losses[task['kind']]+=float(loss.detach())
            processed+=len(example['ids']); supervised+=count
            presentations+=task['kind']=='terminal'
            del loss,loss_sum
            gc.collect(); torch.cuda.empty_cache()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not torch.isfinite(norm):
            atomic_json(output/'failure.json',{'step':step,'reason':'non-finite gradient'})
            raise RuntimeError('Non-finite gradient; previous complete checkpoint retained.')
        optimizer.step(); torch.cuda.synchronize()
        seconds+=time.perf_counter()-started; completed=step
        row={'step':step,'episode_id':entry['episode_id'],'losses':losses,'gradient_norm':float(norm),
             'training_seconds':seconds,'processed_tokens':processed,'supervised_tokens':supervised,
             'terminal_presentations':presentations,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
        records.append(row)
        saved=capture_training_state(model,optimizer)
        saved.update(step=step,training_seconds=seconds,records=records,processed_tokens=processed,
                     supervised_tokens=supervised,terminal_presentations=presentations)
        temporary=checkpoint.with_suffix('.tmp'); torch.save(saved,temporary); temporary.replace(checkpoint)
        effort({'event':'checkpoint_committed','step':step})
        del saved
        with (output/'training.jsonl').open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(row)+'\n')
        if step in (600,1200):
            model.save_pretrained(output/f'adapter_step{step}',safe_serialization=True)
        print(f'{group}/{seed} {step}/1200 terminal={losses["terminal"]:.4f} auxiliary={losses["auxiliary"]:.4f} seconds={seconds:.1f}',flush=True)
        if (stop_after is not None and step>=stop_after) or (output/'STOP_AFTER_CHECKPOINT').exists():
            if step<1200:
                atomic_json(output/'stopped_at_checkpoint.json',{'step':step,'resumable':True,'reason':'execution_stop_request'})
                return
    if (processed,supervised,presentations)!=(budget['input_tokens'],budget['supervised_tokens'],budget['terminal_presentations']):
        raise AssertionError('Actual training budget differs from the frozen plan.')
    model.save_pretrained(output/'adapter',safe_serialization=True)
    atomic_json(output/'result.json',{'completed':True,'group':group,'seed':seed,'completed_updates':completed,
                'training_seconds':seconds,'input_tokens':processed,'supervised_tokens':supervised,
                'terminal_presentations':presentations,'formal_tests_used':False,
                'adapter_sha256':sha(output/'adapter/adapter_model.safetensors'),
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group',choices=list('ABC'),required=True)
    parser.add_argument('--seed',type=int,choices=(401,402,403),required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after',type=int)
    args=parser.parse_args()
    run(args.group,args.seed,args.resume,args.stop_after)
