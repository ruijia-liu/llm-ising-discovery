"""Audit completed main training without loading a model or reserved worlds."""
import argparse
import json
import math
from pathlib import Path

from build_publication_curriculum import atomic_json, read, sha
from discovery.evaluation_profiles import registered_student


def audit_effort(config, result, records, events, entries):
    schedule = config['schedule']
    n = len(schedule)
    if (not result['completed'] or result['completed_updates'] != n or config['maximum_updates'] != n
            or [r['step'] for r in records] != list(range(1, n+1))):
        raise ValueError('Need every committed update of a completed run.')
    totals = {'input_tokens': 0, 'supervised_tokens': 0, 'terminal_presentations': 0}
    for expected, record in zip(schedule, records):
        if record['episode_id'] != expected['episode_id']:
            raise ValueError('Committed updates differ from the registered order.')
        for task in expected['tasks']:
            entry = entries[task['key']]
            for field in ('input_tokens', 'supervised_tokens'):
                totals[field] += entry[field]
            totals['terminal_presentations'] += task['kind'] == 'terminal'
        for field, log_field in (('input_tokens', 'processed_tokens'), ('supervised_tokens', 'supervised_tokens'),
                                 ('terminal_presentations', 'terminal_presentations')):
            if totals[field] != record[log_field]:
                raise ValueError('An effective-update counter differs from its scheduled tasks.')
    if any(result[k] != value or config['budget'][k] != value for k, value in totals.items()):
        raise ValueError('Final counters differ from the registered budget.')
    if result['training_seconds'] != records[-1]['training_seconds']:
        raise ValueError('Final update timing differs from its checkpoint record.')
    invocations, started, completed, commits = {}, {}, {}, set()
    for event in events:
        invocation = event['invocation']
        kind = event['event']
        if kind == 'invocation_start':
            if invocation in invocations or not 0 <= event['resumed_after_step'] <= n:
                raise ValueError('Duplicate or invalid training invocation.')
            invocations[invocation] = event
            continue
        if invocation not in invocations:
            raise ValueError('Effort record has no invocation start.')
        step = event['step']
        if type(step) is not int or not invocations[invocation]['resumed_after_step'] < step <= n:
            raise ValueError('Effort record refers to an invalid or already-restored update.')
        if kind == 'checkpoint_committed':
            for i, task in enumerate(schedule[step-1]['tasks']):
                if (invocation, step, i) not in completed:
                    raise ValueError('Commit receipt precedes a required completed task.')
            if step in commits:
                raise ValueError('A committed update was repeated.')
            commits.add(step)
            continue
        index = event['task_index']
        if type(index) is not int or not 0 <= index < len(schedule[step-1]['tasks']):
            raise ValueError('Invalid task index.')
        task = schedule[step-1]['tasks'][index]
        key = (invocation, step, index)
        if event['key'] != task['key']:
            raise ValueError('Effort record differs from its registered task.')
        if kind == 'task_started':
            if key in started:
                raise ValueError('Duplicate task-start receipt.')
            started[key] = event
        elif kind == 'task_completed':
            if key not in started or key in completed:
                raise ValueError('Missing start or duplicate task completion.')
            if any(event[k] != entries[task['key']][k] for k in ('input_tokens', 'supervised_tokens')):
                raise ValueError('Completed-task tokens differ from the immutable cache.')
            if type(event['seconds']) not in (int, float) or not math.isfinite(event['seconds']) or event['seconds'] < 0:
                raise ValueError('Invalid completed-task time.')
            completed[key] = event
        else:
            raise ValueError('Unknown training effort event.')
    observed = {k: sum(e[k] for e in completed.values()) for k in ('input_tokens', 'supervised_tokens')}
    if any(observed[k] < totals[k] for k in observed):
        raise ValueError('The task log cannot account for the effective update budget; audit the missing effort.')
    return {'effective_updates': n, 'effective_budget': totals, 'invocations': len(invocations),
            'completed_task_presentations': len(completed), 'completed_task_tokens_including_recomputed': observed,
            'completed_tokens_beyond_effective_updates': {k: observed[k]-totals[k] for k in observed},
            'unfinished_task_receipts': [list(k) for k in sorted(set(started)-set(completed))],
            'unfinished_task_cost_is_unknown_not_zero': bool(set(started)-set(completed)),
            'steps_without_effort_commit_receipt': sorted(set(range(1, n+1))-commits),
            'known_completed_task_seconds': sum(e['seconds'] for e in completed.values()),
            'effective_update_seconds_excluding_load_and_checkpoint_io': result['training_seconds'],
            'peak_allocated_gib_across_all_recorded_invocations': max(r['peak_allocated_gib'] for r in records),
            'full_execution_wall_seconds': None,
            'wall_time_note': 'Original queue did not record complete start/end durations; do not substitute update time.'}


def run(group, seed, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserve the previous completed-training audit.')
    profile = registered_student('mainline', group, seed)
    root = Path('results/student_mainline_v1')/f'{group}_seed{seed}'
    config, result = read(root/'config.json'), read(root/'result.json')
    manifest_path = Path('data/student_mainline_v1/manifest.json')
    if sha(manifest_path) != config['manifest_sha256']:
        raise ValueError('The registered training cache manifest changed.')
    manifest = read(manifest_path)
    sources = {**manifest['config']['sources'], **config['sources']}
    if any(sha(name) != digest for name, digest in sources.items()):
        raise ValueError('A frozen training or cache source changed.')
    records, events = ([json.loads(line) for line in (root/name).read_text(encoding='utf-8').splitlines()]
                        for name in ('training.jsonl', 'training_effort.jsonl'))
    summary = audit_effort(config, result, records, events, manifest['entries'])
    payload = {'completed': True, 'group': group, 'seed': seed, 'formal_tests_used': False,
               'training_profile': {k: str(v) if isinstance(v, Path) else v for k, v in profile.items()},
               'summary': summary, 'verified_frozen_sources': sources,
               'inputs_sha256': {str(root/name): sha(root/name) for name in
                   ('config.json', 'result.json', 'training.jsonl', 'training_effort.jsonl')},
               'auditor_sha256': sha(__file__)}
    atomic_json(output/'audit.json', payload)
    print(json.dumps({'group': group, 'seed': seed, **summary}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=('A', 'B', 'C'), required=True)
    parser.add_argument('--seed', choices=(401, 402, 403), type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.group, args.seed, args.output)
