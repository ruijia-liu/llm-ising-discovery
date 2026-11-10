"""Delete only explicit numerical prediction fields from frozen C messages."""
from copy import deepcopy
import json

from .mainline_schedule import paired_schedule
from .student_v2 import packed


GROUP = 'C_no_predictions'
REMOVED_KEYS = {'predicted_delta', 'prediction_errors'}


def remove_explicit_predictions(messages):
    """Return transformed messages and a reversible deletion audit.

    A whitelist of full field paths catches format drift. Observations and
    expressions are preserved, including accidentally correct teacher guesses.
    """
    output, removals = deepcopy(messages), []
    before_count = update_count = 0
    for index, message in enumerate(output):
        if message['role'] != 'assistant':
            continue
        value = json.loads(message['content'])
        allowed = set()
        if 'before_observation' in value:
            before_count += 1
            hypotheses = value['before_observation']['hypotheses']
            if not hypotheses:
                raise ValueError('Missing candidate hypotheses.')
            for j, hypothesis in enumerate(hypotheses):
                path = ('before_observation', 'hypotheses', j, 'predicted_delta')
                allowed.add(path)
                if 'predicted_delta' not in hypothesis:
                    raise ValueError('Missing original numerical prediction.')
        for key in ('update_from_previous_observation', 'update_from_last_observation'):
            if key in value:
                update_count += 1
                allowed.add((key, 'prediction_errors'))
                if 'prediction_errors' not in value[key]:
                    raise ValueError('Missing original prediction error.')

        def visit(node, path=()):
            if isinstance(node, list):
                return [visit(child, path+(j,)) for j, child in enumerate(node)]
            if not isinstance(node, dict):
                return node
            result = {}
            for key, child in node.items():
                current = path+(key,)
                if key in REMOVED_KEYS:
                    if current not in allowed:
                        raise ValueError('Unexpected prediction field; re-audit the schema.')
                    removals.append({'message_index': index, 'path': list(current), 'value': deepcopy(child)})
                else:
                    result[key] = visit(child, current)
            return result

        message['content'] = packed(visit(value))
    if (before_count, update_count) != (24, 24):
        raise ValueError('Expected the complete original 24-round C trajectory.')
    # Restore deleted values at their exact paths: the entire remaining JSON,
    # rather than a selected subset of fields, must equal the original.
    restored = deepcopy(output)
    payloads = {i: json.loads(m['content']) for i, m in enumerate(restored) if m['role'] == 'assistant'}
    for row in removals:
        parent = payloads[row['message_index']]
        for key in row['path'][:-1]:
            parent = parent[key]
        parent[row['path'][-1]] = row['value']
    for i, (old, new) in enumerate(zip(messages, output, strict=True)):
        if old['role'] != new['role']:
            raise AssertionError('Message roles changed.')
        if old['role'] == 'assistant':
            if json.loads(old['content']) != payloads[i]:
                raise AssertionError('Non-prediction content changed.')
        elif old != new:
            raise AssertionError('System instructions or evidence changed.')
    return output, removals


def task_plan(manifest, seed):
    """Match full-C input volume while keeping identical terminal exposures."""
    if seed not in (401, 402, 403):
        raise ValueError('Use a registered seed.')
    rows = {row['episode_id']: row for row in manifest['episodes']}
    records = [{'episode_id': row['episode_id'], 'auxiliary': {
        'B': manifest['entries'][row['auxiliary'][GROUP]],
        'C': manifest['entries'][row['auxiliary']['C']]}}
        for row in manifest['episodes']]
    schedule = paired_schedule(records, seed, passes=2)
    tasks, inputs, supervised, repetitions = [], 0, 0, 0
    candidate_presentations = 0
    for step in schedule['steps']:
        row = rows[step['episode_id']]
        current = [{'key': row['terminal'][str(b)], 'weight': .25, 'kind': 'terminal'}
                   for b in step['B_C_terminal_prefixes']]
        repeat = step['B_auxiliary_repetitions']
        current += [{'key': row['auxiliary'][GROUP], 'weight': 1/repeat, 'kind': 'auxiliary'}
                    for _ in range(repeat)]
        inputs += sum(manifest['entries'][t['key']]['input_tokens'] for t in current)
        supervised += sum(manifest['entries'][t['key']]['supervised_tokens'] for t in current)
        repetitions += repeat
        candidate_presentations += row['candidate_expressions_per_auxiliary']*repeat
        tasks.append({'episode_id': step['episode_id'], 'tasks': current})
    return tasks, {'updates': len(tasks), 'terminal_presentations': len(tasks)*4,
                   'input_tokens': inputs, 'supervised_tokens': supervised,
                   'auxiliary_presentations': repetitions,
                   'candidate_expression_presentations': candidate_presentations,
                   'full_C_auxiliary_presentations': len(tasks),
                   'full_C_candidate_expression_presentations': sum(
                       r['candidate_expressions_per_auxiliary'] for r in rows.values())*2,
                   'relative_auxiliary_input_difference': schedule['relative_auxiliary_input_difference']}
