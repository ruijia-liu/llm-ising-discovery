"""Shorten process supervision, preserving every B/C measurement and causal order."""
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from build_publication_curriculum import atomic_json, sha
from discovery.student_format import compact_messages
from discovery.student_mainline import _localize, auxiliary_messages, terminal_messages
from discovery.student_v2 import encode, load_views, packed
from discovery.teaching import decode_query, expression_from_dict, messages_for_view

SELECTED = (0, 8, 16, 23)
ROOT = Path('data/student_short_bc_v1')


def short_auxiliary(view, group):
    if group not in ('B', 'C'):
        raise ValueError('This development comparison has only B and C.')
    full = messages_for_view(view, group)['messages'][:-1]
    count = 0
    for message in full[1:]:
        if message['role'] != 'assistant':
            continue
        value = json.loads(message['content'])
        if 'experiment' in value:
            if count not in SELECTED:
                value.pop('before_observation', None)
            if count-1 not in SELECTED:
                value.pop('update_from_previous_observation', None)
            count += 1
        message['content'] = packed(value)
    if count != 24:
        raise ValueError('Expected the full 24 extra measurements.')
    if group == 'C':
        full.append({'role': 'assistant', 'content': packed({
            'update_from_last_observation': view['research_process'][-1]['after'],
            'teacher_estimate_before_grading': view['teacher_final_before_grading']})})
    # Rebuild '=' references AFTER selection, from original uncompressed prose.
    compact = compact_messages(full)
    result = [auxiliary_messages(view, group)[0]]
    result += [{'role': m['role'], 'content': packed(_localize(json.loads(m['content'])))} for m in compact[1:]]
    if any('"final_answer"' in m['content'] for m in result):
        raise ValueError('Grader answer leaked into auxiliary process supervision.')
    return result


def without_process(messages):
    result = []
    for message in messages[1:]:
        value = json.loads(message['content'])
        for key in ('before_observation', 'update_from_previous_observation', 'update_from_last_observation', 'teacher_estimate_before_grading'):
            value.pop(key, None)
        if value:
            result.append({'role': message['role'], 'content': value})
    return result


def run():
    if ROOT.exists():
        raise ValueError('Preserve any completed or partial short curriculum.')
    tokenizer = AutoTokenizer.from_pretrained('models/Qwen2.5-0.5B-Instruct', local_files_only=True)
    views = {g: {v['episode_id']: v for sigma in (0, 1) for v in load_views('train', g, sigma=sigma)} for g in ('B', 'C')}
    entries, arrays, episodes, prediction_checks = [], {}, [], 0
    for episode in sorted(views['B']):
        b, c = views['B'][episode], views['C'][episode]
        tasks = {}
        for budget in (8, 32):
            messages = terminal_messages(b, budget)
            if messages != terminal_messages(c, budget):
                raise ValueError('B/C common evidence or answer labels differ.')
            tasks[f'terminal{budget}'] = messages
        tasks.update({g: short_auxiliary(views[g][episode], g) for g in ('B', 'C')})
        if tasks['B'] != auxiliary_messages(b, 'B') or without_process(tasks['B']) != without_process(tasks['C']):
            raise ValueError('Shortening changed B records or shared measurement sequence.')
        before_count = sum('before_observation' in json.loads(m['content']) for m in tasks['C'][1:])
        if before_count != 4:
            raise ValueError('Incorrect number of process points.')
        for index in SELECTED:
            process, experiment = c['research_process'][index], c['experiments'][index]
            query = decode_query(experiment['query'])
            for h, error in zip(process['before']['hypotheses'], process['after']['prediction_errors']):
                predicted = expression_from_dict(h['expression']).delta(*query)
                if not np.isclose(predicted, h['predicted_delta'], atol=1e-9, rtol=0) or not np.isclose(experiment['observed']-predicted, error, atol=1e-9, rtol=0):
                    raise ValueError('A selected pre-observation prediction or update is numerically wrong.')
                prediction_checks += 1
        keys = {}
        for kind, messages in tasks.items():
            key = episode+'_'+kind
            data = encode(tokenizer, messages)
            keys[kind] = key
            arrays[key+'_ids'], arrays[key+'_labels'] = data['ids'], data['labels']
            entries.append({'key': key, 'kind': kind, 'messages': messages, 'input_tokens': len(data['ids']),
                            'supervised_tokens': int(np.count_nonzero(data['labels'][1:] != -100))})
        episodes.append({'episode': episode, 'keys': keys})
    if len(episodes) != 48:
        raise ValueError('Expected 24 worlds and two noise conditions.')
    lookup = {e['key']: e for e in entries}
    rng = np.random.default_rng(701)
    order = [episodes[int(i)] for i in rng.permutation(48)]
    b_total = c_total = 0
    schedule = []
    for episode in order:
        keys = episode['keys']
        b_length, c_length = (lookup[keys[g]]['input_tokens'] for g in ('B', 'C'))
        c_total += c_length
        repetitions = max(1, int(round((c_total-b_total)/b_length)))
        b_total += repetitions*b_length
        schedule.append({**episode, 'B_auxiliary_repetitions': repetitions, 'C_auxiliary_repetitions': 1})
    relative = abs(b_total-c_total)/c_total
    if relative > .01:
        raise ValueError('B/C auxiliary input totals differ by over 1%.')
    sources = ('prepare_short_bc.py', 'SHORT_BC_DEVELOPMENT_PROTOCOL.md', 'discovery/teaching.py',
               'discovery/student_format.py', 'discovery/student_mainline.py', 'discovery/student_v2.py', 'discovery/student_spins.py')
    atomic_json(ROOT/'manifest.json', {'completed': True, 'scope': 'new_short_BC_development_only',
        'episodes': 48, 'physical_worlds': 24, 'sigmas': [0, 1], 'selected_process_round_indices': SELECTED,
        'entries': [{k: v for k, v in e.items() if k != 'messages'} for e in entries],
        'schedule': schedule, 'auxiliary_input_tokens': {'B': b_total, 'C': c_total},
        'relative_auxiliary_input_difference': relative, 'prediction_and_update_pairs_verified': prediction_checks,
        'all_measurements_and_common_targets_matched': True, 'B_auxiliary_messages_unchanged': True,
        'maximum_context_tokens': max(e['input_tokens'] for e in entries),
        'training_performed': False, 'formal_tests_used': False, 'sources': {p: sha(p) for p in sources},
        'data_sha256': {f'data/research_curriculum_v1/views/train/{g}.jsonl': sha(f'data/research_curriculum_v1/views/train/{g}.jsonl') for g in ('B', 'C')}})
    np.savez_compressed(ROOT/'tasks.npz', **arrays)
    with (ROOT/'messages.jsonl').open('w', encoding='utf-8') as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False)+'\n')
    for source in sources:
        path = ROOT/'sources'/source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(source).read_bytes())
    atomic_json(ROOT/'files.json', {p: sha(ROOT/p) for p in ('manifest.json', 'tasks.npz', 'messages.jsonl')})
    print(json.dumps({'episodes': 48, 'tasks': len(entries), 'prediction_checks': prediction_checks,
                      'maximum_context': max(e['input_tokens'] for e in entries), 'auxiliary_tokens_B': b_total,
                      'auxiliary_tokens_C': c_total, 'relative_input_difference': relative}, indent=2))


if __name__ == '__main__':
    run()
