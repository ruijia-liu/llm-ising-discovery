"""Offline, symmetric case selection; never part of a student or teacher prompt."""
from copy import deepcopy
import hashlib
import json


SEEDS = (401, 402, 403)
GROUPS = ('B', 'C')
LEVELS = ('combination', 'four_body')
METHODS = ('llm_active', 'llm_fixed')
SALT = 'ising-behavior-v1'
BEHAVIORS = ('initial_wrong_final_recovered', 'protocol_failure_after_recovery',
             'early_correct_without_test', 'recovered_after_zero_removal',
             'competitive_idea_recovery', 'null_false_discovery')


def keyed_hash(field, value):
    raw = json.dumps([SALT, field, value], ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def identity(pair):
    return {'world_id': pair['world_id'], 'sigma': pair['sigma'], 'seed': pair['seed'],
            'level': pair['level'], 'structure_id': pair['structure_id']}


def ordering(pair):
    fields = ('structure_id', 'world_id', 'seed')
    return tuple(keyed_hash(name, pair[name]) for name in fields) + tuple(str(pair[name]) for name in fields)


def select_cases(pairs):
    """Input pairs contain B/C descriptive summaries, after complete-run validation."""
    cells = set()
    for pair in pairs:
        key = (pair['world_id'], pair['sigma'], pair['seed'])
        if key in cells or pair['seed'] not in SEEDS or pair['sigma'] not in (0, 1):
            raise ValueError('Duplicate or unregistered case cell.')
        cells.add(key)
        for group in GROUPS:
            if type(pair['groups'][group]['tested_idea_recovery']) is not bool:
                raise ValueError('Case selection requires the saved boolean endpoint.')
            if any(type(pair['groups'][group]['behaviors'][name]) is not bool for name in BEHAVIORS):
                raise ValueError('Behavior labels must be explicit booleans.')
    slots = []

    def add(name, predicate):
        eligible = sorted((p for p in pairs if p['sigma'] == 1 and predicate(p)), key=ordering)
        slots.append({'slot': name, 'eligible_seed_world_cells': len(eligible),
                      'eligible_parameter_worlds': len({p['world_id'] for p in eligible}),
                      'eligible_structures': len({p['structure_id'] for p in eligible}),
                      'eligible_seeds': sorted({p['seed'] for p in eligible}),
                      'eligible': [identity(p) for p in eligible],
                      'selected': identity(eligible[0]) if eligible else None})

    for level in LEVELS:
        for label, b, c in (('both_success', True, True), ('only_C_success', False, True),
                            ('only_B_success', True, False), ('neither_success', False, False)):
            add(f'{level}:{label}', lambda p, level=level, b=b, c=c:
                p['level'] == level and p['groups']['B']['tested_idea_recovery'] is b
                and p['groups']['C']['tested_idea_recovery'] is c)
    for group in GROUPS:
        for name in BEHAVIORS:
            add(f'{group}:{name}', lambda p, group=group, name=name:
                p['level'] in (('null',) if name == 'null_false_discovery' else LEVELS)
                and p['groups'][group]['behaviors'][name])
    return slots


def active_summary(artifact):
    row, episode = artifact['result'], artifact['episode']
    score, diagnosis = row['score'], row['grader_diagnostics']
    required = ('full_recovery', 'raw_support_success', 'tested_idea_recovery', 'competitive_idea_recovery',
                'valid_expression', 'final_support_committed_before_measurement', 'final_support_competitively_tested')
    if any(type(score.get(k)) is not bool for k in required):
        raise ValueError('Missing active recovery diagnostics; do not infer favorable values.')
    if type(diagnosis.get('is_null_world')) is not bool or type(diagnosis.get('tool_pruning_helped_full_recovery')) is not bool:
        raise ValueError('Case reports require saved grader-only diagnostics.')
    history = score['hypothesis_history_scores']
    if len(history) != len(episode['snapshots']) or any(a['used'] != b['used'] for a, b in zip(history, episode['snapshots'])):
        raise ValueError('Hypothesis scores do not align with submitted snapshots.')
    for item in history:
        if any(type(item.get(k)) is not bool for k in ('full_recovery', 'raw_support_success')):
            raise ValueError('Missing submitted-hypothesis score.')
    nonnull = not diagnosis['is_null_world']
    if row['level'] == 'null' and nonnull:
        raise ValueError('Null stratum contradicts saved grader diagnosis.')
    if (score['tested_idea_recovery'] != bool(nonnull and score['full_recovery'] and score['final_support_committed_before_measurement'])
            or score['competitive_idea_recovery'] != bool(nonnull and score['full_recovery'] and score['final_support_competitively_tested'])):
        raise ValueError('Active idea endpoint contradicts recovery and commitment components.')
    commits = [e for e in episode['events'] if e['kind'] == 'predictions_committed']
    invalid = not episode['valid_final_answer']
    if invalid == score['valid_expression']:
        raise ValueError('Final answer validity differs from saved score.')
    behaviors = {
        'initial_wrong_final_recovered': bool(nonnull and history and history[0]['used'] == 8
            and not history[0]['raw_support_success'] and score['raw_support_success']
            and score['full_recovery'] and episode['used'] > 8),
        'protocol_failure_after_recovery': bool(nonnull and invalid
            and episode['status'] in ('invalid_reply', 'inference_budget_exhausted')
            and any(h['full_recovery'] for h in history)),
        'early_correct_without_test': bool(nonnull and episode['status'] == 'student_finished'
            and episode['used'] == 8 and score['full_recovery'] and not score['tested_idea_recovery']),
        'recovered_after_zero_removal': bool(nonnull and score['full_recovery']
            and diagnosis['tool_pruning_helped_full_recovery']),
        'competitive_idea_recovery': bool(nonnull and score['competitive_idea_recovery']),
        'null_false_discovery': bool(not nonnull and score['valid_expression'] and score['false_discovery'] is True),
    }
    return {'status': episode['status'], 'used': episode['used'], 'inference': deepcopy(episode['inference']),
            **{k: score[k] for k in required}, 'false_discovery': score['false_discovery'],
            'raw_false_proposal': score['raw_false_proposal'], 'grader_category': diagnosis['category'],
            'truth_term_count': diagnosis['truth_term_count'], 'truth_orders': diagnosis['truth_orders'],
            'first_independent_recovery_used': score['first_independent_recovery_used'],
            'batches': len(commits),
            'example_query_and_alternative_batches': sum(e['idea_diagnostic']['example_query_and_alternative_copied'] for e in commits),
            'competitive_batches': sum(e['idea_diagnostic']['operational_competitive_test'] for e in commits),
            'behaviors': behaviors}


def timeline(artifact):
    """Preserve event ordering and values; no oracle, fitting or new model calls."""
    episode = artifact['episode']
    rows = []
    current_used, pending = 8, None
    history = iter(artifact['result']['score']['hypothesis_history_scores'])
    for index, event in enumerate(episode['events']):
        kind = event['kind']
        out = {'source_event_index': index, 'kind': kind, 'used': current_used}
        if kind == 'initial_evidence':
            out['evidence'] = event['evidence']
        elif kind == 'model_reply':
            # Full messages and exact replies remain in the exported source artifact.
            out.update(stage=event['stage'], raw_reply=event['reply']['text'])
        elif kind == 'current_hypothesis':
            snapshot_score = next(history)
            if event['used'] != current_used or snapshot_score['used'] != current_used:
                raise ValueError('Hypothesis appears under the wrong evidence prefix.')
            out.update(raw_structures=event['raw_structures'], expression=event['expression'],
                       fit=event['fit'], independent_score=snapshot_score)
        elif kind == 'predictions_committed':
            commit = event['commitment']
            if pending is not None or commit['committed_at_used'] != current_used:
                raise ValueError('Case timeline violates commitment ordering.')
            pending = commit
            out.update(commitment=commit, idea_diagnostic=event['idea_diagnostic'])
        elif kind == 'observations_revealed':
            observed = event['observation']
            if (pending is None or pending['before_sha256'] != observed['before_sha256']
                    or pending['ticket'] != observed['ticket']
                    or len(observed['observations']) != len(pending['queries'])):
                raise ValueError('An observation lacks its exact preceding prediction commitment.')
            current_used += len(observed['observations'])
            if observed['used'] != current_used:
                raise ValueError('Case timeline skips measured observations.')
            out.update(used=current_used, observation=observed)
            pending = None
        elif kind in ('invalid_reply', 'episode_failed', 'episode_ended'):
            out['record'] = event
        else:
            raise ValueError('Unrecognized event; do not silently omit behavior.')
        rows.append(deepcopy(out))
    if pending is not None or next(history, None) is not None or current_used != episode['used']:
        raise ValueError('Incomplete hypothesis/measurement timeline.')
    return rows


def validate_run_grid(evaluations):
    expected = {(method, group, seed) for method in METHODS for group in GROUPS for seed in SEEDS}
    keys = [(e['config']['method'], e['config']['group'], e['config']['seed']) for e in evaluations]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('All twelve B/C active/fixed evaluations and three seeds are required exactly once.')
    reference = evaluations[0]
    c0 = reference['config']
    descriptions = lambda e: {(r['world_id'], r['sigma']): (r['level'], r['structure_id']) for r in e['result']['rows']}
    grid = descriptions(reference)
    for e in evaluations:
        c = e['config']
        if c.get('study', 'mainline') != 'mainline':
            raise ValueError('These case slots describe the main study only.')
        if (descriptions(e) != grid or any(c[k] != c0[k] for k in
                ('scope', 'sigmas', 'model_revision', 'calibration_sha256', 'sources'))):
            raise ValueError('Case inputs differ in worlds, clusters, base model, scoring or source versions.')
    by_key = dict(zip(keys, evaluations))
    for group in GROUPS:
        for seed in SEEDS:
            a, f = (by_key[method, group, seed]['config'] for method in METHODS)
            if any(a[k] != f[k] for k in ('adapter_sha256', 'training_config_sha256', 'training_result_sha256')):
                raise ValueError('Active and fixed cases must use the same trained checkpoint.')
    return by_key


def pair_artifacts(records):
    """records[(method, group)] is one already verified artifact for a seed/world."""
    initial, full_fixed = None, None
    identity_fields = ('world_id', 'sigma', 'seed', 'level', 'structure_id')
    reference = records['llm_active', 'B']['result']
    groups = {}
    for group in GROUPS:
        for method in METHODS:
            artifact = records[method, group]
            if any(artifact['result'][k] != reference[k] for k in identity_fields):
                raise ValueError('Case artifacts refer to different worlds or seeds.')
            episode = artifact['episode']
            evidence = episode['events'][0]['evidence'] if method == 'llm_active' else episode['evidence']
            first = {'measurement_noise_sigma': evidence['measurement_noise_sigma'],
                     'used': 8, 'measurements': evidence['measurements'][:8]}
            if initial is not None and first != initial:
                raise ValueError('Case groups did not receive identical initial evidence.')
            initial = first
            if method == 'llm_fixed':
                if full_fixed is not None and full_fixed != evidence:
                    raise ValueError('Fixed-evidence groups did not receive the same measurements.')
                full_fixed = evidence
        active, fixed = records['llm_active', group], records['llm_fixed', group]
        groups[group] = {**active_summary(active), 'fixed_budget_curve': deepcopy(fixed['result']['score']['budget_curve'])}
    return {**{k: reference[k] for k in identity_fields}, 'groups': groups}
