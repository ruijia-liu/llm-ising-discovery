"""Validate completed evaluation artifacts before any paper-level aggregation."""
from collections import Counter
import math
from pathlib import Path

from build_publication_curriculum import read, sha
from .episode_journal import digest


BOOLEAN_SCORES = ('valid_expression', 'raw_support_success', 'prediction_success', 'support_success', 'full_recovery')


def validate_score(score):
    if any(type(score.get(name)) is not bool for name in BOOLEAN_SCORES):
        raise ValueError('Missing or nonboolean recovery metric.')
    if score['full_recovery'] and not all(score[name] for name in ('valid_expression', 'prediction_success', 'support_success')):
        raise ValueError('Full recovery contradicts component metrics.')
    for name in ('uniform_rmse', 'biased_rmse', 'coefficient_max_error'):
        value = score[name]
        if value is None:
            if score['valid_expression']:
                raise ValueError('A valid expression is missing its error metric.')
        elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid error metric.')
    for name in ('false_discovery', 'raw_false_proposal'):
        if score[name] is not None and type(score[name]) is not bool:
            raise ValueError('False-discovery metrics must be boolean or explicitly unavailable.')


def measured_observations(episode, method):
    """Check causal accounting and return observations in charged order."""
    if method == 'llm_active':
        events = episode['events']
        if not events or events[0]['kind'] != 'initial_evidence' or events[-1]['kind'] != 'episode_ended':
            raise ValueError('Active episode has no complete initial/final event envelope.')
        initial = events[0]['evidence']
        if initial['used'] != 8 or len(initial['measurements']) != 8:
            raise ValueError('Missing the eight common initial observations.')
        values = [m['delta'] for m in initial['measurements']]
        used, pending, replies = 8, None, []
        for event in events[1:]:
            kind = event['kind']
            if kind == 'model_reply':
                if event['used'] != used:
                    raise ValueError('Generation was attributed to the wrong evidence budget.')
                replies.append(event['reply'])
            elif kind == 'predictions_committed':
                commit = event['commitment']
                if pending is not None or commit['committed_at_used'] != used:
                    raise ValueError('Prediction commitment did not precede its observations.')
                pending = commit
            elif kind == 'observations_revealed':
                observed = event['observation']
                if (pending is None or observed['ticket'] != pending['ticket']
                        or observed['before_sha256'] != pending['before_sha256']
                        or len(observed['observations']) != len(pending['queries'])):
                    raise ValueError('Observation lacks its exact prior commitment.')
                used += len(observed['observations']); values += observed['observations']
                if observed['used'] != used or used > 32:
                    raise ValueError('Active observation accounting exceeded or skipped budget.')
                pending = None
        if pending is not None or episode['used'] != used or events[-1]['used'] != used:
            raise ValueError('Episode ended with unresolved or inconsistent measurement accounting.')
        costs = episode['inference']
        if (costs['calls'] != len(replies) or costs['input_tokens'] != sum(r['input_tokens'] for r in replies)
                or costs['output_tokens'] != sum(r['output_tokens'] for r in replies)):
            raise ValueError('Recorded generation usage differs from raw replies, including invalid replies.')
        return values, replies
    events = episode['acquisition_events'] if method.endswith('_fixed') else episode['events']
    if not events or events[0]['kind'] != 'initial' or events[0]['used'] != 8 or len(events[0]['observed']) != 8:
        raise ValueError('Missing original initial acquisition event.')
    values, used = list(events[0]['observed']), 8
    for event in events[1:]:
        if (event['kind'] != 'experiment' or event['committed_at_used'] != used
                or len(event['observed']) != len(event['queries'])):
            raise ValueError('Invalid metered experiment history.')
        values += event['observed']; used += len(event['observed'])
        if event['used'] != used or used > 32:
            raise ValueError('Measurement accounting exceeded or skipped budget.')
    if used != 32:
        raise ValueError('Registered fixed-evidence/search evaluation requires 32 measurements.')
    replies = []
    if method.endswith('_fixed'):
        evidence = episode['evidence']
        if evidence['used'] != 32 or [m['delta'] for m in evidence['measurements']] != values:
            raise ValueError('Fixed evidence differs from the actual metered acquisitions.')
        if set(episode['answers']) != {'8', '16', '24', '32'}:
            raise ValueError('Missing a registered fixed-evidence answer.')
        if method == 'llm_fixed':
            replies = [episode['answers'][str(b)]['reply'] for b in (8, 16, 24, 32)]
    return values, replies


def load_completed_evaluation(directory, **expected):
    """No world/answer files are opened; only already completed output artifacts."""
    root = Path(directory)
    result, config = read(root/'result.json'), read(root/'config.json')
    if not result['completed'] or result['scope'] != config['scope'] or result['method'] != config['method']:
        raise ValueError('Evaluation has not completed consistently.')
    view = {**config, 'study': config.get('study', 'mainline')}
    if any(view.get(key) != value for key, value in expected.items()):
        raise ValueError('Evaluation belongs to another study, method, group, seed, or scope.')
    if config['scope'] not in ('development', 'formal') or result['formal_tests_used'] != (config['scope'] == 'formal'):
        raise ValueError('Development and formal evaluations must remain separate.')
    worlds, sigmas = config['worlds'], config['sigmas']
    if len(worlds) != len(set(worlds)) or sigmas != [0, 1]:
        raise ValueError('Duplicate worlds or altered registered noise conditions.')
    if config['scope'] == 'formal' and len(worlds) != 200:
        raise ValueError('Do not aggregate an incomplete formal test subset.')
    for name, checksum in config['sources'].items():
        if sha(root/'sources'/name) != checksum:
            raise ValueError('Saved evaluation source snapshot changed.')
    rows = result['rows']
    required = {(world, sigma) for world in worlds for sigma in sigmas}
    if result['worlds'] != len(worlds) or result['episodes'] != len(required) or len(rows) != len(required):
        raise ValueError('Incomplete result grid; do not drop failed or missing episodes.')
    cells, descriptions, artifacts, resources = set(), {}, {}, []
    for row in rows:
        cell = (row['world_id'], row['sigma'])
        if cell in cells or cell not in required or type(row['sigma']) is not int:
            raise ValueError('Duplicate, unregistered, or malformed evaluation cell.')
        cells.add(cell)
        for name in ('method', 'group', 'seed'):
            if row[name] != config[name]:
                raise ValueError('Result row does not belong to the run metadata.')
        if row.get('study', 'mainline') != view['study']:
            raise ValueError('Different studies were mixed within one result file.')
        description = (row['level'], row['structure_id'])
        if row['world_id'] in descriptions and descriptions[row['world_id']] != description:
            raise ValueError('Mechanism cluster identity changed across noise conditions.')
        descriptions[row['world_id']] = description
        validate_score(row['score'])
        curve = row['score']['budget_curve']
        if [r['budget'] for r in curve] != [8, 16, 24, 32]:
            raise ValueError('Missing or duplicated budget-curve checkpoint.')
        for checkpoint in curve:
            validate_score(checkpoint)
        if config['method'].endswith('_active') and type(row['score'].get('tested_idea_recovery')) is not bool:
            raise ValueError('Missing the prespecified active endpoint.')
        key = row['world_id']+f"-sigma{row['sigma']}"
        path = root/'episodes'/(key+'.json')
        artifact = read(path)
        if artifact['result'] != row:
            raise ValueError('Aggregate score differs from its original episode artifact.')
        observed, replies = measured_observations(artifact['episode'], config['method'])
        if not 8 <= len(observed) <= 32:
            raise ValueError('Invalid total measurement budget.')
        for reply in replies:
            if (type(reply['input_tokens']) is not int or reply['input_tokens'] < 1
                    or type(reply['output_tokens']) is not int or reply['output_tokens'] < 0):
                raise ValueError('Invalid recorded model token usage.')
        if 'execution' in artifact:
            completion = read(root/'private_journals'/key/'completion.json')
            execution = artifact['execution']
            if (completion['episode_sha256'] != digest(artifact['episode'])
                    or completion['logical_oracle_calls'] != len(observed)
                    or completion['logical_model_calls'] != len(replies)
                    or execution['logical_oracle_calls'] != len(observed)
                    or execution['logical_model_calls'] != len(replies)
                    or execution['ambiguous_calls_retried'] != 0):
                raise ValueError('Durable completion/call accounting differs from the original episode.')
        elif config['scope'] == 'formal':
            raise ValueError('Formal episodes require the registered durable execution receipts.')
        artifacts[key] = sha(path)
        resources.append({'world_id': row['world_id'], 'sigma': row['sigma'], 'measurements': len(observed),
                          'model_calls': len(replies), 'input_tokens': sum(r['input_tokens'] for r in replies),
                          'output_tokens': sum(r['output_tokens'] for r in replies)})
    if cells != required or result['full_recoveries'] != sum(r['score']['full_recovery'] for r in rows):
        raise ValueError('Incomplete grid or inconsistent recovery total.')
    if config['scope'] == 'formal':
        counts = Counter(level for level, _ in descriptions.values())
        if counts != {'parameter': 50, 'combination': 50, 'four_body': 50, 'null': 50}:
            raise ValueError('Formal mechanism strata are incomplete.')
    return {'config': config, 'result': result, 'resources': resources,
            'provenance': {'directory': str(root), 'config_sha256': sha(root/'config.json'),
                           'result_sha256': sha(root/'result.json'), 'episodes_sha256': artifacts}}


def verify_pairing(evaluations):
    """Require identical worlds, cluster labels, scoring and interface versions."""
    if not evaluations:
        raise ValueError('No completed evaluations to pair.')
    reference = evaluations[0]
    reference_config = reference['config']
    def shared_sources(config):
        return {k: v for k, v in config['sources'].items() if k != 'PROCESS_ABLATION_PROTOCOL.md'}
    def descriptions(evaluation):
        return {(r['world_id'], r['sigma']): (r['level'], r['structure_id']) for r in evaluation['result']['rows']}
    for current in evaluations[1:]:
        config = current['config']
        if (descriptions(current) != descriptions(reference) or config['scope'] != reference_config['scope']
                or config['method'] != reference_config['method'] or config['sigmas'] != reference_config['sigmas']
                or config.get('model_revision') != reference_config.get('model_revision')
                or config['calibration_sha256'] != reference_config['calibration_sha256']
                or shared_sources(config) != shared_sources(reference_config)):
            raise ValueError('Runs do not share the registered worlds, clusters, interface, scoring or source version.')
