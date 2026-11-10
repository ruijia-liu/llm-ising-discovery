"""Descriptive secondary summaries with explicit missing-value denominators."""
from collections import Counter, defaultdict
import hashlib
import json
import math

import numpy as np


NULL_STRUCTURE_ID = hashlib.sha256(json.dumps(()).encode()).hexdigest()


def distribution(values):
    available = [v for v in values if v is not None]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in available):
        raise ValueError('A descriptive measurement must be finite and nonnegative, or explicitly missing.')
    return {'available': len(available), 'unavailable': len(values)-len(available),
            'mean': float(np.mean(available)) if available else None,
            'median': float(np.median(available)) if available else None,
            'p90': float(np.quantile(available, .9)) if available else None}


def fraction(successes, denominator):
    return {'count': int(successes), 'denominator': denominator,
            'rate': float(successes/denominator) if denominator else None}


def generation_time(replies):
    seconds = [r['generation_seconds'] for r in replies if r.get('generation_seconds') is not None]
    distribution(seconds)  # Do not quietly aggregate corrupt or nonfinite timing records.
    return {'known_generation_seconds': sum(seconds) if seconds else (0. if not replies else None),
            'missing_generation_time_records': len(replies)-len(seconds)}


def is_null(row):
    diagnostic = row.get('grader_diagnostics', {})
    return diagnostic.get('is_null_world', row['structure_id'] == NULL_STRUCTURE_ID)


def behavior(episode, method):
    if method == 'llm_active':
        events = episode['events']
        revealed = {e['observation']['before_sha256'] for e in events if e['kind'] == 'observations_revealed'}
        commits = [e for e in events if e['kind'] == 'predictions_committed' and e['commitment']['before_sha256'] in revealed]
        return {'status': episode['status'], 'revealed_batches': len(revealed),
                'exact_example_batches': sum(e['idea_diagnostic']['exact_format_example_copied'] for e in commits),
                'example_query_and_alternative_batches': sum(e['idea_diagnostic']['example_query_and_alternative_copied'] for e in commits),
                'competitive_batches': sum(e['idea_diagnostic']['operational_competitive_test'] for e in commits)}
    return {'status': 'fixed_evidence_completed' if method.endswith('_fixed') else 'search_budget_completed',
            'revealed_batches': len(episode.get('events', []))-1 if method == 'sparse_active' else None,
            'exact_example_batches': None, 'example_query_and_alternative_batches': None, 'competitive_batches': None}


def summarize(items):
    """Each item supplies a verified result row, actual resources and behavior."""
    groups = defaultdict(list)
    for item in items:
        row = item['row']
        key = (row.get('study', 'mainline'), row['method'], row['group'], row['seed'], row['sigma'], row['level'])
        groups[key].append(item)
    output = []
    for key, members in sorted(groups.items(), key=lambda kv: tuple(str(v) for v in kv[0])):
        study, method, group, seed, sigma, level = key
        rows, scores = [m['row'] for m in members], [m['row']['score'] for m in members]
        n = len(rows)
        if len({r['world_id'] for r in rows}) != n:
            raise ValueError('Duplicate world within a descriptive group/seed stratum.')
        nulls = [m for m in members if is_null(m['row'])]
        nonnull = [m for m in members if not is_null(m['row'])]
        valid_null = [m for m in nulls if m['row']['score']['valid_expression']]
        raw_fp = sum(m['row']['score']['raw_false_proposal'] is True for m in nulls)
        fitted_fp = sum(m['row']['score']['false_discovery'] is True for m in nulls)
        metrics = {name: fraction(sum(s[name] for s in scores), n) for name in
                   ('full_recovery', 'prediction_success', 'raw_support_success', 'support_success')}
        metrics['invalid_final_expression'] = fraction(sum(not s['valid_expression'] for s in scores), n)
        idea_values = [m['row']['score']['tested_idea_recovery'] for m in nonnull if 'tested_idea_recovery' in m['row']['score']]
        metrics['tested_idea_recovery_nonnull'] = fraction(sum(idea_values), len(idea_values))
        errors = {name: distribution([s[name] for s in scores]) for name in ('uniform_rmse', 'biased_rmse', 'coefficient_max_error')}
        null_report = {'worlds': len(nulls), 'valid_answers': len(valid_null), 'invalid_answers': len(nulls)-len(valid_null),
                       'raw_false_proposal_all_nulls': fraction(raw_fp, len(nulls)),
                       'fitted_false_discovery_all_nulls': fraction(fitted_fp, len(nulls)),
                       'raw_false_proposal_given_valid': fraction(raw_fp, len(valid_null)),
                       'fitted_false_discovery_given_valid': fraction(fitted_fp, len(valid_null))}
        first, schedules = [], Counter()
        for member in nonnull:
            row = member['row']
            d = row.get('grader_diagnostics', {})
            if d:
                value, schedule = d['first_independent_recovery_used'], d['first_recovery_observation_schedule']
            elif method == 'llm_active':
                value, schedule = row['score']['first_independent_recovery_used'], 'each_actual_submitted_hypothesis'
            else:
                value = next((b['budget'] for b in row['score']['budget_curve'] if b['full_recovery']), None)
                schedule = 'legacy_only_stored_8_16_24_32_checkpoints'
            if value is not None and not 8 <= value <= 32:
                raise ValueError('First recovery is outside the measurement budget.')
            first.append(value); schedules[schedule] += 1
        curves = []
        for index, budget in enumerate((8, 16, 24, 32)):
            checkpoint = [s['budget_curve'][index] for s in scores]
            curves.append({'budget': budget, 'full_recovery': fraction(sum(s['full_recovery'] for s in checkpoint), n),
                           'raw_support_success': fraction(sum(s['raw_support_success'] for s in checkpoint), n),
                           'invalid_expression': fraction(sum(not s['valid_expression'] for s in checkpoint), n),
                           'uniform_rmse_conditional_on_available': distribution([s['uniform_rmse'] for s in checkpoint]),
                           'biased_rmse_conditional_on_available': distribution([s['biased_rmse'] for s in checkpoint])})
        resources = [m['resources'] for m in members]
        time_available = [r['known_generation_seconds'] for r in resources if r['known_generation_seconds'] is not None]
        pruning = [r['grader_diagnostics']['tool_pruning_helped_full_recovery'] for r in rows if 'grader_diagnostics' in r]
        output.append({'study': study, 'method': method, 'group': group, 'seed': seed, 'sigma': sigma, 'level': level,
                       'worlds': n, 'independent_structure_count': len({r['structure_id'] for r in rows}),
                       'metrics': metrics, 'prediction_errors_conditional_on_available': errors,
                       'null_worlds': null_report, 'budget_curve': curves,
                       'final_measurements': distribution([r['measurements'] for r in resources]),
                       'first_recovery_nonnull_among_recovered': distribution(first),
                       'first_recovery_observation_schedules': dict(schedules),
                       'statuses': dict(Counter(m['behavior']['status'] for m in members)),
                       'error_categories': dict(Counter(r.get('grader_diagnostics', {}).get('category', 'unavailable_in_legacy_output') for r in rows)),
                       'tool_pruning_helped': {**fraction(sum(pruning), len(pruning)), 'unavailable': n-len(pruning)},
                       'behavior_counts': {name: sum(m['behavior'][name] for m in members if m['behavior'][name] is not None)
                                           if any(m['behavior'][name] is not None for m in members) else None
                                           for name in ('revealed_batches', 'exact_example_batches', 'example_query_and_alternative_batches', 'competitive_batches')},
                       'resources': {'measurements': sum(r['measurements'] for r in resources),
                                     'model_calls': sum(r['model_calls'] for r in resources),
                                     'input_tokens': sum(r['input_tokens'] for r in resources),
                                     'output_tokens': sum(r['output_tokens'] for r in resources),
                                     'known_generation_seconds': sum(time_available) if time_available else None,
                                     'missing_generation_time_records': sum(r['missing_generation_time_records'] for r in resources)}})
    return output
