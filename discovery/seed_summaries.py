"""Equal-seed descriptive summaries; never promote repetitions to new mechanisms."""
from collections import Counter, defaultdict

import numpy as np


SEEDS = (401, 402, 403)


def average_fractions(values):
    rates = [values[s]['rate'] for s in SEEDS]
    return {'equal_seed_mean_rate': float(np.mean(rates)) if all(r is not None for r in rates) else None,
            'available_seed_rates': sum(r is not None for r in rates),
            'per_seed': {str(s): values[s] for s in SEEDS}}


def average_distributions(values):
    means = [values[s]['mean'] for s in SEEDS]
    return {'equal_seed_mean_of_available_answer_means': float(np.mean(means)) if all(m is not None for m in means) else None,
            'available_seed_means': sum(m is not None for m in means),
            'per_seed': {str(s): values[s] for s in SEEDS},
            'note': 'Each seed conditions on its own available answers; no pooled or averaged quantile is claimed.'}


def aggregate_seeds(items, cells):
    identities = defaultdict(dict)
    for item in items:
        row = item['row']
        if not row['method'].startswith('llm_'):
            continue
        key = (row.get('study', 'mainline'), row['method'], row['group'], row['sigma'], row['level'])
        seed = row['seed']
        if seed not in SEEDS:
            raise ValueError('Unexpected training seed in the registered descriptive summary.')
        identity = identities[key].setdefault(seed, {})
        if row['world_id'] in identity:
            raise ValueError('Duplicate world in a seed summary.')
        identity[row['world_id']] = row['structure_id']
    grouped = defaultdict(dict)
    for cell in cells:
        if not cell['method'].startswith('llm_'):
            continue
        key = (cell['study'], cell['method'], cell['group'], cell['sigma'], cell['level'])
        if cell['seed'] in grouped[key]:
            raise ValueError('Duplicate per-seed descriptive cell.')
        grouped[key][cell['seed']] = cell
    output = []
    for key, by_seed in sorted(grouped.items(), key=lambda p: tuple(str(v) for v in p[0])):
        study, method, group, sigma, level = key
        grids = identities[key]
        if set(grids) != set(by_seed):
            raise ValueError('Summary seeds differ from the original result rows.')
        first_grid = next(iter(grids.values()))
        if any(grid != first_grid for grid in grids.values()):
            raise ValueError('Seeds do not share the same world and mechanism grid.')
        missing = sorted(set(SEEDS)-set(by_seed))
        common = {'study': study, 'method': method, 'group': group, 'sigma': sigma, 'level': level,
                  'present_seeds': sorted(by_seed), 'missing_seeds': missing,
                  'unique_parameter_worlds': len(first_grid),
                  'independent_structure_count': len(set(first_grid.values())),
                  'seed_world_cells': len(first_grid)*len(by_seed),
                  'weighting': 'Equal means over the three registered seeds; within a stratum, descriptive world rates are unweighted.',
                  'not_the_structure_weighted_confirmatory_estimand': True}
        if missing:
            output.append({**common, 'status': 'incomplete_seeds_no_pooled_estimate'})
            continue
        if any(by_seed[s]['worlds'] != len(first_grid) for s in SEEDS):
            raise ValueError('Summary world count differs from the paired identity grid.')
        metrics = {name: average_fractions({s: by_seed[s]['metrics'][name] for s in SEEDS})
                   for name in by_seed[SEEDS[0]]['metrics']}
        errors = {name: average_distributions({s: by_seed[s]['prediction_errors_conditional_on_available'][name] for s in SEEDS})
                  for name in by_seed[SEEDS[0]]['prediction_errors_conditional_on_available']}
        null_names = ('raw_false_proposal_all_nulls', 'fitted_false_discovery_all_nulls',
                      'raw_false_proposal_given_valid', 'fitted_false_discovery_given_valid')
        nulls = {name: average_fractions({s: by_seed[s]['null_worlds'][name] for s in SEEDS}) for name in null_names}
        nulls['per_seed_world_and_validity_counts'] = {str(s): {k: by_seed[s]['null_worlds'][k]
            for k in ('worlds', 'valid_answers', 'invalid_answers')} for s in SEEDS}
        curves = []
        for index, budget in enumerate((8, 16, 24, 32)):
            snapshots = {s: by_seed[s]['budget_curve'][index] for s in SEEDS}
            if any(v['budget'] != budget for v in snapshots.values()):
                raise ValueError('Different measurement checkpoints were averaged.')
            curves.append({'budget': budget,
                **{name: average_fractions({s: snapshots[s][name] for s in SEEDS})
                   for name in ('full_recovery', 'raw_support_success', 'invalid_expression')},
                **{name: average_distributions({s: snapshots[s][name] for s in SEEDS})
                   for name in ('uniform_rmse_conditional_on_available', 'biased_rmse_conditional_on_available')}})
        totals = {}
        for category in ('statuses', 'error_categories'):
            counts = Counter()
            for cell in by_seed.values():
                counts.update(cell[category])
            totals[category] = dict(counts)
        resources = {name: sum(by_seed[s]['resources'][name] for s in SEEDS) for name in
                     ('measurements', 'model_calls', 'input_tokens', 'output_tokens', 'missing_generation_time_records')}
        times = [by_seed[s]['resources']['known_generation_seconds'] for s in SEEDS]
        resources['known_generation_seconds'] = sum(t for t in times if t is not None) if any(t is not None for t in times) else None
        output.append({**common, 'status': 'complete_three_seed_descriptive_summary',
                       'metrics': metrics, 'prediction_errors_conditional_on_available': errors,
                       'null_worlds': nulls, 'budget_curve': curves,
                       'tool_pruning_helped': average_fractions({s: by_seed[s]['tool_pruning_helped'] for s in SEEDS}),
                       'final_measurements': average_distributions({s: by_seed[s]['final_measurements'] for s in SEEDS}),
                       'first_recovery_nonnull_among_recovered': average_distributions({s: by_seed[s]['first_recovery_nonnull_among_recovered'] for s in SEEDS}),
                       'totals_over_seed_world_cells_not_independent_worlds': totals,
                       'actual_resource_totals_over_all_three_models': resources,
                       'behavior_by_seed': {str(s): by_seed[s]['behavior_counts'] for s in SEEDS},
                       'no_new_significance_test_or_interval': True})
    return output
