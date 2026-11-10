"""Bounded-outcome uncertainty and influence summaries; no new primary test."""
import math

import numpy as np

from .paired_statistics import primary_comparison


LEVELS = ('combination', 'four_body')
SEEDS = (401, 402, 403)


def bounded_interval(point, weights, alpha=.05):
    weights = np.asarray(weights, dtype=float)
    points = np.asarray(point, dtype=float)
    if (not math.isfinite(alpha) or not 0 < alpha < 1 or weights.ndim != 1 or not len(weights)
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or not math.isclose(float(weights.sum()), 1., abs_tol=1e-12, rel_tol=0)
            or not np.isfinite(points).all() or np.any(np.abs(points) > 1)):
        raise ValueError('Finite points in [-1,1], normalized nonnegative weights and alpha in (0,1) required.')
    radius = math.sqrt(2*math.log(2/alpha)*float(np.square(weights).sum()))
    return {'alpha': alpha, 'half_width_before_clipping': radius,
            'interval': np.stack((np.maximum(-1, points-radius), np.minimum(1, points+radius)), axis=-1),
            'sum_squared_weights': float(np.square(weights).sum())}


def summarize_matrices(levels):
    """Each stratum contains arrays (replicates, three fixed seeds, worlds)."""
    if len(levels) != 2 or any(len(values) < 2 for values in levels):
        raise ValueError('Two strata with at least two structures each are required for influence summaries.')
    repetitions = levels[0][0].shape[0]
    if repetitions < 1:
        raise ValueError('At least one complete replicate is required.')
    for group in levels:
        for values in group:
            if (values.ndim != 3 or values.shape[:2] != (repetitions, 3) or values.shape[2] < 1
                    or not np.isfinite(values).all() or not np.isin(values, (-1, 0, 1)).all()):
                raise ValueError('Paired binary differences must have shape (replicates,3,worlds) and values -1/0/1.')
    means = [np.stack([v.mean(axis=2) for v in group], axis=1) for group in levels]
    # Match primary_comparison's sequence of averaging operations.
    seed_effects = np.mean([v.mean(axis=1) for v in means], axis=0)
    point = seed_effects.mean(axis=1)
    structure_means = [v.mean(axis=2) for v in means]
    structure_weights = np.concatenate([np.full(len(group), .5/len(group)) for group in levels])
    world_weights = np.concatenate([np.full(v.shape[2], .5/(len(group)*v.shape[2]))
                                    for group in levels for v in group])
    removed = []
    for k, group in enumerate(means):
        for index in range(group.shape[1]):
            remaining = np.delete(group, index, axis=1).mean(axis=1)
            removed.append(((remaining+means[1-k].mean(axis=1))/2).mean(axis=1))
    leave_one_out = np.stack(removed, axis=1)
    return {'point': point, 'seed_effects': seed_effects,
            'structure_means': np.concatenate(structure_means, axis=1),
            'structure_seed_effects': np.concatenate(means, axis=1),
            'structure_weights': structure_weights, 'world_weights': world_weights,
            'conditional_world_bound': bounded_interval(point, world_weights),
            'independent_structure_assumption_bound': bounded_interval(point, structure_weights),
            'leave_one_structure_out': leave_one_out,
            'leave_one_out_minimum': leave_one_out.min(axis=1), 'leave_one_out_maximum': leave_one_out.max(axis=1)}


def supplement_comparison(rows, expected_counts=None):
    # Reuse the registered completeness validator and point estimate. The single
    # bootstrap draw here is discarded; it is NOT an inferential interval.
    reference = primary_comparison(rows, repetitions=1, expected_counts=expected_counts)
    selected = [r for r in rows if r['group'] in ('B', 'C') and r['mode'] == 'active'
                and r['sigma'] == 1 and r['level'] in LEVELS]
    cells = {(r['group'], r['seed'], r['world_id']): int(r['score']) for r in selected}
    descriptions = {r['world_id']: (r['level'], r['structure_id']) for r in selected}
    levels, metadata = [], []
    for level in LEVELS:
        current = []
        for structure in sorted({s for k, s in descriptions.values() if k == level}):
            worlds = sorted(w for w, description in descriptions.items() if description == (level, structure))
            values = np.asarray([[cells['C', s, w]-cells['B', s, w] for w in worlds] for s in SEEDS])
            current.append(values[None, :, :])
            metadata.append({'level': level, 'structure_id': structure, 'world_ids': worlds, 'worlds': len(worlds)})
        levels.append(current)
    result = summarize_matrices(levels)
    if (abs(float(result['point'][0])-reference['C_minus_B']) > 1e-14
            or not np.allclose(result['seed_effects'][0], [reference['seed_effects'][str(s)] for s in SEEDS], atol=1e-14, rtol=0)):
        raise ValueError('Supplement changed the registered point estimate or seed pairing.')
    for index, row in enumerate(metadata):
        row.update({'seed_effects': dict(zip(map(str, SEEDS), map(float, result['structure_seed_effects'][0, index]))),
                    'mean_difference': float(result['structure_means'][0, index]),
                    'original_weight': float(result['structure_weights'][index]),
                    'weighted_contribution': float(result['structure_means'][0, index]*result['structure_weights'][index]),
                    'point_after_removing_structure': float(result['leave_one_structure_out'][0, index]),
                    'point_shift_after_removing_structure': float(result['leave_one_structure_out'][0, index]-result['point'][0])})
    bounds = {}
    for name in ('conditional_world_bound', 'independent_structure_assumption_bound'):
        bounds[name] = {**result[name], 'interval': result[name]['interval'][0].tolist()}
    return {'analysis_role': 'supplementary_uncertainty_not_a_new_confirmatory_test',
            'C_minus_B': float(result['point'][0]), 'seed_effects': reference['seed_effects'], **bounds,
            'structures': metadata, 'world_weights': result['world_weights'].tolist(),
            'leave_one_structure_out_range': [float(result['leave_one_out_minimum'][0]), float(result['leave_one_out_maximum'][0])],
            'world_bound_target': 'Expected repeat-world performance conditional on selected mechanisms and the three trained seeds; requires independent complete worlds.',
            'structure_bound_target': 'Expected weighted structure-unit performance under independent structure units; this assumption is not established by the formal design.',
            'leave_one_out_is_confidence_interval': False, 'future_training_seed_population_covered': False,
            'primary_rule_replaced': False}
