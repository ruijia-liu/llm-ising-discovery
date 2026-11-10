"""Validation-only calibration and frozen scoring for noisy discovery.

Tolerances are tied to weak-term signal sizes in validation worlds, not fitted
to test success rates. They stay FIXED across all measurement-noise levels.
No structure labels, tolerances, or evaluation results reach query selection.
"""
import hashlib
import json
import math

import numpy as np

from .physics import Grammar
from run_structure_discovery import (FAMILIES, compile_queries, evaluation_queries,
                                     make_world, random_stream)

SCHEMA = 'ising-noise-calibration-v1'


def calibration_digest(calibration):
    payload = {key: value for key, value in calibration.items() if key != 'sha256'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def calibrate(*, seed=91001, worlds_per_family=10, evaluation_count=512):
    """Half a conservative weak-term signal, fixed before test runs.

    For each non-null validation world, measure the standalone RMS contribution
    of its weakest term. Take half the fifth percentile across worlds, separately
    for uniform and biased configurations. Coefficient tolerance is half the
    smallest validation coefficient magnitude. These are engineering accuracy
    targets, not confidence intervals or guaranteed false-discovery bounds.
    """
    if seed < 0 or worlds_per_family < 1 or evaluation_count < 16:
        raise ValueError('Invalid validation seed, world count, or evaluation count.')
    grammar = Grammar()
    rows = []
    index = 0
    for family in FAMILIES:
        for _ in range(worlds_per_family):
            truth = make_world(grammar, family, np.random.default_rng(random_stream(seed, index, 11)))
            queries = evaluation_queries(np.random.default_rng(random_stream(seed, index, 303)), evaluation_count)
            features = [compile_queries(grammar, group) for group in queries]
            terms = []
            for term in truth.terms:
                column = grammar.index[term.points]
                terms.append({'coefficient_abs': abs(term.coefficient),
                              'uniform_rms': float(np.sqrt(np.mean((features[0][:, column]*term.coefficient)**2))),
                              'biased_rms': float(np.sqrt(np.mean((features[1][:, column]*term.coefficient)**2)))})
            rows.append({'world_id': f'validation-{index:04d}', 'family': family,
                         'truth': truth.to_dict(), 'term_signals': terms})
            index += 1
    nonnull = [row for row in rows if row['term_signals']]
    thresholds = {}
    for distribution in ('uniform', 'biased'):
        weakest = [min(t[distribution+'_rms'] for t in row['term_signals']) for row in nonnull]
        thresholds[distribution+'_rmse'] = float(0.5*np.quantile(weakest, 0.05, method='linear'))
    thresholds['coefficient_max_error'] = float(0.5*min(
        t['coefficient_abs'] for row in nonnull for t in row['term_signals']))
    calibration = {'schema': SCHEMA, 'seed': seed, 'worlds_per_family': worlds_per_family,
                   'world_count': len(rows), 'evaluation_count_per_distribution': evaluation_count,
                   'rule': 'half_validation_weakest_term_fifth_percentile',
                   'window': 3, 'maximum_order': 4, 'maximum_terms': 2, 'grammar_size': len(grammar.structures),
                   'thresholds': thresholds, 'validation_worlds': rows,
                   'thresholds_depend_on_noise_sigma': False}
    calibration['sha256'] = calibration_digest(calibration)
    return calibration


def validate_calibration(calibration, test_seed):
    if calibration.get('schema') != SCHEMA or calibration.get('sha256') != calibration_digest(calibration):
        raise ValueError('Calibration is invalid or was modified after it was frozen.')
    if test_seed == calibration['seed']:
        raise ValueError('Validation and test worlds must use separate seeds.')
    if (calibration.get('window'), calibration.get('maximum_order'), calibration.get('maximum_terms'),
        calibration.get('grammar_size'), calibration.get('thresholds_depend_on_noise_sigma')) != (3, 4, 2, 158, False):
        raise ValueError('Calibration does not match the fixed experiment.')
    if any(not math.isfinite(calibration['thresholds'][key]) or calibration['thresholds'][key] <= 0
           for key in ('uniform_rmse', 'biased_rmse', 'coefficient_max_error')):
        raise ValueError('Calibration tolerances must be finite and positive.')


def evaluate_noisy(grammar, truth, hypothesis, evaluation_features, thresholds):
    error = grammar.coefficients(hypothesis)-grammar.coefficients(truth)
    uniform, biased = [float(np.sqrt(np.mean((x @ error)**2))) for x in evaluation_features]
    coefficient_error = float(np.max(np.abs(error)))
    support_success = truth.support == hypothesis.support
    prediction_success = uniform <= thresholds['uniform_rmse'] and biased <= thresholds['biased_rmse']
    return {'uniform_rmse': uniform, 'biased_rmse': biased,
            'coefficient_max_error': coefficient_error,
            'prediction_success': bool(prediction_success), 'support_success': bool(support_success),
            'full_recovery': bool(prediction_success and support_success
                                  and coefficient_error <= thresholds['coefficient_max_error']),
            'false_discovery': bool(not truth.terms and hypothesis.terms)}
