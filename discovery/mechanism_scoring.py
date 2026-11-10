"""Grader-only independent prediction scoring. Never returned to the student."""
import json

import numpy as np

from .noise_evaluation import evaluate_noisy
from .physics import Grammar
from .student_v3 import canonical_proposal
from .teaching import expression_from_dict
from run_structure_discovery import compile_queries, evaluate, evaluation_queries


def independent_features(seed, world_index, count=256):
    grammar = Grammar()
    return [compile_queries(grammar, queries) for queries in
            evaluation_queries(np.random.default_rng([seed, world_index, 303]), count)]


def score_answer(truth, answer, features, sigma, thresholds):
    if answer is None:
        return {'valid_expression': False, 'raw_support_success': False, 'raw_false_proposal': None,
                'prediction_success': False, 'support_success': False, 'full_recovery': False,
                'false_discovery': None, 'uniform_rmse': None, 'biased_rmse': None,
                'coefficient_max_error': None, 'invalid_null_answer': not truth.terms}
    grammar = Grammar()
    fitted = expression_from_dict(answer['expression'])
    raw = canonical_proposal(json.loads(json.dumps(answer['raw_structures'])), grammar)
    score = (evaluate(grammar, truth, fitted, features) if sigma == 0 else
             evaluate_noisy(grammar, truth, fitted, features, thresholds))
    return {**score, 'valid_expression': True, 'raw_support_success': raw == truth.support,
            'raw_false_proposal': bool(not truth.terms and raw), 'invalid_null_answer': False,
            'raw_terms': len(raw), 'fitted_terms': len(fitted.support)}


def score_active(truth, episode, features, sigma, thresholds):
    final = episode['final'] if episode['valid_final_answer'] else None
    result = score_answer(truth, final, features, sigma, thresholds)
    snapshots = [{**{'used': row['used']}, **score_answer(truth, row, features, sigma, thresholds)} for row in episode['snapshots']]
    result['hypothesis_history_scores'] = snapshots
    result['first_independent_recovery_used'] = next((r['used'] for r in snapshots if r['full_recovery']), None)
    final_support = expression_from_dict(final['expression']).support if final else None
    revealed = {e['observation']['before_sha256'] for e in episode['events'] if e['kind'] == 'observations_revealed'}
    committed, competitive = False, False
    for event in episode['events']:
        if event['kind'] != 'predictions_committed' or event['commitment']['before_sha256'] not in revealed:
            continue
        matches = final_support is not None and any(expression_from_dict(f['expression']).support == final_support for f in event['commitment']['fits'])
        committed |= matches
        competitive |= matches and event['idea_diagnostic']['operational_competitive_test']
    result.update(final_support_committed_before_measurement=bool(committed),
                  final_support_competitively_tested=bool(competitive),
                  tested_idea_recovery=bool(truth.terms and result['full_recovery'] and committed),
                  competitive_idea_recovery=bool(truth.terms and result['full_recovery'] and competitive))
    curve = []
    for budget in (8,16,24,32):
        available = [s for s in episode['snapshots'] if s['used'] <= budget]
        answer = available[-1] if available else None
        if not episode['valid_final_answer'] and budget >= episode['used']:
            answer = None
        curve.append({'budget': budget, 'actual_measurements_used': min(budget, episode['used']),
                      **score_answer(truth, answer, features, sigma, thresholds)})
    result['budget_curve'] = curve
    return result
