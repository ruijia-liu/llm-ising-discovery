"""Grader-only decomposition, never added to observations or model prompts."""
import json

from .physics import Grammar
from .student_v3 import canonical_proposal
from .teaching import expression_from_dict


def classify_answer(truth, answer, score):
    if answer is None:
        return {'category': 'invalid_or_missing_expression', 'raw_term_count': None, 'fitted_term_count': None,
                'truth_term_count': len(truth.support), 'raw_orders': None, 'fitted_orders': None,
                'truth_orders': sorted(map(len, truth.support)), 'raw_count_error': None,
                'raw_order_error': None, 'raw_position_error': None, 'zero_terms_removed': None,
                'tool_pruning_helped_full_recovery': False, 'true_structure_removed_after_fit': False}
    raw = canonical_proposal(json.loads(json.dumps(answer['raw_structures'])), Grammar())
    fitted = expression_from_dict(answer['expression']).support
    target = truth.support
    orders, target_orders = sorted(map(len, raw)), sorted(map(len, target))
    count_error = len(raw) != len(target)
    order_error = orders != target_orders
    position_error = not order_error and raw != target
    helped = raw != target and fitted == target and score['full_recovery']
    lost = raw == target and fitted != target
    if score['full_recovery']:
        category = 'recovered_after_zero_term_removal' if helped else 'recovered'
    elif count_error:
        category = 'wrong_term_count'
    elif order_error:
        category = 'wrong_interaction_orders'
    elif position_error:
        category = 'wrong_positions_at_matching_orders'
    elif lost:
        category = 'correct_structure_removed_after_fit'
    else:
        category = 'coefficient_or_prediction_failure'
    return {'category': category, 'raw_term_count': len(raw), 'fitted_term_count': len(fitted),
            'truth_term_count': len(target), 'raw_orders': orders, 'fitted_orders': sorted(map(len, fitted)),
            'truth_orders': target_orders, 'raw_count_error': count_error, 'raw_order_error': order_error,
            'raw_position_error': position_error, 'zero_terms_removed': len(raw)-len(fitted),
            'tool_pruning_helped_full_recovery': bool(helped), 'true_structure_removed_after_fit': lost}


def episode_diagnostics(truth, episode, score, method, features, sigma, thresholds):
    if method == 'llm_active':
        answer = episode['final'] if episode['valid_final_answer'] else None
        first = score['first_independent_recovery_used']
        observation_schedule = 'each_actual_submitted_hypothesis'
    elif method.endswith('_fixed'):
        answer = episode['answers']['32']['answer']
        first = next((r['budget'] for r in score['budget_curve'] if r['full_recovery']), None)
        observation_schedule = 'only_prespecified_8_16_24_32_prefixes'
    else:
        from .mechanism_scoring import score_answer
        snapshots = {int(n): value for n, value in episode['snapshots'].items()}
        final = expression_from_dict(snapshots[32])
        answer = {'expression': final.to_dict(), 'raw_structures': final.support}
        first = None
        for used in sorted(snapshots):
            model = expression_from_dict(snapshots[used])
            candidate = {'expression': model.to_dict(), 'raw_structures': model.support}
            if score_answer(truth, candidate, features, sigma, thresholds)['full_recovery']:
                first = used
                break
        observation_schedule = 'each_search_update'
    return {**classify_answer(truth, answer, score), 'first_independent_recovery_used': first,
            'first_recovery_observation_schedule': observation_schedule,
            'is_null_world': not bool(truth.terms), 'grader_only_never_returned_to_student': True}
