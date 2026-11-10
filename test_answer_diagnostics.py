import unittest

from discovery.answer_diagnostics import classify_answer
from discovery.physics import Expression, Term


class AnswerDiagnosticTests(unittest.TestCase):
    def test_equal_order_wrong_positions_are_distinct_from_wrong_orders(self):
        truth = Expression((Term(((0, 0), (0, 1)), .4),))
        for points, expected in ((((0, 0), (1, 0)), 'wrong_positions_at_matching_orders'),
                                 (((0, 0), (0, 1), (1, 1)), 'wrong_interaction_orders')):
            model = Expression((Term(points, .4),))
            answer = {'raw_structures': model.support, 'expression': model.to_dict()}
            self.assertEqual(classify_answer(truth, answer, {'full_recovery': False})['category'], expected)

    def test_zero_removal_preserves_raw_false_proposal_diagnostic(self):
        truth = Expression()
        answer = {'raw_structures': [[[0, 0]]], 'expression': truth.to_dict()}
        row = classify_answer(truth, answer, {'full_recovery': True})
        self.assertEqual(row['category'], 'recovered_after_zero_term_removal')
        self.assertTrue(row['raw_count_error'])
        self.assertTrue(row['tool_pruning_helped_full_recovery'])
        self.assertEqual(row['zero_terms_removed'], 1)

    def test_invalid_answer_and_numerical_failure_are_not_structural_discoveries(self):
        truth = Expression((Term(((0, 0),), .4),))
        invalid = classify_answer(truth, None, {'full_recovery': False})
        self.assertEqual(invalid['category'], 'invalid_or_missing_expression')
        self.assertIsNone(invalid['raw_term_count'])
        model = Expression((Term(((0, 0),), .7),))
        answer = {'raw_structures': model.support, 'expression': model.to_dict()}
        self.assertEqual(classify_answer(truth, answer, {'full_recovery': False})['category'], 'coefficient_or_prediction_failure')
        answer['expression'] = Expression().to_dict()
        row = classify_answer(truth, answer, {'full_recovery': False})
        self.assertEqual(row['category'], 'correct_structure_removed_after_fit')
        self.assertTrue(row['true_structure_removed_after_fit'])


if __name__ == '__main__':
    unittest.main()
