from copy import deepcopy
import unittest

from analyze_prediction_ablation import ablation_comparison


def fixture(full=True, ablated=False):
    return [{'study': study, 'group': group, 'seed': seed, 'level': level, 'sigma': 1,
             'world_id': f'{level}-{index}', 'structure_id': f'{level}-structure',
             'score': {'tested_idea_recovery': success}}
            for study, group, success in (('mainline', 'C', full), ('prediction_ablation', 'C_no_predictions', ablated))
            for seed in (401, 402, 403) for level in ('combination', 'four_body') for index in (0, 1)]


OPTIONS = {'repetitions': 50, 'expected_counts': {'combination': (2, 1), 'four_body': (2, 1)}}


class AblationAnalysisTests(unittest.TestCase):
    def test_contrast_direction_and_labels_are_explicit(self):
        positive = ablation_comparison(fixture(), **OPTIONS)
        negative = ablation_comparison(fixture(False, True), **OPTIONS)
        self.assertEqual(positive['full_C_minus_C_no_predictions'], 1.)
        self.assertEqual(negative['full_C_minus_C_no_predictions'], -1.)
        self.assertEqual(positive['conditional_95_percentile_interval'], [1., 1.])
        self.assertNotIn('C_minus_B', positive)
        self.assertNotIn('consistent_process_advantage', positive)
        self.assertEqual(positive['world_counts'], {'combination': 2, 'four_body': 2})

    def test_other_groups_duplicate_or_missing_pairs_cannot_enter(self):
        original = fixture()
        wrong = deepcopy(original); wrong[0]['group'] = 'B'
        for rows in (wrong, original[:-1], original+[original[0]]):
            with self.assertRaises(ValueError):
                ablation_comparison(rows, **OPTIONS)

    def test_all_zero_fixture_does_not_claim_an_advantage(self):
        result = ablation_comparison(fixture(False, False), **OPTIONS)
        self.assertFalse(result['positive_interval_and_all_seed_directions'])
        self.assertEqual(result['conditional_95_percentile_interval'], [0., 0.])
        self.assertIn('not_an_additional_confirmatory_test', result['analysis_role'])


if __name__ == '__main__':
    unittest.main()
