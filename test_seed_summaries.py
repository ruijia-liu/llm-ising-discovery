from copy import deepcopy
import unittest

from discovery.secondary_metrics import summarize
from discovery.seed_summaries import aggregate_seeds
from test_secondary_metrics import item


def three_seeds():
    result = []
    for seed in (401, 402, 403):
        for index in (0, 1):
            row = item(f'w{index}', valid=not (seed == 401 and index == 1),
                       recovered=seed == 403, false_proposal=(index == 0 and seed != 403))
            row['row']['seed'] = seed
            score = row['row']['score']
            if score['valid_expression']:
                score['uniform_rmse'] = 10. if seed == 401 else 0.
            result.append(row)
    return result


class SeedSummaryTests(unittest.TestCase):
    def test_three_seeds_are_not_six_independent_worlds(self):
        rows = three_seeds()
        summary = aggregate_seeds(rows, summarize(rows))[0]
        self.assertEqual(summary['unique_parameter_worlds'], 2)
        self.assertEqual(summary['independent_structure_count'], 1)
        self.assertEqual(summary['seed_world_cells'], 6)
        self.assertAlmostEqual(summary['metrics']['full_recovery']['equal_seed_mean_rate'], 1/3)
        self.assertEqual(summary['actual_resource_totals_over_all_three_models']['model_calls'], 24)

    def test_conditioning_is_within_seed_and_missing_answers_are_not_zeros(self):
        rows = three_seeds()
        summary = aggregate_seeds(rows, summarize(rows))[0]
        # 1/1, 1/2, 0/2 averaged equally, rather than pooling into 2/5.
        self.assertEqual(summary['null_worlds']['raw_false_proposal_given_valid']['equal_seed_mean_rate'], .5)
        value = summary['prediction_errors_conditional_on_available']['uniform_rmse']
        self.assertAlmostEqual(value['equal_seed_mean_of_available_answer_means'], 10/3)
        self.assertEqual(value['per_seed']['401']['unavailable'], 1)
        self.assertNotIn('p90', value)

    def test_missing_seed_is_not_silently_dropped_from_a_three_seed_estimate(self):
        rows = [r for r in three_seeds() if r['row']['seed'] != 402]
        summary = aggregate_seeds(rows, summarize(rows))[0]
        self.assertEqual(summary['missing_seeds'], [402])
        self.assertNotIn('metrics', summary)
        self.assertEqual(summary['status'], 'incomplete_seeds_no_pooled_estimate')

    def test_one_all_invalid_seed_makes_the_conditional_average_unavailable(self):
        rows = three_seeds()
        for i, row in enumerate(rows):
            if row['row']['seed'] == 401:
                rows[i] = item(row['row']['world_id'], valid=False)
        summary = aggregate_seeds(rows, summarize(rows))[0]
        self.assertIsNone(summary['prediction_errors_conditional_on_available']['uniform_rmse']['equal_seed_mean_of_available_answer_means'])
        self.assertIsNone(summary['null_worlds']['raw_false_proposal_given_valid']['equal_seed_mean_rate'])
        self.assertEqual(summary['metrics']['invalid_final_expression']['equal_seed_mean_rate'], 1/3)

    def test_changed_worlds_clusters_or_duplicate_cells_are_rejected(self):
        for field in ('world_id', 'structure_id'):
            rows = three_seeds()
            rows[-1]['row'][field] = 'changed'
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'same world and mechanism grid'):
                aggregate_seeds(rows, summarize(rows))
        rows = three_seeds()
        cells = summarize(rows)
        with self.assertRaisesRegex(ValueError, 'Duplicate per-seed'):
            aggregate_seeds(rows, cells+[deepcopy(cells[0])])


if __name__ == '__main__':
    unittest.main()
