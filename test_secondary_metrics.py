from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from build_publication_curriculum import atomic_json, read
from discovery.secondary_metrics import NULL_STRUCTURE_ID, behavior, generation_time, summarize
from report_evaluation_metrics import run
from test_evaluation_results import make_fixture


def item(world, valid=True, null=True, recovered=False, false_proposal=False, diagnostic=True):
    score = {'valid_expression': valid, 'full_recovery': recovered, 'prediction_success': recovered,
             'raw_support_success': recovered and not false_proposal, 'support_success': recovered,
             'raw_false_proposal': false_proposal if valid else None,
             'false_discovery': false_proposal and not recovered if valid else None,
             'uniform_rmse': .4 if valid else None, 'biased_rmse': .6 if valid else None,
             'coefficient_max_error': .2 if valid else None}
    curve = [{'budget': b, **score} for b in (8, 16, 24, 32)]
    row = {'world_id': world, 'structure_id': NULL_STRUCTURE_ID if null else world,
           'study': 'mainline', 'method': 'llm_fixed', 'group': 'B', 'seed': 401, 'sigma': 1,
           'level': 'parameter_development', 'score': {**score, 'budget_curve': curve}}
    if diagnostic:
        row['grader_diagnostics'] = {'is_null_world': null,
            'first_independent_recovery_used': 8 if recovered else None,
            'first_recovery_observation_schedule': 'only_prespecified_8_16_24_32_prefixes',
            'category': 'recovered' if recovered else 'invalid_or_wrong',
            'tool_pruning_helped_full_recovery': false_proposal and recovered}
    return {'row': row, 'resources': {'measurements': 32, 'model_calls': 4, 'input_tokens': 400,
             'output_tokens': 12, 'known_generation_seconds': None, 'missing_generation_time_records': 4},
            'behavior': behavior({}, 'llm_fixed')}


class SecondaryMetricTests(unittest.TestCase):
    def test_invalid_null_answer_is_neither_correct_nor_an_observed_negative(self):
        data = [item('correct', recovered=True), item('wrong', false_proposal=True),
                item('invalid', valid=False)]
        cell = summarize(data)[0]
        self.assertEqual(cell['metrics']['full_recovery']['rate'], 1/3)
        null = cell['null_worlds']
        self.assertEqual(null['invalid_answers'], 1)
        self.assertEqual(null['raw_false_proposal_all_nulls']['rate'], 1/3)
        self.assertEqual(null['raw_false_proposal_given_valid']['rate'], 1/2)
        error = cell['prediction_errors_conditional_on_available']['uniform_rmse']
        self.assertEqual((error['available'], error['unavailable'], error['mean']), (2, 1, .4))
        self.assertIsNone(cell['resources']['known_generation_seconds'])
        self.assertEqual(cell['resources']['missing_generation_time_records'], 12)

    def test_legacy_diagnostics_and_unrecovered_queries_remain_unavailable(self):
        data = [item('legacy', diagnostic=False), item('new', recovered=True, false_proposal=True),
                item('unrecovered', null=False)]
        cell = summarize(data)[0]
        self.assertEqual(cell['null_worlds']['worlds'], 2)
        self.assertEqual(cell['tool_pruning_helped'], {'count': 1, 'denominator': 2, 'rate': .5, 'unavailable': 1})
        first = cell['first_recovery_nonnull_among_recovered']
        self.assertEqual((first['available'], first['unavailable']), (0, 1))
        self.assertIsNone(first['mean'])
        self.assertEqual(cell['error_categories']['unavailable_in_legacy_output'], 1)

    def test_report_does_not_duplicate_worlds_or_mistake_seeds_for_new_mechanisms(self):
        a = item('same-world', null=False, recovered=True)
        with self.assertRaisesRegex(ValueError, 'Duplicate world'):
            summarize([a, deepcopy(a)])
        b = deepcopy(a); b['row']['seed'] = 402
        cells = summarize([a, b])
        self.assertEqual(len(cells), 2)
        self.assertTrue(all(c['worlds'] == c['independent_structure_count'] == 1 for c in cells))

    def test_absent_partial_and_corrupt_generation_times_are_distinct(self):
        self.assertEqual(generation_time([])['known_generation_seconds'], 0)
        self.assertIsNone(generation_time([{}])['known_generation_seconds'])
        self.assertEqual(generation_time([{'generation_seconds': 3.}, {}]),
                         {'known_generation_seconds': 3., 'missing_generation_time_records': 1})
        for value in (-1, float('nan'), float('inf'), '3', True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                generation_time([{'generation_seconds': value}])

    def test_complete_artifact_report_keeps_invalid_rows_and_rejects_duplicate_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture(root/'B')
            run([root/'B'], root/'report', 'development')
            result = read(root/'report/result.json')
            self.assertEqual(len(result['cells']), 2)
            for cell in result['cells']:
                self.assertEqual(cell['metrics']['invalid_final_expression']['rate'], 1)
                self.assertEqual(cell['resources']['output_tokens'], 24)
                self.assertEqual(cell['resources']['missing_generation_time_records'], 8)
                self.assertEqual(cell['tool_pruning_helped']['unavailable'], 2)
            with self.assertRaisesRegex(ValueError, 'Duplicate method'):
                run([root/'B', root/'B'], root/'duplicate', 'development')

    def test_report_integrates_three_paired_seed_artifacts_without_pseudo_replication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = []
            for seed in (401, 402, 403):
                directory = root/f'B_seed{seed}'
                make_fixture(directory)
                config, result = read(directory/'config.json'), read(directory/'result.json')
                config['seed'] = seed
                for row in result['rows']:
                    row['seed'] = seed
                    path = directory/'episodes'/(row['world_id']+f"-sigma{row['sigma']}.json")
                    artifact = read(path); artifact['result'] = row
                    atomic_json(path, artifact)
                atomic_json(directory/'config.json', config)
                atomic_json(directory/'result.json', result)
                directories.append(directory)
            run(directories, root/'report', 'development')
            report = read(root/'report/result.json')
            self.assertEqual(len(report['cells']), 6)
            self.assertEqual(len(report['seed_summaries']), 2)
            for cell in report['seed_summaries']:
                self.assertEqual(cell['unique_parameter_worlds'], 2)
                self.assertEqual(cell['seed_world_cells'], 6)
                self.assertEqual(cell['metrics']['invalid_final_expression']['equal_seed_mean_rate'], 1.)


if __name__ == '__main__':
    unittest.main()
