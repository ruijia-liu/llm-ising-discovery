from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from build_publication_curriculum import atomic_json, read, sha
from discovery.evaluation_results import load_completed_evaluation, verify_pairing
from discovery.evaluation_worlds import fixed_evidence, streams
from discovery.mechanism_scoring import score_answer
from discovery.physics import Expression
from evaluate_mainline import fixed_prediction


def make_fixture(root, group='B'):
    """Temporary synthetic artifact using a deliberately invalid scripted model."""
    root = Path(root)
    (root/'sources').mkdir(parents=True)
    source = root/'sources/fixture.txt'
    source.write_text('Synthetic integrity test; not an experimental student result.\n', encoding='utf-8')
    config = {'scope': 'development', 'method': 'llm_fixed', 'study': 'mainline', 'group': group, 'seed': 401,
              'worlds': ['fixture0', 'fixture1'], 'sigmas': [0, 1], 'sources': {'fixture.txt': sha(source)},
              'calibration_sha256': 'fixture-calibration', 'model_revision': 'fixture-only'}
    rows = []
    for index, world in enumerate(config['worlds']):
        for sigma in (0, 1):
            evidence, events = fixed_evidence(Expression().delta, sigma, streams('development', index))
            answers, curve = {}, []
            for budget in (8, 16, 24, 32):
                answers[str(budget)] = fixed_prediction(evidence, budget, lambda m, maximum: {
                    'text': 'invalid test fixture', 'input_tokens': 100, 'output_tokens': 3})
                curve.append({'budget': budget, **score_answer(Expression(), None, [], sigma, {})})
            row = {'world_id': world, 'level': 'parameter_development', 'sigma': sigma, 'method': 'llm_fixed',
                   'study': 'mainline', 'group': group, 'seed': 401,
                   'structure_id': hashlib.sha256(b'fixture-null').hexdigest(),
                   'score': {**curve[-1], 'budget_curve': curve}}
            rows.append(row)
            atomic_json(root/'episodes'/f'{world}-sigma{sigma}.json', {
                'result': row, 'episode': {'evidence': evidence, 'acquisition_events': events, 'answers': answers}})
    atomic_json(root/'config.json', config)
    atomic_json(root/'result.json', {'completed': True, 'scope': 'development', 'method': 'llm_fixed', 'rows': rows,
                'worlds': 2, 'episodes': 4, 'formal_tests_used': False, 'full_recoveries': 0})


class EvaluationResultTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        make_fixture(self.root/'B')

    def test_invalid_answers_remain_in_complete_grid_and_cost_accounting(self):
        loaded = load_completed_evaluation(self.root/'B', scope='development', study='mainline', group='B')
        self.assertEqual(len(loaded['result']['rows']), 4)
        self.assertEqual(sum(r['measurements'] for r in loaded['resources']), 128)
        self.assertEqual(sum(r['model_calls'] for r in loaded['resources']), 16)
        self.assertEqual(sum(r['output_tokens'] for r in loaded['resources']), 48)
        self.assertTrue(all(not r['score']['full_recovery'] for r in loaded['result']['rows']))

    def test_missing_duplicate_and_inconsistent_aggregate_results_are_rejected(self):
        path = self.root/'B/result.json'
        original = read(path)
        for modify in ('missing', 'duplicate', 'changed_score'):
            bad = deepcopy(original)
            if modify == 'missing':
                bad['rows'].pop()
            elif modify == 'duplicate':
                bad['rows'][-1] = bad['rows'][0]
            else:
                bad['rows'][0]['score']['uniform_rmse'] = 7
            atomic_json(path, bad)
            with self.assertRaises(ValueError):
                load_completed_evaluation(self.root/'B')

    def test_saved_source_or_measurement_history_changes_are_rejected(self):
        source = self.root/'B/sources/fixture.txt'
        original = source.read_bytes()
        source.write_text('changed', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'source snapshot'):
            load_completed_evaluation(self.root/'B')
        source.write_bytes(original)
        path = self.root/'B/episodes/fixture0-sigma0.json'
        artifact = read(path)
        artifact['episode']['acquisition_events'][1]['committed_at_used'] = 7
        atomic_json(path, artifact)
        with self.assertRaisesRegex(ValueError, 'history'):
            load_completed_evaluation(self.root/'B')

    def test_pairing_rejects_different_world_clusters_or_model_versions(self):
        make_fixture(self.root/'C', group='C')
        a, b = (load_completed_evaluation(self.root/g) for g in ('B', 'C'))
        verify_pairing([a, b])
        bad = deepcopy(b)
        bad['result']['rows'][0]['structure_id'] = 'different'
        with self.assertRaises(ValueError):
            verify_pairing([a, bad])
        bad = deepcopy(b)
        bad['config']['model_revision'] = 'another-model'
        with self.assertRaises(ValueError):
            verify_pairing([a, bad])

    def test_development_or_partial_formal_data_cannot_enter_primary_report(self):
        with self.assertRaises(ValueError):
            load_completed_evaluation(self.root/'B', scope='formal')
        config, result = read(self.root/'B/config.json'), read(self.root/'B/result.json')
        config['scope'] = result['scope'] = 'formal'
        result['formal_tests_used'] = True
        atomic_json(self.root/'B/config.json', config)
        atomic_json(self.root/'B/result.json', result)
        with self.assertRaisesRegex(ValueError, 'incomplete formal'):
            load_completed_evaluation(self.root/'B')


if __name__ == '__main__':
    unittest.main()
