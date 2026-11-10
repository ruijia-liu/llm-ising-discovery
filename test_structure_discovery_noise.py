"""Checks for the actual noise distribution and validation/test separation."""
from copy import deepcopy
import unittest

import numpy as np

from discovery.lab import Laboratory
from discovery.noise_evaluation import (calibrate, calibration_digest, evaluate_noisy,
                                        validate_calibration)
from discovery.physics import Expression, Grammar, Term, random_queries
from discovery.search import SparseSearch
from run_structure_discovery import compile_queries, evaluation_queries
from run_structure_discovery_noise import paired_interval


class NoisyDiscoveryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibration = calibrate(seed=101, worlds_per_family=1, evaluation_count=32)

    def test_calibration_is_frozen_and_disjoint(self):
        validate_calibration(self.calibration, 102)
        with self.assertRaises(ValueError):
            validate_calibration(self.calibration, 101)
        changed = deepcopy(self.calibration)
        changed['thresholds']['uniform_rmse'] *= 2
        with self.assertRaises(ValueError):
            validate_calibration(changed, 102)
        self.assertFalse(self.calibration['thresholds_depend_on_noise_sigma'])
        # Tolerances are determined by validation signals, without a noise input.
        worlds = [w for w in self.calibration['validation_worlds'] if w['term_signals']]
        weak = [min(t['uniform_rms'] for t in w['term_signals']) for w in worlds]
        self.assertAlmostEqual(self.calibration['thresholds']['uniform_rmse'], 0.5*np.quantile(weak, 0.05))

    def test_noise_scale_and_independence_and_common_draws(self):
        query = (np.ones((16, 16), dtype=np.int8), (0, 0))
        count = 4096
        oracle = Expression().delta
        low = Laboratory(oracle, noise_sigma=0.5, noise_seed=674,
                         initial_count=count, budget=count)
        high = Laboratory(oracle, noise_sigma=2.0, noise_seed=674,
                          initial_count=count, budget=count)
        errors = np.array(low.initial_observations([query]*count))-oracle(*query)
        high_errors = np.array(high.initial_observations([query]*count))-oracle(*query)
        np.testing.assert_allclose(high_errors, 4*errors, atol=1e-13)
        self.assertLess(abs(errors.mean()), 0.03)
        self.assertLess(abs(errors.std()-0.5), 0.03)
        self.assertLess(abs(np.corrcoef(errors[:-1], errors[1:])[0, 1]), 0.05)
        self.assertEqual(low.used, count)

    def test_wrong_structure_cannot_pass_on_prediction_tolerance_alone(self):
        grammar = Grammar()
        truth = Expression((Term(((0, 0), (1, 0)), 0.4),))
        wrong = Expression((Term(((0, 0), (0, 1)), 0.4),))
        groups = evaluation_queries(np.random.default_rng(29), 32)
        features = [compile_queries(grammar, group) for group in groups]
        loose = {key: 100.0 for key in self.calibration['thresholds']}
        scored = evaluate_noisy(grammar, truth, wrong, features, loose)
        self.assertTrue(scored['prediction_success'])
        self.assertFalse(scored['support_success'])
        self.assertFalse(scored['full_recovery'])
        exact = evaluate_noisy(grammar, truth, truth, features, self.calibration['thresholds'])
        self.assertTrue(exact['full_recovery'])
        self.assertEqual(exact['uniform_rmse'], 0)

    def test_known_noise_penalty_can_reject_small_spurious_signal(self):
        grammar = Grammar()
        queries = random_queries(np.random.default_rng(914), 32)
        features = compile_queries(grammar, queries)
        small_signal = 0.02*features[:, 0]
        clean = SparseSearch(grammar, noise_sigma=0)
        noisy = SparseSearch(grammar, noise_sigma=1)
        self.assertEqual(len(clean.expression(clean.fit(features, small_signal)[0]).terms), 1)
        self.assertEqual(noisy.expression(noisy.fit(features, small_signal)[0]).terms, ())

    def test_bootstrap_pairs_by_seed_and_world_and_ignores_other_noise_levels(self):
        rows = []
        for seed in (1, 2):
            for world in ('w0', 'w1'):
                for strategy in ('random', 'active'):
                    rows.append(dict(seed=seed, family='three_body', world_id=world, strategy=strategy,
                                     noise_sigma=1.0, budget=32, full_recovery=strategy == 'active'))
        rows.append(dict(seed=1, family='three_body', world_id='w0', strategy='random',
                         noise_sigma=2.0, budget=32, full_recovery=True))
        result = paired_interval(rows, 1.0, 'full_recovery')
        self.assertEqual(result['active_minus_random'], 1)
        self.assertEqual(result['stratified_world_bootstrap_95_interval'], [1, 1])


if __name__ == '__main__':
    unittest.main()
