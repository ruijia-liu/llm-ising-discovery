"""Physical identities, query isolation, budget enforcement, and recovery checks."""
import unittest

import numpy as np

from discovery.lab import BudgetExceeded, Laboratory
from discovery.physics import Expression, Grammar, Term, canonical_structure, random_queries
from discovery.search import SparseSearch, run_search


class GeneralizedPhysicsChecks(unittest.TestCase):
    def setUp(self):
        self.grammar = Grammar()
        self.rng = np.random.default_rng(321)

    def test_full_energy_differences_match_every_feature(self):
        # Independent total-energy oracle; cover every grammar shape at a torus edge.
        state, _ = random_queries(self.rng, 1)[0]
        for position in ((0, 15), (7, 8)):
            changed = state.copy()
            changed[position] *= -1
            baseline = Expression().energy(changed)-Expression().energy(state)
            features = self.grammar.features([state], [position])[0]
            for j, points in enumerate(self.grammar.structures):
                model = Expression((Term(points, 0.37),))
                actual = model.energy(changed)-model.energy(state)
                self.assertAlmostEqual(actual, model.delta(state, position), places=10)
                self.assertAlmostEqual(actual, baseline+0.37*features[j], places=10)

    def test_canonicalization_reduces_torus_duplicates_but_keeps_direction(self):
        pair = canonical_structure(((0, 0), (0, 1)))
        self.assertEqual(pair, canonical_structure(((15, 15), (15, 16))))
        self.assertEqual(pair, canonical_structure(((0, 0), (0, 1), (3, 4), (19, 20))))
        self.assertNotEqual(pair, canonical_structure(((0, 0), (1, 0))))
        self.assertEqual(Expression((Term(pair, 0.5), Term(((4, 5), (4, 6)), -0.5))).terms, ())
        self.assertEqual(Expression((Term(((0, 0), (16, 0)), 1),)).terms, ())
        self.assertEqual(len(self.grammar.structures), 158)

    def test_mixed_differences_and_global_parity(self):
        target, j, k, ell = (7, 7), (7, 8), (8, 7), (8, 8)
        state = np.ones((16, 16), dtype=np.int8)
        pair = Expression((Term(((0, 0), (0, 1)), 0.4),))
        triple = Expression((Term(((0, 0), (0, 1), (1, 0)), 0.4),))
        quad = Expression((Term(((0, 0), (0, 1), (1, 0), (1, 1)), 0.4),))

        def mixed(model, third):
            value = 0.0
            for sj, sk in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                config = state.copy()
                config[j], config[k], config[ell] = sj, sk, third
                value += sj*sk*model.delta(config, target)
            return value

        self.assertAlmostEqual(mixed(pair, 1), 0)
        self.assertAlmostEqual(mixed(triple, 1), 8*0.4)
        self.assertAlmostEqual(mixed(triple, 1)-mixed(triple, -1), 0)
        self.assertAlmostEqual(mixed(quad, 1)-mixed(quad, -1), 16*0.4)
        random_state, position = random_queries(self.rng, 1)[0]
        for model in (pair, triple, quad):
            extra = model.delta(random_state, position)-Expression().delta(random_state, position)
            inverted = model.delta(-random_state, position)-Expression().delta(-random_state, position)
            self.assertAlmostEqual(inverted, (-1)**len(model.terms[0].points)*extra)


class LaboratoryChecks(unittest.TestCase):
    def setUp(self):
        self.query = (np.ones((16, 16), dtype=np.int8), (0, 0))
        self.model = Expression((Term(((0, 0), (1, 1), (0, 1)), 0.3),))
        self.calls = 0

        def oracle(s, p):
            self.calls += 1
            return self.model.delta(s, p)

        self.lab = Laboratory(oracle)

    def test_predictions_locked_and_repeated_measurements_charged(self):
        self.lab.initial_observations([self.query]*8)
        prediction = self.model.delta(*self.query)
        state = self.query[0].copy()
        declared = [[prediction]*4]
        ticket = self.lab.commit([(state, (0, 0))]*4, [self.model], declared, 'contrast', 'refit')
        self.assertEqual(self.calls, 8)  # Committing does not query the oracle.
        state[:] = -1
        declared[0][0] = 999
        self.assertEqual(self.lab.reveal(ticket), [prediction]*4)
        self.assertEqual(self.lab.used, 12)
        self.assertEqual(self.lab.events[-1]['predictions'][0][0], prediction)
        with self.assertRaises(ValueError):
            self.lab.reveal(ticket)
        copied_events = self.lab.events
        copied_events[-1]['predictions'][0][0] = 999
        self.assertEqual(self.lab.events[-1]['predictions'][0][0], prediction)

    def test_invalid_and_overbudget_batches_do_not_consume_queries(self):
        with self.assertRaises(ValueError):
            self.lab.reveal(0)
        self.lab.initial_observations([self.query]*8)
        with self.assertRaises(ValueError):
            self.lab.commit([self.query], [self.model], [[12345]], 'reason', 'update')
        with self.assertRaises(BudgetExceeded):
            self.lab.commit([self.query]*25, [self.model], [[0]*25], 'reason', 'update')
        self.assertEqual(self.calls, 8)
        ticket = self.lab.commit([self.query]*24, [self.model],
                                 [[self.model.delta(*self.query)]*24], 'reason', 'update')
        self.lab.reveal(ticket)
        self.assertEqual(self.lab.used, 32)
        with self.assertRaises(BudgetExceeded):
            self.lab.commit([self.query], [self.model], [[0]], 'reason', 'update')
        self.assertEqual(self.calls, 32)

    def test_noise_is_reproducible_and_initial_batch_shared(self):
        a = Laboratory(self.model.delta, noise_sigma=0.2, noise_seed=92)
        b = Laboratory(self.model.delta, noise_sigma=0.2, noise_seed=92)
        first = a.initial_observations([self.query]*8)
        self.assertEqual(first, b.initial_observations([self.query]*8))
        self.assertGreater(np.std(first), 0)


class SearchChecks(unittest.TestCase):
    def test_sparse_recovery_and_null(self):
        grammar = Grammar()
        queries = random_queries(np.random.default_rng(48), 32)
        features = grammar.features([q[0] for q in queries], [q[1] for q in queries])
        truth = Expression((Term(((0, 0), (1, 1), (0, 1), (1, 0)), -0.43),
                            Term(((0, 0), (2, 0)), 0.62)))
        search = SparseSearch(grammar)
        residuals = features @ grammar.coefficients(truth)
        fitted = search.expression(search.fit(features, residuals)[0])
        self.assertEqual(fitted.support, truth.support)
        np.testing.assert_allclose(grammar.coefficients(fitted), grammar.coefficients(truth), atol=1e-10)
        self.assertEqual(search.expression(search.fit(features, np.zeros(32))[0]).terms, ())

    def test_end_to_end_uses_only_interface_and_locks_numerical_predictions(self):
        grammar = Grammar()
        truth = Expression((Term(((0, 0), (0, 1), (1, 0), (1, 1)), 0.55),))
        initial = random_queries(np.random.default_rng(53), 8)
        lab = Laboratory(truth.delta)
        snapshots, events = run_search(lab, grammar, initial, strategy='active', seed=1, pool_size=16)
        self.assertEqual(lab.used, 32)
        self.assertEqual(len(events), 25)
        self.assertEqual(snapshots[32].support, truth.support)
        for event in events[1:]:
            self.assertEqual(event['used'], event['committed_at_used']+len(event['queries']))
            self.assertEqual(len(event['predictions']), len(event['hypotheses']))


if __name__ == '__main__':
    unittest.main()
