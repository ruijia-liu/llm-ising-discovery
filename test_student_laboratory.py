"""Exercise causality, budget charging, hypothesis fitting and public boundaries."""
from copy import deepcopy
import unittest

import numpy as np

from discovery.lab import Laboratory
from discovery.physics import Expression, Grammar, Term, random_queries
from discovery.student_laboratory import (StudentLaboratory, decode_student_query, fit_proposal,
                                          public_query)
from discovery.student_v2 import load_views
from discovery.student_v3 import evidence_only, fit_coefficients


class StudentLaboratoryTests(unittest.TestCase):
    def make_session(self, truth=None):
        truth = truth or Expression((Term(((0, 0), (0, 1)), 0.4),))
        self.calls = []
        def oracle(s, p):
            self.calls.append((s.copy(), p))
            return truth.delta(s, p)
        lab = Laboratory(oracle)
        initial = random_queries(np.random.default_rng(2701), 8)
        return StudentLaboratory(lab, initial), lab

    def proposal(self, count=1):
        queries = []
        for q in random_queries(np.random.default_rng(2702), count):
            public = public_query(q)
            queries.append({'local_spins': public['local_spins'], 'position': public['original_position']})
        return {'hypotheses': [{'structures': [[[0, 0], [0, 1]]]}, {'structures': [[[0, 0]]]}],
                'queries': queries, 'reason': 'Compare two proposed interactions.',
                'update_condition': 'Refit using the new evidence and compare residuals.'}

    def test_preparation_has_no_oracle_calls_and_mutations_cannot_change_commitment(self):
        session, lab = self.make_session()
        proposal = self.proposal(4)
        record = session.prepare(proposal)
        self.assertEqual(len(self.calls), 8)
        self.assertEqual(lab.used, 8)
        expected_queries = deepcopy(record['queries'])
        proposal['queries'][0]['local_spins'][0][0] *= -1
        record['predictions'][0][0] += 999
        reply = session.reveal(record['ticket'])
        self.assertEqual((len(self.calls), lab.used, reply['used']), (12, 12, 12))
        self.assertEqual(lab.events[-1]['queries'], expected_queries)
        committed = session.events()[0]
        self.assertEqual(committed['predictions'], lab.events[-1]['predictions'])
        for expression, predictions in zip(lab.events[-1]['hypotheses'], committed['predictions']):
            from discovery.teaching import expression_from_dict, decode_query
            model = expression_from_dict(expression)
            np.testing.assert_allclose(predictions, [model.delta(*decode_query(q)) for q in expected_queries], atol=1e-12)

    def test_four_measurement_batches_and_repeat_ticket_do_not_bypass_budget(self):
        session, lab = self.make_session()
        for _ in range(6):
            pending = session.prepare(self.proposal(4))
            session.reveal(pending['ticket'])
            with self.assertRaises(ValueError):
                session.reveal(pending['ticket'])
        self.assertEqual(lab.used, 32)
        with self.assertRaises(ValueError):
            session.prepare(self.proposal())
        self.assertEqual(len(self.calls), 32)
        final = session.finish([[[0, 0], [0, 1]]])
        self.assertEqual(final['used'], 32)
        self.assertAlmostEqual(final['expression']['terms'][0]['coefficient'], 0.4)

    def test_partial_evidence_fitter_matches_fixed_32_helper_without_new_structures(self):
        for view in load_views('train'):
            evidence = evidence_only(view)
            structures = [[[0, 0], [0, 1]]]
            fixed, _ = fit_coefficients(evidence, structures)
            flexible, info = fit_proposal(evidence, structures)
            self.assertEqual(fixed.support, flexible.support)
            np.testing.assert_allclose([t.coefficient for t in fixed.terms],
                                       [t.coefficient for t in flexible.terms], atol=1e-12)
            self.assertEqual(info['added_structures'], 0)

    def test_public_replies_have_no_truth_and_finish_can_use_only_eight_observations(self):
        session, lab = self.make_session()
        public = session.evidence()
        self.assertEqual(set(public), {'measurement_noise_sigma', 'used', 'measurements'})
        public['measurements'][0]['delta'] = 999999
        self.assertNotEqual(public, session.evidence())
        final = session.finish([[[0, 0], [0, 1]]])
        self.assertEqual(final['used'], 8)
        self.assertEqual(lab.used, 8)
        with self.assertRaises(ValueError):
            session.prepare(self.proposal())

    def test_equivalent_alternatives_and_malformed_queries_rejected_before_measurement(self):
        session, _ = self.make_session()
        bad = self.proposal()
        bad['hypotheses'][1]['structures'] = [[[2, 2], [2, 3]]]
        with self.assertRaises(ValueError):
            session.prepare(bad)
        bad = self.proposal()
        bad['queries'][0]['local_spins'][0][0] = True
        with self.assertRaises(ValueError):
            session.prepare(bad)
        self.assertEqual(len(self.calls), 8)

    def test_public_crop_preserves_all_158_flip_features_even_at_boundary(self):
        grammar = Grammar()
        for state, position in random_queries(np.random.default_rng(2703), 24):
            public = public_query((state, position))
            restored, other_position = decode_student_query({'local_spins': public['local_spins'], 'position': list(position)})
            np.testing.assert_array_equal(grammar.features([state], [position]), grammar.features([restored], [other_position]))


if __name__ == '__main__':
    unittest.main()
