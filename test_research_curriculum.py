"""Leakage, numerical audit, paired-data, and causal-order checks."""
from copy import deepcopy
import unittest

import numpy as np

from build_research_curriculum import (audit_messages, make_teaching_worlds,
                                       reserve_holdouts, verify_observations, walk_expressions)
from discovery.lab import Laboratory
from discovery.physics import Expression, Term, random_queries
from itertools import combinations
from discovery.teaching import (audit_teacher, audit_views, export_views, messages_for_view,
                                run_teacher)


class TeachingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.truth = Expression((Term(((0, 0), (1, 0), (0, 1)), 0.46),))
        cls.initial = random_queries(np.random.default_rng(811), 8)
        lab = Laboratory(cls.truth.delta, noise_sigma=1.0, noise_seed=821)
        cls.trace = run_teacher(lab, cls.initial, query_seed=831, pool_size=16)

    def test_all_decisions_predictions_and_observations_can_be_replayed(self):
        audit = audit_teacher(self.trace, query_seed=831, pool_size=16)
        self.assertEqual(audit['rounds_replayed'], 24)
        self.assertEqual(audit['predictions_verified_by_total_energy'], 72)
        self.assertEqual(verify_observations(self.trace, self.truth, 821), 32)

    def test_fabricated_prediction_or_selection_is_rejected(self):
        for change in ('prediction', 'selection', 'update'):
            trace = deepcopy(self.trace)
            if change == 'prediction':
                trace['rounds'][0]['before']['hypotheses'][0]['predicted_delta'] += 1
            elif change == 'selection':
                trace['rounds'][0]['before']['selection']['utility'] += 1
            else:
                trace['rounds'][0]['after']['prediction_errors'][0] += 1
            with self.assertRaises(AssertionError):
                audit_teacher(trace, query_seed=831, pool_size=16)

    def test_b_c_share_evidence_and_labels_and_truth_is_attached_afterward(self):
        views = export_views(self.trace, self.truth, 'example')
        audit_views(views)
        messages = {g: messages_for_view(views[g], g) for g in 'ABC'}
        audit_messages(views, messages)
        self.assertEqual(len(messages['B']['messages']), 51)
        self.assertEqual(len(messages['C']['messages']), 51)
        self.assertEqual(len(messages['A']['messages']), 3)
        changed = deepcopy(views)
        changed['C']['experiments'][0]['observed'] += 1
        with self.assertRaises(AssertionError):
            audit_views(changed)
        # Changing a grader label cannot rewrite a past teacher record.
        alternate = export_views(self.trace, Expression(), 'example')
        self.assertEqual(views['C']['research_process'], alternate['C']['research_process'])
        self.assertNotEqual(views['C']['final_answer'], alternate['C']['final_answer'])

    def test_first_decision_does_not_depend_on_future_observation(self):
        counter = [0]

        def changed_future(state, position):
            counter[0] += 1
            return self.truth.delta(state, position)+(10 if counter[0] > 8 else 0)

        lab = Laboratory(changed_future, noise_sigma=1.0, noise_seed=821)
        altered = run_teacher(lab, self.initial, query_seed=831, pool_size=16)
        self.assertEqual(self.trace['initial'], altered['initial'])
        self.assertEqual(self.trace['rounds'][0]['before'], altered['rounds'][0]['before'])
        self.assertEqual(self.trace['rounds'][0]['query'], altered['rounds'][0]['query'])
        self.assertEqual(self.trace['rounds'][0]['before_sha256'], altered['rounds'][0]['before_sha256'])
        self.assertNotEqual(self.trace['rounds'][0]['observed'], altered['rounds'][0]['observed'])

    def test_no_four_body_teaching_and_combination_holdout_excludes_alternatives(self):
        worlds = make_teaching_worlds()
        self.assertEqual(sum(w['split'] == 'train' for w in worlds), 24)
        self.assertEqual(sum(w['split'] == 'validation' for w in worlds), 12)
        models = list(walk_expressions([w['truth'] for w in worlds]))
        self.assertTrue(all(len(p) <= 3 for m in models for p in m.support))
        shown = {m.support for m in models}
        first = reserve_holdouts(worlds, shown)
        candidate = next(w['truth'] for w in first['worlds'] if w['level'] == 'combination')
        # Treat it as an incorrect intermediate hypothesis in C, rather than an answer.
        excluded_support = next(walk_expressions(candidate)).support
        shown.add(excluded_support)
        second = reserve_holdouts(worlds, shown)
        for world in second['worlds']:
            if world['level'] in ('combination', 'four_body'):
                self.assertNotIn(next(walk_expressions(world['truth'])).support, shown)
        views = export_views(self.trace, self.truth, 'example')
        self.assertTrue(all(len(p) <= 3 for m in walk_expressions(views['C']) for p in m.support))

    def test_small_number_of_unseen_combinations_is_reported_honestly(self):
        worlds = make_teaching_worlds()
        models = list(walk_expressions([w['truth'] for w in worlds if w['split'] == 'train']))
        components = sorted({p for m in models for p in m.support})
        pairs = [tuple(sorted(pair)) for pair in combinations(components, 2)]
        shown = set(pairs[7:])
        reserved = reserve_holdouts(worlds, shown)
        self.assertEqual(reserved['unexposed_combinations_available'], 7)
        self.assertEqual(reserved['distinct_reserved_combinations'], 7)
        self.assertEqual(sum(w['level'] == 'combination' for w in reserved['worlds']), 12)


if __name__ == '__main__':
    unittest.main()
