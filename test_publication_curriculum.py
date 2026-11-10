"""Check larger structure splits, exact legacy anchors and audited record reuse."""
from collections import Counter
from copy import deepcopy
import unittest

from build_publication_curriculum import (audit_exposure, coverage_worlds, main_worlds,
                                         make_episode, read, seal_episode, verify_episode)
from build_research_curriculum import reserve_holdouts
from discovery.physics import Expression, Grammar, Term
from discovery.student_v2 import load_views
from discovery.teaching import expression_from_dict


class PublicationCurriculumTests(unittest.TestCase):
    def test_coverage_adds_parameters_without_new_supports(self):
        anchors = {v['episode_id'].split('-')[0]: expression_from_dict(v['final_answer']['expression'])
                   for v in load_views('train')}
        worlds = coverage_worlds()
        self.assertEqual(len(worlds), 96)
        self.assertEqual(Counter(w['anchor_id'] for w in worlds), Counter({key: 4 for key in anchors}))
        self.assertEqual(len({w['world_id'] for w in worlds}), 96)
        for world in worlds:
            model, anchor = expression_from_dict(world['truth']), anchors[world['anchor_id']]
            self.assertEqual(model.support, anchor.support)
            if model.terms:
                self.assertNotEqual(model, anchor)
        self.assertEqual(worlds, coverage_worlds())

    def test_formal_sizes_and_parameter_validation_cover_actual_training(self):
        worlds = main_worlds()
        train = [w for w in worlds if w['split'] == 'train']
        val = [w for w in worlds if w['split'] == 'validation']
        self.assertEqual((len(train), len(val)), (300, 60))
        self.assertEqual(set(Counter(w['family'] for w in train).values()), {50})
        supports = {expression_from_dict(w['truth']).support for w in train}
        self.assertTrue(all(expression_from_dict(w['truth']).support in supports for w in val))
        components = {s for support in supports for s in support}
        self.assertEqual(sum(len(s) == 3 for s in components), 48)
        self.assertTrue(all(len(s) <= 3 for s in components))
        train_answers = {str(w['truth']) for w in train if w['truth']['terms']}
        self.assertTrue(all(str(w['truth']) not in train_answers for w in val if w['truth']['terms']))

    def test_holdout_excludes_intermediate_candidates_and_has_fifty_distinct_combinations(self):
        worlds, shown = main_worlds(), set()
        for world in worlds:
            audit_exposure(world['truth'], shown)
        first = reserve_holdouts(worlds, shown, seed=97401, per_level=50)
        excluded = next(w['truth'] for w in first['worlds'] if w['level'] == 'combination')
        audit_exposure({'incorrect_candidate': excluded}, shown)
        reserved = reserve_holdouts(worlds, shown, seed=97401, per_level=50)
        self.assertEqual(Counter(w['level'] for w in reserved['worlds']),
                         Counter({'parameter': 50, 'combination': 50, 'four_body': 50, 'null': 50}))
        self.assertEqual(reserved['distinct_reserved_combinations'], 50)
        for world in reserved['worlds']:
            if world['level'] in ('combination', 'four_body'):
                self.assertNotIn(expression_from_dict(world['truth']).support, shown)

    def test_four_body_exposure_is_rejected_in_any_nested_field(self):
        model = Expression((Term(((0, 0), (0, 1), (1, 0), (1, 1)), 0.4),))
        with self.assertRaises(AssertionError):
            audit_exposure({'after': {'alternative': model.to_dict()}}, set())

    def test_resuming_cannot_silently_reuse_a_changed_audited_record(self):
        record = seal_episode({'episode_id': 'example', 'before': {'predicted_delta': 4.2},
                               'observed': 4.0, 'audit': {'rounds_replayed': 24}})
        self.assertEqual(verify_episode(record), record)
        changed = deepcopy(record)
        changed['before']['predicted_delta'] += 1
        with self.assertRaises(AssertionError):
            verify_episode(changed)

    def test_noisy_roundoff_tie_replays_actual_policy_and_predictions(self):
        # This exact public seed/world previously stopped on a sub-ULP difference
        # between summing uncertainty separately and incrementally in choose().
        world = main_worlds()[6]
        result = make_episode(world, 1.0, 97301,
                              read('results/structure_discovery_noise/calibration.json')['thresholds'])
        self.assertEqual(result['audit']['rounds_replayed'], 24)
        self.assertEqual(result['audit']['predictions_verified_by_total_energy'], 72)
        self.assertEqual(result['audit']['observations_replayed_by_total_energy'], 32)


if __name__ == '__main__':
    unittest.main()
