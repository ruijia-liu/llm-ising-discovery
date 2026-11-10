"""Verify the boundary between structure proposal, arithmetic and grading."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np
from transformers import AutoTokenizer

from audit_student_validation_scope import classify
from discovery.physics import Expression, Grammar, Term
from discovery.student_v2 import encode, load_views
from discovery.student_v3 import (evidence_only, fit_coefficients, parse_structures,
                                  terminal_messages, terminal_prompt)
from discovery.teaching import expression_from_dict


class StudentV3Tests(unittest.TestCase):
    def test_parser_equivalence_and_invalid_outputs(self):
        grammar = Grammar()
        self.assertEqual(parse_structures('{"structures":[[[2,2],[2,3]],[[0,0],[0,1]]]}', grammar),
                         (((0, 0), (0, 1)),))
        for value in ['{"structures":[[[true,0]]]}', '{"structures":[[[0.0,0]]]}',
                      '{"structures":[[[0,0],[0,4]]]}', '{"structures":[[[0,0],[0,0]]]}',
                      '{"structures":[],"coefficient":1}', 'text {"structures":[]}',
                      '{"structures":[[[0,0]],[[0,1]],[[0,2]]]}']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_structures(value, grammar)

    def test_shared_prompts_and_structure_only_supervision(self):
        tokenizer = AutoTokenizer.from_pretrained('models/Qwen2.5-0.5B-Instruct', local_files_only=True)
        for b, c in zip(load_views('train', 'B'), load_views('train', 'C')):
            messages = terminal_messages(b)
            self.assertEqual(messages, terminal_messages(c))
            self.assertEqual(messages[:-1], terminal_prompt(evidence_only(b)))
            altered = deepcopy(b)
            altered['final_answer']['expression']['terms'] = []
            altered['episode_id'] = 'hidden-id'
            self.assertEqual(messages[:-1], terminal_messages(altered)[:-1])
            example = encode(tokenizer, messages)
            self.assertEqual(tokenizer.decode(example['labels'][example['labels'] != -100]),
                             messages[-1]['content']+'<|im_end|>')
            self.assertEqual(set(json.loads(messages[-1]['content'])), {'structures'})

    def test_all_true_supports_fit_using_visible_patches_without_candidate_features(self):
        for view in load_views('train')+load_views('validation'):
            truth = expression_from_dict(view['final_answer']['expression'])
            with patch.object(Grammar, 'features', side_effect=AssertionError('No candidate enumeration')):
                fitted, info = fit_coefficients(evidence_only(view), truth.support)
            self.assertEqual(fitted.support, truth.support)
            np.testing.assert_allclose([t.coefficient for t in fitted.terms],
                                       [t.coefficient for t in truth.terms], atol=1e-12, rtol=0)
            self.assertTrue(info['identifiable_given_proposal'])
            self.assertEqual(info['measurements_used'], 32)

    def test_wrong_proposals_never_gain_a_missing_structure(self):
        view = load_views('train')[12]
        wrong = (((0, 0), (0, 1)),)
        fitted, info = fit_coefficients(evidence_only(view), wrong)
        self.assertTrue(set(fitted.support).issubset(wrong))
        self.assertEqual(info['candidate_structures_added'], 0)
        self.assertGreater(info['observation_residual_rmse'], 0.01)
        empty, _ = fit_coefficients(evidence_only(view), ())
        self.assertEqual(empty, Expression())

    def test_zero_removal_preserves_record_of_false_proposal(self):
        null = load_views('train')[-1]
        fitted, info = fit_coefficients(evidence_only(null), (((0, 0),),))
        self.assertFalse(fitted.terms)
        self.assertEqual(info['zero_coefficient_structures_removed'], 1)
        self.assertEqual(len(info['proposed_structures']), 1)

    def test_rank_deficient_proposal_is_reported(self):
        evidence = deepcopy(evidence_only(load_views('train')[0]))
        evidence['measurements'] = [deepcopy(evidence['measurements'][0]) for _ in range(32)]
        _, info = fit_coefficients(evidence, (((0, 0),), ((0, 0), (0, 1))))
        self.assertLess(info['design_rank'], 2)
        self.assertFalse(info['identifiable_given_proposal'])

    def test_full_training_covers_parameter_validation(self):
        train, validation = load_views('train'), load_views('validation')
        cohorts = classify(train, validation, [v['episode_id'].split('-')[0] for v in train])
        self.assertEqual(set(cohorts.values()), {'seen_structure'})
        self.assertEqual(len(train), 24)
        self.assertEqual(len(validation), 12)


if __name__ == '__main__':
    unittest.main()
