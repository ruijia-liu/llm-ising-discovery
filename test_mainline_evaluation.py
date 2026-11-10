import unittest
from unittest.mock import patch
from types import SimpleNamespace

from discovery.evaluation_worlds import fixed_evidence,load_worlds,streams
from discovery.lab import Laboratory
from discovery.physics import Expression,Grammar,Term
from discovery.search import run_search
from evaluate_mainline import fixed_prediction, run
from discovery.evaluation_profiles import registered_student


class MainlineEvaluationTests(unittest.TestCase):
    def test_unregistered_cross_study_adapter_is_rejected(self):
        for study, group in (('mainline', 'C_no_predictions'), ('prediction_ablation', 'C'), ('unknown', 'C')):
            with self.assertRaises(ValueError):
                registered_student(study, group, 401)

    def test_unfinished_student_is_rejected_before_world_access(self):
        args = SimpleNamespace(method='llm_active', study='prediction_ablation', group='C_no_predictions', seed=401)
        with patch('evaluate_mainline.registered_student', side_effect=ValueError('no final adapter')):
            with patch('evaluate_mainline.load_worlds') as loader:
                with self.assertRaisesRegex(ValueError, 'no final adapter'):
                    run(args)
                loader.assert_not_called()

    def test_formal_answers_are_not_opened_before_training_guard(self):
        with patch('discovery.evaluation_worlds.require_completed_main_training',side_effect=ValueError('not ready')):
            with patch('discovery.evaluation_worlds.read') as reader:
                with self.assertRaises(ValueError):
                    load_worlds('formal')
                reader.assert_not_called()

    def test_random_and_active_initial_eight_are_identical_including_noise(self):
        truth=Expression((Term(((0,0),(0,1)),.4),))
        inputs=streams('development',2300)
        calls=[]
        def oracle(s,p):
            calls.append(1)
            return truth.delta(s,p)
        evidence,fixed_events=fixed_evidence(oracle,1,inputs)
        self.assertEqual(len(calls),32)
        lab=Laboratory(oracle,noise_sigma=1,noise_seed=inputs['noise_seed'])
        _,active_events=run_search(lab,Grammar(),inputs['initial'],strategy='active',seed=inputs['search_seed'])
        self.assertEqual(fixed_events[0],active_events[0])
        self.assertEqual(len(calls),64)
        self.assertEqual(len(evidence['measurements']),32)

    def test_initial_observation_return_cannot_mutate_private_audit_record(self):
        lab=Laboratory(Expression().delta)
        returned=lab.initial_observations(streams('development',2300)['initial'])
        expected=lab.events[0]['observed'][:]
        returned[0]=999
        returned.append(999)
        self.assertEqual(lab.events[0]['observed'],expected)
        self.assertEqual(lab.used,8)

    def test_fixed_model_only_sees_requested_prefix_and_invalid_is_not_null_success(self):
        truth=Expression()
        evidence,_=fixed_evidence(truth.delta,0,streams('development',2300))
        def bad(messages,limit):
            self.assertEqual(limit,128)
            self.assertIn('"used":8',messages[1]['content'])
            self.assertEqual(messages[1]['content'].count('original_position'),8)
            return {'text':'not a structure','input_tokens':100,'output_tokens':3}
        reply=fixed_prediction(evidence,8,bad)
        self.assertIsNone(reply['answer'])
        self.assertTrue(reply['error'])


if __name__=='__main__':
    unittest.main()
