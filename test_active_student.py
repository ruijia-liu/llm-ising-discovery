from copy import deepcopy
import json
import unittest

import numpy as np

from discovery.active_student import LIMITS, parse_hypothesis, run_active, strict_json
from discovery.lab import Laboratory
from discovery.physics import Expression, Term, random_queries
from discovery.student_laboratory import StudentLaboratory, public_query
from discovery.teaching import expression_from_dict, decode_query


class ScriptedModel:
    def __init__(self, replies):
        self.replies, self.messages = iter(replies), []

    def __call__(self, messages, maximum):
        self.messages.append(deepcopy(messages))
        text = next(self.replies)
        return {'text': text if isinstance(text, str) else json.dumps(text),
                'input_tokens': 100, 'output_tokens': min(maximum, 10), 'generation_seconds': .1}


class ActiveStudentTests(unittest.TestCase):
    def session(self):
        self.measured = []
        self.truth = Expression((Term(((0,0), (0,1)), .4),))
        def oracle(s, p):
            self.measured.append(len(self.measured))
            return self.truth.delta(s, p)
        self.lab = Laboratory(oracle)
        return StudentLaboratory(self.lab, random_queries(np.random.default_rng(100), 8))

    def plan(self, n=4):
        queries = []
        for q in random_queries(np.random.default_rng(101), n):
            v = public_query(q)
            queries.append({'local_spins': v['local_spins'], 'position': v['original_position']})
        return {'action': 'experiment', 'alternative_structures': [[[0,0]]], 'queries': queries,
                'reason': 'Compare these two proposed mechanisms.', 'update_condition': 'Revise if the predictions fail.'}

    def hypothesis(self):
        return {'structures': [[[0,0], [0,1]]]}

    def test_real_commits_precede_oracle_and_exact_predictions_replay(self):
        session = self.session()
        replies = []
        for _ in range(6):
            replies += [self.hypothesis(), self.plan()]
        replies += [self.hypothesis()]
        model = ScriptedModel(replies)
        committed = []
        def writer(event):
            if event['kind'] == 'predictions_committed':
                self.assertEqual(len(self.measured), event['commitment']['committed_at_used'])
                committed.append(event)
        result = run_active(session, model, on_event=writer)
        self.assertEqual(result['status'], 'budget_completed')
        self.assertEqual((result['used'], len(self.measured), len(committed)), (32,32,6))
        self.assertEqual([s['used'] for s in result['snapshots']], [8,12,16,20,24,28,32])
        for event in committed:
            c = event['commitment']
            for fit, predictions in zip(c['fits'], c['predictions']):
                expression = expression_from_dict(fit['expression'])
                for encoded, prediction in zip(c['queries'], predictions):
                    s, p = decode_query(encoded); changed = s.copy(); changed[p] *= -1
                    self.assertAlmostEqual(prediction, expression.energy(changed)-expression.energy(s))
        self.assertEqual(result['final']['expression'], session.events()[-1]['expression'])

    def test_invalid_and_duplicate_alternatives_do_not_measure_or_get_search_fallback(self):
        session = self.session()
        bad = self.plan(); bad['alternative_structures'] = self.hypothesis()['structures']
        model = ScriptedModel([self.hypothesis(), bad, bad])
        result = run_active(session, model)
        self.assertEqual(result['status'], 'invalid_reply')
        self.assertFalse(result['valid_final_answer'])
        self.assertIsNone(result['final'])
        self.assertEqual(len(self.measured), 8)
        self.assertEqual(result['inference']['repairs'], 1)
        self.assertTrue(result['last_valid_hypothesis_for_diagnostics_only'])

    def test_early_finish_does_not_spend_or_count_a_tested_idea(self):
        session = self.session()
        model = ScriptedModel([self.hypothesis(), {'action':'finish'}])
        result = run_active(session, model)
        self.assertEqual(result['status'], 'student_finished')
        self.assertEqual(len(self.measured), 8)
        self.assertFalse(any(e['kind']=='predictions_committed' for e in result['events']))

    def test_repairs_have_no_new_evidence_and_all_generation_is_charged(self):
        session = self.session()
        model = ScriptedModel(['bad json', self.hypothesis(), {'action':'finish'}])
        result = run_active(session, model)
        self.assertEqual(result['inference']['calls'], 3)
        self.assertEqual(result['inference']['output_tokens'], 30)
        self.assertEqual(model.messages[0], model.messages[1][:-2])
        self.assertEqual(len(self.measured), 8)

    def test_inference_exhaustion_and_oversized_batches_are_failures(self):
        session = self.session()
        result = run_active(session, ScriptedModel([self.hypothesis()]), limits={**LIMITS, 'total_generated_tokens':10})
        self.assertEqual(result['status'], 'inference_budget_exhausted')
        self.assertFalse(result['valid_final_answer'])
        session = self.session()
        bad = self.plan(5)
        result = run_active(session, ScriptedModel([self.hypothesis(), bad, bad]))
        self.assertEqual(result['status'], 'invalid_reply')
        self.assertEqual(len(self.measured), 8)

    def test_strict_json_rejects_ambiguous_or_nonfinite_output(self):
        for value in ('{"structures":[],"structures":[]}', '{"structures":NaN}'):
            with self.assertRaises(ValueError):
                parse_hypothesis(value)


if __name__ == '__main__':
    unittest.main()
