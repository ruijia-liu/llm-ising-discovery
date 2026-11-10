from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from build_publication_curriculum import read
from discovery.process_ablation import GROUP, remove_explicit_predictions, task_plan
from discovery.student_mainline import auxiliary_messages, load_group
from train_mainline_student import task_plan as original_plan


class ProcessAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.messages = auxiliary_messages(load_group('data/research_curriculum_publication_v1', 'train', 'C', 1)[0], 'C')

    def test_evidence_experiments_and_hypotheses_are_preserved(self):
        original = deepcopy(self.messages)
        transformed, removals = remove_explicit_predictions(self.messages)
        self.assertEqual(original, self.messages)
        self.assertEqual(sum(row['path'][-1] == 'prediction_errors' for row in removals), 24)
        for old, new in zip(original, transformed, strict=True):
            if old['role'] != 'assistant':
                self.assertEqual(old, new)
                continue
            a, b = json.loads(old['content']), json.loads(new['content'])
            for name in ('experiment', 'teacher_estimate_before_grading'):
                self.assertEqual(a.get(name), b.get(name))
            if 'before_observation' in a:
                before, after = a['before_observation'], b['before_observation']
                self.assertEqual(before['selection'], after['selection'])
                for h, k in zip(before['hypotheses'], after['hypotheses'], strict=True):
                    self.assertEqual({key: value for key, value in h.items() if key != 'predicted_delta'}, k)
            self.assertNotIn('predicted_delta', new['content'])
            self.assertNotIn('prediction_errors', new['content'])
            self.assertNotIn('final_answer', new['content'])

    def test_drift_and_missing_round_are_rejected(self):
        with self.assertRaises(ValueError):
            remove_explicit_predictions(self.messages[:-1])
        changed = deepcopy(self.messages)
        payload = json.loads(changed[2]['content'])
        payload['undocumented_prediction'] = {'predicted_delta': 123}
        changed[2]['content'] = json.dumps(payload)
        with self.assertRaises(ValueError):
            remove_explicit_predictions(changed)

    def test_registered_schedule_matches_full_C_terminal_exposure(self):
        # Build synthetic ablated lengths on actual complete mainline episodes:
        # this tests cumulative rounding and repeats, independently of caching.
        original = read('data/student_mainline_v1/manifest.json')
        ablated = deepcopy(original)
        for row in ablated['episodes']:
            key = row['auxiliary']['C']
            new_key = key+'_without_predictions'
            row['auxiliary'][GROUP] = new_key
            row['candidate_expressions_per_auxiliary'] = 72
            ablated['entries'][new_key] = {**ablated['entries'][key],
                                          'input_tokens': int(.72*ablated['entries'][key]['input_tokens'])}
        for seed in (401, 402, 403):
            plan, budget = task_plan(ablated, seed)
            reference, _ = original_plan(original, 'C', seed)
            self.assertEqual(budget['updates'], 1200)
            self.assertEqual(budget['terminal_presentations'], 4800)
            self.assertLessEqual(abs(budget['relative_auxiliary_input_difference']), .01)
            self.assertGreater(budget['auxiliary_presentations'], 1200)
            for a, b in zip(plan, reference, strict=True):
                self.assertEqual(a['episode_id'], b['episode_id'])
                self.assertEqual(a['tasks'][:4], b['tasks'][:4])
                self.assertAlmostEqual(sum(t['weight'] for t in a['tasks'][4:]), 1)

    def test_training_guard_runs_before_loading_a_model_or_preparing_tasks(self):
        from train_process_ablation import run
        with patch('train_process_ablation.require_completed_main_training', side_effect=ValueError('main not complete')), \
                patch('train_process_ablation.make_student') as model, \
                patch('train_process_ablation.prepare') as prepare:
            with self.assertRaisesRegex(ValueError, 'main not complete'):
                run(401)
            model.assert_not_called()
            prepare.assert_not_called()

    def test_additional_queue_and_resource_probe_do_not_start_during_main_training(self):
        from preflight_robustness_training import run as probe
        from run_process_ablation_queue import run as queue
        with patch('preflight_robustness_training.require_completed_main_training', side_effect=ValueError('main active')), \
                patch('preflight_robustness_training.make_student') as model:
            with self.assertRaisesRegex(ValueError, 'main active'):
                probe()
            model.assert_not_called()
        with patch('run_process_ablation_queue.require_completed_main_training', side_effect=ValueError('main active')), \
                patch('run_process_ablation_queue.subprocess.Popen') as process:
            with self.assertRaisesRegex(ValueError, 'main active'):
                queue()
            process.assert_not_called()


if __name__ == '__main__':
    unittest.main()
