from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from build_publication_curriculum import atomic_json, read
from discovery.run_lock import exclusive_run
import run_evaluation_queue as queue


class EvaluationQueueTests(unittest.TestCase):
    def test_plan_covers_all_groups_seeds_modes_and_shared_baselines_once(self):
        main, ablation = queue.plan('mainline'), queue.plan('prediction_ablation')
        self.assertEqual(len(main['jobs']), 20)
        self.assertEqual(len(ablation['jobs']), 6)
        self.assertEqual(len({j['output'] for j in main['jobs']}), 20)
        self.assertEqual({(j['method'], j['seed']) for j in main['jobs'] if j['group'] == 'sparse'},
                         {('sparse_fixed', None), ('sparse_active', None)})
        self.assertTrue(all(j['group'] == 'C_no_predictions' for j in ablation['jobs']))
        self.assertEqual({j['seed'] for j in ablation['jobs']}, {401, 402, 403})

    def test_training_gate_fails_before_registration_output_or_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'uncreated'
            with patch.object(queue, 'require_completed_main_training', side_effect=ValueError('training incomplete')), \
                 patch.object(queue, 'registered_student') as student, patch.object(queue, 'evaluate') as evaluate:
                with self.assertRaisesRegex(ValueError, 'training incomplete'):
                    queue.execute(queue.plan('mainline', root))
                student.assert_not_called(); evaluate.assert_not_called()
                self.assertFalse(root.exists())

    def test_a_live_controller_holds_an_os_lease_not_just_a_state_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'lease'
            with exclusive_run(path):
                with self.assertRaisesRegex(RuntimeError, 'live evaluation controller'):
                    with exclusive_run(path):
                        self.fail('A second controller acquired a live lease.')
            with exclusive_run(path):
                self.assertTrue(path.exists())  # A stale lock filename does not prevent a new run.

    def test_completed_evaluation_must_match_registered_adapter_and_calibration(self):
        job = queue.plan('mainline')['jobs'][0]
        config = {'sources': {}, 'calibration_sha256': 'original', 'adapter_sha256': 'original-adapter'}
        with patch.object(queue, 'load_completed_evaluation', return_value={'config': config, 'provenance': {}}):
            with self.assertRaisesRegex(ValueError, 'different noise calibration'):
                queue.verified_job(job, 'mainline', {queue.CALIBRATION: 'changed'}, None)
            with self.assertRaisesRegex(ValueError, 'another final adapter'):
                queue.verified_job(job, 'mainline', {queue.CALIBRATION: 'original'}, {'adapter_sha256': 'changed'})

    def mock_environment(self, stack, folder):
        stack.enter_context(patch.object(queue, 'LEASE', Path(folder)/'lease'))
        stack.enter_context(patch.object(queue, 'require_completed_main_training'))
        profile = {k: 'fixture' for k in ('adapter_sha256', 'training_config_sha256', 'training_result_sha256', 'model_revision')}
        stack.enter_context(patch.object(queue, 'registered_student', return_value=profile))
        stack.enter_context(patch.object(queue, 'fingerprint', return_value={'fixture': 'checksum'}))
        stack.enter_context(patch.object(queue, 'verified_job', side_effect=lambda j, *_: {'directory': j['output'], 'fixture': True}))

    def test_failed_attempt_stops_then_explicit_resume_preserves_complete_jobs(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            self.mock_environment(stack, folder)
            spec = queue.plan('mainline', Path(folder)/'outputs')
            calls = []
            def fail_second(args):
                calls.append(args)
                atomic_json(Path(args.output)/'config.json', {'synthetic_fixture': True})
                if len(calls) == 2:
                    raise RuntimeError('synthetic interruption')
                atomic_json(Path(args.output)/'result.json', {'synthetic_fixture': True})
            with patch.object(queue, 'evaluate', side_effect=fail_second):
                with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                    queue.execute(spec)
            self.assertEqual(len(calls), 2)
            root = Path(spec['output_root'])
            self.assertEqual(read(root/'queue_state.json')['status'], 'stopped')
            invocation = next((root/'queue_invocations').iterdir())
            receipt = read(invocation/'001-end.json')
            self.assertEqual(receipt['automatic_retries'], 0)
            self.assertGreaterEqual(receipt['elapsed_wall_seconds'], 0)
            resumed = []
            def finish(args):
                resumed.append(args)
                atomic_json(Path(args.output)/'result.json', {'synthetic_fixture': True})
            with patch.object(queue, 'evaluate', side_effect=finish):
                result = queue.execute(spec, resume=True)
                self.assertEqual(len(resumed), 19)
                self.assertTrue(resumed[0].resume)
                self.assertTrue(all(not a.resume for a in resumed[1:]))
                self.assertEqual(result['jobs'], 20)
                queue.execute(spec, resume=True)
                self.assertEqual(len(resumed), 19)
            self.assertEqual(len(list((root/'queue_invocations').iterdir())), 3)

    def test_changed_source_or_adapter_and_selected_queue_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            self.mock_environment(stack, folder)
            spec = queue.plan('prediction_ablation', Path(folder)/'outputs')
            with patch.object(queue, 'evaluate', side_effect=RuntimeError('fixture stop')):
                with self.assertRaisesRegex(RuntimeError, 'fixture stop'):
                    queue.execute(spec)
            with patch.object(queue, 'fingerprint', return_value={'fixture': 'changed'}):
                with self.assertRaisesRegex(ValueError, 'unchanged queue plan'):
                    queue.execute(spec, resume=True)
            altered = {**spec, 'jobs': spec['jobs'][:1]}
            with self.assertRaisesRegex(ValueError, 'complete registered queue'):
                queue.execute(altered, resume=True)
            profile = {k: 'different' for k in ('adapter_sha256', 'training_config_sha256', 'training_result_sha256', 'model_revision')}
            with patch.object(queue, 'registered_student', return_value=profile):
                with self.assertRaisesRegex(ValueError, 'unchanged queue plan'):
                    queue.execute(spec, resume=True)


if __name__ == '__main__':
    unittest.main()
