from copy import deepcopy
import unittest

from report_mainline_training_effort import audit_effort


def fixture():
    entries = {'example': {'input_tokens': 100, 'supervised_tokens': 10}}
    budget = {'input_tokens': 100, 'supervised_tokens': 10, 'terminal_presentations': 1}
    config = {'maximum_updates': 1, 'budget': budget,
              'schedule': [{'episode_id': 'world', 'tasks': [{'key': 'example', 'kind': 'terminal'}]}]}
    result = {'completed': True, 'completed_updates': 1, 'training_seconds': 3., **budget}
    records = [{'step': 1, 'episode_id': 'world', 'processed_tokens': 100, 'supervised_tokens': 10,
                'terminal_presentations': 1, 'training_seconds': 3., 'peak_allocated_gib': 2.}]
    start = {'invocation': 1, 'event': 'invocation_start', 'resumed_after_step': 0}
    task = {'invocation': 1, 'step': 1, 'task_index': 0, 'key': 'example'}
    events = [start, {**task, 'event': 'task_started'},
              {**task, 'event': 'task_completed', 'input_tokens': 100, 'supervised_tokens': 10, 'seconds': 2.},
              {'invocation': 1, 'event': 'checkpoint_committed', 'step': 1}]
    return config, result, records, events, entries


class TrainingEffortTests(unittest.TestCase):
    def test_completed_task_recomputed_after_interrupt_is_still_charged(self):
        config, result, records, events, entries = fixture()
        aborted = [{**e, 'invocation': 0} for e in events[:-1]]
        summary = audit_effort(config, result, records, aborted+events, entries)
        self.assertEqual(summary['effective_updates'], 1)
        self.assertEqual(summary['completed_task_presentations'], 2)
        self.assertEqual(summary['completed_tokens_beyond_effective_updates']['input_tokens'], 100)
        self.assertIsNone(summary['full_execution_wall_seconds'])

    def test_unfinished_task_cost_remains_unknown(self):
        config, result, records, events, entries = fixture()
        aborted = [{**e, 'invocation': 0} for e in events[:2]]
        summary = audit_effort(config, result, records, aborted+events, entries)
        self.assertTrue(summary['unfinished_task_cost_is_unknown_not_zero'])
        self.assertEqual(summary['unfinished_task_receipts'], [[0, 1, 0]])

    def test_corrupted_budget_and_missing_task_receipts_are_rejected(self):
        config, result, records, events, entries = fixture()
        bad = deepcopy(records); bad[0]['processed_tokens'] = 99
        with self.assertRaises(ValueError):
            audit_effort(config, result, bad, events, entries)
        with self.assertRaisesRegex(ValueError, 'Commit receipt precedes'):
            audit_effort(config, result, records, events[:2]+events[3:], entries)
        with self.assertRaisesRegex(ValueError, 'committed update was repeated'):
            audit_effort(config, result, records, events+events[-1:], entries)


if __name__ == '__main__':
    unittest.main()
