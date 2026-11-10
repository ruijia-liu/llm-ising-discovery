import unittest

from build_publication_curriculum import read
from discovery.mainline_schedule import paired_schedule


class PairedScheduleTests(unittest.TestCase):
    def test_actual_inventory_has_equal_labels_updates_and_matched_B_C_input(self):
        records = read('results/mainline_format_audit/audit.json')['records']
        plan = paired_schedule(records, seed=401, passes=1)
        self.assertEqual(plan['updates_per_arm'], 600)
        self.assertEqual(plan['terminal_presentations_per_arm'], 2400)
        self.assertLess(abs(plan['relative_auxiliary_input_difference']), .01)
        self.assertEqual({s['episode_id'] for s in plan['steps']}, {r['episode_id'] for r in records})
        self.assertTrue(all(s['A_terminal_prefixes'] == [8]*4 for s in plan['steps']))
        self.assertTrue(all(s['B_C_terminal_prefixes'] == [8,16,24,32] for s in plan['steps']))
        self.assertEqual(plan, paired_schedule(records, seed=401, passes=1))

    def test_duplicate_or_missing_inventory_is_rejected(self):
        record = {'episode_id': 'one', 'auxiliary': {'B': {'input_tokens': 100}, 'C': {'input_tokens': 500}}}
        with self.assertRaises(ValueError):
            paired_schedule([record, record], seed=401, passes=1)
        with self.assertRaises(ValueError):
            paired_schedule([], seed=401, passes=1)
        with self.assertRaises(ValueError):
            paired_schedule([record], seed=401, passes=0)


if __name__ == '__main__':
    unittest.main()
