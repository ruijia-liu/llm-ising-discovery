import json
import unittest

from discovery.interface_alignment import neutral_session, practice_cases, prepare_plan
from discovery.student_v2 import packed


class InterfaceAlignmentTests(unittest.TestCase):
    def test_all_targets_execute_and_charge_only_after_commit(self):
        for split in ('train', 'validation'):
            cases = practice_cases(split)
            self.assertEqual(sum(c['answer']['action'] == 'finish' for c in cases), len(cases)//4)
            for case in cases:
                session = neutral_session(case)
                action, committed = prepare_plan(session, case['current_structures'], packed(case['answer']), 2)
                self.assertEqual(action, case['answer'])
                self.assertEqual(session.evidence()['used'], 8)
                if committed:
                    self.assertEqual(session.events()[-1]['kind'], 'committed_before_observation')
                    result = session.reveal(committed['ticket'])
                    self.assertEqual(result['used'], 8+len(action['queries']))
                    self.assertEqual(result['before_sha256'], committed['before_sha256'])

    def test_practice_has_no_mechanism_labels_or_shared_cases(self):
        train, valid = practice_cases('train'), practice_cases('validation')
        self.assertFalse({packed(c['prompt']) for c in train} & {packed(c['prompt']) for c in valid})
        for c in train+valid:
            text = json.dumps(c['prompt'])
            self.assertNotIn('final_answer', text)
            self.assertNotIn('world_id', text)
        with self.assertRaises(ValueError):
            practice_cases('test')


if __name__ == '__main__':
    unittest.main()
