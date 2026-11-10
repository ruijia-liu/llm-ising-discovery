import json
import unittest

from discovery.action_prompt_v2 import planning_prompt_v2, practice_cases_v2
from discovery.active_student import planning_prompt
from discovery.interface_alignment import neutral_session, practice_cases


class ClearActionPromptTests(unittest.TestCase):
    def test_neutral_task_values_and_targets_unchanged(self):
        for split in ('train', 'validation'):
            for old, new in zip(practice_cases(split), practice_cases_v2(split)):
                for key in ('id', 'answer', 'current_structures', 'initial_query_seed'):
                    self.assertEqual(old[key], new[key])
                self.assertNotIn('完整格式示例', str(new['prompt']))
                supplied = json.loads(old['prompt'][1]['content'].split('\n', 2)[1])
                rewritten = json.loads(new['prompt'][1]['content'].split('\n给定计划：\n', 1)[1].split('\n转换规则：', 1)[0])
                self.assertEqual(supplied, rewritten)

    def test_autonomous_payload_stays_public_and_same(self):
        case = practice_cases('train')[1]
        session = neutral_session(case)
        evidence = session.evidence()
        fit = session.fit(case['current_structures'])
        old = planning_prompt(evidence, case['current_structures'], fit, None, 4)
        new = planning_prompt_v2(evidence, case['current_structures'], fit, None, 4)
        self.assertEqual(old[0], new[0])
        self.assertEqual(old[1]['content'].split('\n', 1)[0], new[1]['content'].split('\n', 1)[0])
        self.assertNotIn('完整格式示例', str(new))
        with self.assertRaises(ValueError):
            planning_prompt_v2({**evidence, 'truth': []}, [], {}, None, 4)


if __name__ == '__main__':
    unittest.main()
