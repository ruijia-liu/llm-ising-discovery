"""Protect the information boundary and alignment of the prospective main study."""
from copy import deepcopy
import json
import unittest

from discovery.student_mainline import (PREFIXES, auxiliary_messages, evidence_prefix,
                                        hypothesis_prompt, load_group, terminal_messages)
from discovery.student_format import expand_messages
from discovery.student_v2 import auxiliary_messages as old_auxiliary
from discovery.student_spins import local_query
from discovery.student_laboratory import decode_student_query
from discovery.physics import Grammar
from discovery.student_spins import decode_hex_query


SOURCE = 'data/research_curriculum_publication_v1'


class MainlineFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups = {g: load_group(SOURCE, 'train', g, 1) for g in 'ABC'}

    def test_A_has_no_later_observations_and_B_C_targets_match(self):
        for j in (0, 50, 100, 150, 200, 250):
            a, b, c = (self.groups[g][j] for g in 'ABC')
            self.assertEqual(terminal_messages(a, 8), terminal_messages(b, 8))
            for budget in PREFIXES:
                self.assertEqual(terminal_messages(b, budget), terminal_messages(c, budget))
                evidence = evidence_prefix(b, budget)
                self.assertEqual(len(evidence['measurements']), budget)
                self.assertEqual(terminal_messages(b, budget)[:-1], hypothesis_prompt(evidence))
            with self.assertRaises(ValueError):
                terminal_messages(a, 16)

    def test_future_observations_and_truth_do_not_enter_prefix_prompt(self):
        b = deepcopy(self.groups['B'][200])
        before = hypothesis_prompt(evidence_prefix(b, 8))
        b['final_answer']['expression']['terms'] = []
        b['experiments'][0]['observed'] = 999999
        b['episode_id'] = 'SECRET'
        b['family'] = 'SECRET'
        self.assertEqual(before, hypothesis_prompt(evidence_prefix(b, 8)))
        evidence = evidence_prefix(b, 8)
        evidence['measurements'][0]['truth'] = 'SECRET'
        with self.assertRaises(ValueError):
            hypothesis_prompt(evidence)

    def test_common_actions_observations_and_labels_not_in_auxiliary(self):
        b, c = (self.groups[g][200] for g in 'BC')
        bm, cm = (auxiliary_messages(v, g) for v, g in ((b, 'B'), (c, 'C')))
        self.assertEqual([m for m in bm if m['role'] == 'user'], [m for m in cm if m['role'] == 'user'])
        self.assertFalse(any('final_answer' in m['content'] for m in bm+cm))
        for i in range(24):
            self.assertEqual(json.loads(bm[2+2*i]['content'])['experiment'],
                             json.loads(cm[2+2*i]['content'])['experiment'])
        damaged = deepcopy(c)
        damaged['final_answer']['expression']['terms'] = []
        self.assertEqual(cm, auxiliary_messages(damaged, 'C'))

    def test_local_actions_preserve_every_public_flip_feature(self):
        grammar = Grammar()
        b = self.groups['B'][200]
        old = old_auxiliary(b, 'B')
        new = auxiliary_messages(b, 'B')
        for i in range(24):
            encoded = json.loads(old[2+2*i]['content'])['experiment']['query']
            localized = json.loads(new[2+2*i]['content'])['experiment']['query']
            self.assertEqual(localized, local_query(encoded))
            s, p = decode_hex_query(encoded)
            t, q = decode_student_query({'local_spins': localized['local_spins'], 'position': localized['original_position']})
            import numpy as np
            np.testing.assert_array_equal(grammar.features([s], [p]), grammar.features([t], [q]))


if __name__ == '__main__':
    unittest.main()
