"""The local-spin representation must preserve every public flip feature."""
from copy import deepcopy
import unittest

import numpy as np

from discovery.physics import SIDE, base_flip, encode_query, random_queries
from discovery.student_spins import OFFSETS, evidence_only, local_query, public_coverage, terminal_messages
from discovery.student_v2 import hex_query, load_views


class ExplicitSpinTests(unittest.TestCase):
    def test_all_158_features_survive_crop_including_wrapped_boundaries(self):
        grammar = public_coverage()
        for state, position in random_queries(np.random.default_rng(92551), 96):
            encoded = local_query(hex_query(encode_query((state, position))))
            reconstructed = np.ones((SIDE, SIDE), dtype=np.int8)
            for i, dx in enumerate(OFFSETS):
                for j, dy in enumerate(OFFSETS):
                    reconstructed[(position[0]+dx) % SIDE, (position[1]+dy) % SIDE] = encoded['local_spins'][i][j]
            np.testing.assert_array_equal(grammar.features([state], [position]),
                                          grammar.features([reconstructed], [position]))
            self.assertEqual(base_flip(state, position), base_flip(reconstructed, position))

    def test_crop_does_not_depend_on_true_mechanism_and_pairs_B_C(self):
        for b, c in zip(load_views('train', 'B'), load_views('train', 'C')):
            self.assertEqual(terminal_messages(b), terminal_messages(c))
            modified = deepcopy(b)
            modified['final_answer']['expression']['terms'] = []
            self.assertEqual(evidence_only(b), evidence_only(modified))
            self.assertEqual(terminal_messages(b)[:-1], terminal_messages(modified)[:-1])


if __name__ == '__main__':
    unittest.main()
