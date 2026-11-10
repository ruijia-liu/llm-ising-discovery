"""Validate known-physics preprocessing without relying on learned outputs."""
import copy
import unittest

import numpy as np

from discovery.physics import Expression, Term, random_queries, base_flip
from discovery.student_fastdev import residual_evidence, terminal_prompt
from discovery.student_spins import OFFSETS


class FastDevelopmentTests(unittest.TestCase):
    def evidence(self, expression):
        rng = np.random.default_rng(811)
        queries = random_queries(rng, 32)
        rows = []
        for state, (x, y) in queries:
            patch = [[int(state[(x+dx) % 16, (y+dy) % 16]) for dy in OFFSETS] for dx in OFFSETS]
            rows.append({'query': {'local_spins': patch, 'original_position': [x, y]},
                         'delta': expression.delta(state, (x, y))})
        return {'measurement_noise_sigma': 1.0, 'used': 32, 'measurements': rows}, queries

    def test_reconstruction_across_orders(self):
        for points in (((0, 0),), ((0, 0), (0, 2)), ((0, 0), (1, 0), (1, 1)),
                       ((0, 0), (1, 0), (0, 1), (1, 1))):
            evidence, queries = self.evidence(Expression((Term(points, -.73),)))
            before = copy.deepcopy(evidence)
            converted = residual_evidence(evidence)
            for item, row, (state, pos) in zip(evidence['measurements'], converted['measurements'], queries):
                reconstructed = base_flip(state, pos)+2*int(state[pos])*row[1]*converted['scale']
                self.assertLessEqual(abs(reconstructed-item['delta']), converted['scale']*1.0001e-8)
            self.assertEqual(evidence, before)
            self.assertEqual(converted['normalized_noise_sigma'], .5/converted['scale'])

    def test_null_and_label_rejection(self):
        evidence, _ = self.evidence(Expression())
        converted = residual_evidence(evidence)
        self.assertEqual(converted['scale'], 1)
        self.assertTrue(all(row[1] == 0 for row in converted['measurements']))
        self.assertNotIn('original_position', str(converted))
        with self.assertRaises(ValueError):
            terminal_prompt({**evidence, 'final_answer': 'leaked'}, 'residual')


if __name__ == '__main__':
    unittest.main()
