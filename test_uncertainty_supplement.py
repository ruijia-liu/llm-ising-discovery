from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from analyze_uncertainty_supplement import run
from discovery.uncertainty_supplement import bounded_interval, supplement_comparison


COUNTS = {'combination': (3, 2), 'four_body': (3, 2)}


def fixture():
    return [{'group': group, 'seed': seed, 'level': level, 'structure_id': f'{level}-{world//2}',
             'world_id': f'{level}-{world}', 'sigma': 1, 'mode': 'active', 'score': group=='C' and world<2}
            for group in ('B', 'C') for seed in (401, 402, 403)
            for level in ('combination', 'four_body') for world in range(3)]


class UncertaintySupplementTests(unittest.TestCase):
    def test_weights_influence_and_swap_follow_registered_estimand(self):
        rows=fixture()
        result=supplement_comparison(rows, COUNTS)
        self.assertEqual(result['C_minus_B'], .5)  # Not the naive 2/3 world average.
        self.assertEqual(result['leave_one_structure_out_range'], [.25, .75])
        self.assertEqual(result['world_weights'], [.125, .125, .25]*2)
        self.assertEqual(sum(r['weighted_contribution'] for r in result['structures']), .5)
        reversed_rows=[{**r,'group': 'B' if r['group']=='C' else 'C'} for r in rows]
        reverse=supplement_comparison(reversed_rows, COUNTS)
        self.assertEqual(reverse['C_minus_B'], -.5)
        self.assertEqual(reverse['leave_one_structure_out_range'], [-.75, -.25])
        for name in ('conditional_world_bound','independent_structure_assumption_bound'):
            np.testing.assert_allclose(reverse[name]['interval'], -np.asarray(result[name]['interval'])[::-1])

    def test_incomplete_duplicate_and_relabelled_cells_fail(self):
        original=fixture()
        relabelled=deepcopy(original); relabelled[0]['structure_id']='other'
        nonboolean=deepcopy(original); nonboolean[0]['score']=1
        missing_world=[r for r in original if r['world_id']!='combination-0']
        for rows in (original[:-1], original+[original[0]], relabelled, nonboolean, missing_world):
            with self.assertRaises(ValueError):
                supplement_comparison(rows, COUNTS)

    def test_zero_boundary_keeps_width_and_invalid_weights_fail(self):
        result=bounded_interval(0., np.full(61,1/61))
        self.assertGreater(result['half_width_before_clipping'],0)
        self.assertLess(result['interval'][0],0)
        self.assertGreater(result['interval'][1],0)
        for weights in ([-.1,1.1],[.4,.4],[],[np.nan,1.]):
            with self.assertRaises(ValueError):
                bounded_interval(0.,weights)
        for point in (1.1,np.nan,np.inf):
            with self.assertRaises(ValueError):
                bounded_interval(point,[.5,.5])
        with self.assertRaises(ValueError):
            bounded_interval(0.,[1.],alpha=0)

    def test_exact_enumeration_for_independent_nonidentical_bounded_units(self):
        n=16
        bits=((np.arange(2**n,dtype=np.uint32)[:,None]>>np.arange(n))&1).astype(bool)
        probabilities=np.asarray([.2,.65]*8)
        mass=np.where(bits,probabilities,1-probabilities).prod(axis=1)
        weights=np.asarray([.04,.085]*8)
        target=float(weights@(2*probabilities-1))
        points=(bits.astype(float)*2-1)@weights
        self.assertAlmostEqual(float(mass.sum()),1.)
        for alpha in (.05,.2):
            ci=bounded_interval(points,weights,alpha)['interval']
            covered=(ci[:,0]<=target)&(ci[:,1]>=target)
            self.assertGreaterEqual(float(mass[covered].sum()),1-alpha)

    def test_formal_entry_requires_completed_inputs_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'output'
            with patch('analyze_uncertainty_supplement.load_completed_evaluation',side_effect=ValueError('incomplete')):
                with patch('analyze_uncertainty_supplement.verify_pairing') as paired:
                    with self.assertRaisesRegex(ValueError,'incomplete'):
                        run(Path(temporary)/'missing',output)
                    paired.assert_not_called()
            self.assertFalse(output.exists())


if __name__=='__main__':
    unittest.main()
