import unittest

import numpy as np

from audit_statistical_sensitivity import calculate, compare_reference, generate, paired_draw, sampling_plan


class StatisticalSensitivityTests(unittest.TestCase):
    def test_acceleration_matches_original_on_paired_random_scores(self):
        plan = sampling_plan(37, 7301)
        for shared in (False, True):
            b, c = generate(17, shared, .2, .1, repetitions=3)
            result = calculate(b, c, plan, batch=2)
            for index in range(3):
                compare_reference(b, c, result, index=index, repetitions=37, seed=7301)

    def test_boundaries_and_direction_retain_seed_pairing(self):
        plan = sampling_plan(11)
        zero, one = np.zeros((1, 3, 100), dtype=bool), np.ones((1, 3, 100), dtype=bool)
        for b, c, point in ((zero, zero, 0), (zero, one, 1), (one, zero, -1)):
            result = calculate(b, c, plan)
            np.testing.assert_array_equal(result['interval'], [[point, point]])
            self.assertEqual(bool(result['advantage'][0]), point > 0)
            compare_reference(b, c, result, repetitions=11)
        with self.assertRaises(ValueError):
            calculate(zero.astype(int), one, plan)
        with self.assertRaises(ValueError):
            calculate(zero[:, :2], one[:, :2], plan)

    def test_joint_categories_and_structure_dependence(self):
        class Fixed:
            def random(self, shape):
                return np.asarray([.05, .15, .25, .75])
        b, c = paired_draw(Fixed(), (4,), .2, .3)
        np.testing.assert_array_equal(b, [True, True, False, False])
        np.testing.assert_array_equal(c, [True, False, True, False])
        b, c = generate(8, True, .2, .1, repetitions=2)
        np.testing.assert_array_equal(b[:, 0], b[:, 2])
        np.testing.assert_array_equal(c[:, :, 0], c[:, :, 4])
        with self.assertRaises(ValueError):
            paired_draw(Fixed(), (4,), .8, 1.1)


if __name__ == '__main__':
    unittest.main()
