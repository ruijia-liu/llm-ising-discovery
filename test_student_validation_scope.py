"""A subset of training worlds must not inherit the full corpus split label."""
import unittest

from audit_student_validation_scope import classify
from discovery.student_v2 import TRAIN_IDS, load_views


class ValidationScopeTests(unittest.TestCase):
    def test_full_training_is_parameter_validation_but_subset_is_mixed(self):
        train, validation = load_views('train'), load_views('validation')
        full = classify(train, validation, {v['episode_id'].split('-')[0] for v in train})
        self.assertEqual(set(full.values()), {'seen_structure'})
        subset = classify(train, validation, set(TRAIN_IDS))
        unseen = sorted(k for k, v in subset.items() if v == 'unseen_structure')
        self.assertEqual(unseen, ['w0027-sigma0', 'w0029-sigma0', 'w0031-sigma0', 'w0033-sigma0'])
        self.assertEqual(sum(v == 'seen_structure' for v in subset.values()), 8)


if __name__ == '__main__':
    unittest.main()
