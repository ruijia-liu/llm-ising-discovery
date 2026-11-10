from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from reproduce_training_inputs import compare_manifests, compare_token_file


class TrainingInputReproductionTests(unittest.TestCase):
    def test_equal_ids_with_different_supervision_mask_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            a, b = Path(temporary)/'a.npz', Path(temporary)/'b.npz'
            ids = np.asarray([4, 5, 6], dtype=np.int64)
            labels = np.asarray([-100, 5, 6], dtype=np.int64)
            np.savez_compressed(a, ids=ids, labels=labels)
            np.savez_compressed(b, ids=ids, labels=labels)
            self.assertTrue(compare_token_file(a, b))
            np.savez_compressed(b, ids=ids, labels=np.asarray([4, -100, 6], dtype=np.int64))
            with self.assertRaisesRegex(ValueError, 'supervision labels differ'):
                compare_token_file(a, b)

    def test_changed_token_or_dtype_and_unknown_archive_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            a, b = Path(temporary)/'a.npz', Path(temporary)/'b.npz'
            np.savez_compressed(a, ids=np.asarray([1, 2], dtype=np.int64), labels=np.asarray([-100, 2], dtype=np.int64))
            for ids in (np.asarray([1, 3], dtype=np.int64), np.asarray([1, 2], dtype=np.int32)):
                np.savez_compressed(b, ids=ids, labels=np.asarray([-100, 2], dtype=np.int64))
                with self.assertRaisesRegex(ValueError, 'Token IDs'):
                    compare_token_file(a, b)
            np.savez_compressed(b, ids=[1, 2], labels=[-100, 2], hidden=[7])
            with self.assertRaisesRegex(ValueError, 'archive fields'):
                compare_token_file(a, b)

    def test_only_declared_provenance_metadata_may_change(self):
        original = {'completed': True, 'episodes': [{'episode_id': 'example'}], 'messages_sha256': 'same',
                    'counts': {'episodes': 1}, 'maximum_tokens': 3,
                    'A_references_only_initial8': True, 'B_C_terminal_identical': True,
                    'config': {'curriculum_audit_sha256': 'old', 'model_revision': 'pinned', 'formal_tests_used': False},
                    'entries': {'x': {'path': 'x.npz', 'sha256': 'old-container', 'input_tokens': 3, 'supervised_tokens': 2}}}
        generated = deepcopy(original)
        generated['config']['curriculum_audit_sha256'] = 'new-audit'
        generated['entries']['x']['sha256'] = 'new-container'
        compare_manifests(original, generated)  # Arrays and their own hashes are checked separately.
        for change in ('label-count', 'revision', 'task-set', 'episode-order'):
            bad = deepcopy(generated)
            if change == 'label-count': bad['entries']['x']['supervised_tokens'] = 1
            elif change == 'revision': bad['config']['model_revision'] = 'different'
            elif change == 'task-set': bad['entries']['extra'] = bad['entries']['x']
            else: bad['episodes'].append({'episode_id': 'additional'})
            with self.subTest(change=change), self.assertRaises(ValueError):
                compare_manifests(original, bad)


if __name__ == '__main__':
    unittest.main()
