from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from build_publication_curriculum import sha
from reproduce_curriculum_chain import compare_legacy, require_identical_file


class CurriculumChainComparisonTests(unittest.TestCase):
    def test_source_and_timing_changes_are_disclosed_but_outcome_changes_fail(self):
        original = {'config': {'seed': 93001, 'source_sha256': {'teacher.py': 'old'}},
                    'audit_counts': {'rounds': 24}, 'teacher_outcomes': [{'recovered': False}],
                    'elapsed_seconds': 2.}
        generated = deepcopy(original)
        generated['config']['source_sha256']['teacher.py'] = 'new'
        generated['elapsed_seconds'] = 3.
        self.assertEqual(compare_legacy(original, generated),
                         {'teacher.py': {'original': 'old', 'current': 'new'}})
        generated['teacher_outcomes'][0]['recovered'] = True
        with self.assertRaisesRegex(ValueError, 'outcome summaries'):
            compare_legacy(original, generated)

    def test_changed_random_seed_or_missing_audit_field_cannot_be_ignored(self):
        original = {'config': {'seed': 93001, 'source_sha256': {}}, 'audit_counts': {'rounds': 24}}
        changed = deepcopy(original)
        changed['config']['seed'] = 1
        with self.assertRaisesRegex(ValueError, 'protocol settings'):
            compare_legacy(original, changed)
        changed = deepcopy(original)
        del changed['audit_counts']
        with self.assertRaisesRegex(ValueError, 'audit or outcome'):
            compare_legacy(original, changed)

    def test_byte_identity_and_the_saved_reference_digest_are_both_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            a, b = Path(temporary)/'original.json', Path(temporary)/'new.json'
            a.write_bytes(b'{"count":1}\n')
            b.write_bytes(a.read_bytes())
            checksum = sha(a)
            self.assertEqual(require_identical_file(a, b, checksum), checksum)
            b.write_bytes(b'{ "count": 1 }\n')
            with self.assertRaisesRegex(ValueError, 'Reconstructed file differs'):
                require_identical_file(a, b, checksum)
            a.write_bytes(b.read_bytes())
            with self.assertRaises(ValueError):
                require_identical_file(a, b, checksum)


if __name__ == '__main__':
    unittest.main()
