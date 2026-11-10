"""The coverage contrast must change examples without changing target exposures."""
from collections import Counter
import unittest

from build_publication_curriculum import COVERAGE, read
from train_student_coverage import CONTROL, make_schedule, prepare
from transformers import AutoTokenizer


class CoverageTrainingTests(unittest.TestCase):
    def test_same_update_and_label_counts_with_every_new_world_used(self):
        config = read(CONTROL/'config.json')
        groups = read(COVERAGE/'coverage_groups.json')
        anchors = config['train_world_ids']
        schedule = make_schedule(config['schedule'], anchors, groups)
        reverse = {episode: anchor for anchor, values in groups.items() for episode in values}
        self.assertEqual(len(schedule), 192)
        self.assertTrue(all(len(batch) == 2 for batch in schedule))
        counts = Counter(reverse[e] for batch in schedule for e in batch)
        self.assertEqual(counts, Counter({anchor: 16 for anchor in anchors}))
        used = Counter(e for batch in schedule for e in batch)
        self.assertEqual(set(used), set(reverse))
        self.assertEqual(set(used.values()), {3, 4})
        for old_batch, new_batch in zip(config['schedule'], schedule):
            self.assertEqual([anchors[i] for i in old_batch], [reverse[e] for e in new_batch])

    def test_corpus_and_evaluation_prompts_match_existing_control(self):
        tokenizer = AutoTokenizer.from_pretrained('models/Qwen2.5-0.5B-Instruct', local_files_only=True)
        data, views, examples, groups = prepare(tokenizer)
        self.assertEqual(len(data), 120)
        self.assertEqual((len(examples['train']), len(examples['validation'])), (24, 12))


if __name__ == '__main__':
    unittest.main()
