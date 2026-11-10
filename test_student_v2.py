"""Protect matched terminal supervision and label-free inference in v2."""
from copy import deepcopy
import json
import unittest

import numpy as np
import torch
from transformers import AutoTokenizer

from discovery.student_v2 import (auxiliary_messages, encode, evidence_only, load_views,
                                  normalized_task_loss, terminal_messages, terminal_prompt)


class StudentV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = load_views('train', 'B')
        cls.c = load_views('train', 'C')
        cls.tokenizer = AutoTokenizer.from_pretrained('models/Qwen2.5-0.5B-Instruct', local_files_only=True)

    def test_common_terminal_task_and_inference_are_identical_for_B_C(self):
        for b, c in zip(self.b, self.c):
            self.assertEqual(terminal_messages(b), terminal_messages(c))
            self.assertEqual(terminal_messages(b)[:-1], terminal_prompt(evidence_only(b)))
            changed = deepcopy(b)
            changed['final_answer']['expression']['terms'] = []
            changed['episode_id'] = 'a-label-must-not-enter-input'
            self.assertEqual(terminal_messages(changed)[:-1], terminal_messages(b)[:-1])

    def test_only_terminal_answer_is_supervised(self):
        for view in self.b:
            example = encode(self.tokenizer, terminal_messages(view))
            actual = self.tokenizer.decode(example['labels'][example['labels'] != -100])
            self.assertEqual(actual, example['messages'][-1]['content']+'<|im_end|>')
            self.assertEqual(int(np.count_nonzero(example['labels'] != -100)),
                             sum(label != -100 for label in example['labels']))

    def test_auxiliary_tasks_cannot_repeat_gold_answers(self):
        for group, views in (('B', self.b), ('C', self.c)):
            for view in views:
                before = auxiliary_messages(view, group)
                changed = deepcopy(view)
                changed['final_answer']['expression']['terms'] = []
                self.assertEqual(before, auxiliary_messages(changed, group))
                self.assertFalse(any('"final_answer"' in message['content'] for message in before))
                if group == 'C':
                    self.assertIn('update_from_last_observation', json.loads(before[-1]['content']))

    def test_terminal_gradient_weight_ignores_auxiliary_length(self):
        for group in ('B', 'C'):
            gradients = []
            for length in (10, 10000):
                terminal = torch.tensor(80.0, requires_grad=True)
                auxiliary = torch.tensor(float(length)*2, requires_grad=True)
                loss = normalized_task_loss(terminal, 40, auxiliary, length, group)
                loss.backward()
                self.assertAlmostEqual(loss.item(), 2.0)
                gradients.append(terminal.grad.item())
            self.assertEqual(gradients[0], gradients[1])
            self.assertAlmostEqual(gradients[0], 0.5/40, places=8)

    def test_evidence_keeps_all_measurements_and_hex_is_lossless(self):
        for view in self.b:
            evidence = evidence_only(view)
            records = evidence['measurements']
            queries = view['initial']['queries']+[e['query'] for e in view['experiments']]
            observed = view['initial']['observed']+[e['observed'] for e in view['experiments']]
            self.assertEqual(len(records), 32)
            self.assertEqual([r['delta'] for r in records], observed)
            for record, query in zip(records, queries):
                spins = f"{int(record['query']['hex'], 16):0256b}".replace('1', '+').replace('0', '-')
                self.assertEqual(spins, ''.join(query['rows']))
                self.assertEqual(record['query']['position'], query['position'])


if __name__ == '__main__':
    unittest.main()
