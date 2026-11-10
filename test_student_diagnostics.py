"""Guard the information boundary and answer-mask audit used by diagnostics."""
import copy
import json
import unittest

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from diagnose_student_pilot import ANSWER_PREFIX, MODEL, PREPARED, build_prompt, cached_greedy, final_supervision, jobs
from train_student_pilot import FINAL_INSTRUCTION, selected_data


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = {g: {s: selected_data(PREPARED, g, s) for s in ('train', 'validation')} for g in ('B', 'C')}
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)

    def test_prompts_exclude_grader_answers_and_native_prefix_matches_training(self):
        for group, split, index, mode in jobs():
            messages, prefix = build_prompt(self.data, group, split, index, mode)
            for message in messages:
                if message['role'] == 'system' or message['content'] == FINAL_INSTRUCTION:
                    continue
                body = json.loads(message['content'])
                self.assertNotIn('final_answer', body)
                self.assertNotIn('teacher_estimate_before_grading', body)
            if mode.startswith('native'):
                self.assertEqual(messages, self.data[group][split][index]['messages'][:-1])
            self.assertEqual(prefix, ANSWER_PREFIX if 'prefilled' in mode else '')

    def test_fixed_evidence_prompts_identical_between_groups(self):
        for mode in ('original', 'shared_prefix', 'prefilled_fixed'):
            for split in ('train', 'validation'):
                self.assertEqual(build_prompt(self.data, 'B', split, 0, mode),
                                 build_prompt(self.data, 'C', split, 0, mode))

    def test_audit_detects_accidentally_masked_final_answer(self):
        example = copy.deepcopy(self.data['C']['train'][0])
        self.assertTrue(final_supervision(self.tokenizer, example)['all_final_answer_tokens_supervised'])
        last_message_tokens = len(self.tokenizer.encode(example['messages'][-1]['content'], add_special_tokens=False))
        example['labels'][-last_message_tokens-3:] = -100
        with self.assertRaisesRegex(AssertionError, 'no supervised label'):
            final_supervision(self.tokenizer, example)

    def test_chunked_prefill_keeps_full_context_and_greedy_generation(self):
        torch.set_num_threads(2)
        torch.manual_seed(32)
        config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            max_position_embeddings=128, attention_dropout=0.0,
                            eos_token_id=63, pad_token_id=63)
        model = Qwen2ForCausalLM(config).eval()
        model.generation_config.repetition_penalty = 1.1
        ids = torch.randint(0, 62, (1, 39))
        with torch.inference_mode():
            expected = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=13, do_sample=False)
            actual = cached_greedy(model, ids, 13, [63], chunk_size=7)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
