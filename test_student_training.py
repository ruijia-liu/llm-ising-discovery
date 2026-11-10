"""Verify training mechanics independently of the scientific outcome."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model

from discovery.student_format import compact_messages, expand_messages, normalized_messages, tokenize_assistant_only
from train_student_pilot import bounded_loss, capture_training_state, chunk_feed_forward, make_schedule, offload_saved_activations, parse_expression, restore_training_state, selected_data
from discovery.physics import Grammar


class StudentTrainingTests(unittest.TestCase):
    def test_lossless_round_trip_keeps_signed_spins_and_numbers(self):
        query = {'rows': ['+-'*8, '-+'*8]*8, 'position': [15, 0]}
        expression = {'base': 'nearest_neighbour_J=1', 'terms': [
            {'coefficient': -0.123456789012345, 'offsets': [[0, 0], [0, 2]]}]}
        payload = {'query': query, 'expression': expression, 'reason': '保留这个理由', 'value': -1.23456789012345}
        messages = [{'role': 'system', 'content': '规则'},
                    {'role': 'user', 'content': json.dumps(payload)},
                    {'role': 'assistant', 'content': json.dumps(payload)}]
        compact = compact_messages(messages)
        self.assertEqual(expand_messages(compact), normalized_messages(messages))
        self.assertEqual(json.loads(compact[-1]['content'])['reason'], '=')
        self.assertEqual(len(json.loads(compact[1]['content'])['query']['hex']), 64)

    def test_chunked_loss_and_gradients_equal_full_logits(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            max_position_embeddings=64, attention_dropout=0.0, tie_word_embeddings=True)
        reference = Qwen2ForCausalLM(config).eval()
        chunked = copy.deepcopy(reference)
        chunked.train()
        chunked.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        chunk_feed_forward(chunked, chunk_size=3)
        ids = torch.randint(0, 64, (1, 19))
        labels = ids.clone()
        labels[:, :4] = -100
        labels[:, 9:13] = -100
        ordinary = reference(input_ids=ids, labels=labels).loss
        ordinary.backward()
        total, count = bounded_loss(chunked, ids, labels, head_chunk=3)
        bounded = total/count
        bounded.backward()
        torch.testing.assert_close(bounded, ordinary, rtol=1e-6, atol=1e-6)
        for (name, parameter), (other_name, other) in zip(reference.named_parameters(), chunked.named_parameters()):
            self.assertEqual(name, other_name)
            torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-4, atol=1e-6, msg=name)

    @unittest.skipUnless(Path('models/Qwen2.5-0.5B-Instruct/tokenizer.json').exists(), 'Local tokenizer not downloaded')
    def test_assistant_mask_excludes_observations_and_headers(self):
        tokenizer = AutoTokenizer.from_pretrained('models/Qwen2.5-0.5B-Instruct', local_files_only=True)
        messages = [{'role': 'system', 'content': 'system-only marker'},
                    {'role': 'user', 'content': 'user-only measurement 123'},
                    {'role': 'assistant', 'content': '{"hypothesis":1}'},
                    {'role': 'user', 'content': 'second measurement 456'},
                    {'role': 'assistant', 'content': '{"final":2}'}]
        ids, labels = tokenize_assistant_only(tokenizer, messages)
        self.assertEqual(ids, tokenizer.apply_chat_template(messages, tokenize=True))
        supervised_text = tokenizer.decode([v for v in labels if v != -100])
        self.assertEqual(supervised_text, '{"hypothesis":1}<|im_end|>{"final":2}<|im_end|>')

    @unittest.skipUnless(Path('data/student_training_v1/episodes.npz').exists(), 'Prepare training data first')
    def test_paired_schedules_cover_all_worlds_with_equal_updates(self):
        b = selected_data('data/student_training_v1', 'B', 'train')
        c = selected_data('data/student_training_v1', 'C', 'train')
        for seed in (101, 102, 103):
            schedule = make_schedule(b, c, seed)
            self.assertEqual(len(schedule['B']), 12)
            self.assertEqual(len(schedule['C']), 12)
            self.assertEqual(sorted(i for step in schedule['C'] for i in step), list(range(24)))
            self.assertEqual(set(i for step in schedule['B'] for i in step), set(range(24)))
            self.assertLess(abs(schedule['relative_token_difference']), .01)
            self.assertEqual(schedule, make_schedule(b, c, seed))

    def test_expressions_are_executable_or_explicitly_rejected(self):
        grammar = Grammar()
        self.assertFalse(parse_expression('{"final_answer":{"expression":{"terms":[]}}}', grammar).terms)
        invalid = {'final_answer': {'expression': {'terms': [[float('nan'), [[0, 0]]]]}}}
        with self.assertRaises(ValueError):
            parse_expression(json.dumps(invalid), grammar)
        with self.assertRaises(ValueError):
            parse_expression('I believe this is a pair interaction.', grammar)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_activation_offload_preserves_gradient(self):
        x = torch.arange(8192, dtype=torch.float32, device='cuda').requires_grad_()
        with offload_saved_activations():
            hidden = 2*x
            loss = (hidden*hidden).sum()
        loss.backward()
        torch.testing.assert_close(x.grad, 8*x.detach(), rtol=0, atol=0)

    def test_step_checkpoint_restores_optimizer_weights_and_dropout_rng(self):
        torch.set_num_threads(2)
        config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                            max_position_embeddings=32, attention_dropout=0.0)
        base = Qwen2ForCausalLM(config)

        def make():
            model = get_peft_model(copy.deepcopy(base), LoraConfig(
                r=2, lora_alpha=4, lora_dropout=0.2, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
            return model.train(), torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)

        def step(model, optimizer, ids):
            optimizer.zero_grad()
            loss = model(input_ids=ids, labels=ids).loss
            loss.backward()
            optimizer.step()

        torch.manual_seed(72)
        original, optimizer = make()
        ids = torch.randint(0, 32, (1, 12))
        step(original, optimizer, ids)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.pt'
            torch.save(capture_training_state(original, optimizer), path)
            step(original, optimizer, ids)
            expected = {n: p.detach().clone() for n, p in original.named_parameters() if p.requires_grad}
            resumed, resumed_optimizer = make()  # Deliberately consumes initialization RNG.
            restore_training_state(resumed, resumed_optimizer, torch.load(path, weights_only=True, map_location='cpu'))
            step(resumed, resumed_optimizer, ids)
            for name, parameter in resumed.named_parameters():
                if parameter.requires_grad:
                    torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
