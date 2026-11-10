"""Frozen-model checks of seen worlds, prompt matching and final-answer masks.

No optimizer, training, new laboratory measurements or reserved test files.
Native C histories contain teacher reasoning: they diagnose continuation only,
and MUST NOT be used as a fair fixed-evidence B/C comparison.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoTokenizer, RepetitionPenaltyLogitsProcessor

from train_student_pilot import FINAL_INSTRUCTION, json_write, load_base, parse_expression, selected_data
from discovery.noise_evaluation import evaluate_noisy
from discovery.physics import Grammar
from discovery.teaching import expression_from_dict
from run_structure_discovery import compile_queries, evaluation_queries


MODEL = Path('models/Qwen2.5-0.5B-Instruct')
PREPARED = Path('data/student_training_v1')
RUN = Path('results/student_training_v1')
OUT = Path('results/student_diagnostics_v1')
ANSWER_PREFIX = '{"final_answer":{"expression":{"terms":'
TRAIN_SEEN = (0, 4, 8, 12, 16, 20)
CONTRAST = {'train': (0, 12, 20), 'validation': (0, 6, 10)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_hashes():
    paths = [RUN/'C_seed102/step_checkpoint.pt', RUN/'PAUSED.json',
             Path('results/structure_discovery_noise/calibration.json')]
    for group in ('B', 'C'):
        paths += sorted((RUN/f'{group}_seed101/adapter').glob('*'))
        paths += [RUN/f'{group}_seed101/validation.json', RUN/f'{group}_seed101/training_result.json']
    return {p.as_posix(): digest(p) for p in paths if p.is_file()}


def build_prompt(data, group, split, index, mode):
    source = group if mode.startswith('native') else 'B'
    messages = data[source][split][index]['messages'][:-1]
    # No final answer or grader-provided teacher estimate may enter any prompt.
    for message in messages:
        if message['role'] != 'system':
            body = json.loads(message['content'])
            if any(k in body for k in ('final_answer', 'teacher_estimate_before_grading')):
                raise AssertionError('Grading information in diagnostic prompt.')
    if mode in ('original', 'prefilled_fixed'):
        messages = messages+[{'role': 'user', 'content': FINAL_INSTRUCTION}]
    prefix = ANSWER_PREFIX if 'prefilled' in mode else ''
    return messages, prefix


def final_supervision(tokenizer, example):
    messages = example['messages']
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if not np.array_equal(encoded['input_ids'], example['ids']):
        raise AssertionError('Prepared IDs differ from full training conversation.')
    last = messages[-1]['content']
    start = rendered.rindex(last)
    key = last.index('"final_answer":')
    value_start = key+len('"final_answer":')
    _, end = json.JSONDecoder().raw_decode(last, value_start)
    # Include any token overlapping the answer field (e.g. merged punctuation).
    indices = [i for i, (a, b) in enumerate(encoded['offset_mapping'])
               if a < start+end and b > start+key]
    if not indices or not np.array_equal(example['labels'][indices], example['ids'][indices]):
        raise AssertionError('Some final-answer tokens have no supervised label.')
    final_indices = [i for i, (a, b) in enumerate(encoded['offset_mapping'])
                     if start <= a < b <= start+len(last)+len('<|im_end|>')]
    return {'episode_key': example['entry']['key'], 'input_tokens': len(example['ids']),
            'all_supervised_tokens': int(np.count_nonzero(example['labels'][1:] != -100)),
            'answer_field_supervised_tokens': len(indices),
            'final_message_supervised_tokens': len(final_indices),
            'all_final_answer_tokens_supervised': True,
            'last_user_keys': sorted(json.loads(messages[-2]['content'])),
            'explicit_final_instruction_in_training': any(m['content'] == FINAL_INSTRUCTION for m in messages)}


def audit(tokenizer, data):
    experiment = read_json(RUN/'experiment.json')
    result = {'scope': 'sigma=2, completed seed101; no model updates', 'groups': {},
              'final_answers_identical_B_C': True, 'teacher_histories_are_diagnostic_only': True,
              'original_validation_adds_unseen_final_instruction': True,
              'original_validation_ends_with_two_user_messages': True,
              'C_training_contains_process_history_missing_at_fixed_evidence_evaluation': True}
    for split in ('train', 'validation'):
        for b, c in zip(data['B'][split], data['C'][split]):
            if json.loads(b['messages'][-1]['content'])['final_answer'] != json.loads(c['messages'][-1]['content'])['final_answer']:
                raise AssertionError('B/C gold answers differ.')
    for group in ('B', 'C'):
        episodes = [final_supervision(tokenizer, e) for e in data[group]['train']]
        steps = experiment['schedules']['101'][group]
        counts = {key: sum(episodes[i][key] for step in steps for i in step) for key in
                  ('input_tokens', 'all_supervised_tokens', 'answer_field_supervised_tokens', 'final_message_supervised_tokens')}
        info = read_json(RUN/f'{group}_seed101/training_result.json')
        assert counts['input_tokens'] == info['processed_tokens']
        assert counts['all_supervised_tokens'] == info['supervised_tokens']
        weights = [sum(episodes[i]['answer_field_supervised_tokens'] for i in step) /
                   sum(episodes[i]['all_supervised_tokens'] for i in step) for step in steps]
        counts.update(answer_presentations=sum(map(len, steps)), unique_worlds=len(episodes),
                      answer_token_fraction=counts['answer_field_supervised_tokens']/counts['all_supervised_tokens'],
                      mean_answer_fraction_of_update_loss=float(np.mean(weights)),
                      final_message_max_tokens=max(e['final_message_supervised_tokens'] for e in episodes),
                      final_messages_over_256=sum(e['final_message_supervised_tokens'] > 256 for e in episodes),
                      final_answer_field_max_tokens=max(e['answer_field_supervised_tokens'] for e in episodes))
        result['groups'][group] = {'schedule_counts': counts, 'per_update_answer_loss_fraction': weights, 'episodes': episodes}
    return result


def jobs():
    result = []
    for group in ('B', 'C'):
        for i in TRAIN_SEEN:
            result.append((group, 'train', i, 'original'))
        for split, indices in CONTRAST.items():
            for i in indices:
                for mode in ('shared_prefix', 'prefilled_fixed'):
                    result.append((group, split, i, mode))
                if group == 'C':
                    for mode in ('native_prefix', 'native_prefilled'):
                        result.append((group, split, i, mode))
    return result


def load_views():
    return {split: [v for v in (json.loads(line) for line in
            Path(f'data/research_curriculum_v1/views/{split}/B.jsonl').read_text(encoding='utf-8').splitlines())
            if v['noise_sigma'] == 2.0] for split in ('train', 'validation')}


def summarize(results):
    summaries = []
    for group, split, mode in sorted({(r['group'], r['split'], r['mode']) for r in results}):
        rows = [r for r in results if (r['group'], r['split'], r['mode']) == (group, split, mode)]
        summaries.append({'group': group, 'split': split, 'mode': mode, 'worlds': len(rows),
                          **{key: sum(bool(r.get(key, False)) for r in rows) for key in
                             ('valid_expression', 'support_success', 'prediction_success', 'full_recovery', 'null_world', 'false_discovery', 'hit_token_limit')},
                          'invalid_null_answers': sum(r['null_world'] and not r['valid_expression'] for r in rows)})
    return summaries


def cached_greedy(model, input_ids, max_new_tokens, eos_token_ids, chunk_size=512):
    """Read every prompt token in chunks, retaining all attention keys/values.

    Long full-prefix SDPA falls back to a quadratic allocation on this GPU.
    Chunking prompt queries bounds that workspace; keys/values are never dropped.
    Only used for the native-C long-history diagnostic, never as extra training.
    """
    cache = None
    for start in range(0, input_ids.shape[1], chunk_size):
        part = input_ids[:, start:start+chunk_size]
        end = start+part.shape[1]
        out = model(input_ids=part, past_key_values=cache,
                    attention_mask=torch.ones((1, end), device=input_ids.device, dtype=torch.long),
                    use_cache=True, logits_to_keep=1)
        cache, logits = out.past_key_values, out.logits[:, -1, :]
        del out
    generated = []
    penalty = model.generation_config.repetition_penalty
    processor = RepetitionPenaltyLogitsProcessor(penalty) if penalty != 1.0 else None
    for step in range(max_new_tokens):
        scores = logits.clone().float()
        if processor is not None:
            scores = processor(torch.cat([input_ids, *generated], dim=1), scores)
        token = scores.argmax(-1, keepdim=True)
        generated.append(token)
        if int(token.item()) in eos_token_ids or step+1 == max_new_tokens:
            break
        length = input_ids.shape[1]+step+1
        out = model(input_ids=token, past_key_values=cache,
                    attention_mask=torch.ones((1, length), device=input_ids.device, dtype=torch.long),
                    use_cache=True, logits_to_keep=1)
        cache, logits = out.past_key_values, out.logits[:, -1, :]
        del out
    return torch.cat([input_ids, *generated], dim=1)


def run(audit_only=False, resume=False):
    torch.set_num_threads(4)
    if not (RUN/'PAUSED.json').exists():
        raise RuntimeError('The diagnostic expects the full experiment to remain paused.')
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'results.jsonl').exists() and not audit_only and not resume:
        raise RuntimeError('Preserve existing diagnostic outputs; do not overwrite.')
    data = {g: {s: selected_data(PREPARED, g, s) for s in ('train', 'validation')} for g in ('B', 'C')}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    audit_result = audit(tokenizer, data)
    json_write(OUT/'supervision_audit.json', audit_result)
    print(json.dumps({g: v['schedule_counts'] for g, v in audit_result['groups'].items()}, indent=2), flush=True)
    if audit_only:
        return
    plan = jobs()
    before = protected_hashes()
    if resume:
        saved_plan = read_json(OUT/'plan.json')
        if before != saved_plan['protected_hashes_before']:
            raise AssertionError('Protected model/scoring files changed before resuming diagnostic.')
        if [tuple(j[k] for k in ('group', 'split', 'index', 'mode')) for j in saved_plan['jobs']] != plan:
            raise AssertionError('Diagnostic case selection changed.')
        amendment = OUT/'inference_memory_amendment.json'
        if not amendment.exists():
            json_write(amendment, {'prior_source_sha256': saved_plan['source_sha256'],
                       'resumed_source_sha256': digest(__file__),
                       'failure': 'First long native-C prompt attempted 20.88 GiB SDPA allocation; no answer produced.',
                       'completed_generations_preserved': 26,
                       'change': 'Native-C only: 512-token prompt chunks retaining full KV cache; greedy decoding and budget unchanged.',
                       'training_resumed': False})
    else:
        json_write(OUT/'plan.json', {'source_sha256': digest(__file__), 'model_updates': 0,
               'seed': 101, 'training_world_selection': 'first world of each of six families, sigma=2',
               'contrast_selection': 'first field, three-body and null world in each split',
               'jobs': [dict(zip(('group', 'split', 'index', 'mode'), job)) for job in plan],
               'max_new_tokens_including_fixed_prefix': 256, 'do_sample': False,
               'scoring': 'unchanged strict parser, independent clean-mean queries and frozen thresholds',
               'native_C_is_richer_history_not_fair_B_C_comparison': True,
               'formal_test_worlds_used': False, 'protected_hashes_before': before})
    # Labels are used only after each generation. The prompt builder never receives views.
    views = load_views()
    grammar = Grammar()
    thresholds = read_json('results/structure_discovery_noise/calibration.json')['thresholds']
    results = [json.loads(line) for line in (OUT/'results.jsonl').read_text(encoding='utf-8').splitlines()] if resume else []
    completed = {(r['group'], r['split'], r['index'], r['mode']) for r in results}
    if len(completed) != len(results) or not completed.issubset(set(plan)):
        raise AssertionError('Duplicate or unplanned diagnostic result.')
    features_cache = {}
    for group in ('B', 'C'):
        if all(job in completed for job in plan if job[0] == group):
            continue
        base = load_base(MODEL)
        model = PeftModel.from_pretrained(base, RUN/f'{group}_seed101/adapter', is_trainable=False)
        model.eval()
        model.requires_grad_(False)
        model.config.use_cache = True
        with torch.inference_mode():
            for g, split, index, mode in plan:
                if g != group or (g, split, index, mode) in completed:
                    continue
                messages, prefix = build_prompt(data, group, split, index, mode)
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)+prefix
                encoded = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to('cuda')
                prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False)) if prefix else 0
                budget = 256-prefix_tokens
                start = time.perf_counter()
                eos_ids = model.generation_config.eos_token_id
                eos_ids = [eos_ids] if isinstance(eos_ids, int) else eos_ids
                if mode.startswith('native'):
                    output = cached_greedy(model, encoded['input_ids'], budget, eos_ids)
                else:
                    output = model.generate(**encoded, max_new_tokens=budget, do_sample=False,
                                            pad_token_id=tokenizer.eos_token_id, use_cache=True)
                generated = output[0, encoded['input_ids'].shape[1]:]
                answer = prefix+tokenizer.decode(generated, skip_special_tokens=True)
                view = views[split][index]
                record = {'group': group, 'seed': 101, 'split': split, 'index': index, 'mode': mode,
                          'episode_id': view['episode_id'], 'input_tokens': encoded['input_ids'].shape[1],
                          'output_tokens': len(generated), 'fixed_prefix_tokens': prefix_tokens,
                          'hit_token_limit': len(generated) == budget and int(generated[-1]) not in eos_ids,
                          'prefill_chunk_tokens': 512 if mode.startswith('native') else None,
                          'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                          'text': answer, 'generation_seconds': time.perf_counter()-start,
                          'null_world': not view['final_answer']['expression']['terms'],
                          'richer_teacher_history': mode.startswith('native') and group == 'C'}
                try:
                    hypothesis = parse_expression(answer, grammar)
                    truth = expression_from_dict(view['final_answer']['expression'])
                    key = (split, index)
                    if key not in features_cache:
                        seed = [95102 if split == 'train' else 95101, index]
                        features_cache[key] = [compile_queries(grammar, q) for q in evaluation_queries(np.random.default_rng(seed), 256)]
                    record.update(valid_expression=True, **evaluate_noisy(grammar, truth, hypothesis, features_cache[key], thresholds))
                except (ValueError, KeyError, TypeError, IndexError) as error:
                    record.update(valid_expression=False, full_recovery=False, support_success=False,
                                  prediction_success=False, parsing_error=str(error))
                results.append(record)
                with (OUT/'results.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
                print(f'{len(results)}/{len(plan)} {group} {split} {index} {mode}: '
                      f'valid={record["valid_expression"]} recovered={record["full_recovery"]} '
                      f'tokens={record["output_tokens"]}', flush=True)
                del output, generated, encoded
                gc.collect()
                torch.cuda.empty_cache()
        del model, base
        gc.collect()
        torch.cuda.empty_cache()
    # Reuse already-completed validation, without regenerating or changing scores.
    for group in ('B', 'C'):
        for i, row in enumerate(read_json(RUN/f'{group}_seed101/validation.json')['results']):
            results.append(dict(row, group=group, seed=101, split='validation', index=i,
                                mode='original', reused_prior_validation=True))
    after = protected_hashes()
    if before != after:
        raise AssertionError('A protected training/checkpoint/scoring file changed during inference.')
    json_write(OUT/'results.json', {'model_updates': 0, 'new_generations': len(plan),
               'protected_files_unchanged': True, 'protected_hashes_after': after,
               'formal_test_worlds_used': False, 'summary': summarize(results), 'results': results})
    print('Diagnostics complete; full training remains paused.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Continue incomplete inference; never resumes training.')
    args = parser.parse_args()
    run(args.audit_only, args.resume)
