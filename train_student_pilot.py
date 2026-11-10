"""Local B/C LoRA pilot with full episodes and bounded-memory output loss.

Uses the sigma=2 slice of ALL 24 training worlds and 12 parameter-validation
worlds. Reserved test worlds are never opened. Each C seed sees every training
episode once; B repeats the same episodes to match processed tokens. Twelve
optimizer updates per arm, three seeds, no truncation or reward training.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, get_peft_model_state_dict, set_peft_model_state_dict

from discovery.physics import Expression, Grammar, Term
from discovery.noise_evaluation import evaluate_noisy
from discovery.teaching import expression_from_dict
from run_structure_discovery import compile_queries, evaluation_queries

SEEDS = (101, 102, 103)
CUDA_MEMORY_FRACTION = 0.8


@contextmanager
def offload_saved_activations():
    """Store checkpoint inputs in RAM, avoiding Windows GPU paging in backward.

    Frozen model weights and leaf tensors stay where they are. This changes
    storage only: pack/unpack preserves tensor dtype and numerical values.
    """
    def pack(tensor):
        if tensor.device.type == 'cuda' and not tensor.is_leaf and tensor.numel() > 4096:
            return tensor.device, tensor.detach().to('cpu')
        return tensor

    def unpack(saved):
        if isinstance(saved, tuple):
            device, tensor = saved
            return tensor.to(device)
        return saved

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def selected_data(prepared, group, split):
    prepared = Path(prepared)
    manifest = json.loads((prepared/'manifest.json').read_text(encoding='utf-8'))
    rows = [json.loads(line)['messages'] for line in (prepared/f'{split}_{group}_compact.jsonl').read_text(encoding='utf-8').splitlines()]
    entries = [e for e in manifest['entries'] if e['group'] == group and e['split'] == split]
    selected = [(entry, messages) for entry, messages in zip(entries, rows)
                if json.loads(messages[1]['content'])['measurement_noise_sigma'] == 2.0]
    expected = 24 if split == 'train' else 12
    if len(selected) != expected:
        raise ValueError('Unexpected number of paired heavy-noise episodes.')
    with np.load(prepared/'episodes.npz', allow_pickle=False) as data:
        return [{'entry': entry, 'messages': messages,
                 'ids': data[entry['key']+'_ids'].astype(np.int64),
                 'labels': data[entry['key']+'_labels'].astype(np.int64)} for entry, messages in selected]


def make_schedule(b_data, c_data, seed):
    """Whole episodes; exact update count, near-equal nonpadding input tokens."""
    rng = np.random.default_rng(seed)
    c_order = list(map(int, rng.permutation(len(c_data))))
    c_steps = [c_order[i:i+2] for i in range(0, len(c_order), 2)]
    b_steps, b_order, cursor = [], list(map(int, rng.permutation(len(b_data)))), 0
    c_total, b_total = 0, 0
    for c_step in c_steps:
        c_total += sum(len(c_data[i]['ids']) for i in c_step)
        b_step = []
        while True:
            if cursor == len(b_order):
                b_order = list(map(int, rng.permutation(len(b_data))))
                cursor = 0
            candidate = b_order[cursor]
            length = len(b_data[candidate]['ids'])
            if b_step and abs(b_total+length-c_total) > abs(b_total-c_total):
                break
            b_step.append(candidate)
            b_total += length
            cursor += 1
        b_steps.append(b_step)
    if set(i for step in b_steps for i in step) != set(range(len(b_data))):
        raise AssertionError('Some B training worlds were omitted.')
    if abs(b_total-c_total)/c_total > 0.01:
        raise AssertionError('Processed-token budgets differ by more than 1%.')
    return {'B': b_steps, 'C': c_steps, 'tokens_B': b_total, 'tokens_C': c_total,
            'relative_token_difference': (b_total-c_total)/c_total}


def load_base(model_path):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This local pilot requires a CUDA GPU with bfloat16 support.')
    torch.cuda.set_per_process_memory_fraction(CUDA_MEMORY_FRACTION)
    return AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                                               use_safetensors=True, dtype=torch.bfloat16,
                                               attn_implementation='sdpa').to('cuda')


def make_student(model_path, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    base = load_base(model_path)
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',
                        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'])
    model = get_peft_model(base, config)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    chunk_feed_forward(model)
    return model


def chunk_feed_forward(model, chunk_size=1024):
    """Recompute tokenwise MLP blocks in chunks; attention still sees all tokens."""
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    for layer in base.model.layers:
        original = layer.mlp.forward

        def forward(hidden, original=original):
            if not torch.is_grad_enabled() or hidden.shape[-2] <= chunk_size:
                return original(hidden)
            return torch.cat([checkpoint(original, part, use_reentrant=False)
                              for part in hidden.split(chunk_size, dim=-2)], dim=-2)

        layer.mlp.forward = forward


def bounded_loss(model, ids, labels, head_chunk=128, progress=None):
    """Exact assistant cross-entropy, recomputing small output-vocabulary blocks.

    Does NOT truncate attention/context. Avoids a [sequence, 151936] logits tensor
    that alone exceeds a laptop GPU's memory. Tests compare losses and gradients
    to the ordinary full-logit computation.
    """
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    with offload_saved_activations():
        hidden = base.model(input_ids=ids, use_cache=False).last_hidden_state
    if progress:
        progress('transformer_forward_complete')
    targets = labels[:, 1:].reshape(-1)
    valid = targets != -100
    states = hidden[:, :-1, :].reshape(-1, hidden.shape[-1])[valid]
    targets = targets[valid]
    if not len(targets):
        raise ValueError('No next-token assistant targets.')
    weight = base.lm_head.weight

    def chunk_loss(h, w, target):
        logits = F.linear(h, w)
        return F.cross_entropy(logits.float(), target, reduction='sum')

    total = torch.zeros((), device=hidden.device, dtype=torch.float32)
    for start in range(0, len(targets), head_chunk):
        state, target = states[start:start+head_chunk], targets[start:start+head_chunk]
        if torch.is_grad_enabled():
            total = total+checkpoint(chunk_loss, state, weight, target, use_reentrant=False)
        else:
            total = total+chunk_loss(state, weight, target)
    if progress:
        progress('output_loss_complete')
    return total, int(len(targets))


def tensors(episode):
    return (torch.tensor(episode['ids'], device='cuda')[None],
            torch.tensor(episode['labels'], device='cuda')[None])


def preflight(args):
    torch.set_num_threads(4)
    data = selected_data(args.data, 'C', 'train')
    episode = max(data, key=lambda e: len(e['ids']))
    model = make_student(args.model, 101)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    def progress(stage):
        torch.cuda.synchronize()
        print(json.dumps({'stage': stage, 'seconds': time.perf_counter()-start,
                          'allocated_gib': torch.cuda.memory_allocated()/2**30,
                          'peak_gib': torch.cuda.max_memory_allocated()/2**30}), flush=True)
    progress('starting')
    loss_sum, n = bounded_loss(model, *tensors(episode), args.head_chunk, progress)
    loss = loss_sum/n
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
        raise RuntimeError('Non-finite training preflight.')
    optimizer.step()
    torch.cuda.synchronize()
    result = {'status': 'passed', 'episode': episode['entry']['key'], 'context_tokens': len(episode['ids']),
              'assistant_tokens': n, 'loss': float(loss.detach()), 'gradient_norm': float(gradient_norm),
              'seconds': time.perf_counter()-start, 'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
              'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
              'note': 'Preflight update discarded; all actual arms restart from the same base.'}
    Path(args.output).mkdir(parents=True, exist_ok=True)
    json_write(Path(args.output)/'preflight.json', result)
    print(json.dumps(result, indent=2), flush=True)


def train_arm(args, group, seed, schedule, episodes):
    folder = Path(args.output)/f'{group}_seed{seed}'
    checkpoint_path = folder/'step_checkpoint.pt'
    if folder.exists() and not checkpoint_path.exists():
        raise ValueError('An arm output already exists; never overwrite a completed or partial run.')
    folder.mkdir(parents=True, exist_ok=True)
    model = make_student(args.model, seed)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.0)
    token_total = supervised_total = 0
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    step_records = []
    previous_peak = 0.0
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if (saved['group'], saved['seed'], saved['schedule']) != (group, seed, schedule[group]):
            raise ValueError('Checkpoint does not belong to this exact arm/schedule.')
        restore_training_state(model, optimizer, saved)
        step_records = saved['step_records']
        token_total, supervised_total = saved['processed_tokens'], saved['supervised_tokens']
        start -= saved['elapsed_seconds']
        previous_peak = saved['peak_allocated_gib']
        log = folder/'training.jsonl'
        if log.exists():
            (folder/f'training_before_resume_{time.time_ns()}.jsonl').write_bytes(log.read_bytes())
        log.write_text(''.join(json.dumps(r)+'\n' for r in step_records), encoding='utf-8')
        print(f'Resuming {group} seed={seed} after completed step {len(step_records)}', flush=True)
        del saved
        gc.collect()
        torch.cuda.empty_cache()
    for step, indices in enumerate(schedule[group], 1):
        if step <= len(step_records):
            continue
        optimizer.zero_grad(set_to_none=True)
        normalizer = sum(int(np.count_nonzero(episodes[i]['labels'][1:] != -100)) for i in indices)
        value = 0.0
        for index in indices:
            episode = episodes[index]
            loss_sum, supervised = bounded_loss(model, *tensors(episode), args.head_chunk)
            (loss_sum/normalizer).backward()
            value += float(loss_sum.detach())
            token_total += len(episode['ids'])
            supervised_total += supervised
            del loss_sum
            # Non-reentrant checkpoint frames can retain cyclic references until
            # Python GC runs. Clear them between variable-length full episodes,
            # then release unused cached blocks before the next SDPA workspace.
            gc.collect()
            torch.cuda.empty_cache()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not np.isfinite(value) or not torch.isfinite(gradient_norm):
            raise RuntimeError('Non-finite loss or gradient; arm incomplete.')
        optimizer.step()
        torch.cuda.synchronize()
        record = {'step': step, 'episode_indices': indices, 'loss': value/normalizer,
                  'gradient_norm': float(gradient_norm), 'processed_tokens': token_total,
                  'supervised_tokens': supervised_total, 'elapsed_seconds': time.perf_counter()-start}
        step_records.append(record)
        saved = capture_training_state(model, optimizer)
        saved.update(group=group, seed=seed, schedule=schedule[group], step_records=step_records,
                     processed_tokens=token_total, supervised_tokens=supervised_total,
                     elapsed_seconds=time.perf_counter()-start,
                     peak_allocated_gib=max(previous_peak, torch.cuda.max_memory_allocated()/2**30))
        temporary = folder/'step_checkpoint.tmp'
        torch.save(saved, temporary)
        temporary.replace(checkpoint_path)
        del saved
        with (folder/'training.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record)+'\n')
        print(f'{group} seed={seed} step={step}/{len(schedule[group])} loss={record["loss"]:.4f} '
              f'tokens={token_total} elapsed={record["elapsed_seconds"]:.1f}s', flush=True)
    model.save_pretrained(folder/'adapter', safe_serialization=True)
    info = {'group': group, 'seed': seed, 'completed': True, 'optimizer_updates': len(step_records),
            'processed_tokens': token_total, 'supervised_tokens': supervised_total,
            'episode_presentations': sum(len(step) for step in schedule[group]),
            'unique_training_episodes': len(set(i for step in schedule[group] for i in step)),
            'training_seconds': time.perf_counter()-start,
            'peak_allocated_gib': max(previous_peak, torch.cuda.max_memory_allocated()/2**30),
            'trainable_parameters': sum(p.numel() for p in trainable)}
    json_write(folder/'training_result.json', info)
    del model, optimizer, trainable
    gc.collect()
    torch.cuda.empty_cache()
    return info


def capture_training_state(model, optimizer):
    numpy_rng = np.random.get_state()
    return {'adapter': {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()},
            'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'python_rng': random.getstate(),
            'numpy_rng': (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:])}


def restore_training_state(model, optimizer, saved):
    set_peft_model_state_dict(model, saved['adapter'])
    optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['torch_rng'])
    if saved['cuda_rng']:
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    random.setstate(saved['python_rng'])
    state = saved['numpy_rng']
    np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), *state[2:]))


FINAL_INSTRUCTION = ('实验已结束，禁止继续查询。根据上面的全部观测推断额外作用。'
                     '只输出合法JSON，对象的嵌套键依次为final_answer、expression、terms。'
                     'terms是数组，每项是二元素数组，第一元素是数值系数，第二元素是相对位置数组。'
                     '相对位置数组中的每个元素都是两个整数，分别为行偏移和列偏移。'
                     '最多两项，每项一至四个位置。没有额外作用时terms为空数组。'
                     '必须填写实际数值；不写省略号、占位词、Markdown或解释文字。')


def parse_expression(text, grammar):
    stripped = text.strip()
    if stripped.startswith('```'):
        stripped = stripped.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    raw = json.loads(stripped)
    value = raw['final_answer']['expression']
    terms = value['terms']
    if len(terms) > 2:
        raise ValueError('Too many terms.')
    parsed = []
    for term in terms:
        if isinstance(term, dict):
            coefficient, points = term['coefficient'], term['offsets']
        else:
            coefficient, points = term
        if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)) or not np.isfinite(coefficient):
            raise ValueError('Invalid coefficient.')
        if not 1 <= len(points) <= 4:
            raise ValueError('Invalid order.')
        parsed.append(Term(tuple(tuple(p) for p in points), float(coefficient)))
    expression = Expression(tuple(parsed))
    grammar.coefficients(expression)  # Reject out-of-grammar shapes.
    return expression


def evaluate_arm(args, group, seed=None):
    folder = Path(args.output)/(f'{group}_seed{seed}' if seed is not None else 'untrained_base')
    folder.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = load_base(args.model)
    model = PeftModel.from_pretrained(base, folder/'adapter', is_trainable=False) if seed is not None else base
    model.eval()
    model.config.use_cache = True
    examples = selected_data(args.data, 'B', 'validation')
    views = [json.loads(line) for line in Path('data/research_curriculum_v1/views/validation/B.jsonl').read_text(encoding='utf-8').splitlines()]
    views = [v for v in views if v['noise_sigma'] == 2.0]
    thresholds = json.loads(Path('results/structure_discovery_noise/calibration.json').read_text(encoding='utf-8'))['thresholds']
    grammar = Grammar()
    results = []
    with torch.inference_mode():
        for i, (example, view) in enumerate(zip(examples, views)):
            prompt = example['messages'][:-1]+[{'role': 'user', 'content': FINAL_INSTRUCTION}]
            ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors='pt').to('cuda')
            start = time.perf_counter()
            output = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=256, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id, use_cache=True)
            text = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
            result = {'episode_id': view['episode_id'], 'input_tokens': ids.shape[1],
                      'output_tokens': output.shape[1]-ids.shape[1], 'text': text,
                      'generation_seconds': time.perf_counter()-start,
                      'null_world': not view['final_answer']['expression']['terms']}
            try:
                hypothesis = parse_expression(text, grammar)
                truth = expression_from_dict(view['final_answer']['expression'])
                # Fresh evaluator-only configurations, fixed across models and seeds.
                queries = evaluation_queries(np.random.default_rng([95101, i]), 256)
                features = [compile_queries(grammar, q) for q in queries]
                result.update(valid_expression=True, **evaluate_noisy(grammar, truth, hypothesis, features, thresholds))
            except (ValueError, KeyError, TypeError, IndexError) as error:
                result.update(valid_expression=False, full_recovery=False, support_success=False,
                              prediction_success=False, parsing_error=str(error))
            results.append(result)
            print(f'validation {group} seed={seed}: {i+1}/12 valid={result["valid_expression"]} '
                  f'success={result["full_recovery"]}', flush=True)
    summary = {'group': group, 'seed': seed, 'scope': 'fixed-evidence parameter-validation, sigma=2, 12 worlds',
               'formal_test_worlds_used': False, 'free_exploration_evaluated': False,
               'valid_expressions': sum(r['valid_expression'] for r in results),
               'full_recoveries': sum(r['full_recovery'] for r in results),
               'correct_structures': sum(r['support_success'] for r in results),
               'prediction_successes': sum(r['prediction_success'] for r in results),
               'null_worlds': sum(r['null_world'] for r in results),
               'false_discoveries': sum(r.get('false_discovery', False) for r in results),
               'invalid_null_answers': sum(r['null_world'] and not r['valid_expression'] for r in results),
               'worlds': len(results), 'results': results}
    json_write(folder/'validation.json', summary)
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'experiment.json').exists():
        raise ValueError('Existing experiment; use a new directory rather than overwrite.')
    torch.set_num_threads(4)
    b, c = selected_data(args.data, 'B', 'train'), selected_data(args.data, 'C', 'train')
    schedules = {str(seed): make_schedule(b, c, seed) for seed in SEEDS}
    prepared = json.loads((Path(args.data)/'manifest.json').read_text(encoding='utf-8'))
    config = {'model_id': prepared['model_id'], 'model_revision': prepared['model_revision'],
              'seeds': list(SEEDS), 'groups': ['B', 'C'], 'noise_sigma': 2.0, 'training_worlds': 24,
              'optimizer_updates_per_arm': 12, 'lr': 2e-4, 'lora_rank': 8, 'lora_alpha': 16, 'lora_dropout': 0.05,
              'dtype': 'bfloat16', 'head_loss_chunk': args.head_chunk,
              'checkpoint_activations_in_ram': True, 'mlp_chunk_tokens': 1024,
              'cuda_memory_fraction': CUDA_MEMORY_FRACTION,
              'loss_mask': prepared['loss'], 'token_budget_primary': 'processed nonpadding input tokens, difference <=1%',
              'supervised_token_budgets_identical': False, 'full_context': True, 'truncation': False,
              'inference': {'max_new_tokens': 256, 'do_sample': False, 'final_instruction': FINAL_INSTRUCTION},
              'validation_only': True, 'formal_test_worlds_used': False, 'schedules': schedules,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'prepared_manifest_sha256': hashlib.sha256((Path(args.data)/'manifest.json').read_bytes()).hexdigest()}
    config['evaluation_calibration_sha256'] = hashlib.sha256(
        Path('results/structure_discovery_noise/calibration.json').read_bytes()).hexdigest()
    config['validation_labels_sha256'] = hashlib.sha256(
        Path('data/research_curriculum_v1/views/validation/B.jsonl').read_bytes()).hexdigest()
    config['environment'] = {'packages': {name: importlib.metadata.version(name) for name in
                                         ('torch', 'transformers', 'peft', 'numpy', 'safetensors')},
                             'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}
    json_write(output/'experiment.json', config)
    summaries = [evaluate_arm(args, 'untrained_base')]
    training = []
    for seed in SEEDS:
        order = ('B', 'C') if seed % 2 else ('C', 'B')
        for group in order:
            training.append(train_arm(args, group, seed, schedules[str(seed)], b if group == 'B' else c))
            summaries.append(evaluate_arm(args, group, seed))
            json_write(output/'progress.json', {'completed_training': training, 'validation': summaries})
    json_write(output/'results.json', {'config': config, 'training': training, 'validation': summaries})


def resume(args):
    """Resume completed arms; preserve and restart an incomplete arm from base.

    Run only after the previous process has ended. Training settings and saved
    episode schedules are authoritative; this does not load partial weights.
    """
    output = Path(args.output)
    config = json.loads((output/'experiment.json').read_text(encoding='utf-8'))
    if (output/'results.json').exists():
        raise ValueError('This experiment is already complete.')
    current_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    original = output/'train_student_pilot_original.py'
    if current_hash != config['source_sha256']:
        if not original.exists() or hashlib.sha256(original.read_bytes()).hexdigest() != config['source_sha256']:
            raise ValueError('The originally executed source must be preserved before an implementation amendment.')
    if hashlib.sha256((Path(args.data)/'manifest.json').read_bytes()).hexdigest() != config['prepared_manifest_sha256']:
        raise ValueError('Prepared curriculum changed.')
    if args.head_chunk != config['head_loss_chunk'] or FINAL_INSTRUCTION != config['inference']['final_instruction']:
        raise ValueError('Do not change training/inference parameters while resuming.')
    model_manifest = json.loads((Path(args.model)/'download_manifest.json').read_text(encoding='utf-8'))
    if (model_manifest['model_id'], model_manifest['revision']) != (config['model_id'], config['model_revision']):
        raise ValueError('Base model changed.')
    torch.set_num_threads(4)
    b, c = selected_data(args.data, 'B', 'train'), selected_data(args.data, 'C', 'train')
    summary = json.loads((output/'untrained_base'/'validation.json').read_text(encoding='utf-8'))
    previous_amendment_path = output/'implementation_amendment.json'
    previous_amendment = json.loads(previous_amendment_path.read_text(encoding='utf-8')) if previous_amendment_path.exists() else None
    summaries, training, restarted = [summary], [], []
    if previous_amendment:
        restarted.extend(previous_amendment.get('restarted_partial_arms', []))
    amendment = {'original_source_sha256': config['source_sha256'], 'resumed_source_sha256': current_hash,
                 'change': 'Collect checkpoint cycles; clear CUDA cache per episode; use 80% allocation cap; save step checkpoints with optimizer and RNG.',
                 'cuda_memory_fraction': CUDA_MEMORY_FRACTION,
                 'curriculum_or_training_budget_changed': False, 'restarted_partial_arms': restarted,
                 'previous_implementation_amendment': previous_amendment}
    json_write(output/'implementation_amendment.json', amendment)
    for seed in config['seeds']:
        for group in (('B', 'C') if seed % 2 else ('C', 'B')):
            folder = output/f'{group}_seed{seed}'
            completed = folder/'training_result.json'
            if completed.exists():
                info = json.loads(completed.read_text(encoding='utf-8'))
                if not info['completed'] or info['optimizer_updates'] != 12:
                    raise ValueError('Invalid completed-arm receipt.')
            else:
                if folder.exists() and not (folder/'step_checkpoint.pt').exists():
                    archive = output/'interrupted_attempts'
                    archive.mkdir(exist_ok=True)
                    destination = archive/f'{group}_seed{seed}_{time.time_ns()}'
                    if not folder.resolve().is_relative_to(output.resolve()) or not destination.resolve().is_relative_to(output.resolve()):
                        raise ValueError('Archive paths must stay inside the experiment directory.')
                    folder.rename(destination)
                    restarted.append(str(destination))
                    json_write(output/'implementation_amendment.json', amendment)
                info = train_arm(args, group, seed, config['schedules'][str(seed)], b if group == 'B' else c)
            training.append(info)
            validation = folder/'validation.json'
            summaries.append(json.loads(validation.read_text(encoding='utf-8')) if validation.exists()
                             else evaluate_arm(args, group, seed))
            json_write(output/'progress.json', {'completed_training': training, 'validation': summaries})
    json_write(output/'implementation_amendment.json', amendment)
    json_write(output/'results.json', {'config': config, 'training': training, 'validation': summaries,
                                       'implementation_amendment': amendment})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'run', 'resume'])
    parser.add_argument('--model', default='models/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--data', default='data/student_training_v1')
    parser.add_argument('--output', default='results/student_training_v1')
    parser.add_argument('--head-chunk', type=int, default=128)
    args = parser.parse_args()
    if args.command == 'preflight':
        preflight(args)
    elif args.command == 'resume':
        resume(args)
    else:
        run(args)
