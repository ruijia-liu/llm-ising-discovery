"""Persistent model-only JSONL worker. No oracle, dataset or grading imports."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--adapter')
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from peft import PeftModel
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.8)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, trust_remote_code=False,
             use_safetensors=True, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    model.eval()
    print(json.dumps({'ready': True, 'model': str(Path(args.model).resolve()), 'adapter': args.adapter}), flush=True)
    for line in sys.stdin:
        started = time.perf_counter()
        try:
            request = json.loads(line)
            if set(request) != {'messages', 'max_new_tokens'} or type(request['max_new_tokens']) is not int or not 1 <= request['max_new_tokens'] <= 2048:
                raise ValueError('Invalid inference request.')
            if not isinstance(request['messages'], list) or any(set(m) != {'role', 'content'} or
                    m['role'] not in ('system', 'user', 'assistant') or not isinstance(m['content'], str) for m in request['messages']):
                raise ValueError('Only plain text chat messages are supported.')
            ids = tokenizer.apply_chat_template(request['messages'], tokenize=True, add_generation_prompt=True,
                                                return_tensors='pt').to('cuda')
            if ids.shape[1]+request['max_new_tokens'] > 32768:
                raise ValueError('Context limit exceeded; no truncation allowed.')
            config = GenerationConfig(max_new_tokens=request['max_new_tokens'], do_sample=False,
                                      repetition_penalty=1.0, eos_token_id=tokenizer.eos_token_id,
                                      pad_token_id=tokenizer.eos_token_id, use_cache=True)
            with torch.inference_mode():
                output = model.generate(ids, attention_mask=torch.ones_like(ids), generation_config=config)
            torch.cuda.synchronize()
            reply = {'text': tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True),
                     'input_tokens': int(ids.shape[1]), 'output_tokens': int(output.shape[1]-ids.shape[1]),
                     'generation_seconds': time.perf_counter()-started}
        except Exception as error:
            reply = {'error': type(error).__name__+': '+str(error)}
        print(json.dumps(reply, ensure_ascii=True, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
