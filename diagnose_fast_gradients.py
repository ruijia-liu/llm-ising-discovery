"""Investigate the failed residency equivalence check; never train or score."""
import gc
from pathlib import Path

import torch
from transformers import AutoTokenizer

from build_publication_curriculum import atomic_json
from discovery.student_fastdev import resident_loss
from discovery.student_v2 import encode, load_views
from discovery.student_v3 import terminal_messages
from train_student_pilot import make_student, bounded_loss, tensors


def run(output='results/fast_gradient_diagnostic_v1'):
    output = Path(output)
    if output.exists():
        raise ValueError('Preserve prior diagnosis.')
    output.mkdir()
    (output/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained('models/Qwen2.5-0.5B-Instruct', local_files_only=True)
    examples = [encode(tokenizer, terminal_messages(v)) for v in load_views('train')]
    example = max(examples, key=lambda x: len(x['ids']))
    model = make_student('models/Qwen2.5-0.5B-Instruct', 301)
    model.train()
    results = []
    for dropout in (True, False):
        if not dropout:
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.0
        reference = None
        previous_by_kind = {}
        for name, fn in (('offload', bounded_loss), ('offload_copy', lambda *x: resident_loss(*x, offload=True)),
                         ('resident', resident_loss), ('resident', resident_loss), ('offload', bounded_loss)):
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.manual_seed(912)
            torch.cuda.manual_seed_all(912)
            total, count = fn(model, *tensors(example))
            loss = total/count
            loss.backward()
            gradients = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
            reference = gradients if reference is None else reference
            maxdiffs = sorted(((n, float((g-reference[n]).abs().max())) for n, g in gradients.items()), key=lambda x: -x[1])
            norm = sum(float(g.square().sum()) for g in gradients.values())**.5
            diffnorm = sum(float((g-reference[n]).square().sum()) for n, g in gradients.items())**.5
            row = {'dropout': dropout, 'path': name, 'loss': float(loss.detach()), 'grad_norm': norm,
                   'relative_l2_difference_from_first': diffnorm/norm, 'largest_differences': maxdiffs[:5]}
            if name in previous_by_kind:
                row['same_path_repeat_max_difference'] = max(float((g-previous_by_kind[name][n]).abs().max())
                                                            for n, g in gradients.items())
            previous_by_kind[name] = gradients
            results.append(row)
            atomic_json(output/'result.json', results)
            print(row, flush=True)
            del total, loss


if __name__ == '__main__':
    run()
