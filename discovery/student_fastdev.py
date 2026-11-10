"""Separate development input/loss variants; original study modules stay frozen."""
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from . import student_v3
from .student_v2 import packed


RESIDUAL_SYSTEM = (
    student_v3.SYSTEM.replace(
        'original_position只是原16×16周期格点位置；表达式仍写相对偏移。', '')
    + '观测已按已知H0预处理：u=(ΔE−ΔH0)/(2s_target)，'
      'scale=max(1,max|u|)，每条记录为[5×5自旋数组,u/scale]。'
      '最后一个数保留8位小数；normalized_noise_sigma是该数的已知噪声标准差。'
      '这里只扣除已知最近邻基础项，所有额外项仍需根据这些观测提出。'
)


def residual_evidence(evidence):
    # Validate the same public-only envelope and patches as the shared fitter.
    student_v3.observed_arrays(evidence)
    fields = []
    for item in evidence['measurements']:
        p = item['query']['local_spins']
        center = int(p[2][2])
        neighbours = int(p[1][2])+int(p[3][2])+int(p[2][1])+int(p[2][3])
        fields.append((float(item['delta'])-2*center*neighbours)/(2*center))
    scale = max(1.0, max(abs(x) for x in fields))
    rounded = [round(x/scale, 8) for x in fields]
    return {'used': evidence['used'], 'scale': scale,
            'normalized_noise_sigma': float(evidence['measurement_noise_sigma'])/(2*scale),
            'measurements': [[item['query']['local_spins'], u]
                             for item, u in zip(evidence['measurements'], rounded)]}


def terminal_prompt(evidence, variant):
    if variant == 'raw':
        return student_v3.terminal_prompt(evidence)
    if variant != 'residual':
        raise ValueError('Unknown development variant.')
    return [{'role': 'system', 'content': RESIDUAL_SYSTEM},
            {'role': 'user', 'content': packed(residual_evidence(evidence))+'\n'+student_v3.FINAL_REQUEST}]


def terminal_messages(view, variant):
    original = student_v3.terminal_messages(view)
    return terminal_prompt(student_v3.evidence_only(view), variant)+[original[-1]]


def resident_loss(model, ids, labels, head_chunk=128, offload=False):
    """Same masked CE and checkpoint scheme, with optional activation residency."""
    from train_student_pilot import offload_saved_activations
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    with offload_saved_activations() if offload else nullcontext():
        hidden = base.model(input_ids=ids, use_cache=False).last_hidden_state
    targets = labels[:, 1:].reshape(-1)
    valid = targets != -100
    states = hidden[:, :-1, :].reshape(-1, hidden.shape[-1])[valid]
    targets = targets[valid]
    if not len(targets):
        raise ValueError('No next-token assistant targets.')
    weight = base.lm_head.weight

    def chunk_loss(h, w, target):
        return F.cross_entropy(F.linear(h, w).float(), target, reduction='sum')

    total = torch.zeros((), device=hidden.device, dtype=torch.float32)
    for start in range(0, len(targets), head_chunk):
        state, target = states[start:start+head_chunk], targets[start:start+head_chunk]
        total = total+(checkpoint(chunk_loss, state, weight, target, use_reentrant=False)
                       if torch.is_grad_enabled() else chunk_loss(state, weight, target))
    return total, int(len(targets))
