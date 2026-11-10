"""Explicit local spins sufficient for EVERY term in the public grammar.

The public grammar is any subset of a translated 3x3 window. All positions
entering any allowed single-spin flip are within [-2,2]^2 of the target.
No actual world, coefficient, fitted expression or outcome selects the crop.
"""
import numpy as np

from .physics import Grammar, SIDE
from .student_v2 import FINAL_REQUEST, SYSTEM, packed, evidence_only as hex_evidence

RADIUS = 2
OFFSETS = list(range(-RADIUS, RADIUS+1))
LOCAL_SYSTEM = SYSTEM.replace(
    '构型hex是按行排列的256位自旋，1表示+1，0表示-1；position是翻转位置。',
    '每次测量给出目标自旋周围5×5的明确自旋值，元素为-1或+1。'
    '行列相对偏移均依次为-2,-1,0,1,2，中心[2][2]是被翻转自旋。'
    '公共3×3作用结构的所有平移项，其翻转能量只依赖这25个位置。'
    'original_position只是原16×16周期格点位置；表达式仍写相对偏移。')


def public_coverage():
    grammar = Grammar()
    displacements = grammar.offsets[grammar.factors]
    centred = (displacements+SIDE//2) % SIDE-SIDE//2
    if np.max(np.abs(centred)) > RADIUS:
        raise ValueError('Public grammar changed; local encoding is no longer sufficient.')
    return grammar


def decode_hex_query(query):
    bits = f"{int(query['hex'], 16):0256b}"
    return np.array([1 if b == '1' else -1 for b in bits], dtype=np.int8).reshape(SIDE, SIDE), tuple(query['position'])


def local_query(query):
    state, (x, y) = decode_hex_query(query)
    return {'local_spins': [[int(state[(x+dx) % SIDE, (y+dy) % SIDE]) for dy in OFFSETS] for dx in OFFSETS],
            'original_position': [x, y]}


def evidence_only(view):
    public_coverage()
    evidence = hex_evidence(view)
    return {**evidence, 'measurements': [{'query': local_query(r['query']), 'delta': r['delta']}
                                        for r in evidence['measurements']]}


def terminal_prompt(evidence):
    if set(evidence) != {'measurement_noise_sigma', 'used', 'measurements'}:
        raise ValueError('Only common evidence may enter the terminal task.')
    return [{'role': 'system', 'content': LOCAL_SYSTEM},
            {'role': 'user', 'content': packed(evidence)+'\n'+FINAL_REQUEST}]


def terminal_messages(view):
    terms = [[t['coefficient'], t['offsets']] for t in view['final_answer']['expression']['terms']]
    return terminal_prompt(evidence_only(view))+[{'role': 'assistant', 'content':
            packed({'final_answer': {'expression': {'terms': terms}}})}]
