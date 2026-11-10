"""Structure proposals plus coefficient fitting using the same visible evidence.

The fitter evaluates ONLY supplied structures. It neither enumerates candidate
features nor receives a world label, teacher hypothesis or additional query.
"""
import json

import numpy as np

from .physics import Expression, Grammar, SIDE, Term, base_flip, canonical_structure
from .student_spins import LOCAL_SYSTEM, OFFSETS, evidence_only, public_coverage
from .student_v2 import packed
from .teaching import expression_from_dict

SYSTEM = LOCAL_SYSTEM.replace(
    '能量增量的正负和系数都必须由观测判断。',
    '你负责提出作用位置结构；共用数值程序随后只用这些观测拟合你所提结构的系数，不能补加结构。')
FINAL_REQUEST = ('实验结束，32次测量已经用完。根据全部观测提交最终额外作用的位置结构。'
                 '只输出JSON，唯一键为structures，值为结构数组。'
                 '每个结构是一至四个相对位置的数组，每个位置为两个整数。'
                 '最多两个结构；没有额外作用则structures为空数组。'
                 '不输出系数；表达式默认包含已知H0。不要继续实验，不输出解释或占位符。')


def terminal_prompt(evidence):
    if set(evidence) != {'measurement_noise_sigma', 'used', 'measurements'}:
        raise ValueError('Only shared observations may enter the prompt.')
    return [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': packed(evidence)+'\n'+FINAL_REQUEST}]


def terminal_messages(view):
    support = expression_from_dict(view['final_answer']['expression']).support
    return terminal_prompt(evidence_only(view))+[
        {'role': 'assistant', 'content': packed({'structures': support})}]


def canonical_proposal(structures, grammar):
    if not isinstance(structures, list) or len(structures) > 2:
        raise ValueError('Expected at most two structures.')
    result = set()
    for structure in structures:
        if not isinstance(structure, list) or not 1 <= len(structure) <= 4:
            raise ValueError('Each structure must have one to four positions.')
        for point in structure:
            if (not isinstance(point, list) or len(point) != 2
                    or any(type(value) is not int for value in point)):
                raise ValueError('Position coordinates must be integers, not booleans or floats.')
        points = canonical_structure(structure)
        if not points or points not in grammar.index:
            raise ValueError('Constant or out-of-grammar structure.')
        result.add(points)
    return tuple(sorted(result))


def parse_structures(text, grammar=None):
    # No extraction of convenient fragments from an otherwise invalid answer.
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {'structures'}:
        raise ValueError('Answer must contain only structures.')
    return canonical_proposal(value['structures'], grammar or Grammar())


def observed_arrays(evidence):
    """Reconstruct a sufficient patch from the exact input shown to the model."""
    if set(evidence) != {'measurement_noise_sigma', 'used', 'measurements'}:
        raise ValueError('Fitter accepts only the public evidence envelope.')
    if evidence['used'] != 32 or len(evidence['measurements']) != 32:
        raise ValueError('Expected the existing 32 measurements, no new queries.')
    queries, observed = [], []
    for item in evidence['measurements']:
        patch = np.asarray(item['query']['local_spins'])
        if patch.shape != (5, 5) or not np.all((patch == -1) | (patch == 1)):
            raise ValueError('Invalid explicit spin patch.')
        value = item['delta']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError('Non-finite or nonnumeric measurement.')
        state = np.ones((SIDE, SIDE), dtype=np.int8)
        for i, dx in enumerate(OFFSETS):
            for j, dy in enumerate(OFFSETS):
                state[(2+dx) % SIDE, (2+dy) % SIDE] = patch[i, j]
        queries.append((state, (2, 2)))
        observed.append(float(value))
    return queries, np.asarray(observed)


def fit_coefficients(evidence, structures):
    grammar = public_coverage()
    canonical = canonical_proposal(json.loads(packed(structures)), grammar)
    queries, observed = observed_arrays(evidence)
    known = np.asarray([base_flip(*q) for q in queries])
    y = observed-known
    # Direct physics for proposed terms only: never build all candidate columns.
    columns = [np.asarray([Expression((Term(points, 1.0),)).delta(*q) for q in queries])-known
               for points in canonical]
    x = np.column_stack(columns) if columns else np.empty((len(y), 0))
    if columns:
        coefficients, _, rank, singular = np.linalg.lstsq(x, y, rcond=None)
    else:
        coefficients, rank, singular = np.empty(0), 0, np.empty(0)
    fitted = Expression(tuple(Term(p, float(k)) for p, k in zip(canonical, coefficients)))
    if not set(fitted.support).issubset(canonical):
        raise AssertionError('Numerical fitting must never add a structure.')
    diagnostics = {'measurements_used': len(y), 'extra_queries': 0, 'candidate_structures_added': 0,
                   'proposed_structures': canonical, 'fitted_coefficients_before_zero_removal': coefficients.tolist(),
                   'design_rank': int(rank), 'columns': len(canonical),
                   'identifiable_given_proposal': int(rank) == len(canonical),
                   'singular_values': singular.tolist(),
                   'observation_residual_rmse': float(np.sqrt(np.mean((x @ coefficients-y)**2))),
                   'zero_coefficient_structures_removed': len(canonical)-len(fitted.terms)}
    return fitted, diagnostics
