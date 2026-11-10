"""Prospective common prompts and lossless-for-grammar teaching views.

No model training or test-world access. A is restricted to eight observations;
B and C share exact prefix-to-structure targets. Process records have no gold.
"""
from copy import deepcopy
import json

from .student_laboratory import visible_queries
from .student_spins import local_query
from .student_v2 import auxiliary_messages as hexadecimal_auxiliary, hex_query, packed
from .student_v3 import SYSTEM
from .teaching import expression_from_dict


PREFIXES = (8, 16, 24, 32)
REQUEST = ('根据目前已经看到的测量，提出当前最有根据的额外作用的位置结构。'
           '只输出JSON，唯一键为structures，值为结构数组。每个结构是一至四个相对位置的数组，'
           '每个位置为两个整数；最多两个结构。没有额外作用则structures为空数组。'
           '不输出系数；共用程序只拟合你提出的结构，默认包含已知H0。'
           '这次只提交假设，不申请新测量，不输出解释或占位符。')


def evidence_prefix(view, used):
    if used not in PREFIXES:
        raise ValueError('Use a frozen 8/16/24/32 evidence checkpoint.')
    initial = view['initial']
    if len(initial['queries']) != 8 or len(initial['observed']) != 8:
        raise ValueError('Expected the eight common initial measurements.')
    observations = list(zip(initial['queries'], initial['observed']))
    observations += [(entry['query'], entry['observed']) for entry in view.get('experiments', [])]
    if len(observations) < used:
        raise ValueError('This group has not been given the requested evidence.')
    value = {'measurement_noise_sigma': view['noise_sigma'], 'used': used,
             'measurements': [{'query': local_query(hex_query(q)), 'delta': delta} for q, delta in observations[:used]]}
    visible_queries(value)
    return value


def hypothesis_prompt(evidence):
    # Validate and reconstruct a fresh envelope, preventing labels/IDs from
    # entering through metadata or nested measurements.
    visible_queries(evidence)
    if any(set(item) != {'query', 'delta'} for item in evidence['measurements']):
        raise ValueError('Unexpected observation metadata.')
    return [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': packed(evidence)+'\n'+REQUEST}]


def terminal_messages(view, used):
    label = expression_from_dict(view['final_answer']['expression']).support
    return hypothesis_prompt(evidence_prefix(view, used))+[
        {'role': 'assistant', 'content': packed({'structures': label})}]


def _localize(value):
    if isinstance(value, list):
        return [_localize(v) for v in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {'hex', 'position'}:
        return local_query(value)
    return {key: _localize(child) for key, child in value.items()}


def auxiliary_messages(view, group):
    if group not in ('B', 'C'):
        raise ValueError('A receives only initial evidence and correct final answers.')
    messages = hexadecimal_auxiliary(view, group)
    result = [{'role': 'system', 'content': SYSTEM+
               '\n以下是离线研究教学记录。数值为原记录的完整精度，未截断实验。'
               '表达式terms的每项为[系数,相对位置数组]，默认包含H0。'
               '说明文字字段的值"="表示沿用之前同名字段的文字。'
               '过程中的候选来自当时证据，不保证正确，也不保证仍然合理。'
               '已知噪声时预测是均值，不是每次观测必须精确等于的值。'}]
    result += [{'role': message['role'], 'content': packed(_localize(json.loads(message['content'])))}
               for message in messages[1:]]
    if any('"final_answer"' in message['content'] for message in result):
        raise AssertionError('Grader label leaked into research supervision.')
    return result


def load_group(source, split, group, sigma):
    from pathlib import Path
    if split not in ('train', 'validation') or group not in ('A', 'B', 'C') or sigma not in (0, 1):
        raise ValueError('Only the frozen teaching/development exports are supported.')
    path = Path(source)/'views'/split/(group+'.jsonl')
    return [v for v in (json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())
            if v['noise_sigma'] == sigma]
