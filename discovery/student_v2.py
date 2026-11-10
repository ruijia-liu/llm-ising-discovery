"""Separate evidence-to-answer supervision from B/C record supervision."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .student_format import compact_messages, tokenize_assistant_only
from .teaching import PUBLIC_RULES, messages_for_view

TRAIN_IDS = ('w0000', 'w0004', 'w0008', 'w0012', 'w0016', 'w0020')
FINAL_REQUEST = ('实验结束，32次测量已经用完。根据全部观测提交最终额外作用机制。'
                 '只输出JSON，嵌套键依次为final_answer、expression、terms。'
                 'terms的每项为[数值系数,相对位置数组]，每个位置为两个整数。'
                 '最多两项，每项一至四个位置。没有额外作用则terms为空数组。'
                 '表达式默认包含已知H0。不要继续实验，不输出过程记录、占位符或解释。')
SYSTEM = ('你要从广义Ising翻转测量记录推断额外相互作用。H0是周期边界最近邻耦合，系数为1。'
          '构型hex是按行排列的256位自旋，1表示+1，0表示-1；position是翻转位置。'
          '能量增量的正负和系数都必须由观测判断。表达式位置是相对偏移，不是测量位置。'
          '公共表达语言和预算规则：'+json.dumps(PUBLIC_RULES, ensure_ascii=False, separators=(',', ':')))


def packed(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def load_views(split, group='B', sigma=0.0, source='data/research_curriculum_v1'):
    if split not in ('train', 'validation') or group not in ('B', 'C'):
        raise ValueError('Only teaching train/parameter-validation views may be opened.')
    path = Path(source)/'views'/split/(group+'.jsonl')
    return [v for v in (json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())
            if v['noise_sigma'] == sigma]


def hex_query(query):
    bits = ''.join(query['rows']).replace('+', '1').replace('-', '0')
    if len(bits) != 256 or set(bits)-set('01'):
        raise ValueError('Malformed spin configuration.')
    return {'hex': f'{int(bits, 2):064x}', 'position': list(query['position'])}


def evidence_only(view):
    measurements = [{'query': hex_query(q), 'delta': value} for q, value in
                    zip(view['initial']['queries'], view['initial']['observed'])]
    measurements += [{'query': hex_query(e['query']), 'delta': e['observed']} for e in view['experiments']]
    if len(measurements) != 32:
        raise ValueError('Expected exactly the original 32 measurements.')
    return {'measurement_noise_sigma': view['noise_sigma'], 'used': 32, 'measurements': measurements}


def terminal_prompt(evidence):
    """Does not accept any label, family, world ID, or teacher hypothesis."""
    if set(evidence) != {'measurement_noise_sigma', 'used', 'measurements'}:
        raise ValueError('Only common evidence may enter the terminal task.')
    return [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': packed(evidence)+'\n'+FINAL_REQUEST}]


def terminal_messages(view):
    terms = [[t['coefficient'], t['offsets']] for t in view['final_answer']['expression']['terms']]
    return terminal_prompt(evidence_only(view))+[{'role': 'assistant', 'content':
            packed({'final_answer': {'expression': {'terms': terms}}})}]


def auxiliary_messages(view, group):
    """Teacher/action records have no gold terminal label at all.

    C's last after-observation update is moved into a separate process message,
    before any grader answer could be seen. B receives the same observations.
    This task is separate from the common evidence-to-answer task.
    """
    record = messages_for_view(view, group)['messages'][:-1]
    if group == 'C':
        record.append({'role': 'assistant', 'content': packed({
            'update_from_last_observation': deepcopy(view['research_process'][-1]['after']),
            'teacher_estimate_before_grading': deepcopy(view['teacher_final_before_grading'])})})
    if any('"final_answer"' in m['content'] for m in record):
        raise AssertionError('Gold answer leaked into the auxiliary task.')
    return compact_messages(record)


def encode(tokenizer, messages):
    ids, labels = tokenize_assistant_only(tokenizer, messages)
    if len(ids) > 32768:
        raise ValueError('Full task exceeds context; do not truncate.')
    return {'messages': messages, 'ids': np.array(ids, dtype=np.int64),
            'labels': np.array(labels, dtype=np.int64)}


def objective_weights(group):
    if group not in ('B', 'C', 'common'):
        raise ValueError('Unknown group.')
    return {'terminal': 1.0, 'auxiliary': 0.0} if group == 'common' else {'terminal': 0.5, 'auxiliary': 0.5}


def normalized_task_loss(terminal_sum, terminal_tokens, auxiliary_sum=None, auxiliary_tokens=0, group='common'):
    weights = objective_weights(group)
    if terminal_tokens <= 0:
        raise ValueError('Missing terminal supervision.')
    value = weights['terminal']*terminal_sum/terminal_tokens
    if weights['auxiliary']:
        if auxiliary_sum is None or auxiliary_tokens <= 0:
            raise ValueError('Missing auxiliary supervision.')
        value = value+weights['auxiliary']*auxiliary_sum/auxiliary_tokens
    return value
