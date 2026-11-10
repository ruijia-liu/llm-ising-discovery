"""Neutral wire-format practice, separate from mechanism discovery teaching."""
from copy import deepcopy

import numpy as np

from .active_student import PLAN_REQUEST, parse_plan
from .physics import Expression, random_queries
from .lab import Laboratory
from .student_laboratory import StudentLaboratory
from .student_v2 import packed
from .student_v3 import SYSTEM

FIELD = [[[0, 0]]]


def practice_cases(split):
    if split not in ('train', 'validation'):
        raise ValueError('No test or mechanism worlds belong in interface practice.')
    count, seed = (32, 98611) if split == 'train' else (16, 98612)
    rng = np.random.default_rng(seed)
    cases = []
    for index in range(count):
        finish = index % 4 == 0
        primary = [] if index % 2 else FIELD
        alternative = FIELD if index % 2 else []
        queries = [{'local_spins': rng.choice([-1, 1], size=(5, 5)).tolist(),
                    'position': rng.integers(0, 16, size=2).tolist()} for _ in range(1+index % 2)]
        reason = '按照给定计划提交构型，比较观测与已经锁定的预测。'
        update = '测量后重新检查当前解释；不符合时重新考虑结构与系数。'
        supplied = {'finish_requested': finish, 'current_structures': primary,
                    'given_alternative': alternative, 'given_queries': queries,
                    'given_reason': reason, 'given_update_condition': update, 'maximum_batch': 2}
        prompt = [{'role': 'system', 'content': SYSTEM},
                  {'role': 'user', 'content':
                   '这是共用接口练习，没有隐藏世界，不要求推断机制或选实验。'
                   '请按给定计划转换字段：finish_requested为true时结束；否则原样复制给定备选结构、'
                   '构型、理由和更新条件，按实验接口提交，不添加内容。\n'+packed(supplied)+'\n'+PLAN_REQUEST}]
        answer = {'action': 'finish'} if finish else {
            'action': 'experiment', 'alternative_structures': alternative, 'queries': queries,
            'reason': reason, 'update_condition': update}
        cases.append({'id': f'{split}_{index:02d}', 'prompt': prompt, 'answer': answer,
                      'current_structures': deepcopy(primary), 'initial_query_seed': seed+1000+index,
                      'messages': prompt+[{'role': 'assistant', 'content': packed(answer)}]})
    return cases


def prepare_plan(session, structures, text, maximum_batch):
    """Reuse actual deployed parsing and numerical commitment; no repairs."""
    plan = parse_plan(text, maximum_batch)
    if plan['action'] == 'finish':
        return plan, None
    commitment = session.prepare({'hypotheses': [{'structures': structures},
                                               {'structures': plan['alternative_structures']}],
                                  'queries': plan['queries'], 'reason': plan['reason'],
                                  'update_condition': plan['update_condition']})
    return plan, commitment


def neutral_session(case):
    # A harmless independent pure-H0 oracle used only to exercise the API.
    world = Expression()
    lab = Laboratory(world.delta, noise_sigma=0, noise_seed=0)
    initial = random_queries(np.random.default_rng(case['initial_query_seed']), 8)
    return StudentLaboratory(lab, initial)
