"""Short action schema with no concrete hypothesis or experiment example."""
from copy import deepcopy

from .interface_alignment import practice_cases
from .student_mainline import hypothesis_prompt
from .student_v2 import packed
from .student_v3 import SYSTEM

ACTION_SCHEMA = (
    '只输出一个JSON对象，不要Markdown或其他文字。'
    '结束的对象只能有action键，值为finish。'
    '实验对象必须且只能有action、alternative_structures、queries、reason、update_condition五个键。'
    '实验的action值为experiment；alternative_structures是一组相对位置结构，必须不同于current_structures。'
    'queries是1到maximum_batch个测量的数组，每项只有local_spins和position。'
    'local_spins为5行5列的整数，每个值为-1或1；position为两个0到15的整数。'
    'reason和update_condition是非空字符串。不要省略、增减或改名所需字段。'
)


def practice_cases_v2(split):
    cases = deepcopy(practice_cases(split))
    for case in cases:
        # Exact same supplied values and targets; only the task wording changes.
        old_content = case['prompt'][1]['content']
        supplied = old_content.split('\n', 2)[1]
        case['prompt'] = [{'role': 'system', 'content':
                           '你负责接口格式转换。题目已经给出决定和全部字段，必须照给定计划执行，不自行选实验或结束。'},
                          {'role': 'user', 'content': ACTION_SCHEMA+'\n给定计划：\n'+supplied+
                           '\n转换规则：finish_requested为true时只输出结束对象。'
                           '为false时必须输出实验对象：alternative_structures原样取given_alternative；'
                           'queries原样取given_queries中的全部测量；reason原样取given_reason；'
                           'update_condition原样取given_update_condition。current_structures仅供校验，不是备选结构。'}]
        case['messages'] = case['prompt']+[{'role': 'assistant', 'content': packed(case['answer'])}]
    return cases


def planning_prompt_v2(evidence, structures, fit, previous, maximum_batch):
    hypothesis_prompt(evidence)
    payload = {'evidence': evidence, 'current_structures': structures, 'current_fit': fit,
               'previous_experiment': previous, 'maximum_batch': maximum_batch}
    return [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': packed(payload)+'\n根据这些已有信息，自主决定结束还是继续实验。'
             '继续时提出一个不同且值得检验的竞争解释，并准备构型；不要输出候选机制清单。'
             '共用程序只拟合你提出的结构，在测量前锁定两种解释的数值预测，每次测量单独计费。\n'+ACTION_SCHEMA}]
