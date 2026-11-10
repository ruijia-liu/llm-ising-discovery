"""Model-neutral active loop. Generator sees serialized public messages only.

Arithmetic predicts only student-proposed structures. Invalid replies never
trigger a hidden search or a fallback measurement. The truth is not an argument.
"""
from copy import deepcopy
import hashlib
import json

import numpy as np

from .student_mainline import hypothesis_prompt
from .student_v2 import packed
from .student_v3 import SYSTEM, canonical_proposal
from .student_spins import public_coverage


LIMITS = {'hypothesis_tokens': 128, 'plan_tokens': 1536,
          'total_generated_tokens': 16384, 'maximum_calls': 64,
          'repairs_per_episode': 1, 'maximum_batch': 4}
PLAN_TEMPLATE = {'action': 'experiment', 'alternative_structures': [],
                 'queries': [{'local_spins': [[1]*5 for _ in range(5)], 'position': [0, 0]}],
                 'reason': '比较当前解释与只有已知H0的解释。',
                 'update_condition': '观察若与当前模型的预测不符，重新考虑解释；有噪声时不要求单次观测精确相等。'}
PLAN_REQUEST = (
    '现在由你选择是否继续实验。只输出JSON。若结束，输出{"action":"finish"}。'
    '若继续，必须输出action="experiment"、alternative_structures、queries、reason、update_condition五个键。'
    'alternative_structures与structures使用相同格式，表示一个不同于当前机制且仍值得检验的竞争解释；'
    '它也可以是空数组。不要输出候选机制清单。'
    'queries是1到maximum_batch次测量的数组；每项只有local_spins和position。'
    'local_spins必须是5行5列的整数-1或1，position为16×16晶格上的两个整数。'
    '每次测量（包括重复）消耗一次预算；你可以选择任意合法构型。'
    'reason解释选择这些实验的理由；update_condition说明什么结果会促使你修改当前解释。'
    '共用数值工具会只用已有观测拟合两个机制，并在测量前锁定具体数值预测；不会添加新结构。'
    '预测是噪声分布的均值。不要写Markdown代码块、占位符或其他键。'
    '\n完整格式示例（仅展示接口；不是对当前世界的建议或正确答案；你可以自己准备构型和竞争解释）：'
    +packed(PLAN_TEMPLATE))


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key.')
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('Non-finite JSON number.')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)


def parse_hypothesis(text):
    value = strict_json(text)
    if not isinstance(value, dict) or set(value) != {'structures'}:
        raise ValueError('Expected exactly the structures key.')
    canonical_proposal(value['structures'], public_coverage())
    return value['structures']


def parse_plan(text, maximum_batch):
    value = strict_json(text)
    if not isinstance(value, dict):
        raise ValueError('Decision must be a JSON object.')
    if value == {'action': 'finish'}:
        return value
    if set(value) != {'action', 'alternative_structures', 'queries', 'reason', 'update_condition'} or value['action'] != 'experiment':
        raise ValueError('Use the experiment schema or exactly {"action":"finish"}.')
    if not isinstance(value['queries'], list) or not 1 <= len(value['queries']) <= maximum_batch:
        raise ValueError('The batch crosses the next budget checkpoint or exceeds four measurements.')
    return value


def planning_prompt(evidence, structures, fit, previous, maximum_batch):
    # The evidence validator also rejects metadata such as labels or world IDs.
    hypothesis_prompt(evidence)
    payload = {'evidence': evidence, 'current_structures': structures, 'current_fit': fit,
               'previous_experiment': previous, 'maximum_batch': maximum_batch}
    return [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': packed(payload)+'\n'+PLAN_REQUEST}]


def idea_diagnostic(commitment, sigma):
    fits = commitment['fits']
    compatible = [bool(f['identifiable_given_proposal'] and f['observation_rmse'] <= max(1e-7, 3*sigma)) for f in fits]
    gap = float(np.max(np.ptp(np.asarray(commitment['predictions']), axis=0)))
    distinguishable = gap > max(1e-7, 2*sigma)
    return {'history_compatible': compatible, 'maximum_mean_prediction_gap': gap,
            'operational_competitive_test': bool(all(compatible) and distinguishable
                                                  and commitment['distinct_fitted_hypotheses'] >= 2),
            'rule': 'full rank; history RMSE <= max(1e-7,3*sigma); mean prediction gap > max(1e-7,2*sigma)',
            'not_a_p_value_or_proof_of_identifiability': True}


class ReplyFailure(ValueError):
    pass


class InferenceBudget(ValueError):
    pass


def run_active(session, generate, *, limits=None, on_event=None):
    """generate(messages, max_new_tokens) -> text and exact token usage.

    A JSON round trip, rather than object sharing, is the callback boundary.
    A subprocess transport is provided for real model inference. No eval/exec.
    The callback and optional durable event writer receive no laboratory object.
    """
    limits = deepcopy(limits or LIMITS)
    log, snapshots, last, previous = [], [], None, None
    counters = {'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'repairs': 0, 'generation_seconds': 0.0}

    def emit(record):
        clean = strict_json(packed(record))
        log.append(clean)
        if on_event:
            on_event(deepcopy(clean))

    def request(messages, cap, validator, stage):
        current = deepcopy(messages)
        while True:
            remaining = limits['total_generated_tokens']-counters['output_tokens']
            if remaining <= 0 or counters['calls'] >= limits['maximum_calls']:
                raise InferenceBudget('Inference budget exhausted.')
            maximum = min(cap, remaining)
            public = strict_json(packed(current))
            reply = generate(public, maximum)
            if (not isinstance(reply, dict) or not isinstance(reply.get('text'), str)
                    or type(reply.get('input_tokens')) is not int or reply['input_tokens'] < 1
                    or type(reply.get('output_tokens')) is not int or not 0 <= reply['output_tokens'] <= maximum):
                raise RuntimeError('Transport returned invalid token accounting.')
            counters['calls'] += 1
            counters['input_tokens'] += reply['input_tokens']
            counters['output_tokens'] += reply['output_tokens']
            counters['generation_seconds'] += float(reply.get('generation_seconds', 0))
            emit({'kind': 'model_reply', 'stage': stage, 'used': session.evidence()['used'],
                  'messages': public, 'requested_output_limit': maximum, 'reply': reply})
            try:
                return validator(reply['text'])
            except (ValueError, TypeError, KeyError, IndexError) as error:
                emit({'kind': 'invalid_reply', 'stage': stage, 'error': str(error)})
                if counters['repairs'] >= limits['repairs_per_episode']:
                    raise ReplyFailure(str(error)) from error
                counters['repairs'] += 1
                current += [{'role': 'assistant', 'content': reply['text']},
                            {'role': 'user', 'content': '回答未通过格式/实验接口校验：'+str(error)+
                             '。没有执行新测量。请按原格式重新提交；没有新的观测或正确答案。'}]

    status, final = 'not_finished', None
    emit({'kind': 'initial_evidence', 'evidence': session.evidence(), 'limits': limits})
    try:
        while True:
            evidence = session.evidence()
            structures = request(hypothesis_prompt(evidence), limits['hypothesis_tokens'], parse_hypothesis, 'hypothesis')
            fit = session.fit(structures)
            last = {'used': evidence['used'], 'raw_structures': structures, 'expression': fit['expression'], 'fit': fit}
            snapshots.append(deepcopy(last))
            emit({'kind': 'current_hypothesis', **last})
            if evidence['used'] == 32:
                final, status = session.finish(structures), 'budget_completed'
                break
            next_checkpoint = min(b for b in (16, 24, 32) if b > evidence['used'])
            maximum_batch = min(limits['maximum_batch'], next_checkpoint-evidence['used'])

            def prepare(text):
                plan = parse_plan(text, maximum_batch)
                if plan['action'] == 'finish':
                    return plan, None
                # Only the student's two actual proposals enter arithmetic.
                proposal = {'hypotheses': [{'structures': structures}, {'structures': plan['alternative_structures']}],
                            'queries': plan['queries'], 'reason': plan['reason'], 'update_condition': plan['update_condition']}
                return plan, session.prepare(proposal)

            plan, committed = request(planning_prompt(evidence, structures, fit, previous, maximum_batch),
                                      limits['plan_tokens'], prepare, 'plan')
            if committed is None:
                final, status = session.finish(structures), 'student_finished'
                break
            diagnostic = idea_diagnostic(committed, evidence['measurement_noise_sigma'])
            diagnostic['exact_format_example_copied'] = plan == PLAN_TEMPLATE
            diagnostic['example_query_and_alternative_copied'] = (
                plan['queries'] == PLAN_TEMPLATE['queries'] and plan['alternative_structures'] == [])
            # Durable logging completes BEFORE reveal() makes any oracle call.
            emit({'kind': 'predictions_committed', 'commitment': committed, 'idea_diagnostic': diagnostic})
            observed = session.reveal(committed['ticket'])
            emit({'kind': 'observations_revealed', 'observation': observed})
            previous = {'proposal': committed['proposal'], 'predictions': committed['predictions'],
                        'observed': observed['observations'], 'used': observed['used']}
    except ReplyFailure as error:
        status = 'invalid_reply'
        emit({'kind': 'episode_failed', 'error': str(error)})
    except InferenceBudget as error:
        status = 'inference_budget_exhausted'
        emit({'kind': 'episode_failed', 'error': str(error)})
    result = {'status': status, 'valid_final_answer': final is not None, 'final': final,
              'last_valid_hypothesis_for_diagnostics_only': last, 'used': session.evidence()['used'],
              'snapshots': snapshots, 'inference': counters, 'events': log}
    # Protocol failures are not silently converted into a successful null model.
    emit({'kind': 'episode_ended', 'status': status, 'used': result['used'], 'valid_final_answer': final is not None})
    return result
