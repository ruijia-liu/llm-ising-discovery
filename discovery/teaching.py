"""Causal, executable teaching records from the existing sparse-search policy.

The teacher takes a laboratory capability, never a world or grading label.
Every pre-observation record is computed and hashed before lab.reveal().
Language is templated from actual calculations, not invented research stories.
"""
from copy import deepcopy
import hashlib
import json

import numpy as np

from .physics import Expression, Grammar, Term, base_flip, check_query, encode_query, random_queries
from .search import SparseSearch


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(packed(value).encode('utf-8')).hexdigest()


def decode_query(value):
    rows = value['rows']
    if len(rows) != 16 or any(len(row) != 16 or set(row)-set('+-') for row in rows):
        raise ValueError('Invalid serialized configuration.')
    return check_query([[1 if v == '+' else -1 for v in row] for row in rows], value['position'])


def expression_from_dict(value):
    if value['base'] != 'nearest_neighbour_J=1':
        raise ValueError('Unknown base Hamiltonian.')
    return Expression(tuple(Term(tuple(tuple(p) for p in term['offsets']), term['coefficient'])
                            for term in value['terms']))


def selection_utilities(search, pool_features, committee, observed_features):
    """Instrument the existing choose() formula without changing its policy."""
    scale = max(search.sigma**2, committee[0].sse/len(observed_features), 1e-8)
    plausible = [c for c in committee if c.score <= committee[0].score+9*scale]
    predictions = np.array([pool_features[:, c.support] @ c.coefficients for c in plausible])
    disagreement = predictions.var(axis=0)
    uncertainty = np.zeros(len(pool_features))
    if search.sigma > 0:
        for candidate in plausible:
            if candidate.support:
                old = observed_features[:, candidate.support]
                new = pool_features[:, candidate.support]
                covariance = np.linalg.pinv(old.T @ old)
                uncertainty += search.sigma**2*np.einsum('ni,ij,nj->n', new, covariance, new)/len(plausible)
    return disagreement, uncertainty, len(plausible), scale


def before_record(search, features, residuals, pool, pool_features, committee, choice):
    disagreement, uncertainty, plausible_count, scale = selection_utilities(search, pool_features, committee, features)
    utility = disagreement+uncertainty
    # The instrumentation separately sums uncertainty for reporting, whereas
    # choose() adds it to disagreement incrementally. Near tied maxima can differ
    # by a floating-point ULP solely due to this addition order. Replay the actual
    # frozen policy exactly, then independently verify its utility to roundoff.
    expected = search.choose(pool_features, committee, features)
    tolerance = 16*np.finfo(float).eps*max(1.0, float(np.max(np.abs(utility))))
    if choice != expected or (utility.max() > 1e-12 and utility[choice] < utility.max()-tolerance):
        raise AssertionError('Teaching instrumentation differs from the frozen search policy.')
    hypotheses = []
    for candidate in committee[:3]:
        model = search.expression(candidate)
        hypotheses.append({'expression': model.to_dict(), 'history_sse': candidate.sse,
                           'penalized_score': candidate.score,
                           'within_near_best_band': candidate.score <= committee[0].score+9*scale,
                           'predicted_delta': model.delta(*pool[choice])})
    return {'used': len(residuals), 'hypotheses': hypotheses,
            'anomaly': {'current_fit_rmse': float(np.sqrt(committee[0].sse/len(residuals))),
                        'known_measurement_noise_sigma': search.sigma,
                        'meaning': '只描述当前数据上的拟合残差，不把噪声残差直接判成新机制。'},
            'selection': {'pool_size': len(pool), 'selected_index': choice,
                          'near_best_models_used': plausible_count,
                          'committee_models_available': len(committee),
                          'prediction_disagreement': float(disagreement[choice]),
                          'parameter_uncertainty': float(uncertainty[choice]),
                          'utility': float(utility[choice]), 'maximum_pool_utility': float(utility.max()),
                          'random_tie_fallback': bool(utility.max() <= 1e-12),
                          'reason': '在随机实验池中最大化近优候选预测分歧与参数不确定度之和；没有可利用差异时取随机池首项。',
                          'display_note': '下列最多三个假设是排名靠前的展示项；选择实验可能使用更多候选，备选项不一定仍合理。'},
            'update_condition': '收到观测后用全部已有数据重新拟合并排序；若最优结构改变则修改机制，结构不变时仍可更新系数。单个带噪观测不自动排除假设。'}


def after_record(search, before, value, updated_committee):
    old = expression_from_dict(before['hypotheses'][0]['expression'])
    new = search.expression(updated_committee[0])
    if old.support != new.support:
        change = 'structure_changed'
        explanation = '加入这次观测后，按同一排序规则得到的最优结构改变了。'
    elif not np.allclose(search.grammar.coefficients(old), search.grammar.coefficients(new), atol=1e-10, rtol=0):
        change = 'coefficients_updated'
        explanation = '加入这次观测后保留结构，更新作用强度。'
    else:
        change = 'unchanged'
        explanation = '加入这次观测后，当前最优表达式没有改变。'
    return {'used': before['used']+1, 'prediction_errors': [float(value-h['predicted_delta']) for h in before['hypotheses']],
            'updated_expression': new.to_dict(), 'updated_history_sse': updated_committee[0].sse,
            'change': change, 'explanation': explanation}


def run_teacher(lab, initial_queries, *, query_seed, pool_size=128):
    """Training teacher deliberately proposes order <=3; student syntax allows 4.

    This restriction applies to all training worlds, including failures, and is
    not selected per hidden truth. It keeps explicit four-body teaching absent.
    """
    grammar = Grammar(max_order=3)
    search = SparseSearch(grammar, noise_sigma=lab.noise_sigma)
    rng = np.random.default_rng(query_seed)
    observations = np.asarray(lab.initial_observations(initial_queries))
    features = grammar.features([q[0] for q in initial_queries], [q[1] for q in initial_queries])
    residuals = observations-np.array([base_flip(*q) for q in initial_queries])
    committee = search.fit(features, residuals)
    initial = {'queries': [encode_query(q) for q in initial_queries], 'observed': observations.tolist(), 'used': lab.used}
    rounds = []
    while lab.used < lab.budget:
        pool = random_queries(rng, pool_size)
        pool_features = grammar.features([q[0] for q in pool], [q[1] for q in pool])
        choice = search.choose(pool_features, committee, features)
        before = before_record(search, features, residuals, pool, pool_features, committee, choice)
        before = deepcopy(before)
        commitment = digest({'before': before, 'query': encode_query(pool[choice])})
        models = [expression_from_dict(h['expression']) for h in before['hypotheses']]
        ticket = lab.commit([pool[choice]], models, [[h['predicted_delta']] for h in before['hypotheses']],
                            'Pre-observation record SHA256: '+commitment, before['update_condition'])
        value = lab.reveal(ticket)[0]
        features = np.vstack((features, pool_features[choice]))
        residuals = np.append(residuals, value-base_flip(*pool[choice]))
        committee = search.fit(features, residuals)
        rounds.append({'before': before, 'before_sha256': commitment, 'query': encode_query(pool[choice]),
                       'observed': value, 'after': after_record(search, before, value, committee)})
    return {'initial': initial, 'rounds': rounds,
            'teacher_final_before_grading': search.expression(committee[0]).to_dict(),
            'noise_sigma': lab.noise_sigma, 'used': lab.used}


def assert_same(actual, expected, path='record'):
    """Deep audit, including every recorded floating-point computation."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(path+' keys differ')
        for key in expected:
            assert_same(actual[key], expected[key], path+'.'+str(key))
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(path+' length differs')
        for j, (a, b) in enumerate(zip(actual, expected)):
            assert_same(a, b, path+f'[{j}]')
    elif isinstance(expected, float):
        if not isinstance(actual, (float, int)) or not np.isfinite(actual) or not np.isclose(actual, expected, rtol=1e-9, atol=1e-9):
            raise AssertionError(path+' numeric value differs')
    elif actual != expected:
        raise AssertionError(path+' value differs')


def audit_teacher(trace, *, query_seed, pool_size=128):
    """Replay causal decisions with only prefixes, not truth or future data.

    Also verify every displayed prediction by independently subtracting full
    lattice energies, not just calling the local flip implementation.
    """
    grammar = Grammar(max_order=3)
    search = SparseSearch(grammar, noise_sigma=trace['noise_sigma'])
    initial = trace['initial']
    queries = [decode_query(q) for q in initial['queries']]
    if len(queries) != 8 or initial['used'] != 8 or len(trace['rounds']) != 24 or trace['used'] != 32:
        raise AssertionError('Frozen 8+24 protocol violated.')
    features = grammar.features([q[0] for q in queries], [q[1] for q in queries])
    residuals = np.array(initial['observed'])-np.array([base_flip(*q) for q in queries])
    rng = np.random.default_rng(query_seed)
    predictions_verified = 0
    for record in trace['rounds']:
        committee = search.fit(features, residuals)
        pool = random_queries(rng, pool_size)
        pool_features = grammar.features([q[0] for q in pool], [q[1] for q in pool])
        choice = search.choose(pool_features, committee, features)
        expected_before = before_record(search, features, residuals, pool, pool_features, committee, choice)
        assert_same(record['query'], encode_query(pool[choice]))
        assert_same(record['before'], expected_before)
        if record['before_sha256'] != digest({'before': record['before'], 'query': record['query']}):
            raise AssertionError('Pre-observation commitment changed.')
        state, position = decode_query(record['query'])
        changed = state.copy()
        changed[position] *= -1
        for h in record['before']['hypotheses']:
            model = expression_from_dict(h['expression'])
            assert_same(h['predicted_delta'], float(model.energy(changed)-model.energy(state)))
            predictions_verified += 1
        features = np.vstack((features, pool_features[choice]))
        residuals = np.append(residuals, record['observed']-base_flip(*pool[choice]))
        committee = search.fit(features, residuals)
        assert_same(record['after'], after_record(search, record['before'], record['observed'], committee))
    assert_same(trace['teacher_final_before_grading'], search.expression(committee[0]).to_dict())
    return {'rounds_replayed': len(trace['rounds']), 'predictions_verified_by_total_energy': predictions_verified}


PUBLIC_RULES = {'lattice_side': 16, 'boundary': 'periodic', 'base': 'nearest_neighbour_J=1',
                'allowed_window': [3, 3], 'allowed_max_order': 4, 'maximum_hidden_terms': 2,
                'initial_measurements': 8, 'maximum_measurements': 32,
                'noise': 'independent additive Gaussian, known sigma',
                'offset_coordinates': 'row, column; modulo 16', 'coefficient_convention': 'H=H0-sum(K*translated_products)',
                'no_required_final_confirmation_stage': True}


def export_views(trace, final_answer, episode_id):
    """Only the grader/exporter attaches truth, after teacher trajectory completion."""
    label = {'source': 'evaluator_after_completed_episode', 'expression': final_answer.to_dict()}
    common = {'episode_id': episode_id, 'rules': PUBLIC_RULES, 'noise_sigma': trace['noise_sigma'],
              'initial': trace['initial'], 'final_answer': label}
    a = deepcopy(common)
    b = deepcopy(common)
    b['experiments'] = [{'query': r['query'], 'observed': r['observed'], 'used': r['after']['used']} for r in trace['rounds']]
    c = deepcopy(b)
    c['research_process'] = [{'before': r['before'], 'after': r['after']} for r in trace['rounds']]
    c['teacher_final_before_grading'] = trace['teacher_final_before_grading']
    return {'A': a, 'B': b, 'C': c}


def audit_views(views):
    a, b, c = (views[group] for group in 'ABC')
    shared = {k: v for k, v in c.items() if k not in ('research_process', 'teacher_final_before_grading')}
    if packed(shared) != packed(b) or packed(a) != packed({k: v for k, v in b.items() if k != 'experiments'}):
        raise AssertionError('A/B/C common data or final labels differ.')
    if len(c['research_process']) != len(c['experiments']):
        raise AssertionError('Unpaired process annotations.')
    return digest(b)


SYSTEM_TEXT = ('你正在阅读广义 Ising 模拟实验室的离线教学记录。根据已有证据提出可执行假设与事前预测。'
               '每次翻转能量测量计费一次。表达式中的 offsets 是相对位置，coefficient 是额外作用强度。'
               'measurement_noise_sigma 已知；预测值是候选模型的均值，不保证等于有噪声的观察。'
               '最后的 evaluator_after_completed_episode 是判卷器给出的监督答案，不代表老师事前知道它。'
               '公共规则：'+packed(PUBLIC_RULES))


def messages_for_view(view, group):
    """Model-neutral messages draft. Tokenization/loss masking awaits model choice."""
    messages = [{'role': 'system', 'content': SYSTEM_TEXT},
                {'role': 'user', 'content': packed({'measurement_noise_sigma': view['noise_sigma'], 'initial': view['initial']})}]
    for j, experiment in enumerate(view.get('experiments', [])):
        action = {'experiment': {'query': experiment['query'], 'measurement_count': 1}}
        if group == 'C':
            action['before_observation'] = view['research_process'][j]['before']
            if j:
                action['update_from_previous_observation'] = view['research_process'][j-1]['after']
        messages.append({'role': 'assistant', 'content': packed(action)})
        messages.append({'role': 'user', 'content': packed({'observation': experiment['observed'], 'used': experiment['used']})})
    final = {'final_answer': view['final_answer']}
    if group == 'C':
        final['update_from_last_observation'] = view['research_process'][-1]['after']
        final['teacher_estimate_before_grading'] = view['teacher_final_before_grading']
    messages.append({'role': 'assistant', 'content': packed(final)})
    return {'messages': messages}
