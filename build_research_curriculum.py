"""Build and audit a small, paired A/B/C research curriculum. No model training.

python build_research_curriculum.py
"""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import platform
import time

import numpy as np

from discovery.lab import Laboratory
from discovery.noise_evaluation import evaluate_noisy, validate_calibration
from discovery.physics import Expression, Grammar, Term, canonical_structure, random_queries
from discovery.teaching import (PUBLIC_RULES, assert_same, audit_teacher, audit_views, decode_query,
                                digest, export_views, expression_from_dict, messages_for_view,
                                packed, run_teacher)
from run_structure_discovery import compile_queries, evaluate, evaluation_queries

FAMILIES = ('field', 'direction', 'distance_two', 'three_body', 'mixed', 'null')
NOISE_SIGMAS = (0.0, 0.5, 1.0, 2.0)


def stream(seed, world, purpose):
    return np.random.SeedSequence([seed, world, purpose])


def make_teaching_worlds(seed=93001):
    """24 training worlds + 12 parameter-validation worlds; no four-body labels.

    Fix supports without using teacher outcomes. Validation reuses support
    templates with new coefficients and query/noise streams; it is explicitly
    parameter validation, not a structure-generalization evaluation.
    """
    grammar = Grammar(max_order=3)
    rng = np.random.default_rng(stream(seed, 0, 701))
    field = canonical_structure(((0, 0),))
    horizontal = canonical_structure(((0, 0), (0, 1)))
    vertical = canonical_structure(((0, 0), (1, 0)))
    distant = [p for p in grammar.structures if len(p) == 2 and max(min(v, 16-v) for v in p[1]) == 2]
    triples = [p for p in grammar.structures if len(p) == 3]
    distant = [distant[int(i)] for i in rng.choice(len(distant), 4, replace=False)]
    triples = [triples[int(i)] for i in rng.choice(len(triples), 4, replace=False)]
    templates = {'field': [(field,)]*4, 'direction': [(horizontal,), (vertical,)]*2,
                 'distance_two': [(p,) for p in distant], 'three_body': [(p,) for p in triples],
                 'mixed': [(field, p) for p in triples], 'null': [()]*4}
    worlds = []
    for split, size in (('train', 4), ('validation', 2)):
        for family in FAMILIES:
            for j in range(size):
                index = len(worlds)
                parameter_rng = np.random.default_rng(stream(seed, index, 11))
                support = templates[family][j]
                coefficients = parameter_rng.uniform(0.3, 0.8, len(support))*parameter_rng.choice([-1, 1], len(support))
                truth = Expression(tuple(Term(p, float(c)) for p, c in zip(support, coefficients)))
                worlds.append({'world_index': index, 'world_id': f'w{index:04d}', 'split': split,
                               'family': family, 'truth': truth.to_dict()})
    return worlds


def walk_expressions(value):
    if isinstance(value, dict):
        if value.get('base') == 'nearest_neighbour_J=1' and 'terms' in value:
            yield expression_from_dict(value)
        else:
            for child in value.values():
                yield from walk_expressions(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_expressions(child)


def reserve_holdouts(worlds, shown_supports, *, seed=94001, per_level=12):
    """Create prospective holdouts, excluding even C's intermediate expressions.

    Tests are RESERVED, not run, and their labels stay out of student exports.
    Choosing eligible combinations after the corpus is fixed uses no test scores.
    """
    rng = np.random.default_rng(seed)
    train = [expression_from_dict(w['truth']) for w in worlds if w['split'] == 'train']
    components = sorted({p for model in train for p in model.support})
    pairs = [tuple(sorted(pair)) for pair in combinations(components, 2) if tuple(sorted(pair)) not in shown_supports]
    if not pairs:
        raise ValueError('No unexposed combinations remain; do not claim an unseen-combination test.')
    four = [p for p in Grammar().structures if len(p) == 4]
    four_indices = rng.choice(len(four), per_level, replace=False)
    # Cover distinct eligible supports first; new coefficients give extra worlds,
    # not extra independent mechanism structures. Report both counts explicitly.
    chosen_pairs = np.resize(rng.permutation(len(pairs)), per_level)
    parameter_templates = sorted({model.support for model in train if model.terms})
    parameter_indices = np.resize(rng.permutation(len(parameter_templates)), per_level)
    actual_labels = {packed(w['truth']) for w in worlds}
    reserved = []
    for level in ('parameter', 'combination', 'four_body', 'null'):
        for j in range(per_level):
            if level == 'parameter':
                support = parameter_templates[int(parameter_indices[j])]
            elif level == 'combination':
                support = pairs[int(chosen_pairs[j])]
            elif level == 'four_body':
                support = (four[int(four_indices[j])],)
            else:
                support = ()
            coefficients = rng.uniform(0.3, 0.8, len(support))*rng.choice([-1, 1], len(support))
            model = Expression(tuple(Term(p, float(c)) for p, c in zip(support, coefficients)))
            if model.terms and packed(model.to_dict()) in actual_labels:
                raise AssertionError('Exact teaching answer leaked into a test world.')
            if level in ('combination', 'four_body') and model.support in shown_supports:
                raise AssertionError('A reserved mechanism was already shown in the curriculum.')
            reserved.append({'world_id': f'test-{len(reserved):04d}', 'level': level, 'truth': model.to_dict()})
    return {'status': 'reserved_not_evaluated', 'seed': seed, 'worlds_per_level': per_level,
            'training_answer_components': len(components), 'unexposed_combinations_available': len(pairs),
            'distinct_reserved_combinations': len(set(int(i) for i in chosen_pairs)),
            'exclusion_scope': 'all explicit expressions in both training and validation C views, including alternatives and final labels',
            'worlds': reserved}


def verify_observations(trace, truth, noise_seed):
    """Grader-only audit using full energies and a separate replay of noise draws."""
    rng = np.random.default_rng(noise_seed)
    observations = list(zip(trace['initial']['queries'], trace['initial']['observed']))
    observations += [(r['query'], r['observed']) for r in trace['rounds']]
    for encoded, observed in observations:
        state, position = decode_query(encoded)
        changed = state.copy()
        changed[position] *= -1
        clean = truth.energy(changed)-truth.energy(state)
        assert_same(observed, float(clean+rng.normal(0, trace['noise_sigma'])))
    return len(observations)


def audit_messages(views, messages):
    if messages['B']['messages'][0] != messages['C']['messages'][0]:
        raise AssertionError('B/C have different system instructions.')
    for group in 'ABC':
        current = messages[group]['messages']
        if current[-1]['role'] != 'assistant' or json.loads(current[-1]['content'])['final_answer'] != views[group]['final_answer']:
            raise AssertionError('Final message label differs from view.')
    if [m for m in messages['B']['messages'] if m['role'] == 'user'] != [m for m in messages['C']['messages'] if m['role'] == 'user']:
        raise AssertionError('B/C observation messages differ.')
    for j in range(24):
        b_action = json.loads(messages['B']['messages'][2+2*j]['content'])
        c_action = json.loads(messages['C']['messages'][2+2*j]['content'])
        if b_action['experiment'] != c_action['experiment']:
            raise AssertionError('B/C experiment messages differ.')
        assert_same(c_action['before_observation'], views['C']['research_process'][j]['before'])
        if j == 0 and 'update_from_previous_observation' in c_action:
            raise AssertionError('Future observation update leaked into first action.')
        if j:
            assert_same(c_action['update_from_previous_observation'], views['C']['research_process'][j-1]['after'])


def example_markdown(view, outcome, path):
    lines = ['# 一条实际生成的研究教材', '',
             '这是搜索程序的真实计算记录，不是语言模型编写的研究故事。所有显示的预测均已由完整能量差核验。',
             '选作阅读示例不影响数据集；数据集保留全部轨迹。', '',
             f"教材编号：{view['episode_id']}。测量噪声标准差：{view['noise_sigma']}。",
             f"判卷器最终判断：结构正确={outcome['support_success']}，完整恢复={outcome['full_recovery']}。", '',
             'B 版看到全部相同的构型、位置、观测和最终判卷答案；以下中间过程只出现在 C 版。', '']
    changing = [j for j, r in enumerate(view['research_process']) if r['after']['change'] == 'structure_changed']
    selected = sorted(set([0, 23]+changing[:2]))
    for j in selected:
        process = view['research_process'][j]
        before, after = process['before'], process['after']
        observation = view['experiments'][j]['observed']
        lines += [f"## 第 {j+1} 轮：测量前已使用 {before['used']} 次", '',
                  f"当前解释对已有数据的拟合 RMSE：{before['anomaly']['current_fit_rmse']:.6f}。", '',
                  '### 提交的假设与事前预测', '']
        for k, h in enumerate(before['hypotheses']):
            lines += [f"候选 {k+1}：预测翻转能量 {h['predicted_delta']:.6f}；仍在近优范围内={h['within_near_best_band']}。", '',
                      '```json', json.dumps(h['expression'], ensure_ascii=False, indent=2), '```', '']
        selection = before['selection']
        lines += ['### 为什么选择这次实验', '', selection['reason'], '',
                  f"候选预测分歧贡献 {selection['prediction_disagreement']:.6f}，参数不确定度贡献 {selection['parameter_uncertainty']:.6f}。",
                  '这两个数是程序的选择分数，不是额外实验结果。', '',
                  f"测量位置：{view['experiments'][j]['query']['position']}。完整 16×16 构型保存在对应 JSONL 中。", '',
                  '### 观察后更新', '', f'实际观测：{observation:.6f}。', after['explanation'], '',
                  '```json', json.dumps(after['updated_expression'], ensure_ascii=False, indent=2), '```', '']
    lines += ['## 轨迹结束后的判卷', '', '老师在看答案前的最终估计：', '', '```json',
              json.dumps(view['teacher_final_before_grading'], ensure_ascii=False, indent=2), '```', '',
              '判卷器随后给出的正确机制，B／C 完全相同：', '', '```json',
              json.dumps(view['final_answer'], ensure_ascii=False, indent=2), '```', '',
              '如果两者不同，保留这个失败；不把判卷答案改写成老师事前已经发现的机制。', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def main(args):
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty output directory; existing curriculum will not be overwritten.')
    calibration = json.loads(Path(args.calibration).read_text(encoding='utf-8'))
    validate_calibration(calibration, args.seed)
    worlds = make_teaching_worlds(args.seed)
    output.mkdir(parents=True, exist_ok=True)
    for folder in ('private', 'views/train', 'views/validation', 'messages/train', 'messages/validation'):
        (output/folder).mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parent
    source_files = ('discovery/physics.py', 'discovery/lab.py', 'discovery/search.py', 'discovery/teaching.py',
                    'build_research_curriculum.py')
    import hashlib
    config = {'seed': args.seed, 'holdout_seed': args.holdout_seed, 'rules': PUBLIC_RULES,
              'noise_sigmas': list(NOISE_SIGMAS), 'teacher': 'instrumented sparse search; no language model',
              'teacher_max_order': 3, 'teacher_dictionary_size': 61, 'student_max_order': 4,
              'train_worlds': 24, 'validation_worlds': 12, 'validation_type': 'new parameters on seen support templates',
              'failure_filtering': False, 'forced_confirmation_stage': False,
              'python': platform.python_version(), 'numpy': np.__version__, 'calibration_sha256': calibration['sha256'],
              'source_sha256': {name: hashlib.sha256((source_root/name).read_bytes()).hexdigest() for name in source_files},
              'token_budget_matched': False, 'tokenizer': None, 'training_started': False}
    (output/'protocol.json').write_text(json.dumps(config, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    (output/'private/worlds.json').write_text(json.dumps(worlds, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    handles = {(kind, split, group): (output/kind/split/(group+'.jsonl')).open('w', encoding='utf-8')
               for kind in ('views', 'messages') for split in ('train', 'validation') for group in 'ABC'}
    stats = defaultdict(lambda: {'episodes': 0, 'utf8_bytes': 0, 'characters': 0, 'assistant_characters': 0,
                                 'max_episode_characters': 0})
    counts = Counter()
    outcomes, seen_supports = [], set()
    best_example, example_priority = None, -1
    full_grammar = Grammar()
    start = time.perf_counter()
    try:
        with (output/'private/teacher_traces.jsonl').open('w', encoding='utf-8') as raw_file:
            for world in worlds:
                index = world['world_index']
                truth = expression_from_dict(world['truth'])
                initial = random_queries(np.random.default_rng(stream(args.seed, index, 101)), 8)
                # These evaluation sets stay with the grader and never enter the teacher API.
                evaluation = evaluation_queries(np.random.default_rng(stream(args.seed, index, 303)), 256)
                evaluation_features = [compile_queries(full_grammar, q) for q in evaluation]
                for sigma in NOISE_SIGMAS:
                    lab = Laboratory(truth.delta, noise_sigma=sigma, noise_seed=stream(args.seed, index, 404))
                    trace = run_teacher(lab, initial, query_seed=stream(args.seed, index, 202))
                    audited = audit_teacher(trace, query_seed=stream(args.seed, index, 202))
                    counts.update(audited)
                    counts['observations_replayed_by_total_energy'] += verify_observations(trace, truth, stream(args.seed, index, 404))
                    final_model = expression_from_dict(trace['teacher_final_before_grading'])
                    score = (evaluate(full_grammar, truth, final_model, evaluation_features) if sigma == 0
                             else evaluate_noisy(full_grammar, truth, final_model, evaluation_features, calibration['thresholds']))
                    episode_id = f"{world['world_id']}-sigma{sigma:g}"
                    outcome = {'episode_id': episode_id, 'split': world['split'], 'family': world['family'],
                               'noise_sigma': sigma, **score}
                    outcomes.append(outcome)
                    views = export_views(trace, truth, episode_id)
                    pairing_hash = audit_views(views)
                    messages = {group: messages_for_view(views[group], group) for group in 'ABC'}
                    audit_messages(views, messages)
                    for model in walk_expressions(views['C']):
                        if any(len(p) > 3 for p in model.support):
                            raise AssertionError('Four-body expression leaked into teaching data.')
                        seen_supports.add(model.support)
                    for group in 'ABC':
                        handles['views', world['split'], group].write(packed(views[group])+'\n')
                        serialized = packed(messages[group])+'\n'
                        handles['messages', world['split'], group].write(serialized)
                        current = stats[world['split'], group]
                        current['episodes'] += 1
                        current['utf8_bytes'] += len(serialized.encode('utf-8'))
                        characters = sum(len(m['content']) for m in messages[group]['messages'])
                        current['characters'] += characters
                        current['assistant_characters'] += sum(len(m['content']) for m in messages[group]['messages'] if m['role'] == 'assistant')
                        current['max_episode_characters'] = max(current['max_episode_characters'], characters)
                    structure_updates = sum(r['after']['change'] == 'structure_changed' for r in trace['rounds'])
                    counts['structure_updates'] += structure_updates
                    counts['coefficient_updates'] += sum(r['after']['change'] == 'coefficients_updated' for r in trace['rounds'])
                    counts['paired_episodes_verified'] += 1
                    # Choose only a readable report example; all episodes remain in exports.
                    priority = 10*int(not score['full_recovery'])+min(structure_updates, 5)
                    if world['split'] == 'train' and priority > example_priority:
                        example_priority = priority
                        best_example = (views['C'], outcome)
                    raw_file.write(packed({'episode_id': episode_id, 'world_id': world['world_id'], 'split': world['split'],
                                           'query_seed_components': [args.seed, index, 202], 'trace': trace,
                                           'lab_events': lab.events, 'common_B_C_sha256': pairing_hash,
                                           'grader_outcome_after_episode': outcome})+'\n')
                if (index+1) % 4 == 0:
                    print(f"{index+1}/36 worlds; {(index+1)*4} audited episodes ({time.perf_counter()-start:.1f}s)", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    reserved = reserve_holdouts(worlds, seen_supports, seed=args.holdout_seed)
    (output/'private/reserved_test_worlds.json').write_text(json.dumps(reserved, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    (output/'private/grader_outcomes.json').write_text(json.dumps(outcomes, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    statistics = {split: {group: stats[split, group] for group in 'ABC'} for split in ('train', 'validation')}
    summary = {'config': config, 'audit_counts': dict(counts), 'dataset_statistics': statistics,
               'teacher_outcomes': [{'split': split, 'noise_sigma': sigma,
                                     'episodes': len(selected), 'complete_recoveries': sum(r['full_recovery'] for r in selected),
                                     'correct_structures': sum(r['support_success'] for r in selected),
                                     'false_discoveries': sum(r['false_discovery'] for r in selected)}
                                    for split in ('train', 'validation') for sigma in NOISE_SIGMAS
                                    for selected in [[r for r in outcomes if r['split'] == split and r['noise_sigma'] == sigma]]],
               'reserved_test_counts': dict(Counter(w['level'] for w in reserved['worlds'])),
               'eligible_unseen_combinations': reserved['unexposed_combinations_available'],
               'distinct_reserved_combinations': reserved['distinct_reserved_combinations'],
               'unique_shown_supports': len(seen_supports), 'elapsed_seconds': time.perf_counter()-start}
    (output/'audit_report.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    example_markdown(*best_example, output/'EXAMPLE_ZH.md')
    write_report(summary, output)
    print(json.dumps({'output': str(output.resolve()), 'audit_counts': dict(counts),
                      'reserved_test_counts': summary['reserved_test_counts']}, ensure_ascii=False, indent=2), flush=True)
    return summary


def write_report(summary, output):
    stats = summary['dataset_statistics']
    counts = summary['audit_counts']
    lines = ['# 首批可核验研究教材', '',
             '**已完成教材生成与审计，尚未微调学生，也没有检验过程教学是否优于证据教学。**', '',
             '## 数据规模和不变规则', '',
             '- 24 个训练世界、12 个参数验证世界；每世界有 0／0.5／1／2 四档噪声记录。',
             '- 每组有 96 条训练教材、48 条验证教材；共 144 条原始研究轨迹，按相同世界划分防止跨噪声串入不同集合。',
             '- 每条轨迹仍为 8 次初始测量＋24 次自由选择测量，没有新增最后 8 次强制核验。',
             '- 不过滤误报或失败轨迹。真实机制最多两项，非零系数绝对值仍在 [0.3, 0.8]。',
             '- 老师是普通稀疏搜索程序，使用实际计算的候选、预测、选择分数和更新；不是语言模型编写的故事。',
             '- 首批教材的真实机制与展示的中间假设均为一至三体；学生公共表达语言仍允许四体。', '',
             '## 三组教材', '',
             '| 组别 | 内容 |', '|---|---|',
             '| A | 相同初始观察＋最终判卷答案 |',
             '| B | A＋全部实际实验操作与观测 |',
             '| C | B＋事前假设、数值预测、选择依据、观察后的更新与老师最终估计 |', '',
             'B／C 的所有观测、操作和最终标签逐条相同；系统提示和观测消息也逐条相同。',
             'C 中展示最多三个候选，若某个候选已不在近优范围，会明确标记，不强称它合理。',
             '最终正确标签由判卷器在轨迹结束后补入两组。老师猜错时保留原估计，不改写历史。', '',
             '## 自动核验', '',
             f"- {counts['paired_episodes_verified']} 条轨迹通过 A／B／C 内容配对与消息顺序检查。",
             f"- {counts['rounds_replayed']} 轮实验选择用当时的数据前缀和随机实验池重新计算。",
             f"- {counts['predictions_verified_by_total_energy']} 个事前预测由完整哈密顿量能量差独立核验。",
             f"- {counts['observations_replayed_by_total_energy']} 次观测由私有真值和独立噪声重放核验。",
             f"- 记录了 {counts['structure_updates']} 次结构修改、{counts['coefficient_updates']} 次系数更新。", '',
             '哈希锁定事前记录；重放核验不读取未来观测或最终正确机制。真值仅供结束后的观测核验和判卷。', '',
             '## 老师的结果，全部保留', '',
             '| 集合 | 噪声标准差 | 轨迹数 | 结构正确 | 完整恢复 | 零作用误报数 |', '|---|---:|---:|---:|---:|---:|']
    for row in summary['teacher_outcomes']:
        lines.append(f"| {row['split']} | {row['noise_sigma']:g} | {row['episodes']} | {row['correct_structures']} | "
                     f"{row['complete_recoveries']} | {row['false_discoveries']} |")
    lines += ['', '无噪声完整恢复采用原数值容差；有噪声采用之前冻结的统一精度要求。两者不用于直接比较准确率。',
              '这些是教材生成老师的成绩，不是学生教学效果。', '',
              '## 机制划分和未来测试', '',
              '- 本批验证集使用已见结构的新系数，只用于参数验证；不冒充结构泛化测试。',
              '- 另预留参数变化、新组合、四体、无额外作用各 12 个测试世界，未运行测试，答案仅在 private/ 中。',
              '- 新组合的组成部分出现在训练答案里，但组合本身没有出现在任何训练／验证 C 教材表达式中，连错误候选也排除。',
              f"- 审计后有 {summary['eligible_unseen_combinations']} 个合格未展示组合；12 个组合测试世界包含 {summary['distinct_reserved_combinations']} 种不同组合，其余仅改变系数。",
              '- 不把同一组合的新系数当成新的结构；后续统计应考虑结构重复，并在扩大教材后扩大独立测试结构数量。',
              '- 四体结构没有出现在教材真值、中间假设或最终估计中。公共语言允许四体不等于展示正确四体机制。',
              '- 原来的噪声开发实验曾使用四体世界，因此只主张四体未出现在本批教学数据中，不主张整个项目从未使用四体知识。', '',
              '## 训练预算尚未匹配', '',
              '没有选定学生模型／分词器；不把字符数冒充 token 数。以下是真实字符数，便于发现上下文长度和预算差异。', '',
              '| 集合 | 组 | 教材条数 | 内容字符总数 | 最长一条字符数 |', '|---|---|---:|---:|---:|']
    for split in ('train', 'validation'):
        for group in 'ABC':
            s = stats[split][group]
            lines.append(f"| {split} | {group} | {s['episodes']} | {s['characters']} | {s['max_episode_characters']} |")
    lines += ['', '选择基础模型后还需实际分词、检查上下文长度、确定损失掩码并匹配训练 token／更新预算。',
              'messages/ 是通用多轮消息草稿，包含判卷器提供的监督答案；不是已经完成预算匹配的正式微调实验。',
              '如需增加 B 的训练量，只能重复同一训练教材，不能混入验证或测试数据。', '',
              '## 文件入口', '',
              '- views/train/B.jsonl、C.jsonl：可审阅的结构化配对教材；A.jsonl 为辅助对照。',
              '- views/validation/：同样格式的独立参数验证教材。',
              '- messages/train/ 与 messages/validation/：通用多轮消息草稿。',
              '- [EXAMPLE_ZH.md](EXAMPLE_ZH.md)：一条实际轨迹的中文阅读示例，包括老师最终估计与判卷答案。',
              '- audit_report.json：核验数量、老师结果和实际文件／字符统计。',
              '- protocol.json：冻结规则、源文件哈希与未完成的训练预算状态。',
              '- private/：原始轨迹、私有真值和预留测试世界，未来学生不得访问。', '',
              '当前代码通过接口限制老师所获信息，仍不是可防恶意读文件的进程沙箱。', '']
    (output/'REPORT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=93001)
    parser.add_argument('--holdout-seed', type=int, default=94001)
    parser.add_argument('--calibration', default='results/structure_discovery_noise/calibration.json')
    parser.add_argument('--output', default='data/research_curriculum_v1')
    main(parser.parse_args())
