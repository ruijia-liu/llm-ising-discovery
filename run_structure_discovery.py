"""Run the NOISELESS feasibility pilot; does not train language models.

Example: python run_structure_discovery.py --worlds-per-family 20
Truth, evaluation sets and scoring remain outside run_search().
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import platform
import time

import numpy as np

from discovery.lab import Laboratory
from discovery.physics import Expression, Grammar, SIDE, Term, canonical_structure, random_queries
from discovery.search import run_search

FAMILIES = ('direction', 'distance_two', 'three_body', 'four_body', 'mixed', 'null')
CHECKPOINTS = (8, 16, 24, 32)
PREDICTION_TOLERANCE = 1e-7
COEFFICIENT_TOLERANCE = 1e-6


def random_stream(seed, world_index, purpose):
    return np.random.SeedSequence([seed, world_index, purpose])


def make_world(grammar, family, rng):
    """Private generator; search receives neither this function nor its family."""
    by_order = {k: [p for p in grammar.structures if len(p) == k] for k in range(1, 5)}
    if family == 'null':
        return Expression()
    if family == 'direction':
        possible = [canonical_structure(((0, 0), (0, 1))),
                    canonical_structure(((0, 0), (1, 0)))]
    elif family == 'distance_two':
        possible = [p for p in by_order[2]
                    if max(min(v, SIDE-v) for v in p[1]) == 2]
    elif family == 'three_body':
        possible = by_order[3]
    elif family == 'four_body':
        possible = by_order[4]
    elif family == 'mixed':
        orders = rng.choice([2, 3, 4], size=2, replace=False)
        chosen = [by_order[int(k)][int(rng.integers(len(by_order[int(k)])))] for k in orders]
        possible = None
    else:
        raise ValueError('Unknown world family.')
    if possible is not None:
        count = int(rng.integers(1, 3))
        chosen = [possible[int(j)] for j in rng.choice(len(possible), count, replace=False)]
    coefficients = rng.uniform(0.3, 0.8, len(chosen)) * rng.choice([-1, 1], len(chosen))
    return Expression(tuple(Term(points, float(c)) for points, c in zip(chosen, coefficients)))


def evaluation_queries(rng, count):
    """Two distributions, neither disclosed to the algorithm before it finishes."""
    uniform = random_queries(rng, count)
    biased = []
    for j in range(count):
        density = (0.0, 0.05, 0.15, 0.85, 0.95, 1.0)[j % 6]
        state = np.where(rng.random((SIDE, SIDE)) < density, 1, -1).astype(np.int8)
        position = tuple(int(v) for v in rng.integers(SIDE, size=2))
        biased.append((state, position))
    return uniform, biased


def compile_queries(grammar, queries):
    # Bound temporary-array memory on small machines.
    return np.vstack([grammar.features([q[0] for q in queries[i:i+64]],
                                      [q[1] for q in queries[i:i+64]])
                      for i in range(0, len(queries), 64)])


def evaluate(grammar, truth, hypothesis, evaluation_features):
    difference = grammar.coefficients(hypothesis) - grammar.coefficients(truth)
    rmses = [float(np.sqrt(np.mean((features @ difference)**2))) for features in evaluation_features]
    coefficient_error = float(np.max(np.abs(difference)))
    same_support = hypothesis.support == truth.support
    prediction_success = all(value <= PREDICTION_TOLERANCE for value in rmses)
    return {'uniform_rmse': rmses[0], 'biased_rmse': rmses[1],
            'prediction_success': prediction_success, 'support_success': same_support,
            'coefficient_max_error': coefficient_error,
            'full_recovery': same_support and coefficient_error <= COEFFICIENT_TOLERANCE and prediction_success,
            'false_discovery': not truth.terms and bool(hypothesis.terms)}


def aggregate(rows):
    results = []
    for strategy in ('random', 'active'):
        for family in ('all',) + FAMILIES:
            for budget in CHECKPOINTS:
                selected = [r for r in rows if r['strategy'] == strategy and r['budget'] == budget
                            and (family == 'all' or r['family'] == family)]
                results.append({'strategy': strategy, 'family': family, 'budget': budget,
                                'worlds': len(selected),
                                'prediction_success_rate': float(np.mean([r['prediction_success'] for r in selected])),
                                'support_success_rate': float(np.mean([r['support_success'] for r in selected])),
                                'full_recovery_rate': float(np.mean([r['full_recovery'] for r in selected])),
                                'mean_uniform_rmse': float(np.mean([r['uniform_rmse'] for r in selected])),
                                'mean_biased_rmse': float(np.mean([r['biased_rmse'] for r in selected])),
                                'false_discovery_rate': (float(np.mean([r['false_discovery'] for r in selected]))
                                                         if family == 'null' else None)})
    return results


def paired_comparison(rows):
    results = []
    for budget in CHECKPOINTS:
        random = {r['world_id']: r['full_recovery'] for r in rows
                  if r['budget'] == budget and r['strategy'] == 'random'}
        active = {r['world_id']: r['full_recovery'] for r in rows
                  if r['budget'] == budget and r['strategy'] == 'active'}
        results.append({'budget': budget,
                        'active_only_success': sum(active[w] and not random[w] for w in random),
                        'random_only_success': sum(random[w] and not active[w] for w in random),
                        'both_success': sum(random[w] and active[w] for w in random),
                        'neither_success': sum(not random[w] and not active[w] for w in random)})
    return results


def report(summary):
    labels = {'direction': '方向差异', 'distance_two': '较远二体', 'three_body': '三体',
              'four_body': '四体', 'mixed': '不同阶数组合', 'null': '没有额外作用'}
    lookup = {(r['strategy'], r['family'], r['budget']): r for r in summary['aggregates']}
    lines = ['# 广义 Ising 结构发现：第一版无噪声先导实验', '',
             '**本次只检验普通搜索程序能否在预算内恢复规律；未训练 A／B／C 学生，不能评价过程教学。**', '',
             f"实际完成 {summary['world_count']} 个世界、{summary['run_count']} 条探索轨迹。",
             '两种方法共享每个世界的前 8 次测量，各自最多使用 32 次；每次翻转测量计费一次。', '',
             '## 固定设置', '',
             '- 16×16 周期晶格，已知最近邻基础哈密顿量。',
             '- 公共表达语法：3×3 窗口中一至四个不同位置，平移不变，最多两种隐藏作用。',
             f"- 自动枚举 {summary['grammar_size']} 个规范化结构，不向算法提供真实世界使用的结构名单。",
             '- 非零系数绝对值均匀分布于 [0.3, 0.8]，正负均可。',
             '- 搜索所有空／单项／双项支持集并拟合系数；主动方法从 128 个随机实验中优先选预测分歧大的。',
             f"- 每个模型另用 {summary['evaluation_count_per_distribution']} 个独立均匀构型及相同数量的偏置构型检验。",
             '- 预测通过：两个构型分布的隐藏贡献 RMSE 都不超过 1e-7。',
             '- 完整恢复：预测通过、结构完全正确且最大系数误差不超过 1e-6。',
             '- 阈值为无噪声数值容差，不是从测试成绩选择的噪声阈值。', '',
             '## 预算曲线', '', '| 方法 | 8 次 | 16 次 | 24 次 | 32 次 |', '|---|---:|---:|---:|---:|']
    for strategy, name in [('random', '随机选实验'), ('active', '按候选分歧选实验')]:
        cells = [f"{lookup[strategy, 'all', b]['full_recovery_rate']:.1%}" for b in CHECKPOINTS]
        lines.append('| ' + ' | '.join([name] + cells) + ' |')
    initial_rate = lookup['random', 'all', 8]['full_recovery_rate']
    interpretation = ('本轮初始 8 次就有至少 95% 的世界恢复正确，存在明显天花板效应。'
                      '不能用最终接近满分的结果判断主动实验或过程教学是否有优势。'
                      if initial_rate >= 0.95 else '初始观察未覆盖所有世界；请结合预算曲线和配对结果评估额外实验的作用。')
    lines += ['', '上表为完整恢复比例。均匀和偏置构型上的误差、预测通过率和结构恢复率分别保存在 metrics.json 和 scores.csv。',
              '', interpretation,
              '8 次的成功表示离线检验认为拟合模型正确，不表示学生已完成一次带事前预测的主动验证。',
              '', '## 32 次时分类型结果', '', '| 世界 | 随机选实验 | 按候选分歧选实验 |', '|---|---:|---:|']
    for family in FAMILIES:
        cells = [f"{lookup[s, family, 32]['full_recovery_rate']:.1%}" for s in ('random', 'active')]
        lines.append('| ' + ' | '.join([labels[family]] + cells) + ' |')
    lines += ['', '## 首次成功与虚假发现', '']
    for strategy, name in [('random', '随机方法'), ('active', '主动方法')]:
        details = [r for r in summary['runs'] if r['strategy'] == strategy]
        hits = [r['first_full_recovery'] for r in details if r['first_full_recovery'] is not None]
        median = f'{np.median(hits):g}' if hits else '无成功世界'
        false_rate = lookup[strategy, 'null', 32]['false_discovery_rate']
        lines.append(f'- {name}：成功轨迹的首次完整恢复查询数中位数为 {median}；'
                     f'{len(details)-len(hits)} 条轨迹未成功；零作用世界最终虚假发现率 {false_rate:.1%}。')
    lines += ['', '首次成功只在成功轨迹中统计；8 次以下没有评估点，不代表必须用满 8 次。',
              '每次查询后的模型均被保存并在探索结束后评分；另记录从某点起持续正确的最早查询数。', '',
              '## 解释边界', '',
              '- 这是一个指定表达空间内的可行性试验，不是训练／验证／测试的机制泛化研究。',
              '- “四体”表示搜索程序能处理四体世界，不表示语言模型从三体教学泛化到了四体。',
              '- 初始数据足够时，两种方法会在 8 次就成功；这意味着下一阶段可能存在天花板效应。',
              '- 主动策略只使用有限候选委员会和随机实验池，不是最优实验设计，也不是完整可识别性证明。',
              '- 日志锁定模型、查询和预测后才返回观测；代码通过接口隔离真值，但尚不是防恶意访问的进程沙箱。',
              '- 独立检验构型来自分离的随机流；偏置检验中可重复出现全正／全负构型，且不向搜索程序返回评估结果。',
              '- 本版只运行无噪声实验；噪声成功阈值、语言模型教学与正式机制划分仍未实施。', '',
              '## 输出文件', '',
              '- metrics.json：运行设置、环境、汇总和每条轨迹的首次／持续成功时间。',
              '- scores.csv：每次查询后的离线分数。',
              '- trajectories.jsonl：完整初始观测、锁定预测、后续观测和最终表达式。',
              '- worlds_private.json：评估器的真实机制；未来学生进程不得读取。', '',
              f"运行耗时：{summary['elapsed_seconds']:.1f} 秒。主随机种子：{summary['seed']}。", '']
    return '\n'.join(lines)


def run(args):
    if args.worlds_per_family < 1 or args.eval_count < 16:
        raise ValueError('Use at least one world per family and 16 evaluation queries per distribution.')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    grammar = Grammar()
    start = time.perf_counter()
    rows, run_info, truths = [], [], []
    world_index = 0
    with (output / 'trajectories.jsonl').open('w', encoding='utf-8') as trajectory_file:
        for family in FAMILIES:
            for _ in range(args.worlds_per_family):
                world_id = f'w{world_index:04d}'
                truth = make_world(grammar, family, np.random.default_rng(random_stream(args.seed, world_index, 11)))
                truths.append({'world_id': world_id, 'family': family, 'truth': truth.to_dict()})
                initial = random_queries(np.random.default_rng(random_stream(args.seed, world_index, 101)), 8)
                completed = []
                initial_events = []
                for strategy in ('random', 'active'):
                    lab = Laboratory(truth.delta, noise_seed=random_stream(args.seed, world_index, 404))
                    snapshots, events = run_search(lab, grammar, initial, strategy=strategy,
                                                  seed=random_stream(args.seed, world_index, 202))
                    if lab.used != 32:
                        raise AssertionError('Budget accounting failed.')
                    initial_events.append(events[0])
                    completed.append((strategy, snapshots, events))
                if initial_events[0] != initial_events[1]:
                    raise AssertionError('Methods did not receive identical initial data.')
                # Evaluation is generated and applied only AFTER BOTH explorers finish.
                evaluation = evaluation_queries(np.random.default_rng(random_stream(args.seed, world_index, 303)), args.eval_count)
                evaluation_features = [compile_queries(grammar, queries) for queries in evaluation]
                for strategy, snapshots, events in completed:
                    scores = []
                    for used, hypothesis in snapshots.items():
                        score = evaluate(grammar, truth, hypothesis, evaluation_features)
                        row = dict(world_id=world_id, family=family, strategy=strategy, budget=used, **score)
                        rows.append(row)
                        scores.append(row)
                    successful = [r['budget'] for r in scores if r['full_recovery']]
                    sustained = next((r['budget'] for j, r in enumerate(scores)
                                      if all(later['full_recovery'] for later in scores[j:])), None)
                    run_info.append({'world_id': world_id, 'family': family, 'strategy': strategy,
                                     'first_full_recovery': min(successful) if successful else None,
                                     'sustained_full_recovery': sustained})
                    trajectory_file.write(json.dumps({'world_id': world_id, 'strategy': strategy,
                                                       'events': events,
                                                       'snapshots': {n: h.to_dict() for n, h in snapshots.items()}},
                                                      ensure_ascii=False, allow_nan=False)+'\n')
                world_index += 1
            print(f'{family}: {args.worlds_per_family} worlds completed ({time.perf_counter()-start:.1f}s)', flush=True)
    summary = {'seed': args.seed, 'world_count': world_index, 'run_count': len(run_info),
               'worlds_per_family': args.worlds_per_family, 'grammar_size': len(grammar.structures),
               'grammar_by_order': dict(Counter(len(p) for p in grammar.structures)),
               'window': 3, 'maximum_order': 4, 'maximum_terms': 2, 'coefficient_abs_range': [0.3, 0.8],
               'noise_sigma': 0.0, 'initial_queries': 8, 'query_budget': 32, 'query_pool_size': 128,
               'committee_size': 16, 'evaluation_count_per_distribution': args.eval_count,
               'prediction_tolerance': PREDICTION_TOLERANCE, 'coefficient_tolerance': COEFFICIENT_TOLERANCE,
               'python': platform.python_version(), 'numpy': np.__version__,
               'elapsed_seconds': time.perf_counter()-start,
               'aggregates': aggregate(rows), 'paired_comparison': paired_comparison(rows), 'runs': run_info}
    (output / 'metrics.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    (output / 'worlds_private.json').write_text(json.dumps(truths, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    with (output / 'scores.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'REPORT_ZH.md').write_text(report(summary), encoding='utf-8')
    print(json.dumps({'output': str(output.resolve()), 'paired_comparison': summary['paired_comparison']}, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worlds-per-family', type=int, default=20)
    parser.add_argument('--eval-count', type=int, default=256)
    parser.add_argument('--seed', type=int, default=20260926)
    parser.add_argument('--output', default='results/structure_discovery_v1')
    run(parser.parse_args())
