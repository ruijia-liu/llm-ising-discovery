"""Calibrate then run a paired noise ladder without changing the v1 results.

python run_structure_discovery_noise.py calibrate
python run_structure_discovery_noise.py run --seed 92001 --output results/structure_discovery_noise/seed1
python run_structure_discovery_noise.py summarize --runs results/structure_discovery_noise/seed1
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from discovery.lab import Laboratory
from discovery.noise_evaluation import calibrate, evaluate_noisy, validate_calibration
from discovery.physics import Grammar, random_queries
from discovery.search import run_search
from run_structure_discovery import (CHECKPOINTS, FAMILIES, aggregate, compile_queries,
                                     evaluation_queries, make_world, paired_comparison, random_stream)

ROOT = Path('results/structure_discovery_noise')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def source_hashes():
    root = Path(__file__).resolve().parent
    names = ('run_structure_discovery_noise.py', 'run_structure_discovery.py', 'discovery/physics.py',
             'discovery/lab.py', 'discovery/search.py', 'discovery/noise_evaluation.py')
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def run_noise(args):
    if args.seed < 0 or args.worlds_per_family < 1 or args.eval_count < 16:
        raise ValueError('Invalid seed, world count, or evaluation count.')
    sigmas = [float(value) for value in args.noise_sigmas]
    if len(set(sigmas)) != len(sigmas) or any(not np.isfinite(v) or v <= 0 for v in sigmas):
        raise ValueError('Noise sigmas must be distinct, finite, positive values.')
    calibration_path = Path(args.calibration)
    calibration = json.loads(calibration_path.read_text(encoding='utf-8'))
    validate_calibration(calibration, args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # Do not silently overwrite an existing experiment or a partial run.
    if any((output/name).exists() for name in ('metrics.json', 'trajectories.jsonl', 'run_config.json')):
        raise ValueError('Output already contains a run; choose a fresh directory.')
    config = {'seed': args.seed, 'noise_sigmas': sigmas, 'worlds_per_family': args.worlds_per_family,
              'world_count': len(FAMILIES)*args.worlds_per_family, 'initial_queries': 8, 'query_budget': 32,
              'query_pool_size': 128, 'committee_size': 16, 'window': 3, 'maximum_order': 4, 'maximum_terms': 2,
              'coefficient_abs_range': [0.3, 0.8], 'evaluation_count_per_distribution': args.eval_count,
              'calibration_sha256': calibration['sha256'], 'thresholds': calibration['thresholds'],
              'calibration_world_count': calibration['world_count'], 'calibration_seed': calibration['seed'],
              'source_sha256': source_hashes(), 'python': platform.python_version(), 'numpy': np.__version__,
              'noise_model': 'independent Gaussian additive measurement errors with known sigma',
              'coupling': 'same initial data within sigma; common standard-normal draws across methods and sigmas',
              'search_penalty': 'sigma^2 * (2 log(158) + log(number_of_observations)) per nonzero term',
              'search_changed_from_v1': False}
    write_json(output/'run_config.json', config)  # Frozen before starting test worlds.
    write_json(output/'calibration_used.json', calibration)
    start = time.perf_counter()
    grammar = Grammar()
    rows, runs, truths = [], [], []
    index = 0
    with (output/'trajectories.jsonl').open('w', encoding='utf-8') as logfile:
        for family in FAMILIES:
            for _ in range(args.worlds_per_family):
                world_id = f'w{index:04d}'
                truth = make_world(grammar, family, np.random.default_rng(random_stream(args.seed, index, 11)))
                truths.append({'world_id': world_id, 'family': family, 'truth': truth.to_dict()})
                initial = random_queries(np.random.default_rng(random_stream(args.seed, index, 101)), 8)
                completed = []
                for sigma in sigmas:
                    first_events = []
                    for strategy in ('random', 'active'):
                        lab = Laboratory(truth.delta, noise_sigma=sigma,
                                         noise_seed=random_stream(args.seed, index, 404))
                        snapshots, events = run_search(lab, grammar, initial, strategy=strategy,
                                                      seed=random_stream(args.seed, index, 202))
                        if lab.used != 32:
                            raise AssertionError('Measurement budget violated.')
                        first_events.append(events[0])
                        completed.append((sigma, strategy, snapshots, events))
                    if first_events[0] != first_events[1]:
                        raise AssertionError('Initial noisy measurements differ between methods.')
                # The evaluator is invoked only after every explorer for this world finishes.
                evaluation = evaluation_queries(np.random.default_rng(random_stream(args.seed, index, 303)), args.eval_count)
                evaluation_features = [compile_queries(grammar, queries) for queries in evaluation]
                for sigma, strategy, snapshots, events in completed:
                    trajectory_scores = []
                    for used, hypothesis in snapshots.items():
                        score = evaluate_noisy(grammar, truth, hypothesis, evaluation_features, calibration['thresholds'])
                        row = {'seed': args.seed, 'world_id': world_id, 'family': family,
                               'noise_sigma': sigma, 'strategy': strategy, 'budget': used, **score}
                        rows.append(row)
                        trajectory_scores.append(row)
                    successful = [r['budget'] for r in trajectory_scores if r['full_recovery']]
                    sustained = next((r['budget'] for j, r in enumerate(trajectory_scores)
                                      if all(later['full_recovery'] for later in trajectory_scores[j:])), None)
                    runs.append({'world_id': world_id, 'family': family, 'noise_sigma': sigma, 'strategy': strategy,
                                 'first_full_recovery': min(successful) if successful else None,
                                 'sustained_full_recovery': sustained,
                                 'final_full_recovery': trajectory_scores[-1]['full_recovery']})
                    logfile.write(json.dumps({'world_id': world_id, 'noise_sigma': sigma, 'strategy': strategy,
                                              'events': events,
                                              'snapshots': {n: h.to_dict() for n, h in snapshots.items()}},
                                             ensure_ascii=False, allow_nan=False)+'\n')
                index += 1
            print(f'seed={args.seed} {family}: {args.worlds_per_family} worlds x {len(sigmas)} noise levels '
                  f'completed ({time.perf_counter()-start:.1f}s)', flush=True)
    aggregates = []
    paired = []
    for sigma in sigmas:
        selected = [r for r in rows if r['noise_sigma'] == sigma]
        aggregates.extend(dict(noise_sigma=sigma, **a) for a in aggregate(selected))
        paired.extend(dict(noise_sigma=sigma, **p) for p in paired_comparison(selected))
    metrics = dict(config, run_count=len(runs), elapsed_seconds=time.perf_counter()-start,
                   aggregates=aggregates, paired_comparison=paired, runs=runs)
    write_json(output/'metrics.json', metrics)
    write_json(output/'worlds_private.json', truths)
    with (output/'scores.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'seed': args.seed, 'at_32': [a for a in aggregates
                                                if a['family'] == 'all' and a['budget'] == 32]}, indent=2), flush=True)
    return metrics


def paired_interval(rows, sigma, metric, seed=87123):
    """Stratified paired bootstrap of test worlds, not individual observations.

    Report each noise level separately; reuses across sigmas are not independent
    extra worlds. Strata are generation seed and family; seed-to-seed generality
    is limited to the tested seeds.
    """
    paired = defaultdict(dict)
    for row in rows:
        if row['noise_sigma'] == sigma and row['budget'] == 32:
            paired[(row['seed'], row['family'], row['world_id'])][row['strategy']] = int(row[metric])
    strata = defaultdict(list)
    for (run_seed, family, _), pair in paired.items():
        if set(pair) != {'random', 'active'}:
            raise ValueError('Incomplete method pairing.')
        strata[run_seed, family].append(pair['active']-pair['random'])
    rng = np.random.default_rng(seed)
    differences = np.zeros(2000)
    for values in strata.values():
        values = np.asarray(values)
        differences += rng.choice(values, size=(2000, len(values)), replace=True).sum(axis=1)
    differences /= len(paired)
    low, high = np.quantile(differences, [0.025, 0.975])
    return {'metric': metric, 'active_minus_random': float(np.mean([v for a in strata.values() for v in a])),
            'stratified_world_bootstrap_95_interval': [float(low), float(high)]}


def summarize(run_paths, output):
    manifests, rows = [], []
    for path in map(Path, run_paths):
        manifests.append(json.loads((path/'metrics.json').read_text(encoding='utf-8')))
        with (path/'scores.csv').open(encoding='utf-8', newline='') as handle:
            for raw in csv.DictReader(handle):
                row = {k: (v == 'True' if k in ('prediction_success', 'support_success', 'full_recovery', 'false_discovery')
                           else int(v) if k in ('seed', 'budget') else float(v) if k in (
                               'noise_sigma', 'uniform_rmse', 'biased_rmse', 'coefficient_max_error') else v)
                       for k, v in raw.items()}
                rows.append(row)
    if not manifests:
        raise ValueError('Provide at least one completed run.')
    seeds = [m['seed'] for m in manifests]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Repeated seeds would double-count worlds.')
    for field in ('calibration_sha256', 'thresholds', 'noise_sigmas', 'worlds_per_family', 'source_sha256'):
        if any(m[field] != manifests[0][field] for m in manifests):
            raise ValueError(f'Runs disagree on {field}; do not combine them.')
    sigmas = manifests[0]['noise_sigmas']
    all_aggregates, comparisons = [], []
    for sigma in sigmas:
        selected = [r for r in rows if r['noise_sigma'] == sigma]
        all_aggregates.extend(dict(noise_sigma=sigma, **a) for a in aggregate(selected))
        comparisons.append({'noise_sigma': sigma, **paired_interval(rows, sigma, 'full_recovery')})
    thresholds = manifests[0]['thresholds']
    lookup = {(r['noise_sigma'], r['strategy'], r['family'], r['budget']): r for r in all_aggregates}
    count = sum(m['world_count'] for m in manifests)
    lines = ['# 增加难度：带测量噪声的结构发现实验', '',
             '**本轮只增加测量噪声，保留 3×3 结构范围、最多两项、8 次初始数据和 32 次总预算。**', '',
             f'共 {len(seeds)} 个随机种子、{count} 个不同世界，每个世界在 {len(sigmas)} 档噪声下比较两个方法。',
             '每档噪声下，两方法收到完全相同的初始带噪声数据。各档使用相同世界，不能把它们算成更多独立世界。', '',
             '## 32 次测量后的结果', '',
             '“结构正确”只检查哪些位置共同作用；“完整恢复”还要求系数和新构型预测达到统一精度。', '',
             '| 噪声标准差 | 随机：结构正确 | 主动：结构正确 | 随机：完整恢复 | 主动：完整恢复 |',
             '|---|---:|---:|---:|---:|']
    for sigma in sigmas:
        random = lookup[sigma, 'random', 'all', 32]
        active = lookup[sigma, 'active', 'all', 32]
        lines.append(f"| {sigma:g} | {random['support_success_rate']:.1%} | {active['support_success_rate']:.1%} | "
                     f"{random['full_recovery_rate']:.1%} | {active['full_recovery_rate']:.1%} |")
    lines += ['', '## 完整恢复的预算曲线', '', '| 噪声标准差 | 方法 | 8 次 | 16 次 | 24 次 | 32 次 |',
              '|---|---|---:|---:|---:|---:|']
    for sigma in sigmas:
        for strategy, name in (('random', '随机'), ('active', '主动')):
            rates = [f"{lookup[sigma, strategy, 'all', b]['full_recovery_rate']:.1%}" for b in CHECKPOINTS]
            lines.append(f'| {sigma:g} | {name} | ' + ' | '.join(rates) + ' |')
    lines += ['', '## 连续误差与误报', '',
              '误差针对真实的、无噪声的隐藏能量贡献；测试测量噪声不混入评估目标。', '',
              '| 噪声标准差 | 方法 | 均匀构型平均 RMSE | 偏置构型平均 RMSE | 零作用世界误报率 |',
              '|---|---|---:|---:|---:|']
    for sigma in sigmas:
        for strategy, name in (('random', '随机'), ('active', '主动')):
            a = lookup[sigma, strategy, 'all', 32]
            null = lookup[sigma, strategy, 'null', 32]
            lines.append(f"| {sigma:g} | {name} | {a['mean_uniform_rmse']:.4f} | {a['mean_biased_rmse']:.4f} | "
                         f"{null['false_discovery_rate']:.1%} |")
    lines += ['', '## 按世界类型分开看（32 次完整恢复）', '',
              '| 噪声标准差 | 世界 | 随机 | 主动 |', '|---|---|---:|---:|']
    labels = dict(zip(FAMILIES, ('方向差异', '较远二体', '三体', '四体', '混合', '无额外作用')))
    for sigma in sigmas:
        for family in FAMILIES:
            r = lookup[sigma, 'random', family, 32]['full_recovery_rate']
            a = lookup[sigma, 'active', family, 32]['full_recovery_rate']
            lines.append(f'| {sigma:g} | {labels[family]} | {r:.1%} | {a:.1%} |')
    lines += ['', '## 统一判分标准', '',
              f"先用独立的 {manifests[0]['calibration_world_count']} 个验证世界校准，再冻结校准文件。所有噪声强度、两种方法、所有预算点都使用同一标准。",
              f"- 均匀构型 RMSE ≤ {thresholds['uniform_rmse']:.6f}。",
              f"- 偏置构型 RMSE ≤ {thresholds['biased_rmse']:.6f}。",
              f"- 最大系数误差 ≤ {thresholds['coefficient_max_error']:.6f}，且非零结构完全正确。",
              'RMSE 容忍度取验证世界中较弱作用信号的第 5 百分位的一半；系数容忍度取验证集中最小非零系数的一半。',
              '这是预先定义的工程精度目标，不是统计显著性阈值，也不保证特定误报率。',
              '完整恢复百分比不可直接与 v1 的 1e-7／1e-6 无噪声精确容差比较；结构正确率和原始误差可直接解释。', '',
              '## 配对比较', '',
              '以下为主动减随机的完整恢复率差；按世界配对，在每个种子／世界类型内进行 2,000 次 bootstrap。',
              '区间仅描述本次世界样本，不包含语言模型训练随机性；不把重复噪声条件视为独立样本。', '']
    for c in comparisons:
        low, high = c['stratified_world_bootstrap_95_interval']
        lines.append(f"- σ={c['noise_sigma']:g}：{100*c['active_minus_random']:+.1f} 个百分点，"
                     f'95% 区间 [{100*low:+.1f}, {100*high:+.1f}]。')
    lines += ['', '## 限制和可复核文件', '',
              '- 仍是普通稀疏搜索基线，没有训练 A／B／C 学生，不能解释为过程教学获胜。',
              '- 搜索规则没有根据测试结果调整；它沿用原有的噪声惩罚和有限候选委员会。',
              '- 三档噪声全部保留，不只挑主动方法表现最好的档位。',
              '- 真值、评估和阈值均不传给搜索器；现有接口仍不是防恶意访问的进程沙箱。',
              '- 下一轮可选择中等难度条件，但若据此选择，应把本轮视作开发实验，并使用新的测试世界。',
              '- calibration.json 保存校准规则和验证世界；每个种子目录保存配置、源代码哈希、私有真值、完整轨迹和逐轮分数。',
              '- 本报告没有调整第一版无噪声结果。', '']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'SUMMARY_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    summary = {'seeds': seeds, 'world_count': count, 'run_count': sum(m['run_count'] for m in manifests),
               'noise_sigmas': sigmas, 'calibration_sha256': manifests[0]['calibration_sha256'],
               'thresholds': thresholds, 'aggregates': all_aggregates, 'paired_comparisons': comparisons}
    write_json(output/'summary.json', summary)
    print(json.dumps({'world_count': count, 'paired_comparisons': comparisons, 'report': str(output/'SUMMARY_ZH.md')}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    calibration = subs.add_parser('calibrate')
    calibration.add_argument('--seed', type=int, default=91001)
    calibration.add_argument('--worlds-per-family', type=int, default=10)
    calibration.add_argument('--eval-count', type=int, default=512)
    calibration.add_argument('--output', default=str(ROOT/'calibration.json'))
    run = subs.add_parser('run')
    run.add_argument('--seed', type=int, default=92001)
    run.add_argument('--worlds-per-family', type=int, default=10)
    run.add_argument('--eval-count', type=int, default=256)
    run.add_argument('--noise-sigmas', type=float, nargs='+', default=[0.5, 1.0, 2.0])
    run.add_argument('--calibration', default=str(ROOT/'calibration.json'))
    run.add_argument('--output', default=str(ROOT/'seed1'))
    summary = subs.add_parser('summarize')
    summary.add_argument('--runs', nargs='+', required=True)
    summary.add_argument('--output', default=str(ROOT))
    args = parser.parse_args()
    if args.command == 'calibrate':
        path = Path(args.output)
        if path.exists():
            raise ValueError('Calibration file already exists; use it or choose a new output path.')
        calibration = calibrate(seed=args.seed, worlds_per_family=args.worlds_per_family, evaluation_count=args.eval_count)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, calibration)
        print(json.dumps({'thresholds': calibration['thresholds'], 'sha256': calibration['sha256']}, indent=2))
    elif args.command == 'run':
        run_noise(args)
    else:
        summarize(args.runs, args.output)


if __name__ == '__main__':
    main()
