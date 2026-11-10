"""Outcome-independent synthetic audit of the frozen primary bootstrap.

Does not import a world loader, train a model, or read student/test outcomes.
"""
import argparse
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import sys
import time

import numpy as np

from build_publication_curriculum import atomic_json, sha
from discovery.paired_statistics import primary_comparison


LEVELS = ('combination', 'four_body')
SIZES = ((5,)*6+(4,)*5, (1,)*50)
SEEDS = (401, 402, 403)
PROBABILITIES = ((.20, 0.), (.20, .05), (.20, .10), (.20, .20), (.02, 0.), (.02, .05))
REPETITIONS = 512
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 98901
SIMULATION_SEED = 99071


def layout():
    groups, cursor = [], 0
    for sizes in SIZES:
        current = []
        for size in sizes:
            current.append(np.arange(cursor, cursor+size))
            cursor += size
        groups.append(current)
    return groups


def sampling_plan(repetitions=BOOTSTRAPS, seed=BOOTSTRAP_SEED):
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError('Bootstrap repetitions must be a positive integer.')
    rng, groups = np.random.default_rng(seed), layout()
    # Sentinel 100 is zero-padding, never a resampled world.
    indices = [np.full((repetitions, len(g), max(map(len, g))), 100, dtype=np.int16) for g in groups]
    divisors = [np.empty((repetitions, len(g)), dtype=np.int16) for g in groups]
    for b in range(repetitions):
        for level, structures in enumerate(groups):
            for slot, chosen in enumerate(rng.integers(len(structures), size=len(structures))):
                worlds = structures[int(chosen)]
                drawn = rng.integers(len(worlds), size=len(worlds))
                indices[level][b, slot, :len(worlds)] = worlds[drawn]
                divisors[level][b, slot] = len(SEEDS)*len(worlds)
    return indices, divisors


def calculate(b, c, plan, batch=8):
    if (b.dtype != np.bool_ or c.dtype != np.bool_ or b.shape != c.shape
            or b.ndim != 3 or b.shape[1:] != (3, 100)):
        raise ValueError('Require paired boolean arrays with shape (studies, 3, 100).')
    if type(batch) is not int or batch < 1:
        raise ValueError('Batch size must be a positive integer.')
    differences = c.astype(np.int8)-b.astype(np.int8)
    groups = layout()
    # Use the same order of means as primary_comparison, including seed effects.
    seed_effects = np.mean([np.stack([differences[:, :, w].mean(axis=2) for w in structures], axis=1).mean(axis=1)
                            for structures in groups], axis=0)
    points = seed_effects.mean(axis=1)
    intervals = []
    for start in range(0, len(b), batch):
        # Each resampled structure mean is an integer sum divided by 3*n.
        # Padding only supplies zeros; denominator uses the actual sampled size.
        summed = differences[start:start+batch].sum(axis=1, dtype=np.int16)
        summed = np.pad(summed, ((0, 0), (0, 1)))
        levels = []
        for index, divisor in zip(*plan):
            values = summed[:, index].sum(axis=-1, dtype=np.int64)/divisor
            levels.append(values.mean(axis=-1))
        samples = np.stack(levels, axis=-1).mean(axis=-1)
        intervals.append(np.quantile(samples, [.025, .975], axis=1, method='linear').T)
    ci = np.concatenate(intervals, axis=0)
    return {'point': points, 'seed_effects': seed_effects, 'interval': ci,
            'advantage': (ci[:, 0] > 0) & np.all(seed_effects > 0, axis=1)}


def paired_draw(rng, shape, p_b, p_c):
    if not (0 <= p_b <= 1 and 0 <= p_c <= 1):
        raise ValueError('Success probabilities must lie in [0, 1].')
    q = .5*min(p_b, p_c)+.5*p_b*p_c
    u = rng.random(shape)
    b = u < p_b
    c = (u < q) | ((u >= p_b) & (u < p_b+p_c-q))
    return b, c


def generate(scenario, shared, p_b, difference, repetitions=REPETITIONS):
    rng = np.random.default_rng(np.random.SeedSequence([SIMULATION_SEED, scenario]))
    if shared:
        b, c = paired_draw(rng, (repetitions, 61), p_b, p_b+difference)
        owner = np.repeat(np.arange(61), np.concatenate(SIZES))
        return (np.repeat(b[:, None, owner], 3, axis=1),
                np.repeat(c[:, None, owner], 3, axis=1))
    return paired_draw(rng, (repetitions, 3, 100), p_b, p_b+difference)


def as_rows(b, c):
    rows = []
    for level, structures in zip(LEVELS, layout()):
        for cluster, worlds in enumerate(structures):
            for parameter, world in enumerate(worlds):
                for group, values in (('B', b), ('C', c)):
                    for s, seed in enumerate(SEEDS):
                        rows.append({'group': group, 'seed': seed, 'level': level,
                                     'structure_id': f'{level}-{cluster:02d}',
                                     'world_id': f'{level}-{cluster:02d}-{parameter:02d}',
                                     'sigma': 1, 'mode': 'active', 'score': bool(values[s, world])})
    return rows


def compare_reference(b, c, optimized, index=0, repetitions=BOOTSTRAPS, seed=BOOTSTRAP_SEED):
    reference = primary_comparison(as_rows(b[index], c[index]), repetitions=repetitions, seed=seed)
    actual = [optimized['point'][index], *optimized['interval'][index], *optimized['seed_effects'][index]]
    expected = [reference['C_minus_B'], *reference['conditional_95_percentile_interval'],
                *[reference['seed_effects'][str(s)] for s in SEEDS]]
    error = float(np.max(np.abs(np.asarray(actual)-expected)))
    if error > 2e-15 or bool(optimized['advantage'][index]) != reference['consistent_process_advantage']:
        raise ValueError('Accelerated synthetic calculation differs from the frozen analysis.')
    return {'maximum_absolute_difference': error, 'decision_identical': True,
            'bootstrap_repetitions': repetitions, 'reference': reference}


def proportion(value):
    value = np.asarray(value, dtype=bool)
    p = float(value.mean())
    return {'count': int(value.sum()), 'denominator': len(value), 'fraction': p,
            'monte_carlo_standard_error': float(np.sqrt(p*(1-p)/len(value)))}


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Preserve prior audit directories, including incomplete attempts.')
    sources = ('audit_statistical_sensitivity.py', 'discovery/paired_statistics.py',
               'MAINLINE_STUDY_PROTOCOL.md', 'STATISTICAL_SENSITIVITY_PROTOCOL.md')
    config = {'scope': 'synthetic_design_sensitivity_not_student_results', 'simulation_seed': SIMULATION_SEED,
              'simulation_repetitions': REPETITIONS, 'bootstrap_seed': BOOTSTRAP_SEED,
              'bootstrap_repetitions': BOOTSTRAPS, 'sizes': SIZES, 'probabilities': PROBABILITIES,
              'source_sha256': {p: sha(p) for p in sources}, 'numpy': np.__version__, 'python': sys.version,
              'formal_tests_used': False, 'student_results_used': False, 'gpu_models_used': False}
    atomic_json(output/'config.json', config)
    started, results, checks = time.perf_counter(), [], []
    try:
        plan = sampling_plan()
        np.savez_compressed(output/'bootstrap_indices.npz', combination=plan[0][0], four_body=plan[0][1],
                            combination_divisor=plan[1][0], four_body_divisor=plan[1][1])
        scenario = 0
        for shared in (False, True):
            for p_b, difference in PROBABILITIES:
                b, c = generate(scenario, shared, p_b, difference)
                estimates = calculate(b, c, plan)
                if scenario in (0, 5, 11):
                    checks.append({'scenario': scenario, **compare_reference(b, c, estimates)})
                filename = f'scenario_{scenario:02d}.npz'
                np.savez_compressed(output/filename, B=b, C=c, **estimates)
                ci, point = estimates['interval'], estimates['point']
                row = {'scenario': scenario, 'p_B': p_b, 'p_C': p_b+difference, 'true_difference': difference,
                       'dependence': 'structure_shared' if shared else 'independent_cells',
                       'bias': float((point-difference).mean()), 'empirical_standard_deviation': float(point.std(ddof=1)),
                       'mean_interval_width': float(np.diff(ci, axis=1).mean()),
                       'coverage': proportion((ci[:, 0] <= difference) & (ci[:, 1] >= difference)),
                       'advantage_rule': proportion(estimates['advantage']),
                       'all_zero': proportion(~b.any(axis=(1, 2)) & ~c.any(axis=(1, 2))),
                       'saved_arrays': filename, 'arrays_sha256': sha(output/filename)}
                results.append(row)
                atomic_json(output/'progress.json', {'completed_scenarios': len(results), 'scenarios': results})
                print(f"Scenario {scenario:02d}: {row['dependence']} B={p_b:.2f} delta={difference:.2f}; "
                      f"rule={row['advantage_rule']['count']}/{REPETITIONS}, coverage={row['coverage']['count']}/{REPETITIONS}", flush=True)
                scenario += 1
        zero = np.zeros((1, 3, 100), dtype=bool)
        zero_result = calculate(zero, zero, plan)
        boundary = compare_reference(zero, zero, zero_result)
        for name, checksum in config['source_sha256'].items():
            if sha(name) != checksum:
                raise ValueError('Source or protocol changed during the audit.')
    except BaseException as error:
        atomic_json(output/'failure.json', {'completed': False, 'type': type(error).__name__, 'error': str(error),
                                          'completed_scenarios': len(results)})
        raise
    result = {'completed': True, 'scope': config['scope'], 'scenarios': results,
              'reference_checks': checks, 'all_zero_boundary_reference': boundary,
              'elapsed_seconds': time.perf_counter()-started, 'formal_tests_used': False,
              'student_results_used': False, 'source_sha256': config['source_sha256'],
              'bootstrap_indices_sha256': sha(output/'bootstrap_indices.npz')}
    atomic_json(output/'result.json', result)
    lines = ['# 冻结主要分析的人工成绩压力检查', '',
             '仅人工布尔成绩；未读取正式保留世界或任何学生结果。主训练与主分析均未改变。', '',
             '每层50世界，组合11结构按6×5+5×4的示意布局，四体50结构；三个固定种子。',
             '每情景512次人工研究，每次采用原10,000次bootstrap及原优势规则。',
             '下列“优势”表示模拟中满足判定规则的比例，不是本实验的学生成绩。', '',
             '| 相关性 | B成功概率 | 真实差（百分点） | 判为优势 / 512 | MC标准误（百分点） | 区间覆盖 / 512 | 覆盖MC标准误（百分点） | 平均区间宽（百分点） |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        a, v = r['advantage_rule'], r['coverage']
        label = '单元独立' if r['dependence'] == 'independent_cells' else '结构完全共享'
        lines.append(f"| {label} | {r['p_B']:.2f} | {100*r['true_difference']:.0f} | {a['count']} | "
                     f"{100*a['monte_carlo_standard_error']:.2f} | {v['count']} | "
                     f"{100*v['monte_carlo_standard_error']:.2f} | {100*r['mean_interval_width']:.2f} |")
    lines += ['', '原分析函数与加速实现的三个完整模拟对照及全零边界对照通过，结果及逐次成绩数组均保留。',
              '全零边界得到[0,0]，不判为优势；这个经验区间不能证明总体精确等效。', '',
              '模拟比例的最大Monte Carlo标准误约2.21个百分点；零标准误不保证总体概率恰为0或1。',
              '这些假设不是实测学生分布，不给实际研究贴“功效充足”标签，不把覆盖率称为保证。',
              '同结构相关性、地板效应和少量独立结构须结合实际结果解释；不按模拟成绩改主协议。', '',
              f"本机耗时{result['elapsed_seconds']:.1f}秒，与主训练重叠，不作独占硬件基准。", '']
    (output/'REPORT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(output/'REPORT_ZH.md', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
