"""Assess the fixed uncertainty supplement on preserved synthetic outcomes."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np

from audit_statistical_sensitivity import as_rows, layout, proportion
from build_publication_curriculum import atomic_json, read, sha
from discovery.uncertainty_supplement import summarize_matrices, supplement_comparison


def run(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('Preserve existing outputs, including incomplete attempts.')
    previous = read(source/'result.json')
    if (not previous['completed'] or previous['scope'] != 'synthetic_design_sensitivity_not_student_results'
            or previous['formal_tests_used'] or previous['student_results_used']
            or len(previous['scenarios']) != 12):
        raise ValueError('Require the entire preserved synthetic study, never student outcomes.')
    for name, digest in previous['source_sha256'].items():
        if sha(name) != digest:
            raise ValueError('Original simulation source changed: '+name)
    sources = ('audit_uncertainty_supplement.py', 'discovery/uncertainty_supplement.py',
               'analyze_uncertainty_supplement.py', 'UNCERTAINTY_SUPPLEMENT_PROTOCOL.md',
               'audit_statistical_sensitivity.py', 'discovery/paired_statistics.py')
    config = {'scope': 'supplement_on_preserved_synthetic_outcomes',
              'previous_result_sha256': sha(source/'result.json'),
              'source_sha256': {p: sha(p) for p in sources}, 'numpy': np.__version__, 'python': sys.version,
              'formal_tests_used': False, 'student_results_used': False, 'primary_rule_replaced': False}
    atomic_json(output/'config.json', config)
    start, records, checks = time.perf_counter(), [], []
    try:
        for scenario in previous['scenarios']:
            path = source/scenario['saved_arrays']
            if sha(path) != scenario['arrays_sha256']:
                raise ValueError('Preserved synthetic arrays changed.')
            with np.load(path, allow_pickle=False) as old:
                b, c = old['B'], old['C']
                if b.dtype != np.bool_ or c.dtype != np.bool_ or b.shape != c.shape or b.shape != (512, 3, 100):
                    raise ValueError('Unexpected synthetic binary outcome grid.')
                difference = c.astype(np.int8)-b.astype(np.int8)
                groups = [[difference[:, :, worlds] for worlds in group] for group in layout()]
                current = summarize_matrices(groups)
                if not np.array_equal(current['point'], old['point']):
                    raise ValueError('Supplement changed an original point estimate.')
                if scenario['scenario'] in (0, 5, 11):
                    comparison = supplement_comparison(as_rows(b[0], c[0]))
                    if abs(comparison['C_minus_B']-current['point'][0]) > 1e-14:
                        raise ValueError('Formal-path and batched point estimates differ.')
                    checks.append({'scenario': scenario['scenario'], 'same_point': True,
                                   'same_structure_bound': comparison['independent_structure_assumption_bound']['interval']
                                   ==current['independent_structure_assumption_bound']['interval'][0].tolist()})
                    if not checks[-1]['same_structure_bound']:
                        raise ValueError('Formal-path and batched bounds differ.')
            world_ci = current['conditional_world_bound']['interval']
            structure_ci = current['independent_structure_assumption_bound']['interval']
            delta = scenario['true_difference']
            # In the structure-shared generator, conditioning on the realized
            # structures fixes their complete outcomes. Its conditional target
            # is the weighted realized difference, not the superpopulation delta.
            conditional_target = current['point'] if scenario['dependence']=='structure_shared' else np.full(512, delta)
            filename = f"scenario_{scenario['scenario']:02d}.npz"
            np.savez_compressed(output/filename, point=current['point'], conditional_world_target=conditional_target,
                                structure_population_target=np.full(512, delta), world_interval=world_ci,
                                structure_interval=structure_ci, leave_one_structure_out=current['leave_one_structure_out'],
                                structure_means=current['structure_means'], structure_weights=current['structure_weights'],
                                world_weights=current['world_weights'])
            records.append({'scenario': scenario['scenario'], 'dependence': scenario['dependence'],
                            'p_B': scenario['p_B'], 'true_structure_population_difference': delta,
                            'world_conditional_target': 'realized_structure_weighted_difference' if scenario['dependence']=='structure_shared' else 'prespecified_difference',
                            'world_conditional_coverage': proportion((world_ci[:, 0] <= conditional_target)&(world_ci[:, 1] >= conditional_target)),
                            'independent_structure_population_coverage': proportion((structure_ci[:, 0] <= delta)&(structure_ci[:, 1] >= delta)),
                            'mean_world_interval_width': float(np.diff(world_ci, axis=1).mean()),
                            'mean_structure_interval_width': float(np.diff(structure_ci, axis=1).mean()),
                            'world_half_width_before_clipping': current['conditional_world_bound']['half_width_before_clipping'],
                            'structure_half_width_before_clipping': current['independent_structure_assumption_bound']['half_width_before_clipping'],
                            'mean_leave_one_structure_out_span': float((current['leave_one_out_maximum']-current['leave_one_out_minimum']).mean()),
                            'arrays_sha256': sha(output/filename), 'saved_arrays': filename,
                            'original_arrays_sha256': scenario['arrays_sha256']})
        zero = np.zeros((1, 3, 100), dtype=bool)
        boundary = supplement_comparison(as_rows(zero[0], zero[0]))
        for name, digest in config['source_sha256'].items():
            if sha(name) != digest:
                raise ValueError('Supplement source changed during execution.')
    except BaseException as error:
        atomic_json(output/'failure.json', {'completed': False, 'type': type(error).__name__, 'error': str(error)})
        raise
    result = {'completed': True, 'scope': config['scope'], 'scenarios': records, 'point_and_bound_checks': checks,
              'zero_boundary': boundary, 'source_sha256': config['source_sha256'],
              'previous_result_sha256': config['previous_result_sha256'], 'elapsed_seconds': time.perf_counter()-start,
              'formal_tests_used': False, 'student_results_used': False, 'primary_rule_replaced': False}
    atomic_json(output/'result.json', result)
    lines = ['# 不确定性补充的人工成绩核验', '',
             '复用先前12×512个人工研究，没有重新挑选情景、读取学生结果或修改原分析。', '',
             '两个界的目标不同：世界界条件于已选机制；结构界要求独立结构单元，后者不是已验证的正式总体假设。',
             '在完全共享情景，世界界的条件真值是该次已实现结构的平均差，不能把它当跨新结构的预设差。', '',
             '| 相关性 | B概率 | 结构总体差（百分点） | 条件世界覆盖 / 512 | 独立结构覆盖 / 512 | 世界平均宽（百分点） | 结构平均宽（百分点） |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        label = '单元独立' if r['dependence']=='independent_cells' else '结构完全共享'
        lines.append(f"| {label} | {r['p_B']:.2f} | {100*r['true_structure_population_difference']:.0f} | "
                     f"{r['world_conditional_coverage']['count']} | {r['independent_structure_population_coverage']['count']} | "
                     f"{100*r['mean_world_interval_width']:.2f} | {100*r['mean_structure_interval_width']:.2f} |")
    a = boundary['conditional_world_bound']['half_width_before_clipping']
    b = boundary['independent_structure_assumption_bound']['half_width_before_clipping']
    lines += ['', f'全零点估计为0，两个界分别仍有±{100*a:.2f}和±{100*b:.2f}个百分点半宽，不退化为[0,0]。',
              '这些界很保守；模拟高覆盖不能证明真实数据满足独立性，也不能理解成更有检验力。',
              '逐次区间、明确的目标、结构影响值和Monte Carlo标准误全部保存。零模拟标准误不是总体必然性。',
              '未增加新的显著性判定。正式补充入口仍需六份完整配对结果，目前未执行。', '',
              f"同机CPU核验耗时{result['elapsed_seconds']:.2f}秒。", '']
    (output/'REPORT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(output/'REPORT_ZH.md', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='results/statistical_sensitivity_v1')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.source, args.output)
