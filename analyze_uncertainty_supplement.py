"""Supplement the unchanged main analysis after six formal active runs complete."""
import argparse
from pathlib import Path

from build_publication_curriculum import atomic_json, sha
from discovery.evaluation_results import load_completed_evaluation, verify_pairing
from discovery.uncertainty_supplement import supplement_comparison


def run(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError('Preserve previous supplement outputs.')
    evaluations = [load_completed_evaluation(root/f'{group}_seed{seed}', scope='formal', study='mainline',
                                            method='llm_active', group=group, seed=seed)
                   for group in ('B', 'C') for seed in (401, 402, 403)]
    verify_pairing(evaluations)
    rows = [{**r, 'mode': 'active', 'score': r['score']['tested_idea_recovery']}
            for e in evaluations for r in e['result']['rows']]
    comparison = supplement_comparison(rows)
    result = {'completed': True, 'analysis': 'uncertainty_supplement', 'comparison': comparison,
              'evaluations': [e['provenance'] for e in evaluations],
              'source_sha256': {p: sha(p) for p in ('analyze_uncertainty_supplement.py',
                                  'discovery/uncertainty_supplement.py', 'discovery/paired_statistics.py',
                                  'discovery/evaluation_results.py', 'UNCERTAINTY_SUPPLEMENT_PROTOCOL.md')},
              'primary_analysis_replaced': False}
    atomic_json(output/'result.json', result)
    lines = ['# 主要终点的不确定性补充', '', '原确认性分析仍单独完整报告；本补充不新增胜负判定。', '',
             f"同一结构等权点估计 C−B：{100*comparison['C_minus_B']:+.2f} 个百分点。", '',
             '| 区间 | 下界（百分点） | 上界（百分点） |', '|---|---:|---:|']
    for label, key in (('条件于已选机制、假设完整世界独立', 'conditional_world_bound'),
                       ('额外假设结构单元独立的敏感性界', 'independent_structure_assumption_bound')):
        lower, upper = comparison[key]['interval']
        lines.append(f'| {label} | {100*lower:+.2f} | {100*upper:+.2f} |')
    low, high = comparison['leave_one_structure_out_range']
    lines += ['', '前者不外推到新机制或未来训练种子；后者的独立结构假设未由真实设计自动确立。',
              '宽区间不等于失败，也不证明没有差异。没有跨单元假设时，不从固定成绩表声称总体覆盖保证。', '',
              f'分别移除一个结构后的点估计范围：[{100*low:+.2f}, {100*high:+.2f}] 个百分点。',
              '这只是改变评估集合后的影响描述，不是置信区间；正式结果没有删除任何结构。', '',
              '逐结构世界、逐种子差、权重、贡献和全部移除结果保存在result.json。', '']
    (output/'REPORT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(output/'REPORT_ZH.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='results/mainline_formal/llm_active')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.root, args.output)
