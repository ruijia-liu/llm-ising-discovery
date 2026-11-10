"""Compute the frozen primary contrast only from six complete formal outputs."""
import argparse
from pathlib import Path

from build_publication_curriculum import atomic_json, sha
from discovery.evaluation_results import load_completed_evaluation, verify_pairing
from discovery.paired_statistics import primary_comparison


def run(root, output):
    root, output = Path(root), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserve the existing primary analysis; use an explicitly new audit version.')
    evaluations = [load_completed_evaluation(root/f'{g}_seed{s}', scope='formal', study='mainline',
                                            method='llm_active', group=g, seed=s)
                   for g in ('B', 'C') for s in (401, 402, 403)]
    verify_pairing(evaluations)
    rows = [{**r, 'mode': 'active', 'score': r['score']['tested_idea_recovery']}
            for evaluation in evaluations for r in evaluation['result']['rows']]
    contrast = primary_comparison(rows)
    counts = []
    for evaluation in evaluations:
        c = evaluation['config']
        selected = [r for r in evaluation['result']['rows'] if r['sigma'] == 1 and r['level'] in ('combination', 'four_body')]
        for level in ('combination', 'four_body'):
            subset = [r for r in selected if r['level'] == level]
            counts.append({'group': c['group'], 'seed': c['seed'], 'level': level, 'worlds': len(subset),
                           'tested_idea_recoveries': sum(r['score']['tested_idea_recovery'] for r in subset)})
    result = {'completed': True, 'analysis': 'registered_primary_only', 'endpoint': 'sigma1_active_tested_idea_recovery',
              'contrast': contrast, 'unweighted_world_counts': counts,
              'both_groups_zero_on_primary_endpoint': not any(r['tested_idea_recoveries'] for r in counts),
              'evaluations': [e['provenance'] for e in evaluations],
              'source_sha256': {p: sha(p) for p in ('analyze_mainline_primary.py', 'discovery/evaluation_results.py',
                                  'discovery/paired_statistics.py', 'MAINLINE_STUDY_PROTOCOL.md')},
              'secondary_analyses_completed_by_this_script': False}
    atomic_json(output/'result.json', result)
    lower, upper = contrast['conditional_95_percentile_interval']
    lines = ['# 事前规定的主要比较', '', '本报告仅汇总完整B/C主动评估、三个固定训练种子的确认性终点。', '',
             f"结构等权的 C−B：{100*contrast['C_minus_B']:+.2f} 个百分点；条件于这三个种子的95%区间 "
             f'[{100*lower:+.2f}, {100*upper:+.2f}] 个百分点。', '',
             '组合和四体层等权；层内独立结构等权；同结构参数世界等权。下表为未加权原始成功数，不能替代上述估计。', '',
             '| 组 | 种子 | 层次 | 事前检验idea恢复 / 世界数 |', '|---|---|---|---|']
    lines += [f"| {r['group']} | {r['seed']} | {r['level']} | {r['tested_idea_recoveries']} / {r['worlds']} |" for r in counts]
    lines += ['', '主区间及逐种子方向判定：'+('满足本设置下一致优势的事前规则。' if contrast['consistent_process_advantage'] else
              '不满足本设置下一致优势的事前规则；不能据此宣称两组等效。'), '',
              '此脚本不完成固定证据、次要终点、消融、第二模型或论文全部结论。', '']
    if result['both_groups_zero_on_primary_endpoint']:
        lines += ['两组在此终点均为零，经验bootstrap区间会退化。需结合固定证据、格式和行为分析解释地板效应，',
                  '不能把这个退化区间解读成对未来机制或训练种子的精确零效应保证。', '']
    (output/'REPORT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(output/'REPORT_ZH.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='results/mainline_formal/llm_active')
    parser.add_argument('--output', default='results/mainline_primary_analysis')
    args = parser.parse_args()
    run(args.root, args.output)
