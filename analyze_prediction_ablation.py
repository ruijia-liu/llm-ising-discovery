"""Registered secondary contrast: full C minus C without explicit prediction/error values."""
import argparse
from pathlib import Path

from build_publication_curriculum import atomic_json, sha
from discovery.evaluation_results import load_completed_evaluation, verify_pairing
from discovery.paired_statistics import primary_comparison


def ablation_comparison(rows, **bootstrap_options):
    converted = []
    for row in rows:
        identity = (row.get('study'), row['group'])
        if identity == ('mainline', 'C'):
            group = 'C'
        elif identity == ('prediction_ablation', 'C_no_predictions'):
            group = 'B'
        else:
            raise ValueError('The ablation contrast requires mainline C and the registered prediction ablation only.')
        converted.append({**row, 'group': group, 'mode': 'active',
                          'score': row['score']['tested_idea_recovery']})
    # Reuse exactly the registered pairing/weights/bootstrap, with explicit labels
    # in the returned artifact so this cannot be confused with the main C-B result.
    contrast = primary_comparison(converted, **bootstrap_options)
    effect = contrast.pop('C_minus_B')
    direction = contrast.pop('consistent_process_advantage')
    return {**contrast, 'full_C_minus_C_no_predictions': effect,
            'positive_interval_and_all_seed_directions': direction,
            'analysis_role': 'prespecified_secondary_explanatory_not_an_additional_confirmatory_test'}


def run(main_root, ablation_root, output):
    main_root, ablation_root, output = map(Path, (main_root, ablation_root, output))
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserve the previous ablation analysis.')
    evaluations = []
    for seed in (401, 402, 403):
        evaluations.append(load_completed_evaluation(main_root/f'C_seed{seed}', scope='formal',
            study='mainline', method='llm_active', group='C', seed=seed))
        evaluations.append(load_completed_evaluation(ablation_root/f'C_no_predictions_seed{seed}', scope='formal',
            study='prediction_ablation', method='llm_active', group='C_no_predictions', seed=seed))
    verify_pairing(evaluations)
    rows = [r for evaluation in evaluations for r in evaluation['result']['rows']]
    contrast = ablation_comparison(rows)
    counts = []
    for evaluation in evaluations:
        config = evaluation['config']
        for level in ('combination', 'four_body'):
            subset = [r for r in evaluation['result']['rows'] if r['sigma'] == 1 and r['level'] == level]
            counts.append({'study': config['study'], 'group': config['group'], 'seed': config['seed'],
                'level': level, 'worlds': len(subset),
                'tested_idea_recoveries': sum(r['score']['tested_idea_recovery'] for r in subset)})
    result = {'completed': True, 'analysis': 'registered_prediction_ablation_secondary',
        'endpoint': 'sigma1_active_tested_idea_recovery', 'contrast': contrast,
        'unweighted_world_counts': counts,
        'both_groups_zero_on_endpoint': not any(r['tested_idea_recoveries'] for r in counts),
        'provenance': [e['provenance'] for e in evaluations],
        'sources': {p: sha(p) for p in ('analyze_prediction_ablation.py', 'discovery/paired_statistics.py',
            'discovery/evaluation_results.py', 'PROCESS_ABLATION_PROTOCOL.md', 'MAINLINE_STUDY_PROTOCOL.md')}}
    atomic_json(output/'result.json', result)
    lower, upper = contrast['conditional_95_percentile_interval']
    lines = ['# 显式预测/误差内容消融：预定次要比较', '',
        f"完整C减去删减C：{100*contrast['full_C_minus_C_no_predictions']:+.2f} 个百分点；"
        f'条件于三个种子的95%区间 [{100*lower:+.2f}, {100*upper:+.2f}] 个百分点。', '',
        '组合与四体层等权，层内结构等权，同结构参数世界等权，三个固定种子等权。',
        '下表保留未加权成功数；本比较不替代主要C−B比较，不产生额外确认性显著发现。', '',
        '| 组 | 种子 | 层次 | 事前检验idea恢复 / 世界数 |', '|---|---|---|---|']
    lines += [f"| {r['group']} | {r['seed']} | {r['level']} | {r['tested_idea_recoveries']} / {r['worlds']} |" for r in counts]
    lines += ['', '消融同时删除显式事前预测值和事后预测误差，不能分别归因于其中一项。',
        '其余表达式和选择摘要仍含可推导的预测信息；预算匹配重复删减教材，可能增加中间候选曝光。',
        '没有明确差异不等于两种教材等效；该结果必须结合固定证据、格式失败及行为记录解释。', '']
    if result['both_groups_zero_on_endpoint']:
        lines += ['两组终点均为零，经验区间退化；不能把它当作精确零效应或教学无效的证明。', '']
    (output/'REPORT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(output/'REPORT_ZH.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-root', default='results/mainline_formal/llm_active')
    parser.add_argument('--ablation-root', default='results/process_ablation_formal/llm_active')
    parser.add_argument('--output', default='results/process_ablation_analysis')
    args = parser.parse_args()
    run(args.main_root, args.ablation_root, args.output)
