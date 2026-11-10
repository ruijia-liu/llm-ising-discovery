"""Summarize supplied complete evaluations without treating them as independent seeds."""
import argparse
from collections import defaultdict
from pathlib import Path

from build_publication_curriculum import atomic_json, read, sha
from discovery.evaluation_results import load_completed_evaluation, measured_observations, verify_pairing
from discovery.secondary_metrics import behavior, generation_time, summarize
from discovery.seed_summaries import aggregate_seeds


def run(directories, output, scope):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserve the existing descriptive report.')
    evaluations = [load_completed_evaluation(p, scope=scope) for p in directories]
    paired = defaultdict(list)
    for loaded in evaluations:
        c = loaded['config']
        paired[(c.get('study', 'mainline'), c['method'], c['group'])].append(loaded)
    for runs in paired.values():
        verify_pairing(runs)
    seen, items = set(), []
    for loaded in evaluations:
        config = loaded['config']
        key = (config.get('study', 'mainline'), config['method'], config['group'], config['seed'])
        if key in seen:
            raise ValueError('Duplicate method/group/seed run; do not inflate the denominator.')
        seen.add(key)
        root = Path(loaded['provenance']['directory'])
        resources = {(r['world_id'], r['sigma']): r for r in loaded['resources']}
        for row in loaded['result']['rows']:
            path = root/'episodes'/(row['world_id']+f"-sigma{row['sigma']}.json")
            episode = read(path)['episode']
            _, replies = measured_observations(episode, config['method'])
            resource = {**resources[row['world_id'], row['sigma']],
                        **generation_time(replies)}
            items.append({'row': row, 'resources': resource, 'behavior': behavior(episode, config['method'])})
    cells = summarize(items)
    seed_summaries = aggregate_seeds(items, cells)
    result = {'completed': True, 'scope': scope, 'analysis': 'descriptive_secondary_only',
              'completeness_scope': 'only_the_supplied_complete_runs_not_a_claim_that_the_whole_study_is_finished',
              'cells': cells, 'seed_summaries': seed_summaries, 'provenance': [e['provenance'] for e in evaluations],
              'sources': {p: sha(p) for p in ('report_evaluation_metrics.py', 'discovery/secondary_metrics.py',
                          'discovery/seed_summaries.py',
                          'discovery/evaluation_results.py', 'SECONDARY_ANALYSIS_PROTOCOL.md')}}
    atomic_json(output/'result.json', result)
    lines = ['# 次要描述性指标', '', f'范围：{scope}；仅汇总所提供且已核验完整的运行。各训练种子分别列出，不充作独立机制。', '',
             '| 方法 | 组 | 种子 | σ | 层次 | 完整恢复 | 原始结构正确 | 非法终答 |',
             '|---|---|---|---|---|---|---|---|']
    for row in cells:
        m = row['metrics']; n = row['worlds']
        label = row['seed'] if row['seed'] is not None else '不适用'
        lines.append(f"| {row['method']} | {row['group']} | {label} | {row['sigma']} | {row['level']} | "
                     f"{m['full_recovery']['count']}/{n} | {m['raw_support_success']['count']}/{n} | {m['invalid_final_expression']['count']}/{n} |")
    if seed_summaries:
        lines += ['', '三种子描述性汇总（种子等权；层内世界成功率，不替代主要分析的结构等权估计）：', '',
                  '| 方法 | 组 | σ | 层次 | 独立结构 / 参数世界 | 完整恢复率 | 状态 |',
                  '|---|---|---|---|---|---|---|']
        for row in seed_summaries:
            complete = not row['missing_seeds']
            value = row['metrics']['full_recovery']['equal_seed_mean_rate'] if complete else None
            rate = f'{100*value:.2f}%' if value is not None else '不可用'
            status = '三个种子齐全' if complete else '缺少种子 '+','.join(map(str, row['missing_seeds']))
            lines.append(f"| {row['method']} | {row['group']} | {row['sigma']} | {row['level']} | "
                         f"{row['independent_structure_count']} / {row['unique_parameter_worlds']} | {rate} | {status} |")
    lines += ['', 'JSON同时记录：误差可用/缺失分母、空世界原始与拟合后误报及非法回答、首次恢复的检查频率、',
              '预算曲线、失败分类、工具删零项效果、实验行为和实际token/测量消耗。', '',
              '预测误差均值条件于存在可执行表达式，不能忽略非法率来作优劣断言；无误报也不等于全部回答正确。',
              '跨种子条件误差均值先在每个种子内计算，再等权平均；不把分位数的平均冒充总体分位数。',
              '缺少任何注册种子时不生成三种子平均；部分种子没有可用误差/条件误报率时，该跨种子量保留不可用。',
              '本报告没有新的确认性终点或显著性检验；主要结论使用独立的事前注册比较。', '']
    (output/'REPORT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(output/'REPORT_ZH.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+')
    parser.add_argument('--scope', choices=('development', 'formal'), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.directories, args.output, args.scope)
