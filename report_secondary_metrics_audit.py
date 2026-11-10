"""Check that new grader-only diagnostics preserve prior development trajectories."""
from pathlib import Path

from build_publication_curriculum import atomic_json, read, sha
from discovery.evaluation_results import load_completed_evaluation
from report_evaluation_metrics import run


def audit():
    output = Path('results/secondary_metrics_audit')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserve the previous audit; do not overwrite evidence.')
    comparisons, directories = [], []
    for method in ('sparse_fixed', 'sparse_active'):
        previous = Path(f'results/mainline_development_{method}_v3_journal')
        current = Path(f'results/mainline_development_{method}_v4_diagnostics')
        before, after = (load_completed_evaluation(p, scope='development', method=method) for p in (previous, current))
        old = {(r['world_id'], r['sigma']): r for r in before['result']['rows']}
        new = {(r['world_id'], r['sigma']): r for r in after['result']['rows']}
        if set(old) != set(new):
            raise ValueError('Regression comparison must use the same full development subset.')
        for key, row in new.items():
            if {k: v for k, v in row.items() if k != 'grader_diagnostics'} != old[key]:
                raise ValueError('Existing scores or metadata changed; do not call this a diagnostics-only change.')
            name = key[0]+f'-sigma{key[1]}.json'
            if read(previous/'episodes'/name)['episode'] != read(current/'episodes'/name)['episode']:
                raise ValueError('An original trajectory changed.')
            if not row['grader_diagnostics']['grader_only_never_returned_to_student']:
                raise ValueError('Missing grader-only provenance.')
        comparisons.append({'method': method, 'conditions': len(new), 'episodes_identical': True,
                            'scores_identical': True, 'previous': before['provenance'], 'current': after['provenance']})
        directories.append(current)
    run(directories, output/'descriptive_report', 'development')
    atomic_json(output/'audit.json', {'completed': True, 'formal_tests_used': False,
        'conditions_compared': sum(r['conditions'] for r in comparisons), 'comparisons': comparisons,
        'sources': {p: sha(p) for p in ('report_secondary_metrics_audit.py', 'report_evaluation_metrics.py',
            'discovery/secondary_metrics.py', 'discovery/answer_diagnostics.py', 'evaluate_mainline.py',
            'test_secondary_metrics.py', 'test_answer_diagnostics.py', 'SECONDARY_ANALYSIS_PROTOCOL.md')}})
    (output/'REPORT_ZH.md').write_text(
        '# 次要指标开发核验\n\n'
        '固定证据与主动搜索各12个原开发条件，24个条件的完整实验轨迹和原有分数逐项相同。\n'
        '新增项只在判卷结束后记录错误类型和首次恢复时点，不返回学生。\n\n'
        '描述性报告同时保留非法回答、条件误差的缺失分母、原始/拟合后误报、'
        '工具删零项效果和实际资源记录。旧记录缺少诊断或生成时间时保留不可用标记。\n\n'
        '这是小型开发回归，不是正式测试，也不提供A/B/C的教学效果结论。\n', encoding='utf-8')
    print(output/'REPORT_ZH.md')


if __name__ == '__main__':
    audit()
