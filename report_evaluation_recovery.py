"""Record exact development invariance after adding durable evaluation journals."""
from pathlib import Path

from build_publication_curriculum import atomic_json, read, sha
from discovery.evaluation_results import load_completed_evaluation


def run():
    comparisons = []
    for mode in ('fixed', 'active'):
        old_root = Path(f'results/mainline_development_sparse_{mode}_v2')
        new_root = Path(f'results/mainline_development_sparse_{mode}_v3_journal')
        old = load_completed_evaluation(old_root, scope='development', method='sparse_'+mode)
        new = load_completed_evaluation(new_root, scope='development', method='sparse_'+mode)
        prior = {(r['world_id'], r['sigma']): r for r in old['result']['rows']}
        for row in new['result']['rows']:
            key = row['world_id']+f"-sigma{row['sigma']}"
            a, b = read(old_root/'episodes'/(key+'.json')), read(new_root/'episodes'/(key+'.json'))
            if row['score'] != prior[row['world_id'], row['sigma']]['score'] or a['episode'] != b['episode']:
                raise AssertionError('Recovery implementation changed a paired episode or score.')
        comparisons.append({'mode': mode, 'paired_conditions': len(new['result']['rows']),
                            'all_scores_identical': True, 'complete_episodes_identical': True,
                            'new_run_metered_observations': sum(r['measurements'] for r in new['resources']),
                            'old_provenance': old['provenance'], 'new_provenance': new['provenance']})
    output = Path('results/evaluation_recovery_audit')
    atomic_json(output/'audit.json', {'completed': True, 'comparisons': comparisons, 'formal_tests_used': False,
                'source_sha256': {p: sha(p) for p in ('report_evaluation_recovery.py', 'discovery/evaluation_results.py')},
                'scope': 'Development search invariance, not a language-model teaching-effect result.'})
    lines = ['# 评估恢复与开发结果核验', '',
             '固定/主动搜索各比较12个原开发条件，合计24个。新旧分数、完整实验轨迹、拟合结果逐项一致。',
             '每次完整搜索仍为32次测量；新两组各记录384次。所有既有失败和误报原样保留。', '',
             '自动检查覆盖：真实子进程退出后重建、六类协议事件边界、部分初始批/揭示批、',
             '错误回答与一次修复预算、未决调用拒绝重复、并发写入和校验和、事前预测落盘、',
             '固定证据与主动搜索重建。结果汇总入口另检查完整网格、来源、原始分数及实际计费。', '',
             '未决预约不自动重试；干净模拟器回执位于私有控制目录，学生只看到原带噪观测。',
             '这些是工程验证与开发对照，不是正式学生效果分数。正式保留测试未开启。', '',
             '核验入口：`python -m unittest test_episode_journal test_mainline_evaluation test_evaluation_results -v`。',
             '结果比较入口：`python report_evaluation_recovery.py`。', '']
    (output/'REPORT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(output/'REPORT_ZH.md')


if __name__ == '__main__':
    run()
