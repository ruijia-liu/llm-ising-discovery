"""Read-only comparison of the one-shot coverage run and its fixed v3 control."""
from pathlib import Path

from build_publication_curriculum import atomic_json, read, sha


def paired_counts(before, after, metric):
    left = {r['episode_id']: bool(r.get(metric, False)) for r in before['results']}
    right = {r['episode_id']: bool(r.get(metric, False)) for r in after['results']}
    if set(left) != set(right):
        raise AssertionError('Unpaired evaluated worlds.')
    return {'both_success': sum(left[k] and right[k] for k in left),
            'only_control_success': sum(left[k] and not right[k] for k in left),
            'only_coverage_success': sum(not left[k] and right[k] for k in left),
            'both_failure': sum(not left[k] and not right[k] for k in left)}


def run():
    directory = Path('results/student_training_coverage')
    result, config = read(directory/'result.json'), read(directory/'config.json')
    control = read('results/student_training_v3/result.json')
    if result['completed_updates'] != 192 or result['terminal_presentations'] != 384:
        raise AssertionError('Prespecified budget incomplete.')
    if sha('results/student_training_v3/result.json') != config['control_result_sha256']:
        raise AssertionError('Control results changed.')
    if not all(sha(p) == expected for p, expected in config['old_result_hashes'].items()):
        raise AssertionError('A protected prior result changed.')
    if not all(sha(directory/'sources'/p) == expected for p, expected in config['source_sha256'].items()):
        raise AssertionError('Source snapshot differs.')
    paired = {}
    samples = [('相同24道原训练题', control['training_checks'][-1], result['anchor_training_check']),
               ('相同12道开发验证题', control['validation'], result['validation'])]
    lines = ['# 一次固定训练预算的数据覆盖对照', '',
             '旧对照使用24个训练世界，新运行轮换120个同结构参数世界。两者同一模型、seed301、'
             '192次更新、384次答案呈现和逐步结构标签顺序；旧对照不重训。',
             '这是单次开发诊断，没有A/B/C教学比较、正式测试或显著性结论。', '',
             '| 检查数据 | 24世界对照：原始结构正确 | 120世界：原始结构正确 | 对照：拟合后恢复 | 120世界：拟合后恢复 |',
             '|---|---:|---:|---:|---:|']
    for name, old, new in samples:
        if old['worlds'] != new['worlds']:
            raise AssertionError('Evaluation sizes differ.')
        paired[name] = {key: paired_counts(old, new, key) for key in ('raw_support_success', 'full_recovery')}
        n = new['worlds']
        lines.append(f'| {name} | {old["raw_support_success"]}/{n} | {new["raw_support_success"]}/{n} | '
                     f'{old["full_recovery"]}/{n} | {new["full_recovery"]}/{n} |')
    old, new = control['validation'], result['validation']
    delta = new['nonempty_recoveries']-old['nonempty_recoveries']
    interpretation = ('同预算下新增参数例题在这组开发题上改善了恢复，但单种子小样本不能建立普遍结论。' if delta > 0 else
                      '同预算下新增参数例题没有提高这组开发题的非空机制恢复；不能仅靠扩充同结构数据解释或解决当前困难。')
    lines += ['', f'非空开发题：对照 {old["nonempty_recoveries"]}/{old["nonempty_worlds"]}，'
              f'新运行 {new["nonempty_recoveries"]}/{new["nonempty_worlds"]}。'+interpretation, '',
              '训练题检查仍只用原来24个世界，不能当作120个训练世界的总准确率。'
              '12道开发题已经用于此前调试，不能称为未使用的确认测试。', '',
              '| 开发验证空机制检查 | 对照 | 新运行 |', '|---|---:|---:|',
              f'| 原始错误非空提议 | {old["raw_false_proposal"]} | {new["raw_false_proposal"]} |',
              f'| 拟合后仍错误非空 | {old["false_discovery"]} | {new["false_discovery"]} |',
              f'| 无效回答 | {old["invalid_null_answers"]} | {new["invalid_null_answers"]} |', '',
              '空机制误报是保留的结果，不作为继续实验的门槛。无噪声拟合可将错误项系数变为零，'
              '因此原始提议与拟合后分数分别报告。', '',
              '| 资源 | 对照 | 新运行 |', '|---|---:|---:|',
              f'| 输入token | {control["processed_tokens"]} | {result["processed_tokens"]} |',
              f'| 监督token | {control["supervised_tokens"]} | {result["supervised_tokens"]} |',
              f'| 训练秒数 | {control["training_seconds"]:.1f} | {result["training_seconds"]:.1f} |', '',
              '数值工具只拟合学生提出的项，没有搜索或添加结构。旧模型、检查点与分数已按哈希验证保留。',
              '按事前协议，到此结束覆盖诊断，继续正式教学对照的格式、资源和主动实验接入；不再延长本轮训练或选择新种子追逐分数。', '']
    atomic_json(directory/'comparison.json', {'paired': paired, 'interpretation': interpretation,
                'result_sha256': sha(directory/'result.json'), 'control_sha256': config['control_result_sha256'],
                'old_results_preserved': True, 'source_snapshots_match': True,
                'formal_tests_used': False, 'confirmatory_significance_test': False})
    (directory/'REPORT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines[:13]), flush=True)


if __name__ == '__main__':
    run()
