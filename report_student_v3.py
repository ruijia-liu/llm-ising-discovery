"""Report the completed structure-plus-fit check without changing any scores."""
import argparse
import hashlib
import json
from pathlib import Path

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run(output):
    output = Path(output)
    result, config = read(output/'result.json'), read(output/'config.json')
    baseline = read(output/'fixed_evidence_baseline.json')
    train, validation = result['training_checks'][-1], result['validation']
    receipt = {'raw_result_sha256': hashlib.sha256((output/'result.json').read_bytes()).hexdigest(),
               'source_snapshots_match': all(hashlib.sha256((output/'sources'/p).read_bytes()).hexdigest() == value
                                             for p, value in config['source_sha256'].items()),
               'protected_old_files_unchanged': all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == value
                                                   for p, value in config['protected_old_hashes'].items())}
    if not receipt['source_snapshots_match'] or not receipt['protected_old_files_unchanged']:
        raise AssertionError('Provenance check failed.')
    lines = ['# v3：模型提出结构、程序拟合系数', '',
             f'已完成共同任务检查：{result["completed_updates"]} 次更新，单个训练种子301，'
             '24个无噪声训练世界、12个开发验证世界。', '',
             '**这是结构模型与数值拟合程序组成的系统成绩；不替换模型独立输出完整公式的 v2 分数。**', '',
             '| 数据 | 合法回答 | 拟合前结构完全正确 | 拟合后机制恢复 | 其中非空机制恢复 | 原严格基准恢复 |',
             '|---|---:|---:|---:|---:|---:|']
    for title, summary in [('训练题', train), ('已见结构、强度变化的开发验证题', validation)]:
        n = summary['worlds']
        lines.append(f'| {title} | {summary["valid_expression"]}/{n} | {summary["raw_support_success"]}/{n} | '
                     f'{summary["full_recovery"]}/{n} | {summary["nonempty_recoveries"]}/{summary["nonempty_worlds"]} | '
                     f'{summary["exact_recoveries"]}/{n} |')
    lines += ['', f'训练门槛（22/24合法、16/20非空恢复）：**{"通过" if result["training_gate_passed"] else "未通过"}**。'
              f'开发验证门槛（10/12合法、5/10非空恢复）：**{"通过" if result["validation_gate_passed"] else "未通过"}**。', '']
    if result['ready_for_small_B_C']:
        lines.append('两项技术门槛均通过，可进入预先约定的小规模B/C配对实验。此结果本身不能说明过程教学有效。')
    else:
        lines.append('尚未满足启动小规模B/C的两项技术门槛；没有恢复完整B/C训练。'
                     '保留本轮负结果，不继续增加相同配置的训练轮数。这不是对过程教学效果的否定。')
        if result['training_gate_passed']:
            lines += ['', '下一步建议先检查训练数据覆盖：在已学结构内增加不同强度与构型的训练例子，'
                      '预先固定一个小规模数据量对照，再判断参数泛化是否改善。'
                      '这是待检验的修正方向，不能把当前失败直接归因于样本数量。'
                      '本轮没有额外生成这些教材或启动新训练。']
    lines += ['', '## 误报与数值程序', '',
              '| 数据 | 空机制世界数 | 模型原始非空提议数 | 拟合后仍有错误机制 | 空机制题无效回答 |',
              '|---|---:|---:|---:|---:|']
    for title, summary in [('训练', train), ('开发验证', validation)]:
        lines.append(f'| {title} | {summary["null_worlds"]} | {summary["raw_false_proposal"]} | '
                     f'{summary["false_discovery"]} | {summary["invalid_null_answers"]} |')
    lines += ['', '无噪声空机制题的额外能量残差为零，拟合可能把错误提议的系数算为零。'
              '因此拟合后的低误报不能单独证明模型没有编造结构。两种误报均保留，零误报不是训练门槛。', '',
              '程序只计算学生所提结构的特征，不枚举其它候选，不新增查询；'
              '满列秩表示这些系数在给定结构下可以估计，不表示该结构已被唯一证明。'
              '机制恢复还要求结构、系数及新构型预测同时通过。', '',
              '## 学习进展与预算', '',
              '| 更新步数 | 合法回答 | 拟合前结构正确 | 非空训练机制恢复 |',
              '|---:|---:|---:|---:|']
    for check in result['training_checks']:
        lines.append(f'| {check["step"]} | {check["valid_expression"]}/24 | {check["raw_support_success"]}/24 | '
                     f'{check["nonempty_recoveries"]}/20 |')
    generation_seconds = sum(c['generation_seconds'] for c in result['training_checks'])+validation['generation_seconds']
    lines += ['', f'- 终答展示 {result["terminal_presentations"]} 次；输入 token {result["processed_tokens"]:,}；'
              f'监督 token {result["supervised_tokens"]:,}。',
              f'- 梯度计算与优化累计 {result["training_seconds"]:.1f} 秒；检查时生成累计 {generation_seconds:.1f} 秒。'
              '这些分项不含全部文件准备、物理评分和保存时间，不等于完整墙钟耗时。',
              '- 每世界仍为原有32次测量；贪心输出上限128 token。没有增加实验，也没有让模型看到独立评分结果。',
              '- 每世界评分使用256个均匀、256个偏置新构型。训练题仍然是见过的证据，成功可能包括记忆。', '',
              '## 相同证据的非语言模型对照', '',
              f'稀疏搜索恢复训练 {baseline["summary"]["train"]["full_recoveries"]}/24、'
              f'开发验证 {baseline["summary"]["validation"]["full_recoveries"]}/12，'
              '也通过原严格无噪声标准。其结构搜索能力与仅拟合学生提议的数值程序不同，不能混为一谈。']
    error_path = output/'error_analysis.json'
    if error_path.exists():
        errors = read(error_path)
        if errors['raw_result_sha256'] != receipt['raw_result_sha256']:
            raise AssertionError('Error analysis refers to a different result.')
        categories = {'invalid_answer': '回答无法执行',
                      'exact_structure_recovered': '原始结构正确且恢复成功',
                      'correct_structure_failed_numerical_check': '结构正确但数值检查失败',
                      'recovered_after_zero_coefficient_removal': '删除零系数项后恢复',
                      'wrong_term_count_or_orders': '作用项数或阶数错误',
                      'correct_term_count_and_orders_wrong_positions': '项数和阶数正确，但位置错误'}
        lines += ['', '## 错在哪里', '', '| 类型 | 训练题 | 开发验证题 |', '|---|---:|---:|']
        for key, title in categories.items():
            lines.append(f'| {title} | {errors["summary"]["train"].get(key, 0)} | '
                         f'{errors["summary"]["validation"].get(key, 0)} |')
        lines += ['', '此表仅重新归类已有输出，没有新增推理、测量或改动原分数。']
    lines += ['', '## 逐题结果', '',
              '| 数据 | 世界 | 原始结构正确 | 拟合后恢复 | 均匀RMSE | 偏置RMSE |',
              '|---|---|---|---|---:|---:|']
    for summary in (train, validation):
        for row in summary['results']:
            label = '训练' if row['split'] == 'train' else '开发验证'
            uniform = f'{row["uniform_rmse"]:.6g}' if row['valid_expression'] else '无效'
            biased = f'{row["biased_rmse"]:.6g}' if row['valid_expression'] else '无效'
            lines.append(f'| {label} | {row["episode_id"]} | {"是" if row.get("raw_support_success") else "否"} | '
                         f'{"是" if row["full_recovery"] else "否"} | {uniform} | {biased} |')
    lines += ['', '## 能说明什么', '',
              '本轮同时改变了输出任务、数值工具、训练世界数与总训练量；与v2差异不能单独归因于某一项。'
              '本轮也没有过程教学组或自由选实验测试，不能回答C是否优于B。', '',
              '相对于本轮实际使用的完整24个训练世界，12个验证世界均为已见完整结构组合。'
              '它们已用于先前开发诊断，不是封存测试集。组合泛化与四体正式测试仍未使用。', '',
              'v1暂停点、旧模型与v2原始评分的哈希均保持一致；本次执行源码与配置已归档。', '',
              '入口：[运行前协议](../../STUDENT_STRUCTURE_FIT_PROTOCOL.md)、[原始结果](result.json)、'
              '[配置](config.json)、[相同证据稀疏搜索](fixed_evidence_baseline.json)。', '']
    (output/'REPORT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    (output/'report_audit.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(output/'REPORT_ZH.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/student_training_v3')
    run(parser.parse_args().output)
