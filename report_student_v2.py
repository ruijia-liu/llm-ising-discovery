"""Report the completed positive control without implying a B/C effect."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run(output):
    output = Path(output)
    result, config, audit = [read(output/name) for name in ('result.json', 'config.json', 'data_audit.json')]
    explicit = config.get('input_representation') == 'local-spins'
    lines = ['# v2 独立终答任务：小规模学习检查', '',
             '**范围更正：** 相对于实际六题训练子集，验证包含8道已见结构参数题、4道未见结构题。'
             '旧记录中的“参数验证”名称过宽，原分数不变；详见 validation_scope_audit.json。', '',
             f'本轮完成 {result["completed_updates"]} 个更新，技术门槛：**{"通过" if result["learning_gate_passed"] else "未通过"}**。'
             '这是一组共同终答任务的训练，不是 B/C 对比。', '',
             'v1 两个完整模型、原验证结果及 C/102 第 5 步保存点哈希均未改变。'
             '本轮从相同基础模型重新开始，没有续训旧 B/C 权重。', '',
             '## 已实施的修改', '',
             '- 两组独立终答任务完全相同，训练和评估共用提示生成函数，明确说明实验结束。',
             '- 终答只含机制表达式；损失只训练终答及结束符，不再由长过程记录稀释。',
             '- B/C 辅助教材另存，移除判卷器正确终答。C 的最后修正保留在辅助教材中。',
             '- 配对训练接口分别归一化终答和辅助损失，预设各占 0.5；本轮只训练共同终答，权重为 1。',
             '- 训练使用 6 个既有无噪声世界，每题保留全部 32 次测量；不按教师成败筛选。',
             '- 解码重复惩罚统一设为 1.0，保留重复坐标和数字。',
             ('- 本版使用公开表达语言决定的 5×5 局部明确 ±1 自旋，已检查它保留全部158个语法项的翻转特征；未提供拟合参数或候选答案。' if explicit else
              '- 构型仍使用原来的无损 hex 表示。'), '',
             '## 学习进展', '',
             '| 更新步数 | 可执行 | 结构正确 | 非空机制恢复 | 全部机制恢复 | 原严格无噪声评分 |',
             '|---|---:|---:|---:|---:|---:|']
    for check in result['training_checks']:
        lines.append(f'| {check["step"]} | {check["valid_expression"]}/6 | {check["support_success"]}/6 | '
                     f'{check["nonempty_recoveries"]}/5 | {check["full_recovery"]}/6 | {check["exact_recoveries"]}/6 |')
    validation = result['validation']
    lines += ['', '技术门槛在运行前固定：至少 5/6 可执行，且至少 4/5 个非空训练机制恢复。'
              '恢复需结构正确、两个独立预测 RMSE 均≤0.01、系数最大误差≤0.002。'
              '这只是学习检查的数值容差；原无噪声基准的更严格评分也完整保留，旧阈值没有修改。', '',
              '停止步数由训练题的固定检查点决定。随后才做一次混合验证：'
              f'可执行 {validation["valid_expression"]}/12，结构正确 {validation["support_success"]}/12，'
              f'开发容差下完整恢复 {validation["full_recovery"]}/12，'
              f'原严格评分 {validation["exact_recoveries"]}/12。'
              '验证结果没有用于调参或选择训练步数；结构组合/四体正式测试仍未开启。', '']
    baseline = output/'fixed_evidence_baseline.json'
    if baseline.exists():
        b = read(baseline)
        lines += [f'非语言模型参照：稀疏搜索使用同样的 32 次观测，恢复 {b["full_recoveries"]}/6，'
                  f'原严格评分也是 {b["exact_recoveries"]}/6，没有追加查询。', '']
    latest = result['training_checks'][-1]
    lines += ['## 误报与限制', '',
              f'最后训练题检查：1 个空机制世界，明确误报 {latest["false_discovery"]}，'
              f'无效答案 {latest["invalid_null_answers"]}。'
              f'参数验证：2 个空机制世界，明确误报 {validation["false_discovery"]}，'
              f'无效答案 {validation["invalid_null_answers"]}。误报未被过滤，也不是门槛必须为零的条件。', '',
              '六题学习检查可能只反映记忆，不能证明结构泛化或研究能力。'
              '多项训练接口修正同时实施，不能从 v1/v2 差异单独归因于某一项修改。'
              '本轮未训练 B/C 辅助任务，也没有测量过程教学效应。', '',
              f'累计训练计算时间 {result["training_seconds"]:.1f} 秒（另有评估、保存和准备开销）；'
              f'终答呈现 {2*result["completed_updates"]} 次。每题 token 数见 data_audit.json；'
              '每步训练损失、输入量、梯度及保存点均已保留。', '',
              '## 后续', '',
              ('技术门槛通过，下一步可以做独立的小规模 B/C 配对训练。需要保持两组终答次数和权重相同，'
               '继续报告计算预算差异；参数验证若仍差，不能直接跳到结构发现结论。' if result['learning_gate_passed'] else
               '技术门槛未通过，不扩大同一配置。下一步应检查自旋输入表示和模型能力，'
               '对表示的改动必须由公开表达语言决定，不允许使用真实机制选择相关自旋。'), '',
              '复查：config.json、data_audit.json、training.jsonl、train_step*.json、validation_step*.json、result.json。', '']
    (output/'REPORT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(output/'REPORT_ZH.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/student_training_v2')
    run(parser.parse_args().output)
