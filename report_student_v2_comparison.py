"""Combine the completed repair and conditional input check, without regrading."""
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run():
    folders = [Path('results/student_training_v2'), Path('results/student_training_v2_spins')]
    entries = [(name, folder, read(folder/'result.json')) for name, folder in zip(('hex构型', '明确±1自旋'), folders)]
    lines = ['# v2 修订与输入表示检查汇总', '',
             '**验证范围更正：** 对实际六题训练子集，12道验证题是8道已见结构的参数变化题加4道未见结构题，'
             '此前统称“参数验证”不准确。原分数不变，见[分层核查](VALIDATION_SCOPE_CORRECTION_ZH.md)。', '',
             '**终答任务的修订已落实，完成了两次独立的小规模训练。两次都从相同基础模型开始，'
             '使用相同六道无噪声训练题、种子、更新上限和目标答案。旧实验与断点完整保留。**', '',
             '两轮都是共同终答任务检查；尚未训练 B/C 过程教学对照，也没有运行正式结构测试。', '',
             '| 检查 | 更新数 | 训练题可执行 | 非空训练机制恢复 | 全部训练机制恢复 | 混合验证恢复 |',
             '|---|---:|---:|---:|---:|---:|']
    for name, folder, result in entries:
        last = result['training_checks'][-1]
        validation = result['validation']
        lines.append(f'| {name} | {result["completed_updates"]} | {last["valid_expression"]}/6 | '
                     f'{last["nonempty_recoveries"]}/5 | {last["full_recovery"]}/6 | {validation["full_recovery"]}/12 |')
    lines += ['', '“恢复”要求作用结构正确、独立预测与系数误差都通过运行前固定的开发容差。'
              '原无噪声严格评分也保留在每轮结果中；不能把本表和 v1 高噪声结果当作同条件比较。', '',
              '原严格训练评分：'+ '，'.join(f'{name} {result["training_checks"][-1]["exact_recoveries"]}/6'
                                          for name, _, result in entries)+'。'
              '混合验证中的非空机制恢复：'+ '，'.join(f'{name} {result["validation"]["nonempty_recoveries"]}/10'
                                                    for name, _, result in entries)+'。', '',
              '## 已解决什么', '',
              '1. 训练与评估共用同一个终答提示，明确说明实验结束；输出只含机制表达式。',
              '2. 终答单独监督，不被长篇过程记录稀释。两组的终答任务、曝光次数和显式权重可保持一致。',
              '3. B/C 辅助材料已单独生成，移除了判卷器终答；过程损失和终答损失分别归一化。'
              '本轮只执行共同终答训练，没有将辅助材料效果冒充为 B/C 比较结果。', '',
              '## 输入表示检查', '',
              '第二轮把难读的 hex 换成目标附近5×5个明确±1自旋。该窗口由公开3×3表达语言推导，'
              '没有使用真实机制挑选自旋；测试验证全部158个允许作用的翻转特征和H0都保持不变。'
              '没有提供候选机制、乘积特征或拟合答案。两版原观测值、顺序和32次预算完全一致。', '',
              '| 版本 | 训练输入 token | 终答出现次数 | 训练计算秒数 |', '|---|---:|---:|---:|']
    for name, folder, result in entries:
        records = [json.loads(line) for line in (folder/'training.jsonl').read_text(encoding='utf-8').splitlines()]
        last_record = records[-1]
        lines.append(f'| {name} | {last_record["processed_tokens"]:,} | {last_record["terminal_presentations"]} | {result["training_seconds"]:.1f} |')
    lines += ['', '用时为训练计算累计，另有保存与评估开销。输入表示改变使 token 数略有变化，实际数值如上。', '',
              '## 是否只是算错系数', '',
              '补充诊断固定模型提出的结构，只用已有测量重新拟合系数；不搜索新结构、不追加测量、不更新模型。'
              '**这些结果不替换学生原评分。**', '',
              '| 版本 | 原训练恢复 → 程序拟合后 | 原验证恢复 → 程序拟合后 |', '|---|---:|---:|']
    for name, folder, _ in entries:
        diagnostic = read(folder/'coefficient_diagnostic.json')['summary']
        cells = [f'{diagnostic[s]["original_recoveries"]} → {diagnostic[s]["after_arithmetic_diagnostic"]}' for s in ('train', 'validation')]
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines += ['', '稀疏表达式搜索使用同样六题、同样观测能严格恢复6/6，因此这些训练题的失败不能简单归因于信息不足。', '',
              '## 决定与解释', '']
    passed = [name for name, _, r in entries if r['learning_gate_passed']]
    if passed:
        lines += ['预定技术门槛已通过的版本：'+','.join(passed)+'。这仅表示能学会少量训练题，'
                  '可能包含记忆；参数变化和新结构仍需要独立检验。下一阶段是小规模 B/C 配对比较。']
    else:
        lines += ['两版均未达到“至少5/6可执行、至少4/5非空训练机制恢复”的预定门槛，'
                  '**未恢复完整 B/C 批次**。格式修复有作用，但还不能认为模型已经可靠掌握机制推断。', '',
                  '这不证明0.5B模型永远学不会，也不证明过程教学无效；它只说明本预算下的两套实现尚未通过学习检查。'
                  '后续若增加训练量、改变模型或引入数值拟合，应作为单独的开发实验，并继续保留失败结果。']
    lines += ['', '无作用世界的误报和无效答案均保留，未把“误报为零”设为继续实验的条件。'
              '两轮评分使用未见构型；验证答案不返给模型、未参与训练或步数选择。', '',
              '代码检查：终答掩码、训练/评估提示一致、B/C终答一致、辅助任务无判卷终答、'
              '辅助长度不稀释终答梯度、全部公开作用的局部表示等价，以及数值拟合不增加结构。', '',
              '详细记录：[hex版](REPORT_ZH.md)、[明确自旋版](../student_training_v2_spins/REPORT_ZH.md)。'
              '协议见项目根目录 STUDENT_TRAINING_V2_PROTOCOL.md 与 STUDENT_INPUT_CHECK_PROTOCOL.md。', '']
    path = folders[0]/'COMPARISON_ZH.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(path)


if __name__ == '__main__':
    run()
