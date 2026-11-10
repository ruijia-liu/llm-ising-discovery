"""Write a readable snapshot from append-only training logs; no model access."""
import argparse
from datetime import datetime
import json
from pathlib import Path


def status(output):
    output = Path(output)
    config = json.loads((output/'experiment.json').read_text(encoding='utf-8'))
    paused = (output/'PAUSED.json').exists()
    diagnostic = output.parent/'student_diagnostics_v1/REPORT_ZH.md'
    diagnostic_done = output.name == 'student_training_v1' and diagnostic.exists()
    v2_started = output.name == 'student_training_v1' and (output.parent/'student_training_v2/config.json').exists()
    rows, completed = [], 0
    for seed in config['seeds']:
        for group in ('B', 'C'):
            folder = output/f'{group}_seed{seed}'
            log = folder/'training.jsonl'
            steps = []
            if log.exists():
                for line in log.read_text(encoding='utf-8').splitlines():
                    try:
                        steps.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # A snapshot can coincide with an unfinished final write.
            validation = folder/'validation.json'
            result = json.loads(validation.read_text(encoding='utf-8')) if validation.exists() else None
            if result:
                state = '训练和验证完成'
                completed += 1
            elif (folder/'training_result.json').exists():
                state = '训练完成，正在验证'
            elif steps:
                state = '已暂停，保存点保留' if paused else '训练中'
            elif folder.exists():
                state = '已暂停，未完成第一步' if paused else '开始第一步训练'
            else:
                state = '未开始，整批暂停' if paused else '等待前面的组完成'
            valid = f'{result["valid_expressions"]}/12' if result else '—'
            recovery = f'{result["full_recoveries"]}/12' if result else '—'
            rows.append(f'| {group} / {seed} | {state} | {len(steps)}/12 | {valid} | {recovery} |')
    lines = ['# 学生训练进度', '',
             f'更新时间：{datetime.now().astimezone().isoformat(timespec="seconds")}。这是一次快照，不自动刷新。', '',
             f'已完成训练及验证：**{completed}/6 组**。', '',
             ('当前状态：v1 整批训练继续暂停；独立 v2 开发实验另行记录。' if v2_started else
              '当前状态：小规模诊断已完成，整批训练继续暂停；未重新训练。' if diagnostic_done else
              '当前状态：按用户要求暂停整批训练，正在检查训练题表现和训练/验证格式匹配。') if paused else '当前状态：整批试跑进行中。', '',
             'B 学习证据记录；C 学习相同证据，加上老师的假设、预测和修正过程。',
             '每组使用相同的 24 个训练世界；验证只看另外 12 个参数变化世界，正式结构测试仍未启用。', '',
             '| 组别 / 种子 | 状态 | 训练步数 | 可执行回答 | 完整恢复机制 |',
             '|---|---|---:|---:|---:|', *rows, '',
             '“可执行”只表示表达式能被程序计算；“完整恢复”还要求结构、系数和独立预测都通过检查。',
             '当前数据不足以判断过程教学是否有效。全部结束后才生成正式试跑报告。', '']
    if diagnostic_done:
        lines += ['[暂停后的诊断报告](../student_diagnostics_v1/REPORT_ZH.md)记录训练题回测、终答监督审查及下一版修订建议。', '']
    if v2_started:
        lines += ['[v2 修订和输入表示检查](../student_training_v2/COMPARISON_ZH.md)单独记录，不覆盖 v1。', '']
    path = output/'STATUS_ZH.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/student_training_v1')
    status(parser.parse_args().output)
