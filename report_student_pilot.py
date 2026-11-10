"""Render completed local student results without hiding invalid answers."""
import argparse
import json
from pathlib import Path

import numpy as np


def report(output):
    output = Path(output)
    data = json.loads((output/'results.json').read_text(encoding='utf-8'))
    validation, training = data['validation'], data['training']
    lines = ['# 第一轮学生训练试跑', '',
             '已完成同一个基础模型的 B（证据教材）与 C（过程教材）训练，各 3 个随机种子。',
             '使用全部 24 个训练世界的强噪声版本（σ=2）。保留失败教材；没有要求先消除误报。', '',
             '**这是 12 个参数变化验证世界上的固定证据检查，不是未见结构的正式测试。**',
             '每个模型看到相同的 32 次观测，之后输出机制，不追加查询。正式测试集仍未使用。', '',
             '## 学生实际表现', '',
             '| 模型 | 可执行回答 | 结构正确 | 预测达标 | 完整恢复 | 空世界误报 | 空世界无效回答 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in validation:
        name = '未训练基础模型' if row['seed'] is None else f'{row["group"]}，种子 {row["seed"]}'
        n, null = row['worlds'], row['null_worlds']
        values = [f'{row[k]}/{n}' for k in ('valid_expressions', 'correct_structures', 'prediction_successes', 'full_recoveries')]
        values += [f'{row[k]}/{null}' for k in ('false_discoveries', 'invalid_null_answers')]
        lines.append('| '+name+' | '+' | '.join(values)+' |')
    baseline_path = output/'fixed_evidence_baselines.json'
    if baseline_path.exists():
        baselines = json.loads(baseline_path.read_text(encoding='utf-8'))
        if baselines['source_sha256'] != data['config']['validation_labels_sha256']:
            raise ValueError('Baseline evidence differs from student evidence.')
        for row in baselines['baselines']:
            name = {'always_H0': '始终回答无额外作用', 'sparse_search': '稀疏表达式搜索（固定证据）'}[row['group']]
            n, null = row['worlds'], row['null_worlds']
            values = [f'{row[k]}/{n}' for k in ('valid_expressions', 'correct_structures', 'prediction_successes', 'full_recoveries')]
            values += [f'{row["false_discoveries"]}/{null}', f'0/{null}']
            lines.append('| '+name+' | '+' | '.join(values)+' |')
    lines += ['', '完整恢复同时要求结构正确、独立构型预测达标、系数误差达标。',
              '无效或被输出上限截断的答案计入失败；空世界的无效回答单列，不能当成“没有误报”。', '',
              '两种非语言模型参照也只使用相同的 32 次证据，没有追加查询。稀疏搜索的候选来自公共表达语言，',
              '拟合函数不接收世界真值；其计算预算不与语言模型 token 数匹配。', '',
              '## 未见构型上的预测误差', '',
              '| 模型 | 可计误差的回答数 | 平均均匀构型 RMSE | 平均偏置构型 RMSE |',
              '|---|---:|---:|---:|']
    for row in validation:
        name = '未训练基础模型' if row['seed'] is None else f'{row["group"]}，种子 {row["seed"]}'
        valid = [r for r in row['results'] if r['valid_expression']]
        errors = [f'{np.mean([r[k] for r in valid]):.4f}' if valid else '不适用' for k in ('uniform_rmse', 'biased_rmse')]
        lines.append(f'| {name} | {len(valid)}/{row["worlds"]} | '+ ' | '.join(errors)+' |')
    lines += ['', 'RMSE 只对可执行表达式定义；这里的均值以可执行回答为条件，不能据此忽略无效回答。', '',
              '## 失败发生在哪里', '',
              '| 模型 | 格式或表达语言无效 | 结构错误 | 结构正确但参数/预测未达标 | 完整恢复 | 非空世界恢复 |',
              '|---|---:|---:|---:|---:|---:|']
    for row in validation:
        name = '未训练基础模型' if row['seed'] is None else f'{row["group"]}，种子 {row["seed"]}'
        counts = [sum(not r['valid_expression'] for r in row['results']),
                  sum(r['valid_expression'] and not r['support_success'] for r in row['results']),
                  sum(r['support_success'] and not r['full_recovery'] for r in row['results']),
                  row['full_recoveries']]
        if sum(counts) != row['worlds']:
            raise ValueError('Failure categories must partition every validation answer.')
        nonnull = [r for r in row['results'] if not r['null_world']]
        nonnull_success = sum(r['full_recovery'] for r in nonnull)
        lines.append('| '+name+' | '+' | '.join(map(str, counts))+f' | {nonnull_success}/{len(nonnull)} |')
    lines += ['', '这几个类别互斥。结构正确但系数不准，与没有构造出正确相互作用，属于不同问题。',
              '在空世界输出非零额外项仍计为误报，不因其系数较小而事后改成零。', '',
              '## 配对比较', '', '| 种子 | 仅 B 恢复 | 仅 C 恢复 | 都恢复 | 都失败 |', '|---|---:|---:|---:|---:|']
    paired = []
    for seed in data['config']['seeds']:
        b, c = [next(r for r in validation if r['group'] == g and r['seed'] == seed) for g in ('B', 'C')]
        bm, cm = [{r['episode_id']: r for r in arm['results']} for arm in (b, c)]
        if bm.keys() != cm.keys():
            raise ValueError('Validation worlds are not paired.')
        cases = [(bm[k]['full_recovery'], cm[k]['full_recovery']) for k in bm]
        counts = [sum((x, y) == target for x, y in cases) for target in ((True, False), (False, True), (True, True), (False, False))]
        paired.append({'seed': seed, 'only_B': counts[0], 'only_C': counts[1], 'both': counts[2], 'neither': counts[3]})
        lines.append('| '+str(seed)+' | '+' | '.join(map(str, counts))+' |')
    lines += ['', '三个种子反复评估同一批世界，不能把 36 次结果当作 36 个独立世界。',
              '本试跑不据此宣称 C 优于 B，也不判断模型内部学会了哪种研究能力。', '',
              '## 实际训练消耗', '', '| 组别/种子 | 更新步数 | 输入 token | 监督目标 token | 完整教材呈现次数 | 耗时/分钟 | PyTorch 峰值分配/GiB |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in training:
        lines.append(f'| {row["group"]}/{row["seed"]} | {row["optimizer_updates"]} | {row["processed_tokens"]:,} | '
                     f'{row["supervised_tokens"]:,} | {row["episode_presentations"]} | {row["training_seconds"]/60:.1f} | '
                     f'{row["peak_allocated_gib"]:.2f} |')
    lines += ['', 'B 只重复同一批教材来匹配输入 token。两组更新次数相同，完整上下文没有截断。',
              '输入 token 匹配不等于监督目标 token、FLOPs 或墙钟时间相同，长上下文的 C 计算成本更高。',
              '所有模型使用相同的贪心解码和 256 个新 token 上限；无效答案保留原文。', '',
              '表中耗时只计完成版本的训练。另有格式预检查、显存预检和中断尝试，相关记录均保留，',
              '部分失败尝试没有完整计时；不能把表格相加当作本项目的总运行时间。', '',
              '显存列为 PyTorch 实际分配的张量峰值，不包含其保留缓存、驱动和桌面应用的占用。', '',
              '## 如何理解这一轮', '',
              '这轮首先检查教材是否能用于训练、模型能否产生可执行表达式，以及误报和失败如何被记录。',
              '训练量很小，模型也很小；十六进制构型还要求模型完成编码理解。若恢复失败，不能据此否定过程教学。',
              '下一阶段应根据保存的原始输出区分格式失败、系数估计失败和结构失败，另行登记改动，',
              '之后再做自由选实验与未见结构测试。不要在这一批正式保留测试世界上反复调参。', '',
              '## 文件', '',
              '- `experiment.json`：训练前锁定的设置与顺序。',
              '- `preflight.json`：丢弃的显存/梯度预检。',
              '- `results.json`：全部指标、原始回答与训练消耗。',
              '- `fixed_evidence_baselines.json`：相同证据上的稀疏搜索与恒定 H0 参照。',
              '- 每组目录下的 `training.jsonl`、`training_result.json`、`validation.json` 与 `adapter/`。',
              '- [完整协议](../../STUDENT_TRAINING_PILOT_PROTOCOL.md)。', '']
    (output/'SUMMARY_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    (output/'paired_comparison.json').write_text(json.dumps(paired, indent=2)+'\n', encoding='utf-8')
    print(output/'SUMMARY_ZH.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/student_training_v1')
    report(parser.parse_args().output)
