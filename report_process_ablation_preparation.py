"""Write a factual preparation report without student or heldout scoring."""
from pathlib import Path

from build_publication_curriculum import read, sha


def run():
    root = Path('data/student_process_ablation_v1')
    manifest, audit = read(root/'manifest.json'), read(root/'audit.json')
    if not manifest['completed'] or sha(root/'audit.json') != manifest['audit_sha256']:
        raise ValueError('Preparation audit changed.')
    lines = ['# 显式数值预测消融：准备核验', '',
             '状态：600条教材和三个种子的计划已核验；补充组尚未训练、尚未评分。', '',
             '仅删除事前逐假设预测值及逐次预测误差。系统说明、观测、实验、全部候选表达式及其系数、',
             '排名、更新、教师最终估计保持一致；终答任务直接复用主缓存。', '',
             f"总计删除 {audit['deleted_fields']['predicted_delta']:,} 个预测值字段和 "
             f"{audit['deleted_fields']['prediction_errors']:,} 个误差数组字段；最长删减过程为 {audit['maximum_ablated_context']:,} token。", '',
             '| 种子 | 更新/终答呈现 | 输入token | 监督token | 辅助记录呈现 | 辅助输入差异 |',
             '|---|---:|---:|---:|---:|---:|']
    for seed, row in audit['budgets'].items():
        b = row['ablated']
        lines.append(f"| {seed} | {b['updates']}/{b['terminal_presentations']} | {b['input_tokens']:,} | "
                     f"{b['supervised_tokens']:,} | {b['auxiliary_presentations']} | {100*b['relative_auxiliary_input_difference']:+.5f}% |")
    b = audit['budgets']['401']['full_C']
    lines += ['', f"完整C每种子输入 {b['input_tokens']:,} token，监督 {b['supervised_tokens']:,} token，辅助记录呈现1200次。", '',
              '补齐输入量增加了删减轨迹的重复，因此中间候选表达式曝光也增加：完整C每种子81,592次，',
              '删减组为94,360 / 94,336 / 94,332次。这里只计事前候选列表，不计更新表达式。',
              '辅助损失仍平均归一化为权重1，终答次数和权重一致。输入量相近不等于FLOPs或内容曝光完全匹配。', '',
              '保留的表达式及选择摘要仍含可推导的预测信息；消融仅检验显式数值预测/误差的增量作用。', '',
              '代码入口：`train_process_ablation.py`、`run_process_ablation_queue.py`；默认检查主九个训练完成后才运行。',
              '共同评估入口：`evaluate_mainline.py --study prediction_ablation --group C_no_predictions`。',
              '冻结条件见 `PROCESS_ABLATION_PROTOCOL.md`。任何效果结论都必须等待实际训练和配对评估。', '',
              f"教材manifest SHA256：`{sha(root/'manifest.json')}`。", '']
    output = Path('results/process_ablation_preparation')
    output.mkdir(parents=True, exist_ok=True)
    (output/'REPORT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(output/'REPORT_ZH.md')


if __name__ == '__main__':
    run()
