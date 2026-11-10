"""Write the diagnostic report from frozen inference and supervision audit."""
import json
from pathlib import Path

from diagnose_student_pilot import CONTRAST, OUT, RUN, read_json


def run():
    result = read_json(OUT/'results.json')
    audit = read_json(OUT/'supervision_audit.json')
    baseline = read_json(OUT/'seen_world_baselines.json')
    rows = result['results']
    seen_counts = {g: sum(r['full_recovery'] for r in rows if r['group'] == g and r['split'] == 'train'
                         and r['mode'] == 'original') for g in ('B', 'C')}
    lines = ['# 暂停训练后的诊断', '',
             f'**原提示下，抽查的训练题完整恢复率为 B：{seen_counts["B"]}/6，C：{seen_counts["C"]}/6。'
             '问题已经出现在见过的题上，不能只归因于验证题泛化失败。**'
             '这里判断的是能否按任务要求交出正确终答，不据此断言模型内部完全没有获得相关知识。', '',
             '这次只做已保存模型的推理与数据审查，没有重新训练，没有增加实验室测量，也没有打开正式结构测试世界。', '',
             '整批实验保持暂停。B/101、C/101 已完成并保留；C/102 在第 5 个更新的保存点暂停，优化器与随机数状态完整保留。'
             '诊断前后模型文件、保存点、原验证结果及评分阈值的 SHA-256 全部一致。', '',
             '## 训练题与验证题', '',
             '使用相同的原验证提示：全部 32 次测量记录，最后追加终答指令。'
             '训练题预先选每类第一个世界，共 6 个：单体、方向差异、距离二、三体、组合、无额外作用；'
             '噪声标准差均为 2。验证题沿用已完成的 12 个结果，没有重复生成或改分。', '',
             '| 模型 | 数据 | 可执行表达式 | 结构正确 | 独立预测通过 | 完整恢复 |',
             '|---|---|---:|---:|---:|---:|']
    for group in ('B', 'C'):
        for split in ('train', 'validation'):
            s = next(s for s in result['summary'] if (s['group'], s['split'], s['mode']) == (group, split, 'original'))
            n = s['worlds']
            lines.append(f'| {group}/101 | {"见过的训练题" if split == "train" else "参数验证题"} | '
                         +' | '.join(f'{s[k]}/{n}' for k in ('valid_expression', 'support_success', 'prediction_success', 'full_recovery'))+' |')
    lines += ['', '“可执行”只表示程序能够计算表达式，不表示机制正确。完整恢复要求结构、系数误差和独立预测全部通过原来的冻结标准。'
              '评分对每个世界使用新的 256 个均匀构型及 256 个偏置构型，测量结果不反馈给模型。', '',
              '同样 6 道训练题，稀疏表达式搜索完整恢复 '
              +str(next(s['full_recovery'] for s in baseline['summary'] if s['group'] == 'sparse_search'))
              +' 道；始终回答 H0 的基线恢复 1 道（无额外作用题）。它们只使用相同证据，未追加查询。', '',
              '## 提示格式的小范围对照', '',
              '每个分区预先固定 3 题：单体、三体、无额外作用。下表每格为“可执行数 / 完整恢复数”，分母均为 3。'
              '这些是开发诊断，不能当作新的正式模型排名。', '',
              '| 提示条件 | B 训练 | B 验证 | C 训练 | C 验证 |',
              '|---|---:|---:|---:|---:|']
    descriptions = {
        'original': '原验证提示（相同三题子集）',
        'shared_prefix': '共同证据，不追加终答指令',
        'prefilled_fixed': '原验证提示 + 固定 JSON 外壳',
        'native_prefix': 'C 原训练历史，含教师过程记录',
        'native_prefilled': 'C 原训练历史 + 固定 JSON 外壳',
    }
    for mode, label in descriptions.items():
        cells = []
        for group in ('B', 'C'):
            for split in ('train', 'validation'):
                subset = [r for r in rows if r['group'] == group and r['split'] == split and r['mode'] == mode
                          and r['index'] in CONTRAST[split]]
                cells.append(f'{sum(r["valid_expression"] for r in subset)} / {sum(r["full_recovery"] for r in subset)}' if subset else '—')
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    successes = [r for r in rows if r['mode'] != 'original' and r['full_recovery']]
    lines += ['', f'这些提示变式共出现 {len(successes)} 次完整恢复，其中 '
              f'{sum(not r["null_world"] for r in successes)} 次来自有额外作用的世界。'
              '因此需单独检查非空机制，不能仅凭格式通过或空机制题的得分宣布学会发现规律。']
    continuing = [r for r in rows if r['group'] == 'C' and r['mode'] == 'native_prefix'
                  and r['text'].startswith('{"before_observation":')]
    lines += ['', f'C 恢复原训练历史的 6 次续写中，有 {len(continuing)} 次以 `before_observation` 开头，'
              '继续生成过程记录。输入已经含 `used=32`；这提示当前模型没有可靠学会在实验结束时交终答。']
    lines += ['', '“固定 JSON 外壳”只补写 `{"final_answer":{"expression":{"terms":`，'
              '不提供系数、自旋位置、作用项数或正确机制。补写占 8 token，从原来的 256 token 输出预算中扣除，'
              '剩余内容由模型生成。保持原有严格解析器；没有从坏答案里挑取片段重新判对。', '',
              '**C 原训练历史包含教师先前推断，信息比共同证据丰富。** 这两行只用于检查训练格式下能否续写；'
              '即使分数提高，也不能计作 C 优于 B 的公平实验结果。最终标准答案及最后一轮教师估计均已从提示移除。', '',
              '## 最终答案确实参与了训练，但权重不均', '',
              '逐 token 对照原始对话、实际输入数组和损失标签，全部 24 个训练世界的终答字段都未被屏蔽。'
              'B/C 的最终正确答案完全一致，保存的数据没有截断。', '',
              '| 已完成的种子 101 | B | C |', '|---|---:|---:|']
    b = audit['groups']['B']['schedule_counts']
    c = audit['groups']['C']['schedule_counts']
    for label, key in [('输入 token', 'input_tokens'), ('全部受监督 token', 'all_supervised_tokens'),
                       ('终答字段受监督 token', 'answer_field_supervised_tokens'), ('正确终答出现次数', 'answer_presentations')]:
        lines.append(f'| {label} | {b[key]:,} | {c[key]:,} |')
    lines.append('| 终答在每步损失中的平均占比 | '
                 +f'{100*b["mean_answer_fraction_of_update_loss"]:.3f}% | {100*c["mean_answer_fraction_of_update_loss"]:.3f}% |')
    ratio = b['mean_answer_fraction_of_update_loss']/c['mean_answer_fraction_of_update_loss']
    lines += ['', f'现实现按每一步的全部助手 token 平均损失，C 的终答相对损失权重约为 B 的 1/{ratio:.2f}。'
              '字段统计包含 `final_answer` 的键、expression 和监督来源标签；该比例是损失项的配额，不等于测出的梯度贡献或学习效果。'
              '总输入 token 接近并不保证终答监督强度接近。C 较低的总体训练损失也不能直接解释为推断更好。', '',
              '## 发现的格式差异', '',
              '1. 训练记录在最后一个测量结果之后直接给标准答案；验证多加一条训练时没有的终答请求，形成连续两条 user 消息。',
              '2. C 训练时之前的 assistant 消息含教师假设、预测和修正；原验证提示采用 B 格式，移除了这些教师过程。'
              '固定证据比较本身需要公平输入，但当前训练没有专门教 C 在这种输入下作最终判断。',
              '3. B 最后一条助手消息仅含终答；C 同一条消息还要输出最后修正与教师估计。'
              f'C 有 {c["final_messages_over_256"]}/24 条完整训练终消息超过验证的 256 token 上限，最长 {c["final_message_max_tokens"]} token；'
              f'其中正确终答字段本身最长只有 {c["final_answer_field_max_tokens"]} token。'
              '但原 C 验证并非因达到长度上限才失败，短输出已经出现错误嵌套或类型。', '',
              '## 误报与失败原样保留', '',
              '| 原提示条件 | 无额外作用题数 | 明确误报 | 无效答案 |', '|---|---:|---:|---:|']
    for s in result['summary']:
        if s['mode'] == 'original':
            lines.append(f'| {s["group"]} / {s["split"]} | {s["null_world"]} | {s["false_discovery"]} | {s["invalid_null_answers"]} |')
    lines += ['', '无效答案属于失败，不能因未产生可评分机制而宣称它没有误报。没有用本轮结果重新调整成功阈值。', '',
              '## 恢复完整实验之前的修订', '',
              '这次小检查不足以支持直接续跑剩余种子。现有训练题已出现失败，并伴随终答监督不均和输入格式差异；'
              '继续旧配置会重复这些问题。旧实验作为 v1 留存，不在同一批次中途换教材或损失函数。', '',
              '下一版先做一个独立的小验证：', '',
              '1. 单独构建“共同证据 → 终答”的任务，两组使用相同提示、相同终答结构，与固定证据评估完全一致。'
              'C 的研究过程教学保留为另外的训练任务；终答不再混入教师估计和最后修正。',
              '2. 两组终答任务使用相同样本、出现次数和显式损失系数，另记过程任务的损失；'
              '继续报告输入 token、监督 token、更新数、实际用时及终答配额，不只匹配总输入长度。',
              '3. 先用少量无噪声题检查训练管线，确认见过的题能输出可执行且物理预测通过的机制，再检查参数验证题和高噪声题。'
              '必须同时保留有作用与无作用世界；恢复训练的条件不是消灭误报。若仍失败，再检查模型容量、训练量和输入表示。', '',
              '本次没有执行这些训练修改，也没有重新训练。格式对照不等于对训练方法的因果消融；'
              '一个种子、少量高噪声题只能定位当前实现的问题，不能判断显式研究过程教学是否有效。', '',
              '## 长提示的运行修正', '',
              '前 26 次推理完成后，第一次 C 原训练历史提示触发显存不足，尚未产生答案。'
              '已完成结果原样保留，剩余推理从中断处继续。长历史改为每段 512 token 读取，保留所有注意力缓存；'
              '不截断上下文，不改变输出预算，保持原模型的贪心解码与 1.1 重复惩罚。'
              '该改动只用于 C 原训练历史的诊断条件。小型模型测试确认分段读取与一次读取产生相同答案；'
              '不宣称不同精度及所有输入下均逐位相同。运行修正记录见 `inference_memory_amendment.json`。', '',
              '## 可复查文件', '',
              '- `plan.json`：生成前固定的题目、条件、代码摘要、模型与保存点摘要。',
              '- `supervision_audit.json`：逐题 token 统计、每步终答损失占比。',
              '- `results.jsonl`：48 次新推理的完整原文、严格解析错误、物理误差。',
              '- `results.json`：新推理、沿用的验证结果、汇总及前后文件一致性检查。',
              '- `seen_world_baselines.json`：相同训练题与证据的两个非语言模型参照。', '',
              '复现代码位于项目根目录 `diagnose_student_pilot.py`、`diagnose_student_baselines.py`、'
              '`report_student_diagnostics.py`。诊断拒绝覆盖已有推理结果。'
              '四个针对性测试覆盖提示中无判卷答案、B/C 共同输入一致、审查能检出被错误屏蔽的终答，以及分段读取保留上下文。', '']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    decision = {'full_training_status': 'paused', 'new_training_performed': False,
                'completed_models_used': ['B_seed101', 'C_seed101'],
                'preserved_partial_model': 'C_seed102', 'preserved_updates': 5,
                'resume_current_configuration_recommended': False,
                'seen_world_full_recoveries_original_prompt': seen_counts,
                'reason': 'Seen-world failures, unequal terminal-answer supervision and training/evaluation format mismatch.',
                'next_step': 'Separate follow-up with matched terminal-answer tasks and loss allocation; small noiseless feasibility check before scaling.',
                'zero_false_discoveries_required': False}
    (OUT/'decision.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(OUT/'REPORT_ZH.md')


if __name__ == '__main__':
    run()
