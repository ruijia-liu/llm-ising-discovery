"""Report completed integration/baseline checks, never formal teaching effects."""
import hashlib
import json
from pathlib import Path

from build_publication_curriculum import atomic_json,read,sha


def run():
    null_id=hashlib.sha256(json.dumps(()).encode()).hexdigest()
    baseline={m:read(f'results/mainline_development_sparse_{m}_v2/result.json') for m in ('fixed','active')}
    probes={v:read(f'results/active_student_development_{v}/result.json') for v in ('v1','v2')}
    rows=[]
    for method,result in baseline.items():
        for sigma in (0,1):
            selected=[r for r in result['rows'] if r['sigma']==sigma]
            null=[r for r in selected if r['structure_id']==null_id]
            rows.append({'method':method,'sigma':sigma,'worlds':len(selected),
                         'recoveries':sum(r['score']['full_recovery'] for r in selected),
                         'null_worlds':len(null),'false_discoveries':sum(bool(r['score']['false_discovery']) for r in null),
                         'budget_recoveries':{str(b):sum(next(c for c in r['score']['budget_curve'] if c['budget']==b)['full_recovery'] for r in selected) for b in (8,16,24,32)}})
        before=read(f'results/mainline_development_sparse_{method}/result.json')
        new={(r['world_id'],r['sigma']):r['score'] for r in result['rows']}
        if any(r['score']!=new[r['world_id'],r['sigma']] for r in before['rows']):
            raise AssertionError('Initial-log correction changed a numerical score.')
    paired=[]
    for sigma in (0,1):
        a={r['world_id']:r['score']['full_recovery'] for r in baseline['active']['rows'] if r['sigma']==sigma}
        b={r['world_id']:r['score']['full_recovery'] for r in baseline['fixed']['rows'] if r['sigma']==sigma}
        assert set(a)==set(b)
        paired.append({'sigma':sigma,'active_only':sum(a[w] and not b[w] for w in a),
                       'fixed_only':sum(b[w] and not a[w] for w in a),
                       'both':sum(a[w] and b[w] for w in a),'neither':sum(not a[w] and not b[w] for w in a)})
    output=Path('results/mainline_development_report')
    atomic_json(output/'summary.json',{'baseline':rows,'paired':paired,'formal_tests_used':False,
                'old_common_world_scores_unchanged_after_log_fix':True,
                'active_integration':{v:{k:r[k] for k in ('worlds','valid_final_answers','full_recoveries','tested_idea_recoveries','measured_flips')} for v,r in probes.items()},
                'source_result_sha256':{f'{m}_v2':sha(f'results/mainline_development_sparse_{m}_v2/result.json') for m in baseline}})
    lines=['# 正式管线的开发核验', '',
           '这里没有正式A/B/C效果或保留测试成绩。下面两类检查使用不同数据，不能横向比较恢复率。', '',
           '## 非语言模型：全部60个开发验证世界', '',
           '固定证据为32次预先随机测量；主动搜索与它共享初始8次及配对噪声流，自选追加24次。', '',
           '| 方法 | σ | 8次恢复 | 16次 | 24次 | 32次 | 空机制误报 |','|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        curve=row['budget_recoveries']
        lines.append(f'| {row["method"]} | {row["sigma"]} | {curve["8"]}/60 | {curve["16"]}/60 | '
                     f'{curve["24"]}/60 | {curve["32"]}/60 | {row["false_discoveries"]}/{row["null_worlds"]} |')
    lines+=['', '这些结果证明当前开发数据在给定语法和预算内对程序搜索相当可恢复，不保证语言模型能学会。',
            '主动并非每一题都优于固定测量；配对输赢计数保存在summary.json。误报继续保留。', '',
            '## 冻结旧学生：六个见过的训练世界', '',
            '| 接口版本 | 正常结束 | 最终独立恢复 | 事前检验idea恢复 | 追加测量 |','|---|---:|---:|---:|---:|']
    for version,result in probes.items():
        lines.append(f'| {version} | {result["valid_final_answers"]}/6 | {result["full_recoveries"]}/6 | '
                     f'{result["tested_idea_recoveries"]}/6 | {result["measured_flips"]-48} |')
    lines+=['', '补完整格式示例后，旧模型仍没有真正提交并执行一次追加实验。不能把接口试跑说成学生已学会主动研究。',
            '这组旧模型仅接受过最终结构回答训练；正式B/C将接受各自完整的行动/观测及过程教材。',
            '这不是新的学习门槛，不延长旧模型训练，不替它补选实验；正式研究按冻结协议继续。', '',
            '## 审计', '',
            '初始观测日志的列表别名问题已修复；旧12个共用开发条件的数值分数逐项相同。',
            '旧结果及日志保留，修复后完整60世界重跑保存在_v2目录。接口顺序、精确预测、预算、',
            '初始观测配对、格式失败不伪装为H0、正式测试访问门槛均有相应回归测试。', '']
    (output/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines[:16]))


if __name__=='__main__':
    run()
