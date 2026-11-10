"""Classify validation relative to the ACTUAL subset used to train each run.

The curriculum's validation is parameter-only relative to all 24 train worlds;
this does not imply parameter-only validation after selecting six train worlds.
No model inference, score change or new measurements.
"""
import hashlib
import json
from pathlib import Path

from discovery.student_v2 import load_views
from discovery.teaching import expression_from_dict


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify(train_views, validation_views, train_ids):
    selected = [v for v in train_views if v['episode_id'].split('-')[0] in train_ids]
    if len(selected) != len(train_ids):
        raise ValueError('Actual training subset could not be reconstructed.')
    supports = {expression_from_dict(v['final_answer']['expression']).support for v in selected}
    return {v['episode_id']: ('seen_structure' if expression_from_dict(v['final_answer']['expression']).support in supports
                              else 'unseen_structure') for v in validation_views}


def run():
    train, validation = load_views('train'), load_views('validation')
    lines = ['# v2 验证范围更正', '',
             '此前将12道验证题统称为“参数验证题”不准确：这个名称只适用于原来的完整24个训练世界。'
             'v2实际只训练6个世界，因此必须相对于这6个世界重新分层。', '',
             '**实际是8道见过结构的参数变化题，以及4道未见结构题。原生成文本、预测误差和评分均未改变。**', '',
             '| 版本 | 验证层次 | 题数 | 原机制恢复 | 程序只拟合系数后 |', '|---|---|---:|---:|---:|']
    for folder, name in ((Path('results/student_training_v2'), 'hex'),
                         (Path('results/student_training_v2_spins'), '明确自旋')):
        config, result = read(folder/'config.json'), read(folder/'result.json')
        cohorts = classify(train, validation, set(config['train_world_ids']))
        original = {r['episode_id']: r for r in result['validation']['results']}
        fitted = {r['episode_id']: r for r in read(folder/'coefficient_diagnostic.json')['results'] if r['split'] == 'validation'}
        summary = {}
        for cohort in ('seen_structure', 'unseen_structure'):
            ids = [key for key, value in cohorts.items() if value == cohort]
            summary[cohort] = {'worlds': len(ids), 'episode_ids': ids,
                              'null_worlds': sum(original[i]['null_world'] for i in ids),
                              'original_recoveries': sum(original[i]['full_recovery'] for i in ids),
                              'original_nonempty_recoveries': sum(original[i]['full_recovery'] and not original[i]['null_world'] for i in ids),
                              'fitted_recoveries_supplement_only': sum(fitted[i]['full_recovery'] for i in ids),
                              'fitted_nonempty_recoveries_supplement_only': sum(fitted[i]['full_recovery'] and not original[i]['null_world'] for i in ids)}
            s = summary[cohort]
            label = '见过结构，参数变化' if cohort == 'seen_structure' else '未见结构'
            lines.append(f'| {name} | {label} | {s["worlds"]} | {s["original_recoveries"]} | {s["fitted_recoveries_supplement_only"]} |')
        receipt = {'classification_uses_actual_training_subset': True, 'scores_changed': False,
                   'raw_result_sha256': sha(folder/'result.json'), 'cohorts': cohorts, 'summary': summary}
        (folder/'validation_scope_audit.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines += ['', '两版原始验证成功都只来自2个空机制世界。对明确自旋版，固定学生提出的结构、'
              '再由程序拟合系数，可恢复3/6个非空的已见结构参数变化题；4道未见结构题仍为0/4。'
              '这是补充诊断，不替换学生原分数，也不证明教学方法有效。', '',
              '后续先用完整24个训练世界建立结构提出与数值拟合的共同基线，再比较B/C。'
              '新的验证分层必须由实际训练结构自动计算；正式四体及组合测试继续保留。', '']
    path = Path('results/student_training_v2/VALIDATION_SCOPE_CORRECTION_ZH.md')
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(path)


if __name__ == '__main__':
    run()
