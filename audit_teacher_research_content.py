"""Audit whether the completed teacher corpus actually contains model revision."""
from collections import Counter
import json
from pathlib import Path

from build_publication_curriculum import atomic_json, read, sha
from discovery.teaching import expression_from_dict


def run():
    source = Path('data/research_curriculum_publication_v1')
    audit = read(source/'audit_report.json')
    manifest = {p.replace('\\', '/'): h for p, h in audit['export_sha256'].items()}
    summary, hashes = [], {}
    for split in ('train', 'validation'):
        path = source/'views'/split/'C.jsonl'
        hashes[str(path)] = sha(path)
        if hashes[str(path)] != manifest[path.relative_to(source).as_posix()]:
            raise AssertionError('Teaching export changed.')
        counts = {0.0: Counter(), 1.0: Counter()}
        for line in path.read_text(encoding='utf-8').splitlines():
            view = json.loads(line)
            counter = counts[view['noise_sigma']]
            truth = expression_from_dict(view['final_answer']['expression'])
            rounds = view['research_process']
            first = expression_from_dict(rounds[0]['before']['hypotheses'][0]['expression'])
            counter['worlds'] += 1
            counter['first8_correct_support'] += first.support == truth.support
            counter['worlds_with_structure_update'] += any(r['after']['change'] == 'structure_changed' for r in rounds)
            for record in rounds:
                counter['rounds'] += 1
                counter['at_least_two_near_best_displayed'] += sum(h['within_near_best_band'] for h in record['before']['hypotheses']) >= 2
                counter['positive_committee_prediction_disagreement'] += record['before']['selection']['prediction_disagreement'] > 1e-9
                counter['structure_updates'] += record['after']['change'] == 'structure_changed'
        summary += [{'split': split, 'sigma': sigma, **count} for sigma, count in counts.items()]
    output = Path('results/teacher_research_content')
    result = {'source_sha256': hashes, 'summary': summary, 'test_worlds_opened': False,
              'interpretation': 'Noiseless records mainly confirm an already recovered structure. Noise creates more real competition and revision.',
              'not_student_results': True, 'filtering_or_regenerating_episodes': False}
    atomic_json(output/'audit.json', result)
    lines = ['# 教材里有多少真实的竞争与修正', '',
             '这是教师过程的描述性审计，使用最终标签只作事后判卷；不改变老师已经做过的决策。', '',
             '| 集合 | σ | 初始8次已恢复结构 | 出现结构修正的世界 | 有候选预测分歧的轮数 |',
             '|---|---:|---:|---:|---:|']
    for row in summary:
        lines.append(f'| {row["split"]} | {row["sigma"]:g} | {row["first8_correct_support"]}/{row["worlds"]} | '
                     f'{row["worlds_with_structure_update"]}/{row["worlds"]} | '
                     f'{row["positive_committee_prediction_disagreement"]}/{row["rounds"]} |')
    lines += ['', '无噪声训练世界中298/300个在初始8次后就已得到正确结构，后面大多是确认。',
              '因此无噪声成绩可以检验管线和结构预测，但单靠它难以充分检验学会修改假设或选择有区分力的实验。',
              'σ=1的教材包含更多仍可竞争的模型与实际修正，正式协议应预先明确它在主分析中的地位。',
              '这些数字不说明学生是否已学会，也不能预判C会胜出。', '',
              '展示的备选并不总是近优或仍合理；不能把每一条过程记录都称为经过有效竞争检验的新idea。',
              '所有世界和失败轨迹继续保留，不依照这些统计筛掉无变化记录或生成更好看的结果。', '']
    (output/'REPORT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
