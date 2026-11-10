"""Select symmetric, auditable cases from complete saved B/C evaluations only."""
import argparse
from collections import defaultdict
from pathlib import Path
import json

from build_publication_curriculum import atomic_json, read, sha
from discovery.behavior_cases import (GROUPS, METHODS, SEEDS, SALT, identity, keyed_hash,
                                     pair_artifacts, select_cases, timeline, validate_run_grid)
from discovery.evaluation_results import load_completed_evaluation, verify_pairing


SOURCES = ('report_behavior_cases.py', 'discovery/behavior_cases.py', 'discovery/evaluation_results.py',
           'BEHAVIOR_CASE_PROTOCOL.md', 'MAINLINE_STUDY_PROTOCOL.md', 'SECONDARY_ANALYSIS_PROTOCOL.md')


def plain(value):
    """Treat model strings as literal table contents, never Markdown instructions."""
    rendered = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    return rendered.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('|', '&#124;').replace('`', '&#96;')


def case_markdown(card):
    pair = card['pair']
    lines = [f"# Paired behavior case: {pair['world_id']} / seed {pair['seed']} / sigma {pair['sigma']}", '',
             'Selected deterministically under the saved case protocol. This is an illustration, not a representative sample or a new statistical test.', '',
             'Selection slots: '+', '.join(card['selection_slots']), '',
             f"Structure ID: `{pair['structure_id']}`. Stratum: `{pair['level']}`.", '',
             'The original ground-truth expression was not copied into these evaluator artifacts; it is not inferred from its hash.', '',
             '| Group | Active status | Measurements | Calls / output tokens | Full recovery | Tested idea | Competitive idea | Grader category |',
             '|---|---|---:|---:|---|---|---|---|']
    for group in GROUPS:
        s = pair['groups'][group]
        lines.append(f"| {group} | {s['status']} | {s['used']} | {s['inference']['calls']} / {s['inference']['output_tokens']} | "
                     f"{s['full_recovery']} | {s['tested_idea_recovery']} | {s['competitive_idea_recovery']} | {s['grader_category']} |")
    lines += ['', 'Fixed evidence (the same 32 observations for B/C; later observations can differ from their active choices):', '',
              '| Group | Budget | Valid | Raw support correct | Full recovery |', '|---|---:|---|---|---|']
    for group in GROUPS:
        for r in pair['groups'][group]['fixed_budget_curve']:
            lines.append(f"| {group} | {r['budget']} | {r['valid_expression']} | {r['raw_support_success']} | {r['full_recovery']} |")
    for group in GROUPS:
        s = pair['groups'][group]
        lines += ['', f'## {group}: recorded active sequence', '',
                  f"Example-query/alternative copies: {s['example_query_and_alternative_batches']} batches; "
                  f"operationally competitive tests: {s['competitive_batches']} batches.", '',
                  'Independent scores below were assigned after the episode and were never returned to the student. '
                  'Reason/update text is a literal model output, not an assessment of internal reasoning.', '',
                  '| Source event | Used | Event | Recorded contents |', '|---:|---:|---|---|']
        for r in card['timelines'][group]:
            kind = r['kind']
            if kind == 'current_hypothesis':
                detail = {'raw': r['raw_structures'], 'fitted': r['expression'],
                          'post_episode_full_recovery': r['independent_score']['full_recovery']}
            elif kind == 'predictions_committed':
                c = r['commitment']
                detail = {'hypotheses': c['proposal']['hypotheses'], 'predictions': c['predictions'],
                          'reason': c['proposal']['reason'], 'update_condition': c['proposal']['update_condition'],
                          'competitive': r['idea_diagnostic']['operational_competitive_test']}
            elif kind == 'observations_revealed':
                detail = r['observation']
            elif kind == 'model_reply':
                detail = {'stage': r['stage'], 'raw_reply': r['raw_reply']}
            elif kind == 'initial_evidence':
                detail = {'initial_observations': [m['delta'] for m in r['evidence']['measurements']]}
            else:
                detail = r['record']
            lines.append(f"| {r['source_event_index']} | {r['used']} | {kind} | {plain(detail)} |")
    lines += ['', 'The adjacent JSON preserves complete source artifacts, including queries, original messages, '
              'fitted values and all generation replies. No measurements or scores were recomputed for this card.', '']
    return '\n'.join(lines)


def run(root, output, scope='formal'):
    root, output = Path(root), Path(output)
    if scope not in ('formal', 'development'):
        raise ValueError('A report must explicitly retain its development/formal scope.')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserve prior case reports; use a new output directory.')
    sources = {p: sha(p) for p in SOURCES}
    evaluations = [load_completed_evaluation(root/method/f'{group}_seed{seed}', scope=scope, study='mainline',
                    method=method, group=group, seed=seed)
                   for method in METHODS for group in GROUPS for seed in SEEDS]
    for method in METHODS:
        verify_pairing([e for e in evaluations if e['config']['method'] == method])
    runs = validate_run_grid(evaluations)

    def records_for(world, sigma, seed):
        records, provenance = {}, {}
        key = world+f'-sigma{sigma}'
        for method in METHODS:
            for group in GROUPS:
                e = runs[method, group, seed]
                p = Path(e['provenance']['directory'])/'episodes'/(key+'.json')
                checksum = sha(p)
                if checksum != e['provenance']['episodes_sha256'][key]:
                    raise ValueError('An episode changed after complete-run verification.')
                records[method, group] = read(p)
                provenance[f'{method}:{group}'] = {'path': str(p), 'sha256': checksum}
        return records, provenance

    first = evaluations[0]['result']['rows']
    worlds = sorted((r['world_id'], r['sigma']) for r in first)
    pairs = []
    for seed in SEEDS:
        for world, sigma in worlds:
            records, _ = records_for(world, sigma, seed)
            pairs.append(pair_artifacts(records))
    slots = select_cases(pairs)
    selected = defaultdict(list)
    for slot in slots:
        if slot['selected'] is not None:
            s = slot['selected']
            selected[s['world_id'], s['sigma'], s['seed']].append(slot['slot'])
    by_cell = {(p['world_id'], p['sigma'], p['seed']): p for p in pairs}
    cards = []
    for cell, labels in sorted(selected.items()):
        records, provenance = records_for(*cell)
        pair = by_cell[cell]
        filename = keyed_hash('case', identity(pair))[:24]
        card = {'scope': scope, 'illustration_only_not_a_teaching_effect_estimate': True,
                'selection_slots': labels, 'pair': pair, 'sources': provenance,
                'timelines': {g: timeline(records['llm_active', g]) for g in GROUPS},
                'complete_artifacts': {f'{m}:{g}': records[m, g] for m in METHODS for g in GROUPS}}
        json_path, md_path = output/'cases'/(filename+'.json'), output/'cases'/(filename+'.md')
        atomic_json(json_path, card)
        md_path.write_text(case_markdown(card), encoding='utf-8')
        cards.append({**identity(pair), 'selection_slots': labels, 'json': str(json_path.relative_to(output)),
                      'markdown': str(md_path.relative_to(output)), 'json_sha256': sha(json_path), 'markdown_sha256': sha(md_path)})
    if any(sha(p) != checksum for p, checksum in sources.items()):
        raise ValueError('Case-selection source changed during extraction.')
    result = {'completed': True, 'scope': scope, 'formal_student_results_used': scope == 'formal',
              'new_model_calls': 0, 'new_measurements': 0, 'selection_salt': SALT,
              'paired_seed_world_noise_cells': len(pairs), 'slots': slots, 'selected_cards': cards,
              'all_pair_summaries': pairs, 'evaluations': [e['provenance'] for e in evaluations],
              'source_sha256': sources, 'scientific_case_interpretation_completed': False}
    atomic_json(output/'result.json', result)
    lines = ['# 配对行为案例：确定性提取', '', f'范围：{scope}。这是案例清单，不是新的显著性检验或教学效果估计。', '',
             '全体B/C、三个种子及固定/主动输出均已核验；没有重新调用模型或进行测量。', '',
             '| 槽位 | 合格种子×世界单元 | 参数世界 | 结构 | 选中世界 / 种子 |', '|---|---:|---:|---:|---|']
    for slot in slots:
        s = slot['selected']
        label = f"{s['world_id']} / {s['seed']}" if s else '无'
        lines.append(f"| {slot['slot']} | {slot['eligible_seed_world_cells']} | {slot['eligible_parameter_worlds']} | {slot['eligible_structures']} | {label} |")
    lines += ['', '重叠槽位共用卡片；没有案例的槽位保持为空。全体合格ID和配对摘要在JSON中保留。', '',
              '以下卡片包含两组主动轨迹及同一世界的固定证据分数，供结合总体结果作科学解释：', '']
    lines += [f"- [{c['world_id']} / seed {c['seed']}]({Path(c['markdown']).as_posix()})" for c in cards]
    lines += ['', '选中案例不是随机代表样本；不能仅凭单例、自然语言理由或格式成功推断内部研究能力。',
              '本入口仅提取证据，未自动完成行为解释或论文结论。', '']
    (output/'REPORT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(output/'REPORT_ZH.md')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='results/mainline_formal')
    parser.add_argument('--output', default='results/mainline_behavior_cases')
    parser.add_argument('--scope', choices=('formal', 'development'), default='formal')
    args = parser.parse_args()
    run(args.root, args.output, args.scope)
