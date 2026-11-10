"""Measure structural exposure using development exports, never test answers."""
from itertools import combinations
import json
from pathlib import Path

from build_publication_curriculum import atomic_json, read, sha
from build_research_curriculum import walk_expressions
from discovery.teaching import expression_from_dict


MAIN = Path('data/research_curriculum_publication_v1')


def summarize_exposure(components, labeled, stages):
    possible = {tuple(sorted(pair)) for pair in combinations(components, 2)}
    seen = set(labeled)
    rows = [{'scope': 'formal_training_final_answers_only', 'exposed_supports': len(seen),
             'newly_exposed_pairs': len(possible & seen), 'eligible_unseen_pairs': len(possible-seen)}]
    for name, supports in stages:
        previous = len(possible-seen)
        seen.update(supports)
        rows.append({'scope': name, 'exposed_supports': len(seen),
                     'newly_exposed_pairs': previous-len(possible-seen),
                     'eligible_unseen_pairs': len(possible-seen)})
    return {'training_components': len(components), 'possible_distinct_pairs': len(possible),
            'cumulative_exposure': rows, 'all_exposed_supports': len(seen)}


def run():
    audit = read(MAIN/'audit_report.json')
    if not audit['completed'] or audit['tests_measured_or_scored']:
        raise AssertionError('Expected the completed, untested formal curriculum.')
    hashes, stages = {}, []
    exports = {key.replace('\\', '/'): value for key, value in audit['export_sha256'].items()}
    labels, components = set(), set()
    for name, root, split in [
        ('formal_training_all_C_content', MAIN, 'train'),
        ('plus_formal_validation_all_C_content', MAIN, 'validation'),
        ('plus_legacy_training', Path('data/research_curriculum_v1'), 'train'),
        ('plus_legacy_validation', Path('data/research_curriculum_v1'), 'validation'),
        ('plus_coverage_training', Path('data/research_curriculum_coverage'), 'train'),
        ('plus_coverage_validation', Path('data/research_curriculum_coverage'), 'validation'),
    ]:
        path = root/'views'/split/'C.jsonl'
        hashes[str(path)] = sha(path)
        if root == MAIN and sha(path) != exports[path.relative_to(MAIN).as_posix()]:
            raise AssertionError('Curriculum changed after its audit.')
        seen = set()
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                view = json.loads(line)
                if name == 'formal_training_all_C_content':
                    support = expression_from_dict(view['final_answer']['expression']).support
                    labels.add(support)
                    components.update(support)
                for model in walk_expressions(view):
                    if any(len(s) > 3 for s in model.support):
                        raise AssertionError('A fourth-order structure occurs in teaching material.')
                    seen.add(model.support)
        stages.append((name, seen))
    result = summarize_exposure(components, labels, stages)
    if result['cumulative_exposure'][-1]['eligible_unseen_pairs'] != audit['reserved_tests']['unexposed_combinations_available']:
        raise AssertionError('Exposure audit disagrees with the frozen reservation.')
    result.update(source_sha256=hashes, reserved_test_answers_opened=False,
                  formal_distinct_training_answers=len(labels),
                  reserved_combination_worlds=audit['reserved_tests']['counts']['combination'],
                  reserved_distinct_combinations=audit['reserved_tests']['distinct_reserved_combinations'],
                  caveat='Cumulative order attributes first exposure, not causal responsibility for leakage.')
    output = Path('results/publication_exposure_audit')
    atomic_json(output/'audit.json', result)
    rows = result['cumulative_exposure']
    names = ['只排除正式训练的最终答案', '再排除正式训练的全部 C 内容', '再排除正式验证的全部 C 内容',
             '再排除旧训练教材', '再排除旧验证教材', '再排除覆盖诊断训练教材', '再排除覆盖诊断验证教材']
    lines = ['# 结构曝光审计：答案之外的假设也算见过', '',
             f'正式训练答案含 {len(components)} 种组成项，共可组成 {result["possible_distinct_pairs"]} 种不同的二项机制。', '',
             '| 逐步扩大排除范围 | 仍未出现的二项机制数 | 本步新排除 |', '|---|---:|---:|']
    lines += [f'| {name} | {row["eligible_unseen_pairs"]} | {row["newly_exposed_pairs"]} |' for name, row in zip(names, rows)]
    lines += ['', '以上是累积排除；交换来源顺序会改变“本步新排除”数量，但不会改变最终剩余数量。', '',
              '只检查最终正确标签，会把教材中已明确展示过的错误候选误称为未见机制。',
              '这个审计只读取已有教学导出文件和保留集摘要，没有打开保留测试的答案或运行测试。', '',
              f'冻结的组合测试有 {result["reserved_combination_worlds"]} 个参数世界，只有 '
              f'{result["reserved_distinct_combinations"]} 个独立结构。统计必须保留结构聚类，不能当成50个独立新机制。',
              '四体测试的50种结构均未在教学内容中出现。组合层的推断范围较窄，不能用重复参数掩盖这一限制。', '']
    (output/'REPORT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
