"""Audited coverage diagnostic and full publication-sized curriculum.

No student training or reserved-test oracle calls. Each completed episode is an
atomic resumable record; labels are attached only after the teacher finishes.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from build_research_curriculum import (FAMILIES, audit_messages, reserve_holdouts,
                                      stream, verify_observations, walk_expressions)
from discovery.lab import Laboratory
from discovery.physics import Expression, Grammar, Term, canonical_structure, random_queries
from discovery.student_v2 import load_views
from discovery.teaching import (audit_teacher, audit_views, export_views, expression_from_dict,
                                messages_for_view, packed, run_teacher)
from run_structure_discovery import compile_queries, evaluate, evaluation_queries
from discovery.noise_evaluation import evaluate_noisy

LEGACY = Path('data/research_curriculum_v1')
COVERAGE = Path('data/research_curriculum_coverage')
MAIN = Path('data/research_curriculum_publication')
SOURCES = ('build_publication_curriculum.py', 'build_research_curriculum.py', 'discovery/teaching.py',
           'discovery/search.py', 'discovery/physics.py', 'discovery/lab.py', 'PUBLICATION_DATA_PROTOCOL.md')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def seal_episode(record):
    return {**record, 'audited_record_sha256': hashlib.sha256(packed(record).encode('utf-8')).hexdigest()}


def verify_episode(record):
    payload = {k: v for k, v in record.items() if k != 'audited_record_sha256'}
    expected = hashlib.sha256(packed(payload).encode('utf-8')).hexdigest()
    if record.get('audited_record_sha256') != expected:
        raise AssertionError('Completed episode changed after its audit.')
    return record


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def parameterized(support, seed, index):
    rng = np.random.default_rng(stream(seed, index, 11))
    coefficients = rng.uniform(0.3, 0.8, len(support))*rng.choice([-1, 1], len(support))
    return Expression(tuple(Term(points, float(k)) for points, k in zip(support, coefficients)))


def main_worlds(seed=97301, train_per_family=50, validation_per_family=10):
    grammar = Grammar(max_order=3)
    rng = np.random.default_rng(stream(seed, 0, 701))
    field = canonical_structure(((0, 0),))
    directions = [canonical_structure(((0, 0), (0, 1))), canonical_structure(((0, 0), (1, 0)))]
    distance = [s for s in grammar.structures if len(s) == 2 and max(min(v, 16-v) for v in s[1]) == 2]
    triples = [s for s in grammar.structures if len(s) == 3]
    distance = [distance[int(i)] for i in rng.permutation(len(distance))]
    triples = [triples[int(i)] for i in rng.permutation(len(triples))]
    templates = {'field': [(field,)], 'direction': [(s,) for s in directions],
                 'distance_two': [(s,) for s in distance], 'three_body': [(s,) for s in triples],
                 'mixed': [tuple(sorted((field, s))) for s in triples], 'null': [()]}
    worlds = []
    for split, count in (('train', train_per_family), ('validation', validation_per_family)):
        for family in FAMILIES:
            available = templates[family][:train_per_family]
            for j in range(count):
                index = 2000+len(worlds)
                support = available[j % len(available)]
                truth = parameterized(support, seed, index)
                worlds.append({'world_index': index, 'world_id': f'w{index:04d}', 'split': split,
                               'family': family, 'truth': truth.to_dict()})
    train_supports = {expression_from_dict(w['truth']).support for w in worlds if w['split'] == 'train'}
    if any(expression_from_dict(w['truth']).support not in train_supports for w in worlds if w['split'] == 'validation'):
        raise AssertionError('Parameter validation contains an untrained structure.')
    return worlds


def coverage_worlds(seed=97101):
    worlds = []
    anchors = load_views('train')
    for variant in range(1, 5):
        for anchor, view in enumerate(anchors):
            index = 1000+(variant-1)*24+anchor
            support = expression_from_dict(view['final_answer']['expression']).support
            worlds.append({'world_index': index, 'world_id': f'w{index:04d}', 'split': 'train',
                           'family': FAMILIES[anchor//4], 'anchor_id': view['episode_id'].split('-')[0],
                           'variant': variant, 'truth': parameterized(support, seed, index).to_dict()})
    return worlds


def make_episode(world, sigma, seed, thresholds):
    index = world['world_index']
    truth = expression_from_dict(world['truth'])
    initial = random_queries(np.random.default_rng(stream(seed, index, 101)), 8)
    lab = Laboratory(truth.delta, noise_sigma=sigma, noise_seed=stream(seed, index, 404))
    # No world label, truth, family or future observations reach the teacher.
    trace = run_teacher(lab, initial, query_seed=stream(seed, index, 202))
    audit = audit_teacher(trace, query_seed=stream(seed, index, 202))
    audit['observations_replayed_by_total_energy'] = verify_observations(trace, truth, stream(seed, index, 404))
    episode_id = f'{world["world_id"]}-sigma{sigma:g}'
    views = export_views(trace, truth, episode_id)
    pairing = audit_views(views)
    messages = {g: messages_for_view(views[g], g) for g in 'ABC'}
    audit_messages(views, messages)
    grammar = Grammar()
    features = [compile_queries(grammar, q) for q in
                evaluation_queries(np.random.default_rng(stream(seed, index, 303)), 256)]
    estimate = expression_from_dict(trace['teacher_final_before_grading'])
    score = (evaluate(grammar, truth, estimate, features) if sigma == 0 else
             evaluate_noisy(grammar, truth, estimate, features, thresholds))
    return {'episode_id': episode_id, 'world_id': world['world_id'], 'split': world['split'],
            'family': world['family'], 'noise_sigma': sigma, 'trace': trace, 'lab_events': lab.events,
            'audit': audit, 'pairing_sha256': pairing, 'grader_outcome_after_episode': score}


def audit_exposure(value, shown):
    for model in walk_expressions(value):
        if any(len(s) > 3 for s in model.support):
            raise AssertionError('Four-body expression in teaching material.')
        shown.add(model.support)


def exposure_from(directory, shown):
    for split in ('train', 'validation'):
        with (Path(directory)/'views'/split/'C.jsonl').open(encoding='utf-8') as handle:
            for line in handle:
                audit_exposure(json.loads(line), shown)


def build(stage, output=None, resume=False):
    if stage not in ('coverage', 'main'):
        raise ValueError('Unknown stage.')
    output = Path(output or (COVERAGE if stage == 'coverage' else MAIN))
    seed = 97101 if stage == 'coverage' else 97301
    sigmas = (0.0,) if stage == 'coverage' else (0.0, 1.0)
    if (output/'audit_report.json').exists():
        raise ValueError('Preserve the already completed curriculum.')
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError('Use --resume for this exact incomplete build.')
    if stage == 'main' and not (COVERAGE/'audit_report.json').exists():
        raise ValueError('Build coverage records first so holdouts exclude all development exposure.')
    worlds = coverage_worlds(seed) if stage == 'coverage' else main_worlds(seed)
    calibration = read('results/structure_discovery_noise/calibration.json')
    dependencies = [LEGACY/'views'/s/(g+'.jsonl') for s in ('train', 'validation') for g in 'ABC']
    if stage == 'main':
        dependencies += [COVERAGE/'views'/s/'C.jsonl' for s in ('train', 'validation')]
    config = {'stage': stage, 'seed': seed, 'holdout_seed': 97401, 'noise_sigmas': sigmas,
              'new_worlds': len(worlds), 'training_worlds': 120 if stage == 'coverage' else 300,
              'validation_worlds': 12 if stage == 'coverage' else 60, 'failure_filtering': False,
              'teacher_max_order': 3, 'student_max_order': 4, 'queries_per_episode': 32,
              'source_sha256': {s: sha(s) for s in SOURCES},
              'dependency_sha256': {str(p): sha(p) for p in dependencies},
              'noise_calibration_sha256': sha('results/structure_discovery_noise/calibration.json')}
    # Round trip tuples so resume compares the exact serialized configuration.
    config = json.loads(json.dumps(config))
    if resume:
        if config != read(output/'protocol.json'):
            raise ValueError('Source/configuration/data changed since the build began.')
    else:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output/'protocol.json', config)
        atomic_json(output/'private/worlds.json', worlds)
        for source in SOURCES:
            snapshot = output/'sources'/source
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(Path(source).read_bytes())
    start = time.perf_counter()
    for index, world in enumerate(worlds):
        for sigma in sigmas:
            path = output/'private/episodes'/f'{world["world_id"]}-sigma{sigma:g}.json'
            if not path.exists():
                atomic_json(path, seal_episode(make_episode(world, sigma, seed, calibration['thresholds'])))
            else:
                saved = verify_episode(read(path))
                if (saved['world_id'], saved['noise_sigma'], saved['split']) != (world['world_id'], sigma, world['split']):
                    raise AssertionError('Existing episode does not match this build.')
        if (index+1) % 12 == 0:
            print(f'{stage}: {index+1}/{len(worlds)} new worlds audited; seconds={time.perf_counter()-start:.1f}', flush=True)
    # Rebuild derived exports from complete immutable episode records on resume.
    entries = {(split, g): [] for split in ('train', 'validation') for g in 'ABC'}
    if stage == 'coverage':
        for split, g in entries:
            entries[split, g].extend(json.loads(line) for line in (LEGACY/'views'/split/(g+'.jsonl')).read_text(encoding='utf-8').splitlines()
                                      if json.loads(line)['noise_sigma'] == 0.0)
    shown, audit_counts, outcomes, episode_hashes = set(), Counter(), [], {}
    for world in worlds:
        for sigma in sigmas:
            path = output/'private/episodes'/f'{world["world_id"]}-sigma{sigma:g}.json'
            saved = verify_episode(read(path))
            views = export_views(saved['trace'], expression_from_dict(world['truth']), saved['episode_id'])
            if audit_views(views) != saved['pairing_sha256']:
                raise AssertionError('Paired evidence changed after audit.')
            audit_exposure(views['C'], shown)
            audit_counts.update(saved['audit'])
            outcomes.append({k: saved[k] for k in ('episode_id', 'split', 'family', 'noise_sigma', 'grader_outcome_after_episode')})
            for g in 'ABC':
                entries[world['split'], g].append(views[g])
            episode_hashes[path.name] = sha(path)
    export_hashes = {}
    for (split, g), views in entries.items():
        for kind in ('views', 'messages'):
            path = output/kind/split/(g+'.jsonl')
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name+'.tmp')
            with temporary.open('w', encoding='utf-8') as handle:
                for view in views:
                    handle.write(packed(view if kind == 'views' else messages_for_view(view, g))+'\n')
            temporary.replace(path)
            export_hashes[str(path.relative_to(output))] = sha(path)
    if stage == 'coverage':
        mapping = {v['episode_id'].split('-')[0]: [v['episode_id']] for v in load_views('train')}
        for world in worlds:
            mapping[world['anchor_id']].append(world['world_id']+'-sigma0')
        atomic_json(output/'coverage_groups.json', mapping)
    exposure_from(LEGACY, shown)
    exposure_from(COVERAGE if stage == 'main' else output, shown)
    holdout_summary = None
    if stage == 'main':
        reserved = reserve_holdouts(worlds, shown, seed=97401, per_level=50)
        atomic_json(output/'private/reserved_test_worlds.json', reserved)
        holdout_summary = {k: v for k, v in reserved.items() if k != 'worlds'}
        holdout_summary['counts'] = dict(Counter(w['level'] for w in reserved['worlds']))
        holdout_summary['private_file_sha256'] = sha(output/'private/reserved_test_worlds.json')
    if any(sha(p) != expected for p, expected in config['dependency_sha256'].items()):
        raise AssertionError('A source curriculum changed.')
    counts = {split: {g: len(entries[split, g]) for g in 'ABC'} for split in ('train', 'validation')}
    teacher_summary = []
    for split in ('train', 'validation'):
        for sigma in sigmas:
            rows = [r for r in outcomes if r['split'] == split and r['noise_sigma'] == sigma]
            teacher_summary.append({'split': split, 'sigma': sigma, 'new_episodes': len(rows),
                                    'full_recoveries': sum(r['grader_outcome_after_episode']['full_recovery'] for r in rows),
                                    'false_discoveries': sum(r['grader_outcome_after_episode']['false_discovery'] for r in rows)})
    report = {'stage': stage, 'completed': True, 'config': config, 'new_audit_counts': dict(audit_counts),
              'episode_counts': counts, 'teacher_summary_new_episodes': teacher_summary,
              'reserved_tests': holdout_summary, 'unique_exposed_supports_including_development': len(shown),
              'export_sha256': export_hashes, 'episode_sha256': episode_hashes,
              'elapsed_seconds_this_invocation': time.perf_counter()-start,
              'tests_measured_or_scored': False, 'student_training_started_by_builder': False}
    atomic_json(output/'audit_report.json', report)
    lines = [f'# {stage} 教材扩充与审计', '',
             f'训练世界 {config["training_worlds"]}，开发验证世界 {config["validation_worlds"]}；噪声条件 {list(sigmas)}。', '',
             f'新生成并完整审计 {len(outcomes)} 条研究轨迹。没有过滤教师失败，没有运行学生训练或保留测试。', '',
             '| 集合 | 噪声 | 新轨迹 | 教师完整恢复 | 空机制误报数 |', '|---|---:|---:|---:|---:|']
    for r in teacher_summary:
        lines.append(f'| {r["split"]} | {r["sigma"]} | {r["new_episodes"]} | {r["full_recoveries"]} | {r["false_discoveries"]} |')
    if stage == 'coverage':
        lines += ['', '旧24个训练和12个开发验证世界原样保留。上表只统计新增96个世界，不能当完整数据集成绩。']
    else:
        lines += ['', f'保留测试每类50个，共200个；可用未曝光组合 {holdout_summary["unexposed_combinations_available"]} 种，'
                  f'实际保留不同组合 {holdout_summary["distinct_reserved_combinations"]} 种。',
                  '组合排除覆盖新教材、旧开发教材和覆盖诊断的所有C表达式；四体只存在于保留测试。',
                  '两个噪声条件是相同世界的两次独立预算实验，每次32次测量；不能合并为64次给学生。']
    lines += ['', '来源、配置、逐轨迹哈希与全部核验计数见 audit_report.json；真值和轨迹仅在 private/。', '']
    (output/'REPORT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Completed {stage}: {counts}; holdout summary={holdout_summary}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('coverage', 'main'), required=True)
    parser.add_argument('--output')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    build(args.stage, args.output, args.resume)
