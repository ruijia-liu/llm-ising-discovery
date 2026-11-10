"""Rebuild calibration, legacy, coverage and formal teaching in a fresh work tree.

Only source files are copied into the child working directory. Original exports
are read by the parent for comparison, never supplied to the builders. No GPU
model is loaded and no reserved student world is measured or scored.
"""
import argparse
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

from build_publication_curriculum import atomic_json, read, sha
from reproduce_publication_curriculum import verify_rebuild


ROOT = Path(__file__).resolve().parent
COPY_SOURCES = ('build_research_curriculum.py', 'build_publication_curriculum.py',
    'run_structure_discovery.py', 'run_structure_discovery_noise.py', 'discovery/__init__.py',
    'discovery/lab.py', 'discovery/physics.py', 'discovery/search.py', 'discovery/teaching.py',
    'discovery/noise_evaluation.py', 'discovery/student_v2.py', 'discovery/student_format.py',
    'PUBLICATION_DATA_PROTOCOL.md')
REFERENCE_PATHS = {'legacy': 'data/research_curriculum_v1',
                   'coverage': 'data/research_curriculum_coverage',
                   'formal': 'data/research_curriculum_publication_v1'}
EXPORTS = tuple(f'{kind}/{split}/{group}.jsonl' for kind in ('views', 'messages')
                for split in ('train', 'validation') for group in 'ABC')
CALIBRATION = 'results/structure_discovery_noise/calibration.json'


def compare_legacy(original, generated):
    """Only timing and recorded source versions may differ, never audit outcomes."""
    a = {k: v for k, v in original.items() if k not in ('config', 'elapsed_seconds')}
    b = {k: v for k, v in generated.items() if k not in ('config', 'elapsed_seconds')}
    if a != b:
        raise ValueError('Legacy audit or outcome summaries differ.')
    a = {k: v for k, v in original['config'].items() if k != 'source_sha256'}
    b = {k: v for k, v in generated['config'].items() if k != 'source_sha256'}
    if a != b:
        raise ValueError('Legacy seeds, calibration or protocol settings differ.')
    return {name: {'original': original['config']['source_sha256'].get(name), 'current': checksum}
            for name, checksum in generated['config']['source_sha256'].items()
            if original['config']['source_sha256'].get(name) != checksum}


def require_identical_file(original, generated, expected=None):
    original_hash, generated_hash = sha(original), sha(generated)
    if original_hash != generated_hash or (expected is not None and original_hash != expected):
        raise ValueError('Reconstructed file differs: '+str(generated))
    return original_hash


def run(output):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT) or output == ROOT:
        raise ValueError('Use a new reproduction directory inside this project.')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserve existing reproduction artifacts, including failures.')
    prefix = Path(sys.prefix).resolve()
    distributions = list(importlib.metadata.distributions())
    names = {d.metadata['Name'].lower().replace('_', '-') for d in distributions}
    if (sys.prefix == sys.base_prefix or np.__version__ != '2.3.5'
            or names-{'numpy', 'pip', 'setuptools', 'wheel'}
            or not all(Path(d.locate_file('')).resolve().is_relative_to(prefix) for d in distributions)
            or not Path(np.__file__).resolve().is_relative_to(prefix)):
        raise ValueError('Use the isolated minimal NumPy 2.3.5 environment.')
    references = {stage: read(ROOT/path/'audit_report.json') for stage, path in REFERENCE_PATHS.items()}
    if references['legacy']['audit_counts']['paired_episodes_verified'] != 144:
        raise ValueError('Missing the original complete legacy curriculum.')
    for stage, expected_count in (('coverage', 96), ('formal', 720)):
        audit = references[stage]
        if not audit['completed'] or len(audit['episode_sha256']) != expected_count:
            raise ValueError('Missing a complete curriculum reference: '+stage)
        for name, checksum in audit['config']['source_sha256'].items():
            if sha(ROOT/REFERENCE_PATHS[stage]/'sources'/name) != checksum:
                raise ValueError('Original curriculum source snapshot changed.')
    # The coverage snapshot preserves all five source versions recorded by legacy.
    for name, checksum in references['legacy']['config']['source_sha256'].items():
        if sha(ROOT/REFERENCE_PATHS['coverage']/'sources'/name) != checksum:
            raise ValueError('Legacy source provenance no longer matches the preserved coverage snapshot.')
    source_hashes = {p: sha(ROOT/p) for p in (*COPY_SOURCES, 'reproduce_curriculum_chain.py',
                                            'reproduce_publication_curriculum.py')}
    config = {'scope': 'calibration_and_complete_curriculum_dependency_chain_from_source',
        'packages': {d.metadata['Name']: d.version for d in distributions},
        'python': sys.version, 'source_sha256': source_hashes,
        'reference_audit_sha256': {stage: sha(ROOT/path/'audit_report.json') for stage, path in REFERENCE_PATHS.items()},
        'reference_calibration_sha256': sha(ROOT/CALIBRATION), 'copied_files': list(COPY_SOURCES),
        'preexisting_data_copied_to_child_workspace': False,
        'formal_tests_measured_or_scored': False, 'gpu_models_loaded': False,
        'note': 'Source isolation is for reproducibility; it is not an adversarial filesystem sandbox.'}
    atomic_json(output/'config.json', config)
    workspace = output/'workspace'
    for name in COPY_SOURCES:
        destination = workspace/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())
        if sha(destination) != source_hashes[name]:
            raise ValueError('A source copy differs from its recorded version.')
    if (workspace/'data').exists() or (workspace/'results').exists():
        raise ValueError('The builder workspace must initially contain source only.')
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONNOUSERSITE='1')
    commands, stages = [], []
    started = time.perf_counter()

    def execute(stage, arguments):
        before = time.perf_counter()
        log_path = output/(stage+'.log')
        with log_path.open('w', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, '-u', *arguments], cwd=workspace,
                env=env, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            atomic_json(output/'execution.json', {'status': 'running', 'stage': stage,
                        'pid': child.pid, 'started_unix': time.time(), 'completed_stages': stages})
            print(f'{stage} started, child PID {child.pid}.', flush=True)
            exit_code = child.wait()
        commands.append({'stage': stage, 'arguments': arguments, 'working_directory': 'workspace',
                         'exit_code': exit_code, 'elapsed_seconds': time.perf_counter()-before,
                         'log_sha256': sha(log_path)})
        atomic_json(output/'commands.json', {'commands': commands})
        if exit_code:
            raise RuntimeError(f'{stage} failed with code {exit_code}; preserve its log.')

    try:
        execute('calibration', ['run_structure_discovery_noise.py', 'calibrate', '--seed', '91001',
                '--worlds-per-family', '10', '--eval-count', '512', '--output', CALIBRATION])
        checksum = require_identical_file(ROOT/CALIBRATION, workspace/CALIBRATION,
                                          config['reference_calibration_sha256'])
        stages.append({'stage': 'calibration', 'entire_file_identical': True, 'sha256': checksum,
                       'worlds': 60, 'evaluation_count_per_distribution': 512})
        for stage, arguments in (
            ('legacy', ['build_research_curriculum.py', '--seed', '93001', '--holdout-seed', '94001',
                        '--output', REFERENCE_PATHS['legacy']]),
            ('coverage', ['build_publication_curriculum.py', '--stage', 'coverage', '--output', REFERENCE_PATHS['coverage']]),
            ('formal', ['build_publication_curriculum.py', '--stage', 'main', '--output', REFERENCE_PATHS['formal']])):
            execute(stage, arguments)
            reference, target = ROOT/REFERENCE_PATHS[stage], workspace/REFERENCE_PATHS[stage]
            original, generated = references[stage], read(target/'audit_report.json')
            changes = compare_legacy(original, generated) if stage == 'legacy' else verify_rebuild(original, generated)
            hashes = {}
            for name in EXPORTS:
                expected = None if stage == 'legacy' else original['export_sha256'][str(Path(name))]
                hashes[name] = require_identical_file(reference/name, target/name, expected)
            hashes['private/worlds.json'] = require_identical_file(reference/'private/worlds.json', target/'private/worlds.json')
            if stage == 'legacy':
                for name in ('private/teacher_traces.jsonl', 'private/grader_outcomes.json'):
                    hashes[name] = require_identical_file(reference/name, target/name)
                count = sum(1 for _ in (target/'private/teacher_traces.jsonl').open(encoding='utf-8'))
                if count != 144:
                    raise ValueError('Legacy trace count differs.')
            else:
                for name, checksum in original['episode_sha256'].items():
                    relative = 'private/episodes/'+name
                    hashes[relative] = require_identical_file(reference/relative, target/relative, checksum)
                count = len(original['episode_sha256'])
                if stage == 'coverage':
                    hashes['coverage_groups.json'] = require_identical_file(reference/'coverage_groups.json', target/'coverage_groups.json')
                else:
                    # Compare only to the recorded digest, never parse original heldout answers.
                    if sha(target/'private/reserved_test_worlds.json') != original['reserved_tests']['private_file_sha256']:
                        raise ValueError('Formal reserved-definition digest differs from the saved audit.')
            row = {'stage': stage, 'episodes_identical': count, 'exports_identical': len(EXPORTS),
                   'generated_audit_sha256': sha(target/'audit_report.json'), 'source_differences': changes,
                   'actual_file_sha256': hashes,
                   'audit_counts': generated['audit_counts'] if stage == 'legacy' else generated['new_audit_counts']}
            stages.append(row)
            atomic_json(output/'comparisons.json', {'completed_stages': stages})
            print(f'{stage} verified: {count} full episodes and {len(EXPORTS)} exports identical.', flush=True)
        if any(sha(ROOT/name) != h for name, h in source_hashes.items()):
            raise ValueError('Original source changed during the reproduction.')
        if any(sha(workspace/name) != source_hashes[name] for name in COPY_SOURCES):
            raise ValueError('Copied builder source changed during the reproduction.')
        if any(sha(ROOT/REFERENCE_PATHS[stage]/'audit_report.json') != h
               for stage, h in config['reference_audit_sha256'].items()):
            raise ValueError('A reference audit changed during comparison.')
    except BaseException as error:
        atomic_json(output/'failure.json', {'completed': False, 'error_type': type(error).__name__,
                    'error': str(error), 'elapsed_seconds': time.perf_counter()-started, 'commands': commands,
                    'completed_stages': stages})
        raise
    result = {'completed': True, 'scope': config['scope'], 'stages': stages, 'commands': commands,
              'total_teacher_episodes_identical': sum(s.get('episodes_identical', 0) for s in stages),
              'total_curriculum_exports_identical': sum(s.get('exports_identical', 0) for s in stages),
              'calibration_entire_file_identical': True, 'historical_development_sources_regenerated': True,
              'preexisting_data_copied_to_child_workspace': False, 'formal_reserved_definition_digest_identical': True,
              'original_reserved_world_files_parsed': False, 'formal_tests_measured_or_scored': False,
              'gpu_models_loaded': False, 'full_student_study_reproduced': False,
              'elapsed_seconds': time.perf_counter()-started,
              'limits': ['Same machine and operating system, independently installed numerical environment.',
                         'Original files are comparison references only; training/model outputs are not reconstructed.',
                         'Human-edited narrative reports and the historical legacy reserved definitions are not byte comparisons.',
                         'CPU reconstruction overlaps live GPU training; this is not a controlled speed benchmark.']}
    atomic_json(output/'result.json', result)
    atomic_json(output/'execution.json', {'status': 'completed', 'elapsed_seconds': result['elapsed_seconds']})
    (output/'REPORT_ZH.md').write_text(
        '# 校准与全部教材依赖链重建\n\n'
        '独立NumPy环境在新的工作目录仅从复制的源码启动，先生成校准文件，再生成早期教材、'
        '覆盖教材及正式教材。没有把原有教材或校准数值复制给子生成程序。\n\n'
        '60世界的完整校准文件与原件逐字节一致。144条早期、96条新增覆盖、720条正式教师轨迹，'
        '共960条完整轨迹及36个教材/消息导出文件与原件逐字节一致。'
        '覆盖阶段额外复用的36条无噪声历史轨迹不重复计作新生成轨迹。\n\n'
        '全部原教材用于父进程的结果核对，不是重建输入。源码的已知版本差异逐项保存；'
        '正式保留定义摘要一致，没有解析原保留世界文件、测量保留世界或评价学生。\n\n'
        '本次仍是同机数值和教材复现。没有重训GPU模型，也没有复现正式学生结果；'
        '不能据此宣称整项研究已经复现或达到投稿条件。详见result.json、各阶段日志及workspace。\n',
        encoding='utf-8')
    print(output/'REPORT_ZH.md', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
