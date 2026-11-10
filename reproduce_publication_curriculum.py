"""Regenerate all formal teaching episodes in the isolated numerical environment.

Historical development exports remain explicit inputs to the exposure exclusion.
No student is trained and no reserved world is measured, scored, or parsed from
the original reserved file. Only its previously recorded digest is compared.
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

from build_publication_curriculum import atomic_json, read, sha, SOURCES


REFERENCE = Path('data/research_curriculum_publication_v1')


def verify_rebuild(reference, generated):
    for field in ('episode_counts', 'new_audit_counts', 'teacher_summary_new_episodes',
                  'reserved_tests', 'unique_exposed_supports_including_development',
                  'export_sha256', 'episode_sha256'):
        if reference[field] != generated[field]:
            raise ValueError('Curriculum reproduction differs in '+field)
    a = {k: v for k, v in reference['config'].items() if k != 'source_sha256'}
    b = {k: v for k, v in generated['config'].items() if k != 'source_sha256'}
    if a != b or generated['tests_measured_or_scored'] or generated['student_training_started_by_builder']:
        raise ValueError('Curriculum settings, inputs or reproduction scope changed.')
    return {name: {'original': reference['config']['source_sha256'][name], 'current': digest}
            for name, digest in generated['config']['source_sha256'].items()
            if reference['config']['source_sha256'].get(name) != digest}


def run(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserve all previous reproduction artifacts, including failed runs.')
    prefix = Path(sys.prefix).resolve()
    distributions = list(importlib.metadata.distributions())
    names = {d.metadata['Name'].lower().replace('_', '-') for d in distributions}
    if (sys.prefix == sys.base_prefix or np.__version__ != '2.3.5'
            or names-{'numpy', 'pip', 'setuptools', 'wheel'}
            or not all(Path(d.locate_file('')).resolve().is_relative_to(prefix) for d in distributions)
            or not Path(np.__file__).resolve().is_relative_to(prefix)):
        raise ValueError('Use the separately installed minimal NumPy 2.3.5 environment.')
    original = read(REFERENCE/'audit_report.json')
    if not original['completed'] or len(original['episode_sha256']) != 720:
        raise ValueError('Need the completed 720-episode formal curriculum reference.')
    for name, digest in original['config']['source_sha256'].items():
        if sha(REFERENCE/'sources'/name) != digest:
            raise ValueError('An original curriculum source snapshot changed.')
    for name, digest in original['config']['dependency_sha256'].items():
        if sha(name) != digest:
            raise ValueError('Historical development dependencies changed.')
    config = {'scope': 'all_720_formal_teaching_episodes_from_source_with_preserved_historical_development_inputs',
        'numpy': np.__version__, 'packages': {d.metadata['Name']: d.version for d in distributions},
        'reference_audit_sha256': sha(REFERENCE/'audit_report.json'),
        'source_sha256': {name: sha(name) for name in (*SOURCES, 'reproduce_publication_curriculum.py')},
        'dependency_sha256': original['config']['dependency_sha256'],
        'formal_tests_measured_or_scored': False, 'gpu_models_loaded': False}
    atomic_json(output/'config.json', config)
    target = output/'curriculum'
    started = time.perf_counter()
    try:
        with (output/'build.log').open('w', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, '-u', 'build_publication_curriculum.py',
                '--stage', 'main', '--output', str(target)], stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            atomic_json(output/'execution.json', {'status': 'builder_started', 'pid': child.pid,
                                                'started_unix': time.time()})
            print(f'Curriculum reconstruction started; child PID {child.pid}.', flush=True)
            code = child.wait()
        if code:
            raise RuntimeError(f'Curriculum reconstruction exited with code {code}; inspect the preserved build.log.')
        generated = read(target/'audit_report.json')
        changes = verify_rebuild(original, generated)
        # Check actual bytes, not merely agreement between two manifest dictionaries.
        for name, digest in original['episode_sha256'].items():
            if sha(target/'private/episodes'/name) != digest or sha(REFERENCE/'private/episodes'/name) != digest:
                raise ValueError('A full teacher episode differs from its reference digest.')
        for name, digest in original['export_sha256'].items():
            if sha(target/name) != digest or sha(REFERENCE/name) != digest:
                raise ValueError('A teaching-view/message export differs from its reference digest.')
        if sha(target/'private/worlds.json') != sha(REFERENCE/'private/worlds.json'):
            raise ValueError('Regenerated training/development worlds differ.')
        # Never read the original reserved-world contents in this reproduction.
        if sha(target/'private/reserved_test_worlds.json') != original['reserved_tests']['private_file_sha256']:
            raise ValueError('Regenerated reserved-definition digest differs from the existing audit.')
        if any(sha(name) != digest for name, digest in config['source_sha256'].items()):
            raise ValueError('A source changed during reconstruction.')
    except BaseException as error:
        atomic_json(output/'failure.json', {'completed': False, 'error_type': type(error).__name__,
                    'error': str(error), 'elapsed_seconds': time.perf_counter()-started})
        raise
    result = {'completed': True, 'scope': config['scope'], 'episodes_identical': len(original['episode_sha256']),
        'exports_identical': len(original['export_sha256']), 'teacher_audit_counts': generated['new_audit_counts'],
        'source_differences_from_original': changes, 'reserved_definition_digest_identical': True,
        'formal_tests_measured_or_scored': False, 'gpu_models_loaded': False,
        'historical_development_sources_regenerated': False,
        'reference_audit_sha256': config['reference_audit_sha256'], 'generated_audit_sha256': sha(target/'audit_report.json'),
        'build_log_sha256': sha(output/'build.log'), 'elapsed_seconds': time.perf_counter()-started}
    atomic_json(output/'result.json', result)
    atomic_json(output/'execution.json', {'status': 'completed', 'elapsed_seconds': result['elapsed_seconds']})
    (output/'REPORT_ZH.md').write_text(
        '# 正式教材从源码重建\n\n'
        '在独立安装的最小NumPy环境中，从固定种子重新生成300训练、60开发验证世界，'
        '两种噪声共720条完整教师轨迹。720条轨迹及12个A/B/C教材和消息导出文件与原件逐字节一致。\n\n'
        '数值预测、因果决策、翻转能量和B/C证据配对重新审计，失败教师与误报完整保留。'
        '训练/开发世界定义相同，重新生成的保留世界定义摘要也与原审计一致；没有测量或评分保留世界。\n\n'
        '历史小规模开发教材和覆盖诊断仍作为机制曝光排除的固定输入，本次没有重新生成这两个历史来源。'
        '原生成代码与当前代码的哈希差异单列于result.json；不把字节一致说成两版代码完全相同。\n\n'
        '本次没有加载GPU模型或训练学生，不能替代完整模型训练、正式评估和论文复现。\n', encoding='utf-8')
    print(output/'REPORT_ZH.md', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
