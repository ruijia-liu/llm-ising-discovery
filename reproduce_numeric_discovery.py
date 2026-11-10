"""Reproduce numerical discovery checks in an independently installed environment.

This deliberately does not claim to reproduce LLM training or heldout results.
No GPU model is loaded and no reserved test world is opened.
"""
import argparse
import importlib.metadata
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

from build_publication_curriculum import atomic_json, read, sha
from discovery.evaluation_results import load_completed_evaluation


TESTS = ('test_structure_discovery', 'test_structure_discovery_noise', 'test_student_laboratory',
         'test_active_student', 'test_episode_journal', 'test_evaluation_results',
         'test_answer_diagnostics', 'test_secondary_metrics', 'test_seed_summaries',
         'test_evaluation_queue', 'test_ablation_analysis', 'test_training_effort', 'test_behavior_cases',
         'test_curriculum_chain_reproduction', 'test_training_input_reproduction', 'test_statistical_sensitivity',
         'test_uncertainty_supplement')


def run(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserve a prior reproduction, including failed attempts.')
    prefix = Path(sys.prefix).resolve()
    packages = list(importlib.metadata.distributions())
    names = {d.metadata['Name'].lower().replace('_', '-') for d in packages}
    if (sys.prefix == sys.base_prefix or np.__version__ != '2.3.5'
            or names-{'numpy', 'pip', 'setuptools', 'wheel'}
            or not all(Path(d.locate_file('')).resolve().is_relative_to(prefix) for d in packages)
            or not Path(np.__file__).resolve().is_relative_to(prefix)):
        raise ValueError('Use a clean, isolated environment with only pinned NumPy and packaging tools.')
    sources = ('reproduce_numeric_discovery.py', 'capture_discovery_environment.py', 'evaluate_mainline.py',
               'discovery/evaluation_results.py', 'discovery/episode_journal.py', 'discovery/physics.py',
               'discovery/lab.py', 'discovery/search.py', 'requirements-discovery-numeric-lock.txt',
               'discovery/answer_diagnostics.py', 'discovery/secondary_metrics.py', 'discovery/seed_summaries.py',
               'report_evaluation_metrics.py', 'run_evaluation_queue.py', 'discovery/run_lock.py',
               'analyze_prediction_ablation.py', 'report_mainline_training_effort.py',
               'report_behavior_cases.py', 'discovery/behavior_cases.py', 'BEHAVIOR_CASE_PROTOCOL.md',
               'reproduce_curriculum_chain.py', 'reproduce_publication_curriculum.py',
               'reproduce_training_inputs.py', 'audit_statistical_sensitivity.py',
               'discovery/paired_statistics.py', 'STATISTICAL_SENSITIVITY_PROTOCOL.md',
               'discovery/uncertainty_supplement.py', 'audit_uncertainty_supplement.py',
               'analyze_uncertainty_supplement.py', 'UNCERTAINTY_SUPPLEMENT_PROTOCOL.md')+tuple(name+'.py' for name in TESTS)
    config = {'scope': 'independent-numerical-environment-only', 'numpy': np.__version__,
              'packages': {d.metadata['Name']: d.version for d in packages},
              'source_sha256': {p: sha(p) for p in sources}, 'formal_tests_used': False, 'gpu_models_used': False}
    atomic_json(output/'config.json', config)
    commands = []

    def execute(name, args):
        start = time.perf_counter()
        with (output/(name+'.log')).open('w', encoding='utf-8') as log:
            child = subprocess.run([sys.executable, *args], stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        row = {'name': name, 'arguments': args, 'exit_code': child.returncode,
               'seconds': time.perf_counter()-start, 'log_sha256': sha(output/(name+'.log'))}
        commands.append(row)
        atomic_json(output/'execution_progress.json', {'commands': commands})
        if child.returncode:
            atomic_json(output/'failure.json', {'failed_step': name, 'completed': False, 'commands': commands})
            raise RuntimeError('Reproduction failed; preserve logs before any correction.')
        print(f'Completed {name}, seconds={row["seconds"]:.1f}', flush=True)

    execute('environment', ['capture_discovery_environment.py', '--output', str(output/'environment.json')])
    execute('tests', ['-m', 'unittest', '-v', *TESTS])
    test_log = (output/'tests.log').read_text(encoding='utf-8')
    matches = re.findall(r'Ran (\d+) tests? in ', test_log)
    if not matches or '\nOK' not in test_log:
        raise AssertionError('The test command did not produce a complete passing unittest summary.')
    comparisons = []
    for mode in ('fixed', 'active'):
        target = output/('sparse_'+mode)
        execute('sparse_'+mode, ['-u', 'evaluate_mainline.py', '--scope', 'development', '--method', 'sparse_'+mode,
                               '--limit', '1', '--output', str(target)])
        reference_root = Path(f'results/mainline_development_sparse_{mode}_v4_diagnostics')
        reference = load_completed_evaluation(reference_root, scope='development', method='sparse_'+mode)
        reproduced = load_completed_evaluation(target, scope='development', method='sparse_'+mode)
        prior = {(r['world_id'], r['sigma']): r for r in reference['result']['rows']}
        if {(r['world_id'], r['sigma']) for r in reproduced['result']['rows']} != set(prior):
            raise AssertionError('Reproduced evaluation covers different conditions.')
        for row in reproduced['result']['rows']:
            name = row['world_id']+f"-sigma{row['sigma']}.json"
            a, b = read(reference_root/'episodes'/name), read(target/'episodes'/name)
            if a['episode'] != b['episode'] or row['score'] != prior[row['world_id'], row['sigma']]['score']:
                raise AssertionError('Clean-environment result differs from the reference.')
        comparisons.append({'mode': mode, 'conditions': len(prior), 'scores_and_full_episodes_identical': True,
                            'reference': reference['provenance'], 'reproduction': reproduced['provenance']})
    if any(sha(p) != h for p, h in config['source_sha256'].items()):
        raise AssertionError('Source changed during clean-environment reproduction.')
    atomic_json(output/'result.json', {'completed': True, 'scope': config['scope'], 'tests_passed': int(matches[-1]),
                'commands': commands, 'comparisons': comparisons, 'formal_tests_used': False, 'gpu_models_used': False,
                'full_training_and_inference_reproduction_completed': False})
    (output/'REPORT_ZH.md').write_text('\n'.join([
        '# 独立数值环境复现', '',
        f"独立新环境只有NumPy及安装工具；未继承主训练环境的包。{matches[-1]}项相关检查全部通过。",
        '固定/主动搜索各12个开发条件，完整轨迹、拟合结果、分数与原环境逐项相同。', '',
        '这是数值实验室、恢复与汇总管线的干净环境复现。没有加载GPU模型，没有运行保留测试；',
        '**并未完成语言模型训练/推理或整篇论文的独立复现。** 主训练环境未安装、升级或删除任何包。', '',
        '详细版本、源码哈希、执行日志和逐条件对照在同目录JSON与log文件中。', '']), encoding='utf-8')
    print(output/'REPORT_ZH.md', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/reproducibility_numeric_v1')
    run(parser.parse_args().output)
