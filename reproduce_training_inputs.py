"""Rebuild all SFT inputs from regenerated curriculum in a fresh model environment.

CPU/tokenizer only: no weights are copied, loaded, trained or evaluated. The
original training cache remains an immutable comparison reference.
"""
import argparse
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

from build_publication_curriculum import atomic_json, read, sha
from reproduce_curriculum_chain import COPY_SOURCES, EXPORTS


ROOT = Path(__file__).resolve().parent
CURRICULUM = Path('data/research_curriculum_publication_v1')
CACHE = Path('data/student_mainline_v1')
MODEL = Path('models/Qwen2.5-0.5B-Instruct')
EXTRA_SOURCES = ('prepare_mainline_training.py', 'discovery/student_mainline.py', 'discovery/student_v3.py',
                 'discovery/student_spins.py', 'discovery/student_laboratory.py', 'discovery/mainline_schedule.py',
                 'train_mainline_student.py', 'train_student_pilot.py',
                 'MAINLINE_STUDY_PROTOCOL.md', 'ACTIVE_STUDENT_PROTOCOL.md')
ENV_REFERENCE = Path('results/reproducibility_environment/training_environment.json')


def package_versions():
    return {d.metadata['Name'].lower().replace('_', '-'): d.version for d in importlib.metadata.distributions()}


def check_environment():
    prefix = Path(sys.prefix).resolve()
    reference = read(ROOT/ENV_REFERENCE)
    expected = {p['name'].lower().replace('_', '-'): p['version'] for p in reference['packages']}
    if prefix == (ROOT/'.venv-discovery').resolve() or sys.prefix == sys.base_prefix:
        raise ValueError('Use a separately installed model environment, not the live training environment.')
    if package_versions() != expected or not all(Path(d.locate_file('')).resolve().is_relative_to(prefix)
            for d in importlib.metadata.distributions()):
        raise ValueError('The independent environment must exactly match all pinned study package versions.')
    import torch
    import transformers
    import peft
    if torch.cuda.is_initialized():
        raise ValueError('This CPU-only reconstruction must not initialize CUDA.')
    if not all(Path(m.__file__).resolve().is_relative_to(prefix) for m in (np, torch, transformers, peft)):
        raise ValueError('An imported library came from outside the isolated environment.')
    value = (torch.tensor([1., 2.], device='cpu')*2).numpy()
    if not np.array_equal(value, [2., 4.]) or torch.cuda.is_initialized():
        raise ValueError('CPU torch/NumPy interoperability check failed.')
    checked = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True,
                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if checked.returncode:
        raise ValueError('Dependency verification failed: '+checked.stdout+checked.stderr)
    return {'python': sys.version, 'reference_sha256': sha(ROOT/ENV_REFERENCE), 'packages': expected,
            'all_imports_and_distributions_inside_environment': True, 'pip_check': checked.stdout.strip(),
            'torch_version': torch.__version__, 'torch_compiled_cuda_version': torch.version.cuda,
            'cpu_torch_numpy_interoperability': True, 'cuda_initialized': False,
            'model_weights_loaded': False, 'gpu_execution_verified': False}


def compare_token_file(original, generated):
    with np.load(original, allow_pickle=False) as a, np.load(generated, allow_pickle=False) as b:
        if set(a.files) != {'ids', 'labels'} or set(b.files) != set(a.files):
            raise ValueError('Token archive fields changed.')
        for name in ('ids', 'labels'):
            if a[name].dtype != b[name].dtype or a[name].shape != b[name].shape or not np.array_equal(a[name], b[name]):
                raise ValueError('Token IDs or supervision labels differ: '+name)
    return sha(original) == sha(generated)


def compare_manifests(original, generated):
    # Actual array values and archive hashes are checked separately below.
    for name in ('completed', 'episodes', 'messages_sha256', 'counts', 'maximum_tokens',
                 'A_references_only_initial8', 'B_C_terminal_identical'):
        if original[name] != generated[name]:
            raise ValueError('Rebuilt training manifest differs in '+name)
    a = {k: v for k, v in original['config'].items() if k != 'curriculum_audit_sha256'}
    b = {k: v for k, v in generated['config'].items() if k != 'curriculum_audit_sha256'}
    if a != b or set(original['entries']) != set(generated['entries']):
        raise ValueError('Training configuration or task set changed.')
    for key, a in original['entries'].items():
        b = generated['entries'][key]
        if {k: v for k, v in a.items() if k != 'sha256'} != {k: v for k, v in b.items() if k != 'sha256'}:
            raise ValueError('Training token counts or cache paths changed: '+key)


def run(output, curriculum_chain):
    output, curriculum_chain = Path(output).resolve(), Path(curriculum_chain).resolve()
    if not output.is_relative_to(ROOT) or output == ROOT:
        raise ValueError('Use a new directory inside this project.')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserve previous cache reconstructions, including failures.')
    environment = check_environment()
    chain = read(curriculum_chain/'result.json')
    if (not chain['completed'] or not chain['historical_development_sources_regenerated']
            or chain['total_teacher_episodes_identical'] != 960 or chain['formal_tests_measured_or_scored']):
        raise ValueError('Require the verified complete curriculum-chain reconstruction.')
    rebuilt_curriculum = curriculum_chain/'workspace'/CURRICULUM
    formal_stage = next(s for s in chain['stages'] if s['stage'] == 'formal')
    if sha(rebuilt_curriculum/'audit_report.json') != formal_stage['generated_audit_sha256']:
        raise ValueError('Regenerated curriculum audit changed.')
    audit = read(rebuilt_curriculum/'audit_report.json')
    original = read(ROOT/CACHE/'manifest.json')
    if not original['completed'] or len(original['entries']) != 3600:
        raise ValueError('Need the complete original 3,600-task cache reference.')
    source_names = tuple(dict.fromkeys((*COPY_SOURCES, *EXTRA_SOURCES)))
    model_manifest = read(ROOT/MODEL/'download_manifest.json')
    if model_manifest['revision'] != original['config']['model_revision']:
        raise ValueError('Tokenizer revision differs from the original training cache.')
    tokenizer_names = tuple(name for name in model_manifest['files'] if name != 'model.safetensors')
    if any(Path(name).suffix in ('.safetensors', '.bin', '.pt') for name in tokenizer_names):
        raise ValueError('Do not copy model weights into the tokenizer-only workspace.')
    config = {'scope': 'all_main_training_inputs_from_regenerated_curriculum_cpu_only',
              'chain_result_sha256': sha(curriculum_chain/'result.json'),
              'reference_manifest_sha256': sha(ROOT/CACHE/'manifest.json'),
              'curriculum_audit_sha256': sha(rebuilt_curriculum/'audit_report.json'),
              'source_sha256': {p: sha(ROOT/p) for p in (*source_names, 'reproduce_training_inputs.py')},
              'tokenizer_sha256': {name: model_manifest['files'][name]['sha256'] for name in tokenizer_names},
              'model_revision': model_manifest['revision'], 'copied_model_weights': False,
              'formal_tests_used': False}
    atomic_json(output/'config.json', config)
    atomic_json(output/'environment.json', environment)
    workspace = output/'workspace'
    for name in source_names:
        destination = workspace/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())
    for name in (*EXPORTS, 'audit_report.json'):
        source, destination = rebuilt_curriculum/name, workspace/CURRICULUM/name
        if name != 'audit_report.json' and sha(source) != audit['export_sha256'][str(Path(name))]:
            raise ValueError('A regenerated teaching export changed.')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    for name, checksum in config['tokenizer_sha256'].items():
        source, destination = ROOT/MODEL/name, workspace/MODEL/name
        if sha(source) != checksum:
            raise ValueError('Pinned tokenizer file changed: '+name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env['PYTHONNOUSERSITE'] = '1'
    started = time.perf_counter()
    try:
        with (output/'build.log').open('w', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, '-u', 'prepare_mainline_training.py'], cwd=workspace,
                env=env, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            atomic_json(output/'execution.json', {'status': 'running', 'pid': child.pid,
                                                'started_unix': time.time()})
            print(f'CPU training-input reconstruction started; child PID {child.pid}.', flush=True)
            code = child.wait()
        if code:
            raise RuntimeError(f'Training-input builder exited with {code}; inspect preserved build.log.')
        target = workspace/CACHE
        generated = read(target/'manifest.json')
        compare_manifests(original, generated)
        if generated['config']['curriculum_audit_sha256'] != config['curriculum_audit_sha256']:
            raise ValueError('The rebuilt cache did not preserve its actual curriculum audit provenance.')
        if (sha(ROOT/CACHE/'messages.jsonl') != original['messages_sha256']
                or sha(target/'messages.jsonl') != original['messages_sha256']):
            raise ValueError('Serialized task messages differ.')
        archive_identical = 0
        for key, entry in original['entries'].items():
            a, b = ROOT/CACHE/entry['path'], target/generated['entries'][key]['path']
            if sha(a) != entry['sha256'] or sha(b) != generated['entries'][key]['sha256']:
                raise ValueError('A cached archive differs from its own manifest.')
            archive_identical += compare_token_file(a, b)
        # Importing this module does not call run(), construct a model or initialize CUDA.
        from train_mainline_student import task_plan
        plans = []
        for group in 'ABC':
            for seed in (401, 402, 403):
                a, b = task_plan(original, group, seed), task_plan(generated, group, seed)
                if a != b:
                    raise ValueError('Sample order, objective weights or token budget changed.')
                plans.append({'group': group, 'seed': seed, 'schedule_and_budget_identical': True, 'budget': b[1]})
        import torch
        if torch.cuda.is_initialized():
            raise ValueError('CUDA was unexpectedly initialized during a CPU-only reconstruction.')
        if any(sha(ROOT/p) != h for p, h in config['source_sha256'].items()):
            raise ValueError('Source changed during reconstruction.')
        if any(sha(workspace/p) != config['source_sha256'][p] for p in source_names):
            raise ValueError('Copied source changed during reconstruction.')
        if sha(ROOT/CACHE/'manifest.json') != config['reference_manifest_sha256']:
            raise ValueError('Original training manifest changed.')
    except BaseException as error:
        atomic_json(output/'failure.json', {'completed': False, 'error_type': type(error).__name__,
                    'error': str(error), 'elapsed_seconds': time.perf_counter()-started})
        raise
    result = {'completed': True, 'scope': config['scope'], 'tasks_with_identical_ids_and_labels': len(original['entries']),
              'archives_byte_identical': archive_identical, 'messages_byte_identical': True,
              'episode_order_identical': True, 'maximum_tokens': generated['maximum_tokens'], 'plans': plans,
              'reference_curriculum_audit_sha256': original['config']['curriculum_audit_sha256'],
              'regenerated_curriculum_audit_sha256': generated['config']['curriculum_audit_sha256'],
              'metadata_difference': 'Rebuilt curriculum audit has its own timing/source-version provenance; it is not replaced with the original digest.',
              'generated_manifest_sha256': sha(target/'manifest.json'), 'build_log_sha256': sha(output/'build.log'),
              'elapsed_seconds': time.perf_counter()-started, 'formal_tests_used': False,
              'model_weights_loaded': False, 'cuda_initialized': False, 'model_training_or_inference_reproduced': False}
    atomic_json(output/'result.json', result)
    atomic_json(output/'execution.json', {'status': 'completed', 'elapsed_seconds': result['elapsed_seconds']})
    (output/'REPORT_ZH.md').write_text(
        '# 独立模型环境与学生训练输入重建\n\n'
        '使用另行安装、与主环境29个包版本完全相同的环境，CPU导入和依赖检查通过，未初始化CUDA。'
        '只复制固定版本分词器文件，不复制或加载模型权重。\n\n'
        '从整条链重建出的正式教材重新生成3,600个训练任务；消息字节、所有token ID、监督标签、'
        '600条轨迹顺序以及A/B/C三种子共九份训练计划和预算全部一致。'
        f'其中{archive_identical}/3600个压缩数组文件逐字节一致；数组本身全部逐项相同。\n\n'
        '重建教材审计文件因计时和已披露源码修正具有不同摘要，原样记录，不伪装成原审计版本。'
        '主训练输入和环境均未覆盖。本次仍未运行GPU训练/推理或正式测试，不能称为模型结果复现。\n',
        encoding='utf-8')
    print(output/'REPORT_ZH.md', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--curriculum-chain', default='results/reproducibility_curriculum_chain_v1')
    args = parser.parse_args()
    run(args.output, args.curriculum_chain)
