"""Audit queue coverage and reporting on existing development artifacts, without formal access."""
import ast
from pathlib import Path

from build_publication_curriculum import atomic_json, read, sha
from discovery.evaluation_worlds import require_completed_main_training
from report_evaluation_metrics import run as summarize_evaluations
from run_evaluation_queue import plan, RUNTIME_FILES


def run():
    output = Path('results/evaluation_preparation_v1')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserve the completed preparation audit.')
    tree = ast.parse(Path('evaluate_mainline.py').read_text(encoding='utf-8'))
    # Literal declarations only; do not execute the evaluator to discover its sources.
    declared = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'source_files' for t in node.targets):
            declared.update(ast.literal_eval(node.value))
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == 'source_files':
            declared.update(ast.literal_eval(node.value))
    missing = declared-set(RUNTIME_FILES)-{'PROCESS_ABLATION_PROTOCOL.md'}
    if not declared or missing:
        raise ValueError('Queue fingerprint misses declared evaluator dependencies: '+repr(missing))
    plans = {study: plan(study) for study in ('mainline', 'prediction_ablation')}
    for study, specification in plans.items():
        atomic_json(output/(study+'_plan.json'), specification)
    try:
        require_completed_main_training()
        training_gate = {'main_training_complete': True, 'detail': 'All nine registered final checkpoints verified.'}
    except ValueError as error:
        training_gate = {'main_training_complete': False, 'detail': str(error)}
    directories = [f'results/mainline_development_{method}_v4_diagnostics' for method in ('sparse_fixed', 'sparse_active')]
    summarize_evaluations(directories, output/'development_summary', 'development')
    prior = read('results/secondary_metrics_audit/descriptive_report/result.json')
    current = read(output/'development_summary/result.json')
    if prior['cells'] != current['cells']:
        raise ValueError('Per-seed development summaries changed while adding seed aggregation.')
    if current['seed_summaries']:
        raise ValueError('A deterministic search baseline was replicated into artificial training seeds.')
    result = {'completed': True, 'formal_tests_used': False, 'gpu_models_loaded': False,
        'mainline_evaluation_jobs': len(plans['mainline']['jobs']),
        'ablation_evaluation_jobs': len(plans['prediction_ablation']['jobs']),
        'training_gate_snapshot': training_gate,
        'declared_evaluator_sources_covered': sorted(declared),
        'development_conditions': 24, 'existing_descriptive_cells_unchanged': True,
        'synthetic_unit_test_results_are_not_student_results': True,
        'sources': {p: sha(p) for p in ('report_evaluation_preparation.py', 'run_evaluation_queue.py',
            'discovery/run_lock.py', 'report_evaluation_metrics.py', 'discovery/seed_summaries.py',
            'analyze_prediction_ablation.py', 'test_evaluation_queue.py', 'test_seed_summaries.py',
            'test_secondary_metrics.py', 'test_ablation_analysis.py')},
        'development_report_sha256': sha(output/'development_summary/result.json')}
    atomic_json(output/'audit.json', result)
    (output/'REPORT_ZH.md').write_text(
        '# 评估与三种子分析准备核验\n\n'
        '主队列计划20项评估：A/B/C×三个种子×固定/主动两种模式，再加两种共享搜索基线。\n'
        '显式预测内容消融计划6项评估；不把同一搜索基线重复当成消融模型。\n'
        '每项正式评估都包含200世界×两种噪声。计划生成不打开测试世界、不加载GPU模型。\n\n'
        '当前主训练访问检查：'+training_gate['detail']+'\n\n'
        '队列持有单写者锁；完成记录核验后跳过，部分记录需显式恢复，失败不自动重试。'
        '逐次调用保存墙钟耗时；仅有开始记录的中断保留未知完成状态，不将缺失耗时填零。\n\n'
        '三种子汇总仅在401/402/403和世界网格完整时形成等权平均，独立结构和参数世界不随种子数增加。'
        '已有24个开发条件的逐种子描述性单元完全不变；搜索基线没有生成虚假的三种子结果。\n\n'
        '消融比较入口已接通相同结构等权与配对bootstrap；它属于次要分析。'
        '尚无正式学生结果，以上均不证明任何教法更好。\n', encoding='utf-8')
    print(output/'REPORT_ZH.md')


if __name__ == '__main__':
    run()
