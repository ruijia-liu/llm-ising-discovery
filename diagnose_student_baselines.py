"""Non-LLM references on the six preselected seen-world diagnostic cases."""
import numpy as np

from diagnose_student_pilot import OUT, TRAIN_SEEN, digest, load_views, read_json
from train_student_pilot import json_write
from discovery.physics import Expression, Grammar
from discovery.noise_evaluation import evaluate_noisy
from discovery.teaching import expression_from_dict
from evaluate_fixed_evidence_baselines import infer_sparse
from run_structure_discovery import compile_queries, evaluation_queries


def run():
    path = OUT/'seen_world_baselines.json'
    if path.exists():
        raise ValueError('Preserve the existing baseline run.')
    grammar = Grammar()
    thresholds = read_json('results/structure_discovery_noise/calibration.json')['thresholds']
    views = load_views()['train']
    results = []
    for index in TRAIN_SEEN:
        view = views[index]
        inferred = infer_sparse(view['initial'], view['experiments'], view['noise_sigma'], grammar)
        truth = expression_from_dict(view['final_answer']['expression'])
        features = [compile_queries(grammar, q) for q in evaluation_queries(np.random.default_rng([95102, index]), 256)]
        for name, expression in (('always_H0', Expression()), ('sparse_search', inferred)):
            results.append({'group': name, 'episode_id': view['episode_id'], 'index': index,
                            'null_world': not truth.terms, 'expression': expression.to_dict(),
                            **evaluate_noisy(grammar, truth, expression, features, thresholds)})
    summary = [{ 'group': name, 'worlds': len(TRAIN_SEEN),
                **{k: sum(r[k] for r in results if r['group'] == name) for k in
                   ('support_success', 'prediction_success', 'full_recovery', 'false_discovery')}}
               for name in ('always_H0', 'sparse_search')]
    json_write(path, {'source_sha256': digest(__file__), 'extra_measurements': 0,
                     'formal_test_worlds_used': False, 'summary': summary, 'results': results})
    print(summary)


if __name__ == '__main__':
    run()
