"""Check that the six noiseless evidence sets are identifiable by sparse search."""
import numpy as np

from discovery.student_v2 import TRAIN_IDS, load_views
from discovery.physics import Grammar
from discovery.teaching import expression_from_dict
from discovery.noise_evaluation import evaluate_noisy
from evaluate_fixed_evidence_baselines import infer_sparse
from run_structure_discovery import compile_queries, evaluate, evaluation_queries
from train_student_pilot import json_write
from train_student_v2 import OUTPUT, TOLERANCE


def run():
    path = OUTPUT/'fixed_evidence_baseline.json'
    if path.exists():
        raise ValueError('Preserve existing baseline output.')
    grammar = Grammar()
    rows = []
    for view in load_views('train'):
        if view['episode_id'].split('-')[0] not in TRAIN_IDS:
            continue
        hypothesis = infer_sparse(view['initial'], view['experiments'], 0.0, grammar)
        truth = expression_from_dict(view['final_answer']['expression'])
        index = int(view['episode_id'][1:5])
        features = [compile_queries(grammar, q) for q in evaluation_queries(np.random.default_rng([95201, index]), 256)]
        rows.append({'episode_id': view['episode_id'], 'expression': hypothesis.to_dict(),
                     **evaluate_noisy(grammar, truth, hypothesis, features, TOLERANCE),
                     'original_exact_benchmark': evaluate(grammar, truth, hypothesis, features)})
    result = {'worlds': len(rows), 'extra_queries': 0, 'full_recoveries': sum(r['full_recovery'] for r in rows),
              'exact_recoveries': sum(r['original_exact_benchmark']['full_recovery'] for r in rows), 'results': rows}
    json_write(path, result)
    print({k: v for k, v in result.items() if k != 'results'})


if __name__ == '__main__':
    run()
