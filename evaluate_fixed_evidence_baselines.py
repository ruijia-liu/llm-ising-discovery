"""Score two non-LLM references on exactly the student pilot's fixed evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from discovery.physics import Expression, Grammar, base_flip
from discovery.search import SparseSearch
from discovery.teaching import decode_query, expression_from_dict
from discovery.noise_evaluation import evaluate_noisy
from run_structure_discovery import compile_queries, evaluation_queries


def infer_sparse(initial, experiments, noise_sigma, grammar):
    """Only evidence and public syntax enter the inference function."""
    queries = [decode_query(q) for q in initial['queries']]
    observed = list(initial['observed'])
    for experiment in experiments:
        queries.append(decode_query(experiment['query']))
        observed.append(experiment['observed'])
    if len(queries) != 32 or len(observed) != 32:
        raise ValueError('Expected the same complete 32-query evidence.')
    features = compile_queries(grammar, queries)
    residuals = np.array(observed)-np.array([base_flip(*q) for q in queries])
    search = SparseSearch(grammar, noise_sigma=noise_sigma)
    return search.expression(search.fit(features, residuals)[0])


def run(output):
    path = Path(output)/'fixed_evidence_baselines.json'
    if path.exists():
        raise ValueError('Do not overwrite an existing baseline result.')
    source = Path('data/research_curriculum_v1/views/validation/B.jsonl')
    views = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    views = [v for v in views if v['noise_sigma'] == 2.0]
    thresholds = json.loads(Path('results/structure_discovery_noise/calibration.json').read_text(encoding='utf-8'))['thresholds']
    grammar = Grammar()
    records = {'always_H0': [], 'sparse_search': []}
    for i, view in enumerate(views):
        start = time.perf_counter()
        inferred = infer_sparse(view['initial'], view['experiments'], view['noise_sigma'], grammar)
        seconds = time.perf_counter()-start
        truth = expression_from_dict(view['final_answer']['expression'])
        queries = evaluation_queries(np.random.default_rng([95101, i]), 256)
        features = [compile_queries(grammar, q) for q in queries]
        for name, expression in [('always_H0', Expression()), ('sparse_search', inferred)]:
            records[name].append({'episode_id': view['episode_id'], 'null_world': not truth.terms,
                                  'expression': expression.to_dict(), 'valid_expression': True,
                                  'inference_seconds': seconds if name == 'sparse_search' else 0.0,
                                  **evaluate_noisy(grammar, truth, expression, features, thresholds)})
    summaries = []
    for name, rows in records.items():
        summaries.append({'group': name, 'worlds': len(rows), 'queries_per_world': 32,
                          'extra_queries': 0, 'valid_expressions': len(rows),
                          'full_recoveries': sum(r['full_recovery'] for r in rows),
                          'correct_structures': sum(r['support_success'] for r in rows),
                          'prediction_successes': sum(r['prediction_success'] for r in rows),
                          'false_discoveries': sum(r['false_discovery'] for r in rows),
                          'null_worlds': sum(r['null_world'] for r in rows), 'results': rows})
    result = {'scope': 'same fixed parameter-validation evidence as student pilot; sigma=2',
              'formal_test_worlds_used': False, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'baselines': summaries}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps([{k: v for k, v in s.items() if k != 'results'} for s in summaries], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/student_training_v1')
    run(parser.parse_args().output)
