"""Supplement only: fit coefficients for the model's unchanged proposed support.

No candidate search, new measurements or model updates. This is a post-hoc
arithmetic diagnostic and must never replace the student's original score.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from discovery.physics import Expression, Grammar, Term, base_flip
from discovery.teaching import decode_query, expression_from_dict
from discovery.student_v2 import load_views
from discovery.noise_evaluation import evaluate_noisy
from run_structure_discovery import compile_queries, evaluate, evaluation_queries
from train_student_pilot import json_write, parse_expression
from train_student_v2 import TOLERANCE


def fit_coefficients(view, hypothesis, grammar):
    if not hypothesis.terms:
        return hypothesis
    queries = [decode_query(q) for q in view['initial']['queries']]
    observed = list(view['initial']['observed'])
    for item in view['experiments']:
        queries.append(decode_query(item['query']))
        observed.append(item['observed'])
    columns = [grammar.index[term.points] for term in hypothesis.terms]
    x = compile_queries(grammar, queries)[:, columns]
    y = np.array(observed)-np.array([base_flip(*q) for q in queries])
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    return Expression(tuple(Term(term.points, float(k)) for term, k in zip(hypothesis.terms, coefficients)))


def run(output):
    output = Path(output)
    path = output/'coefficient_diagnostic.json'
    if path.exists():
        raise ValueError('Keep the existing arithmetic diagnostic.')
    experiment = json.loads((output/'result.json').read_text(encoding='utf-8'))
    grammar = Grammar()
    rows = []
    for check in [experiment['training_checks'][-1], experiment['validation']]:
        views = {v['episode_id']: v for v in load_views(check['split'])}
        for original in check['results']:
            row = {'episode_id': original['episode_id'], 'split': original['split'],
                   'original_full_recovery': original['full_recovery'],
                   'valid_original_expression': original['valid_expression']}
            if original['valid_expression']:
                view = views[original['episode_id']]
                proposal = parse_expression(original['text'], grammar)
                # Do NOT pass view's label to the fitting function.
                evidence = {k: view[k] for k in ('initial', 'experiments')}
                fitted = fit_coefficients(evidence, proposal, grammar)
                truth = expression_from_dict(view['final_answer']['expression'])
                world = int(view['episode_id'][1:5])
                features = [compile_queries(grammar, q) for q in evaluation_queries(np.random.default_rng([95201, world]), 256)]
                row.update(fitted_expression=fitted.to_dict(),
                           **evaluate_noisy(grammar, truth, fitted, features, TOLERANCE),
                           original_exact_benchmark=evaluate(grammar, truth, fitted, features))
            else:
                row['full_recovery'] = False
            rows.append(row)
    summary = {split: {'worlds': sum(r['split'] == split for r in rows),
                      'original_recoveries': sum(r['original_full_recovery'] for r in rows if r['split'] == split),
                      'after_arithmetic_diagnostic': sum(r['full_recovery'] for r in rows if r['split'] == split)}
               for split in ('train', 'validation')}
    json_write(path, {'supplement_only_original_scores_unchanged': True,
                     'support_candidates_added': 0, 'extra_measurements': 0,
                     'model_updates': 0, 'summary': summary, 'results': rows})
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/student_training_v2')
    run(parser.parse_args().output)
