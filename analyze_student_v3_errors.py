"""Read-only error classification of completed v3 proposals, without new probes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from discovery.student_v2 import load_views
from discovery.student_v3 import parse_structures
from discovery.teaching import expression_from_dict


def classify_answer(row, truth):
    if not row['valid_expression']:
        return 'invalid_answer'
    proposed = parse_structures(row['text'])
    if proposed == truth.support:
        return 'exact_structure_recovered' if row['full_recovery'] else 'correct_structure_failed_numerical_check'
    if row['full_recovery']:
        return 'recovered_after_zero_coefficient_removal'
    if sorted(map(len, proposed)) != sorted(map(len, truth.support)):
        return 'wrong_term_count_or_orders'
    return 'correct_term_count_and_orders_wrong_positions'


def run(output):
    output = Path(output)
    path = output/'result.json'
    source_bytes = path.read_bytes()
    result = json.loads(source_bytes)
    checks = [result['training_checks'][-1], result['validation']]
    summaries, rows = {}, []
    for check in checks:
        views = {v['episode_id']: v for v in load_views(check['split'])}
        counts = Counter()
        for row in check['results']:
            truth = expression_from_dict(views[row['episode_id']]['final_answer']['expression'])
            category = classify_answer(row, truth)
            counts[category] += 1
            rows.append({'split': row['split'], 'episode_id': row['episode_id'],
                         'null_world': row['null_world'], 'category': category})
        summaries[check['split']] = dict(counts)
    audit = {'raw_result_sha256': hashlib.sha256(source_bytes).hexdigest(), 'model_updates': 0,
             'new_measurements': 0, 'new_model_inferences': 0, 'scores_changed': False,
             'summary': summaries, 'results': rows}
    (output/'error_analysis.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(summaries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/student_training_v3')
    run(parser.parse_args().output)
