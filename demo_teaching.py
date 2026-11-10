"""Generate one audited A/B/C teaching episode from synthetic Ising measurements.

Runs on CPU with NumPy only. This is a worked example, not an LLM benchmark.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from discovery.lab import Laboratory
from discovery.physics import Expression, Term, random_queries
from discovery.student_mainline import evidence_prefix, hypothesis_prompt
from discovery.teaching import (
    audit_teacher, audit_views, export_views, messages_for_view, run_teacher,
)


def main(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # The oracle remains in the laboratory; the teacher receives measurements.
    truth = Expression((Term(((0, 0), (0, 1), (1, 0)), 0.46),))
    initial = random_queries(np.random.default_rng(811), 8)
    lab = Laboratory(truth.delta, noise_sigma=1.0, noise_seed=821)
    trace = run_teacher(lab, initial, query_seed=831, pool_size=16)
    audit = audit_teacher(trace, query_seed=831, pool_size=16)
    # Gold labels are attached after the complete teacher trajectory.
    views = export_views(trace, truth, 'synthetic-demo-001')
    audit_views(views)

    def write(name, value):
        (output/name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
            encoding='utf-8',
        )

    for group in 'ABC':
        write(f'{group}_view.json', views[group])
        write(f'{group}_messages.json', messages_for_view(views[group], group))
    # This inference prompt contains observations only, without labels or truth.
    request = {'messages': hypothesis_prompt(evidence_prefix(views['B'], 32)),
               'max_new_tokens': 128}
    write('inference_request.json', request)
    write('audit.json', {
        **audit, 'used_measurements': lab.used, 'paired_views_verified': True,
        'LLM_training_or_inference_performed': False,
        'teacher_final_before_grading': trace['teacher_final_before_grading'],
    })
    print(json.dumps({'output': str(output), **audit,
                      'used_measurements': lab.used}, ensure_ascii=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='output/teaching_demo')
    main(parser.parse_args().output)
