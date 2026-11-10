"""The arithmetic diagnostic fits only the student's supplied structures."""
import unittest

import numpy as np

from diagnose_student_parameters import fit_coefficients
from discovery.physics import Expression, Grammar, Term
from discovery.student_v2 import load_views
from discovery.teaching import expression_from_dict


class ParameterDiagnosticTests(unittest.TestCase):
    def test_recovers_coefficients_from_same_evidence_when_support_is_right(self):
        grammar = Grammar()
        views = load_views('train')
        for index in (0, 4, 16):
            view = views[index]
            truth = expression_from_dict(view['final_answer']['expression'])
            proposed = Expression(tuple(Term(t.points, 7.0) for t in truth.terms))
            evidence = {k: view[k] for k in ('initial', 'experiments')}
            fitted = fit_coefficients(evidence, proposed, grammar)
            np.testing.assert_allclose(grammar.coefficients(fitted), grammar.coefficients(truth), atol=1e-10, rtol=0)

    def test_cannot_add_missing_structures_or_search_candidates(self):
        grammar = Grammar()
        view = load_views('train')[0]
        proposed = Expression((Term(((0, 0), (0, 1)), 1.0),))
        fitted = fit_coefficients({k: view[k] for k in ('initial', 'experiments')}, proposed, grammar)
        self.assertTrue(set(fitted.support).issubset(proposed.support))
        self.assertNotIn(((0, 0),), fitted.support)


if __name__ == '__main__':
    unittest.main()
