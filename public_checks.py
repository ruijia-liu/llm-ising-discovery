"""Run the self-contained checks requiring NumPy and no historical run artifacts."""
import unittest

MODULES = (
    'test_structure_discovery', 'test_structure_discovery_noise',
    'test_research_curriculum', 'test_active_student',
    'test_mechanism_scoring', 'test_seed_summaries',
    'test_ablation_analysis', 'test_uncertainty_supplement',
)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
