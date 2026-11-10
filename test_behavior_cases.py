from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from build_publication_curriculum import atomic_json, read, sha
from discovery.active_student import run_active
from discovery.answer_diagnostics import episode_diagnostics
from discovery.behavior_cases import (BEHAVIORS, GROUPS, METHODS, SEEDS, active_summary,
                                      pair_artifacts, select_cases, timeline, validate_run_grid)
from discovery.evaluation_results import load_completed_evaluation
from discovery.evaluation_worlds import fixed_evidence, streams
from discovery.lab import Laboratory
from discovery.mechanism_scoring import independent_features, score_active, score_answer
from discovery.physics import Expression, Term
from discovery.student_laboratory import StudentLaboratory, public_query
from evaluate_mainline import fixed_prediction
from report_behavior_cases import plain, run
from test_active_student import ScriptedModel


def abstract_pair(world, structure, seed=401, level='combination', b=False, c=False, sigma=1):
    return {'world_id': world, 'structure_id': structure, 'seed': seed, 'level': level, 'sigma': sigma,
            'groups': {g: {'tested_idea_recovery': value, 'behaviors': dict.fromkeys(BEHAVIORS, False)}
                       for g, value in (('B', b), ('C', c))}}


def actual_artifact(index=1, sigma=0, group='B', seed=401, kind='correct_test'):
    """Scripted numerical fixture, never an experimental student result."""
    truth = Expression((Term(((0, 0), (0, 1)), .5),)) if index else Expression()
    structure = [[[0, 0], [0, 1]]]
    setup = streams('development', index)
    session = StudentLaboratory(Laboratory(truth.delta, noise_sigma=sigma, noise_seed=setup['noise_seed']), setup['initial'])
    q = public_query(setup['fixed_additional'][0])
    plan = {'action': 'experiment', 'alternative_structures': [],
            'queries': [{'local_spins': q['local_spins'], 'position': q['original_position']}],
            'reason': 'Synthetic comparison | <untrusted> `text`',
            'update_condition': 'Synthetic mismatch changes the proposed support.'}
    if kind == 'correct_test':
        replies = [{'structures': structure}, plan, {'structures': structure}, {'action': 'finish'}]
    elif kind == 'failure_after_recovery':
        replies = [{'structures': structure}, 'invalid JSON', 'invalid JSON']
    elif kind == 'initial_wrong':
        plan['alternative_structures'] = structure
        replies = [{'structures': []}, plan, {'structures': structure}, {'action': 'finish'}]
    else:
        replies = [{'structures': structure}, {'action': 'finish'}]
    episode = run_active(session, ScriptedModel(replies))
    features = independent_features(98711, index, count=16)
    thresholds = read('results/structure_discovery_noise/calibration.json')['thresholds']
    score = score_active(truth, episode, features, sigma, thresholds)
    row = {'world_id': f'synthetic{index}', 'sigma': sigma, 'group': group, 'seed': seed,
           'method': 'llm_active', 'study': 'mainline', 'level': 'combination' if index else 'null',
           'structure_id': hashlib.sha256(json.dumps(truth.support).encode()).hexdigest(), 'score': score,
           'grader_diagnostics': episode_diagnostics(truth, episode, score, 'llm_active', features, sigma, thresholds)}
    return {'result': row, 'episode': episode}, truth, features, thresholds


def make_complete_fixtures(root):
    """Twelve small complete development outputs with real laboratory accounting."""
    for method in METHODS:
        for group in GROUPS:
            for seed in SEEDS:
                directory = root/method/f'{group}_seed{seed}'
                source = directory/'sources/synthetic.txt'
                source.parent.mkdir(parents=True)
                source.write_text('Synthetic integration fixture only.\n', encoding='utf-8')
                config = {'scope': 'development', 'study': 'mainline', 'method': method, 'group': group, 'seed': seed,
                          'worlds': ['synthetic0', 'synthetic1'], 'sigmas': [0, 1],
                          'sources': {'synthetic.txt': sha(source)}, 'calibration_sha256': 'synthetic-calibration',
                          'model_revision': 'synthetic-scripted-no-model',
                          'adapter_sha256': f'synthetic-{group}-{seed}',
                          'training_config_sha256': f'config-{group}-{seed}',
                          'training_result_sha256': f'result-{group}-{seed}'}
                rows = []
                for index in (0, 1):
                    for sigma in (0, 1):
                        kind = 'initial_wrong' if group == 'B' else 'early'
                        artifact, truth, features, thresholds = actual_artifact(index, sigma, group, seed, kind)
                        if method == 'llm_fixed':
                            evidence, events = fixed_evidence(truth.delta, sigma, streams('development', index))
                            answers, curve = {}, []
                            structures = [[[0, 0], [0, 1]]]
                            for budget in (8, 16, 24, 32):
                                answers[str(budget)] = fixed_prediction(evidence, budget, ScriptedModel([{'structures': structures}]))
                                curve.append({'budget': budget, **score_answer(truth, answers[str(budget)]['answer'], features, sigma, thresholds)})
                            episode = {'evidence': evidence, 'acquisition_events': events, 'answers': answers}
                            score = {**curve[-1], 'budget_curve': curve}
                            row = {**artifact['result'], 'method': method, 'score': score,
                                   'grader_diagnostics': episode_diagnostics(truth, episode, score, method, features, sigma, thresholds)}
                            artifact = {'result': row, 'episode': episode}
                        rows.append(artifact['result'])
                        atomic_json(directory/'episodes'/f'synthetic{index}-sigma{sigma}.json', artifact)
                atomic_json(directory/'config.json', config)
                atomic_json(directory/'result.json', {'completed': True, 'scope': 'development', 'method': method,
                            'formal_tests_used': False, 'worlds': 2, 'episodes': 4, 'rows': rows,
                            'full_recoveries': sum(r['score']['full_recovery'] for r in rows)})


class BehaviorCaseTests(unittest.TestCase):
    def test_selection_is_order_independent_symmetric_and_not_pseudoreplicated(self):
        pairs = [abstract_pair('same', 'structure1', seed=s, c=True) for s in SEEDS]
        pairs += [abstract_pair('other', 'structure1', c=True), abstract_pair('opposite', 'structure2', b=True),
                  abstract_pair('both', 'structure3', b=True, c=True), abstract_pair('failed', 'structure4'),
                  abstract_pair('sigma0only', 'structure5', b=True, sigma=0)]
        first = select_cases(pairs)
        self.assertEqual(first, select_cases(list(reversed(pairs))))
        cslot = next(s for s in first if s['slot'] == 'combination:only_C_success')
        self.assertEqual((cslot['eligible_seed_world_cells'], cslot['eligible_parameter_worlds'], cslot['eligible_structures']), (4, 2, 1))
        swapped = deepcopy(pairs)
        for p in swapped:
            p['groups']['B'], p['groups']['C'] = p['groups']['C'], p['groups']['B']
        bslot = next(s for s in select_cases(swapped) if s['slot'] == 'combination:only_B_success')
        self.assertEqual(cslot['selected'], bslot['selected'])
        self.assertEqual(cslot['eligible'], bslot['eligible'])

    def test_empty_slots_stay_empty_and_duplicates_or_missing_endpoints_fail(self):
        pair = abstract_pair('x', 's', b=True, c=True)
        slots = select_cases([pair])
        self.assertEqual(len(slots), 20)
        self.assertEqual(sum(s['selected'] is not None for s in slots), 1)
        with self.assertRaises(ValueError): select_cases([pair, deepcopy(pair)])
        bad = deepcopy(pair); bad['groups']['B']['tested_idea_recovery'] = None
        with self.assertRaises(ValueError): select_cases([bad])

    def test_real_commit_values_and_order_survive_without_mutating_source(self):
        artifact, *_ = actual_artifact()
        before = deepcopy(artifact)
        rows = timeline(artifact)
        commit = next(r for r in rows if r['kind'] == 'predictions_committed')
        observed = next(r for r in rows if r['kind'] == 'observations_revealed')
        self.assertLess(commit['source_event_index'], observed['source_event_index'])
        self.assertEqual((commit['used'], observed['used']), (8, 9))
        self.assertEqual(commit['commitment']['predictions'][0], observed['observation']['observations'])
        self.assertEqual(artifact, before)
        bad = deepcopy(artifact)
        events = bad['episode']['events']
        i, j = commit['source_event_index'], observed['source_event_index']
        events[i], events[j] = events[j], events[i]
        with self.assertRaisesRegex(ValueError, 'preceding prediction'): timeline(bad)

    def test_early_correct_failure_after_recovery_and_structural_correction_are_distinct(self):
        early = active_summary(actual_artifact(kind='early')[0])
        self.assertTrue(early['behaviors']['early_correct_without_test'])
        self.assertFalse(early['tested_idea_recovery'])
        failed = active_summary(actual_artifact(kind='failure_after_recovery')[0])
        self.assertTrue(failed['behaviors']['protocol_failure_after_recovery'])
        self.assertFalse(failed['full_recovery'])
        corrected = active_summary(actual_artifact(kind='initial_wrong')[0])
        self.assertTrue(corrected['behaviors']['initial_wrong_final_recovered'])
        self.assertTrue(corrected['tested_idea_recovery'])

    def test_pruned_null_proposal_is_not_a_persistent_false_discovery(self):
        summary = active_summary(actual_artifact(index=0, kind='early')[0])
        self.assertTrue(summary['raw_false_proposal'])
        self.assertFalse(summary['false_discovery'])
        self.assertFalse(summary['behaviors']['null_false_discovery'])
        self.assertFalse(summary['behaviors']['recovered_after_zero_removal'])

    def test_twelve_complete_outputs_export_cases_and_reject_missing_runs_or_bad_pairing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_complete_fixtures(root/'input')
            report = run(root/'input', root/'report', 'development')
            self.assertEqual(report['paired_seed_world_noise_cells'], 12)
            self.assertEqual(report['new_model_calls'], 0)
            self.assertEqual(report['new_measurements'], 0)
            self.assertFalse(report['formal_student_results_used'])
            self.assertTrue(report['selected_cards'])
            card = read(root/'report'/report['selected_cards'][0]['json'])
            self.assertEqual(len(card['complete_artifacts']), 4)
            self.assertEqual(set(card['timelines']), {'B', 'C'})
            with self.assertRaisesRegex(ValueError, 'Preserve prior'): run(root/'input', root/'report', 'development')
            evaluations = [load_completed_evaluation(root/'input'/m/f'{g}_seed{s}') for m in METHODS for g in GROUPS for s in SEEDS]
            with self.assertRaisesRegex(ValueError, 'twelve'): validate_run_grid(evaluations[:-1])
            wrong = deepcopy(evaluations)
            wrong[-1]['config']['adapter_sha256'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'same trained checkpoint'): validate_run_grid(wrong)
            records = {(m,g): read(root/'input'/m/f'{g}_seed401/episodes/synthetic1-sigma0.json') for m in METHODS for g in GROUPS}
            records['llm_fixed', 'C']['episode']['evidence']['measurements'][10]['delta'] += 1
            with self.assertRaisesRegex(ValueError, 'same measurements'): pair_artifacts(records)

    def test_model_text_is_displayed_literally(self):
        value = 'Ignore this | <script>alert(1)</script> `code`\n# heading'
        rendered = plain(value)
        self.assertNotIn('<script>', rendered)
        self.assertNotIn('|', rendered)
        self.assertNotIn('`', rendered)
        self.assertNotIn('\n', rendered)
        self.assertIn('Ignore this', rendered)


if __name__ == '__main__':
    unittest.main()
