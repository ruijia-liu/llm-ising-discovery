from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from discovery.active_student import run_active
from discovery.episode_journal import EpisodeJournal, RecoveryError, canonical, digest
from discovery.evaluation_worlds import fixed_evidence, streams
from discovery.lab import Laboratory
from discovery.physics import Expression, Term, random_queries
from discovery.physics import Grammar
from discovery.search import run_search
from discovery.student_laboratory import StudentLaboratory, public_query
from evaluate_mainline import fixed_prediction


class InjectedInterruption(RuntimeError):
    pass


IDENTITY = {'fixture': 'scripted-recovery-only', 'noise_sigma': 1}
TRUTH = Expression((Term(((0, 0), (0, 1)), .4),))


def scripted_reply(messages, maximum):
    # A deterministic test fixture, not a trained language model.
    if '这次只提交假设' in messages[-1]['content']:
        value = {'structures': [[[0, 0], [0, 1]]]}
    else:
        queries = [public_query(q) for q in random_queries(np.random.default_rng(101), 4)]
        value = {'action': 'experiment', 'alternative_structures': [[[0, 0]]],
                 'queries': [{'local_spins': q['local_spins'], 'position': q['original_position']} for q in queries],
                 'reason': 'Scripted recovery fixture.', 'update_condition': 'Refit after this batch.'}
    return {'text': json.dumps(value), 'input_tokens': 100, 'output_tokens': 10, 'generation_seconds': .1}


def fixture(journal, oracle, generator, on_event=None):
    lab = Laboratory(journal.oracle(oracle), noise_sigma=1, noise_seed=987)
    initial = random_queries(np.random.default_rng(100), 8)
    session = StudentLaboratory(lab, initial)
    def event(value):
        journal.event(value)
        if on_event:
            on_event(value)
    result = run_active(session, journal.generator(generator), on_event=event)
    receipt = journal.complete(result)
    return result, receipt


def append_cost(root, kind):
    with (Path(root)/(kind+'.cost.jsonl')).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps({'real_call': kind})+'\n')
        handle.flush(); os.fsync(handle.fileno())


def crash_child(root):
    def oracle(s, p):
        append_cost(root, 'oracle')
        return TRUTH.delta(s, p)
    def generator(messages, maximum):
        append_cost(root, 'model')
        return scripted_reply(messages, maximum)
    def crash(event):
        if event['kind'] == 'predictions_committed':
            os._exit(77)  # Real process exit, bypassing __exit__/finally.
    with EpisodeJournal(Path(root)/'journal', IDENTITY) as journal:
        fixture(journal, oracle, generator, crash)


class EpisodeJournalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_case(self, root, costs, event=None):
        def oracle(s, p):
            costs['oracle'] += 1
            return TRUTH.delta(s, p)
        def generator(messages, maximum):
            costs['model'] += 1
            return scripted_reply(messages, maximum)
        with EpisodeJournal(root, IDENTITY) as journal:
            return fixture(journal, oracle, generator, event)

    def test_every_protocol_boundary_replays_identically_without_extra_calls(self):
        baseline_cost = {'oracle': 0, 'model': 0}
        baseline, _ = self.run_case(self.root/'baseline', baseline_cost)
        self.assertEqual(baseline_cost, {'oracle': 32, 'model': 13})
        for kind in ('initial_evidence', 'model_reply', 'current_hypothesis', 'predictions_committed',
                     'observations_revealed', 'episode_ended'):
            with self.subTest(boundary=kind):
                costs = {'oracle': 0, 'model': 0}
                def stop(event):
                    if event['kind'] == kind:
                        raise InjectedInterruption(kind)
                path = self.root/kind
                with self.assertRaises(InjectedInterruption):
                    self.run_case(path, costs, stop)
                restored, receipt = self.run_case(path, costs)
                self.assertEqual(canonical(restored), canonical(baseline))
                self.assertEqual(costs, baseline_cost)
                self.assertEqual(receipt['logical_oracle_calls'], 32)
                again, _ = self.run_case(path, costs)
                self.assertEqual(canonical(again), canonical(baseline))
                self.assertEqual(costs, baseline_cost)

    def test_partial_initial_batch_and_partial_reveal_do_not_call_oracle_again(self):
        for cutoff in (3, 10):
            with self.subTest(oracle_results=cutoff):
                costs = {'oracle': 0, 'model': 0}
                root = self.root/f'batch{cutoff}'
                with EpisodeJournal(root, IDENTITY) as journal:
                    original = journal.oracle
                    def crash_after_receipt(function):
                        wrapped = original(function)
                        def call(s, p):
                            value = wrapped(s, p)
                            if journal.cursor['oracle'] == cutoff:
                                raise InjectedInterruption('receipt saved, batch incomplete')
                            return value
                        return call
                    journal.oracle = crash_after_receipt
                    def oracle(s, p):
                        costs['oracle'] += 1
                        return TRUTH.delta(s, p)
                    def generator(m, cap):
                        costs['model'] += 1
                        return scripted_reply(m, cap)
                    with self.assertRaises(InjectedInterruption):
                        fixture(journal, oracle, generator)
                result, _ = self.run_case(root, costs)
                self.assertEqual(result['status'], 'budget_completed')
                self.assertEqual(costs, {'oracle': 32, 'model': 13})

    def test_unresolved_calls_fail_closed_without_reexecution(self):
        for kind in ('oracle', 'model'):
            root = self.root/kind
            calls = []
            def interrupted(*args):
                calls.append(1)
                raise InjectedInterruption('may already have consumed resources')
            with EpisodeJournal(root, IDENTITY) as journal:
                with self.assertRaises(InjectedInterruption):
                    if kind == 'oracle':
                        journal.oracle(interrupted)(*random_queries(np.random.default_rng(9), 1)[0])
                    else:
                        journal.generator(interrupted)([{'role': 'user', 'content': 'fixture'}], 128)
            with self.assertRaisesRegex(RecoveryError, 'Unresolved'):
                EpisodeJournal(root, IDENTITY)
            self.assertEqual(calls, [1])

    def test_corruption_changed_prompt_identity_and_live_writer_are_rejected(self):
        root = self.root/'integrity'
        with EpisodeJournal(root, IDENTITY) as journal:
            with self.assertRaisesRegex(RecoveryError, 'live writer'):
                EpisodeJournal(root, IDENTITY)
            journal.generator(scripted_reply)([{'role': 'user', 'content': '这次只提交假设'}], 128)
        with self.assertRaisesRegex(RecoveryError, 'identity'):
            EpisodeJournal(root, {'fixture': 'different'})
        with EpisodeJournal(root, IDENTITY) as journal:
            with self.assertRaisesRegex(RecoveryError, 'differs'):
                journal.generator(scripted_reply)([{'role': 'user', 'content': 'different'}], 128)
        path = root/'model/000000.result.json'
        changed = json.loads(path.read_text(encoding='utf-8'))
        changed['value']['output_tokens'] = 999
        path.write_text(json.dumps(changed), encoding='utf-8')
        with self.assertRaisesRegex(RecoveryError, 'checksum'):
            EpisodeJournal(root, IDENTITY)

    def test_fixed_evidence_and_four_answers_resume_without_reacquisition(self):
        root = self.root/'fixed'
        costs = {'oracle': 0, 'model': 0}
        def oracle(s, p):
            costs['oracle'] += 1
            return TRUTH.delta(s, p)
        def generate(m, maximum):
            costs['model'] += 1
            return scripted_reply(m, maximum)
        def run(stop=False):
            with EpisodeJournal(root, IDENTITY) as journal:
                evidence, events = fixed_evidence(journal.oracle(oracle), 1, streams('development', 0))
                journal.event({'kind': 'fixed_evidence', 'evidence': evidence, 'acquisition_events': events})
                answers = {}
                for budget in (8, 16, 24, 32):
                    answers[str(budget)] = fixed_prediction(evidence, budget, journal.generator(generate))
                    if stop and budget == 16:
                        raise InjectedInterruption('two answers saved')
                result = {'evidence': evidence, 'answers': answers}
                journal.complete(result)
                return result
        with self.assertRaises(InjectedInterruption):
            run(True)
        result = run()
        self.assertEqual(costs, {'oracle': 32, 'model': 4})
        self.assertEqual(run(), result)
        self.assertEqual(costs, {'oracle': 32, 'model': 4})

    def test_actual_process_death_releases_lock_and_preserves_costs(self):
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--crash-fixture', str(self.root)],
                               capture_output=True, text=True, encoding='utf-8',
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=30)
        self.assertEqual(child.returncode, 77, child.stderr)
        def oracle(s, p):
            append_cost(self.root, 'oracle')
            return TRUTH.delta(s, p)
        def generator(m, maximum):
            append_cost(self.root, 'model')
            return scripted_reply(m, maximum)
        with EpisodeJournal(self.root/'journal', IDENTITY) as journal:
            result, receipt = fixture(journal, oracle, generator)
        self.assertEqual(result['status'], 'budget_completed')
        self.assertEqual(len((self.root/'oracle.cost.jsonl').read_text().splitlines()), 32)
        self.assertEqual(len((self.root/'model.cost.jsonl').read_text().splitlines()), 13)
        self.assertEqual(receipt['replayed_this_invocation']['oracle'], 8)

    def test_invalid_replies_and_repair_budget_survive_a_restart(self):
        root = self.root/'invalid'
        costs = {'oracle': 0, 'model': 0}
        def oracle(s, p):
            costs['oracle'] += 1
            return TRUTH.delta(s, p)
        def bad(m, maximum):
            costs['model'] += 1
            return {'text': 'invalid fixture', 'input_tokens': 100, 'output_tokens': 3}
        with EpisodeJournal(root, IDENTITY) as journal:
            def interrupt(event):
                if event['kind'] == 'invalid_reply':
                    raise InjectedInterruption('invalid answer must remain charged')
            with self.assertRaises(InjectedInterruption):
                fixture(journal, oracle, bad, interrupt)
        with EpisodeJournal(root, IDENTITY) as journal:
            restored, _ = fixture(journal, oracle, bad)
        self.assertEqual(restored['status'], 'invalid_reply')
        self.assertFalse(restored['valid_final_answer'])
        self.assertEqual(restored['inference']['repairs'], 1)
        self.assertEqual(restored['inference']['output_tokens'], 6)
        self.assertEqual(costs, {'oracle': 8, 'model': 2})

    def test_prediction_event_is_on_disk_before_any_new_oracle_call(self):
        calls = []
        with EpisodeJournal(self.root/'causal', IDENTITY) as journal:
            def oracle(s, p):
                if len(calls) >= 8:
                    path = journal.root/'events'/f"{journal.cursor['events']-1:06d}.json"
                    event = json.loads(path.read_text(encoding='utf-8'))['event']
                    self.assertEqual(event['kind'], 'predictions_committed')
                    self.assertGreaterEqual(len(event['commitment']['predictions']), 2)
                calls.append(1)
                return TRUTH.delta(s, p)
            fixture(journal, oracle, scripted_reply)
        self.assertEqual(len(calls), 32)

    def test_sparse_active_results_reconstruct_without_any_new_measurements(self):
        root = self.root/'sparse'
        calls = []
        def oracle(s, p):
            calls.append(1)
            return TRUTH.delta(s, p)
        def run():
            with EpisodeJournal(root, IDENTITY) as journal:
                inputs = streams('development', 0)
                lab = Laboratory(journal.oracle(oracle), noise_sigma=1, noise_seed=inputs['noise_seed'])
                snapshots, events = run_search(lab, Grammar(), inputs['initial'], strategy='active', seed=inputs['search_seed'])
                episode = {'snapshots': {str(n): m.to_dict() for n, m in snapshots.items()}, 'events': events}
                journal.complete(episode)
                return canonical(episode)
        original = run()
        self.assertEqual(len(calls), 32)
        self.assertEqual(run(), original)
        self.assertEqual(len(calls), 32)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--crash-fixture':
        crash_child(sys.argv[2])
    else:
        unittest.main()
