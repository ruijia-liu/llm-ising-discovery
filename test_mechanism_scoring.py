import unittest

from discovery.active_student import run_active
from discovery.mechanism_scoring import independent_features, score_active
from discovery.paired_statistics import primary_comparison
from test_active_student import ActiveStudentTests, ScriptedModel


class ScoringTests(unittest.TestCase):
    def test_correct_early_finish_is_not_a_tested_idea(self):
        helper = ActiveStudentTests()
        session = helper.session()
        episode = run_active(session,ScriptedModel([helper.hypothesis(),{'action':'finish'}]))
        score = score_active(helper.truth,episode,independent_features(98701,0,16),0,{})
        self.assertTrue(score['full_recovery'])
        self.assertFalse(score['tested_idea_recovery'])
        self.assertTrue(all(r['full_recovery'] for r in score['budget_curve']))

    def test_precommitted_correct_structure_counts_only_after_measurement(self):
        helper = ActiveStudentTests(); session = helper.session()
        episode = run_active(session,ScriptedModel([helper.hypothesis(),helper.plan(),helper.hypothesis(),{'action':'finish'}]))
        score = score_active(helper.truth,episode,independent_features(98701,0,16),0,{})
        self.assertTrue(score['tested_idea_recovery'])
        self.assertFalse(score['competitive_idea_recovery'])  # field is incompatible with the visible horizontal data
        episode['events'] = [e for e in episode['events'] if e['kind']!='observations_revealed']
        self.assertFalse(score_active(helper.truth,episode,independent_features(98701,0,16),0,{})['tested_idea_recovery'])


class PairedStatisticsTests(unittest.TestCase):
    def rows(self):
        return [{'group':g,'seed':seed,'world_id':f'{level}-{i}','structure_id':str(i//2),
                 'level':level,'sigma':1,'mode':'active','score':g=='C'}
                for g in 'BC' for seed in (401,402,403) for level in ('combination','four_body') for i in range(4)]

    def test_pairing_cluster_counts_and_constant_known_effect(self):
        counts={'combination':(4,2),'four_body':(4,2)}
        result = primary_comparison(self.rows(),repetitions=100,expected_counts=counts)
        self.assertEqual(result['C_minus_B'],1)
        self.assertEqual(result['conditional_95_percentile_interval'],[1,1])
        self.assertEqual(result['cluster_counts'],{'combination':2,'four_body':2})
        self.assertTrue(result['consistent_process_advantage'])
        swapped = [{**r,'score':not r['score']} for r in self.rows()]
        self.assertEqual(primary_comparison(swapped,repetitions=100,expected_counts=counts)['C_minus_B'],-1)

    def test_missing_seed_duplicate_or_missing_pair_cannot_be_ignored(self):
        rows = self.rows()
        for bad in (rows[:-1],rows+[rows[0]],[r for r in rows if r['seed']!=403]):
            with self.assertRaises(ValueError):
                primary_comparison(bad,repetitions=10)

    def test_whole_missing_worlds_across_all_arms_are_rejected(self):
        rows=[r for r in self.rows() if r['world_id']!='combination-0']
        with self.assertRaises(ValueError):
            primary_comparison(rows,repetitions=10,expected_counts={'combination':(4,2),'four_body':(4,2)})


if __name__=='__main__':
    unittest.main()
