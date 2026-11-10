import unittest

from build_publication_curriculum import read
from train_mainline_student import task_plan


class MainlineTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=read('data/student_mainline_v1/manifest.json')

    def test_final_schedule_matches_answers_and_objective_weights(self):
        plans={g:task_plan(self.manifest,g,401) for g in 'ABC'}
        for i in range(1200):
            a,b,c=(plans[g][0][i] for g in 'ABC')
            self.assertEqual(a['episode_id'],b['episode_id'])
            self.assertEqual(b['episode_id'],c['episode_id'])
            self.assertEqual(b['tasks'][:4],c['tasks'][:4])
            self.assertTrue(all(t['key'].endswith('terminal08') for t in a['tasks']))
            for current in (a,b,c):
                self.assertAlmostEqual(sum(t['weight'] for t in current['tasks'] if t['kind']=='terminal'),1)
            for current in (b,c):
                self.assertAlmostEqual(sum(t['weight'] for t in current['tasks'] if t['kind']=='auxiliary'),1)
        for g in 'ABC':
            self.assertEqual(plans[g][1]['terminal_presentations'],4800)
        self.assertLess(abs(plans['B'][1]['input_tokens']/plans['C'][1]['input_tokens']-1),.01)


if __name__=='__main__':
    unittest.main()
