import math
import statistics
import unittest

from util import load


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.br = load("scripts/build_results.py", "build_results_stats")

    def test_student_t_multiplier(self):
        self.assertEqual(self.br.t95(10), 2.262)
        self.assertEqual(self.br.t95(3), 4.303)
        self.assertEqual(self.br.t95(500), 1.96)  # z fallback off-table

    def test_sem_uses_sample_stdev(self):
        self.assertAlmostEqual(self.br.sem([1, 2, 3]),
                               statistics.stdev([1, 2, 3]) / math.sqrt(3))
        self.assertIsNone(self.br.sem([1]))
        self.assertIsNone(self.br.sem([]))

    def test_mean_ignores_missing(self):
        self.assertEqual(self.br.mean([1, None, 3]), 2.0)
        self.assertIsNone(self.br.mean([None]))

    def test_wilson_interval_matches_published_bounds(self):
        lo, hi = self.br.wilson(19, 33)
        self.assertAlmostEqual(lo, 0.4081, places=3)
        self.assertAlmostEqual(hi, 0.7276, places=3)
        self.assertEqual(self.br.wilson(0, 0), (None, None))

    def test_cluster_bootstrap_is_deterministic(self):
        groups = [[1, 1, 1], [0, 0, 0], [1, 0, 0]]
        a = self.br.cluster_bootstrap(groups, self.br.cluster_run_rate)
        b = self.br.cluster_bootstrap(groups, self.br.cluster_run_rate)
        self.assertEqual(a, b)

    def test_cluster_bootstrap_matches_published_coding_bounds(self):
        # per-task pass counts from results/coding_summary.csv (task_id order)
        counts = [0, 0, 3, 0, 1, 0, 3, 3, 3, 3, 3]
        groups = [[1] * c + [0] * (3 - c) for c in counts]
        lo, hi = self.br.cluster_bootstrap(groups, self.br.cluster_run_rate)
        self.assertAlmostEqual(round(lo, 4), 0.2727, places=4)
        self.assertAlmostEqual(round(hi, 4), 0.8485, places=4)


if __name__ == "__main__":
    unittest.main()
