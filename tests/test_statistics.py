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


if __name__ == "__main__":
    unittest.main()
