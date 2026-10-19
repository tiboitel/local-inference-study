import csv
import math
import statistics
import unittest

from util import ROOT, load


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

    def test_mtp_comparison_is_generated_and_tbased(self):
        with (ROOT / "results" / "mtp_runs.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        off = [float(r["tg_t_s"]) for r in rows if r["arm"] == "off"]
        on = [float(r["tg_t_s"]) for r in rows if r["arm"] == "on"]
        with (ROOT / "results" / "mtp_comparison.csv").open() as fh:
            comp = next(csv.DictReader(fh))
        se = math.sqrt(statistics.stdev(off) ** 2 / len(off)
                       + statistics.stdev(on) ** 2 / len(on))
        half = self.br.t95(min(len(off), len(on))) * se
        self.assertAlmostEqual(float(comp["difference"]), 23.77, places=2)
        self.assertAlmostEqual(float(comp["se_diff"]), se, places=2)
        self.assertAlmostEqual(float(comp["ci95_halfwidth"]), half, places=2)
        self.assertEqual(comp["df"], "9")
        self.assertAlmostEqual(float(comp["df_welch"]), 9.66, places=2)
        self.assertIn("unpaired", comp["method"])

    def test_block_means_are_present(self):
        with (ROOT / "results" / "mtp_blocks.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual([r["block"] for r in rows], ["1", "2"])
        for row in rows:
            self.assertEqual((row["n_off"], row["n_on"]), ("5", "5"))
            self.assertGreater(float(row["difference"]), 0)


if __name__ == "__main__":
    unittest.main()
