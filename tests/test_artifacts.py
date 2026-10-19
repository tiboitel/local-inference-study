import unittest

from util import ROOT

FIGURES = [
    "fig_coding_passrate.png",
    "fig_deep_vs_depth0.png",
    "fig_moe_cache.png",
    "fig_mtp_ab.png",
    "fig_prefill_vs_ctx.png",
]
TABLES = [
    "coding_eval.csv", "coding_overall.csv", "coding_summary.csv",
    "deep_vs_depth0.csv", "moe_cache.csv", "mtp_ab.csv", "mtp_runs.csv",
    "ncmoe_probe_29b.csv", "ncmoe_sweep_28b.csv", "prefill_vs_ctx.csv",
]


class ArtifactTests(unittest.TestCase):
    def test_expected_result_tables_exist(self):
        for name in TABLES:
            path = ROOT / "results" / name
            self.assertTrue(path.exists(), name)
            self.assertGreater(path.stat().st_size, 20, name)

    def test_expected_figures_exist_and_are_non_empty(self):
        for name in FIGURES:
            path = ROOT / "figures" / name
            self.assertTrue(path.exists(), name)
            self.assertGreater(path.stat().st_size, 1000, name)

    def test_provenance_manifests_exist(self):
        for rel in ("data/manifest.json", "results/manifest.json"):
            path = ROOT / rel
            self.assertTrue(path.exists(), rel)
            self.assertGreater(path.stat().st_size, 100, rel)


if __name__ == "__main__":
    unittest.main()
