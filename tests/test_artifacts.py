import json
import unittest

from util import ROOT

FIGURES = [
    "fig_coding_passrate.png",
    "fig_deep_vs_depth0.png",
    "fig_moe_cache.png",
    "fig_mtp_ab.png",
    "fig_mtp_runs.png",
    "fig_overview.png",
    "fig_prefill_vs_ctx.png",
]
DIAGRAMS = ["fig_setup.svg"]
TABLES = [
    "coding_eval.csv", "coding_overall.csv", "coding_summary.csv",
    "deep_vs_depth0.csv", "moe_cache.csv", "mtp_ab.csv", "mtp_blocks.csv",
    "mtp_comparison.csv", "mtp_runs.csv", "ncmoe_probe_29b.csv",
    "ncmoe_sweep_28b.csv", "prefill_vs_ctx.csv",
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
        for name in DIAGRAMS:
            path = ROOT / "figures" / name
            self.assertTrue(path.exists(), name)
            self.assertGreater(path.stat().st_size, 500, name)

    def test_traceability_report_is_published(self):
        path = ROOT / "results" / "traceability.txt"
        self.assertTrue(path.exists(), path)
        text = path.read_text()
        self.assertIn("result: OK", text)
        self.assertIn("reproducible: no", text)

    def test_provenance_manifests_exist(self):
        for rel in ("data/manifest.json", "results/manifest.json"):
            path = ROOT / rel
            self.assertTrue(path.exists(), rel)
            self.assertGreater(path.stat().st_size, 100, rel)

    def test_manifest_counts_only_kept_runs(self):
        counts = json.loads((ROOT / "data/manifest.json").read_text())["counts"]
        self.assertEqual(counts["mtp_runs"], 20)
        self.assertEqual(counts["coding_runs"], 33)
        self.assertEqual(counts["probe_configs"], 3)


if __name__ == "__main__":
    unittest.main()
