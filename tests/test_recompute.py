import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import fixture
from util import ROOT, load


class RecomputeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = fixture.make_repo(Path(self.tmp.name))
        self.raw = self.root / "data" / "raw"
        self.results = self.root / "results"
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_results.py"),
             "--raw", str(self.raw),
             "--sources", str(self.root / "data" / "sources-bench.json"),
             "--out", str(self.results)],
            cwd=ROOT, check=True, capture_output=True)
        self.ct = load("scripts/check_traceability.py", "check_traceability")
        self.ct.ROOT = self.root
        self.ct.DATA = self.root / "data"
        self.ct.RES = self.results
        self.ct.RAW = self.raw
        self.ct.RAW_MTP = self.raw / "mtp"
        self.ct.RAW_CODING = self.raw / "coding"
        self.ct.RAW_P4 = self.raw / "p4"
        self.mtp = {p.name: json.loads(p.read_text())
                    for p in sorted((self.raw / "mtp").glob("*-b*-r*.json"))}
        self.coding = {p.name: json.loads(p.read_text())
                       for p in sorted((self.raw / "coding").glob("run-*.json"))}

    def tearDown(self):
        self.tmp.cleanup()

    def test_mtp_recompute_matches_built_tables(self):
        runs, agg, comp, blocks = self.ct.recompute_mtp(self.mtp)
        self.assertEqual(len(runs), 20)
        self.assertEqual([a["arm"] for a in agg], ["off", "on"])
        self.assertEqual([a["n_kept"] for a in agg], [10, 10])
        self.assertAlmostEqual(agg[0]["tg_mean"], 10.3, places=2)
        self.assertAlmostEqual(agg[1]["tg_mean"], 20.3, places=2)
        self.assertEqual(comp[0]["df"], 9)
        self.assertEqual(blocks[0]["n_off"], 5)
        self.ct.problems.clear()
        self.ct.stage_recompute(self.mtp, self.coding)
        self.assertEqual(self.ct.problems, [])

    def test_coding_recompute_matches_built_tables(self):
        runs, tasks, overall = self.ct.recompute_coding(self.coding)
        self.assertEqual(len(runs), 33)
        self.assertEqual(len(tasks), 11)
        self.assertEqual(overall["runs"], 33)
        self.assertEqual(overall["passes"], 30)
        self.assertEqual(overall["tests_run"], 132)
        self.assertEqual(overall["tasks_total"], 11)
        self.assertLessEqual(overall["cluster_lo"], overall["cluster_hi"])
        self.assertLessEqual(overall["cluster_all_lo"], overall["cluster_all_hi"])

    def test_schema_and_invariants_accept_fixture(self):
        self.ct.problems.clear()
        self.ct.stage_schema(self.mtp, self.coding)
        self.ct.stage_invariants(self.mtp, self.coding)
        self.assertEqual(self.ct.problems, [])

    def test_missing_raw_trace_is_detected(self):
        self.mtp.pop("on-b1-r1.json")
        self.ct.problems.clear()
        self.ct.stage_integrity(self.mtp, self.coding)
        self.assertTrue(any("MTP raw/runs mismatch" in p
                            for p in self.ct.problems))


if __name__ == "__main__":
    unittest.main()
