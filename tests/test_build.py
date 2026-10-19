import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import fixture
from util import ROOT, load

SCRIPT = ROOT / "scripts" / "build_results.py"


def digest(directory: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.glob("*.csv"))}


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = fixture.make_repo(Path(self.tmp.name))
        self.raw = self.root / "data" / "raw"
        self.sources = self.root / "data" / "sources-bench.json"
        self.br = load("scripts/build_results.py", "build_results_build")

    def tearDown(self):
        self.tmp.cleanup()

    def run_build(self, out: Path):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--raw", str(self.raw),
             "--sources", str(self.sources), "--out", str(out)],
            cwd=ROOT, capture_output=True, text=True)

    def test_validate_accepts_fixture(self):
        self.assertEqual(self.br.validate_inputs(self.raw, self.sources), [])

    def test_validate_rejects_missing_trace(self):
        (self.raw / "mtp" / "on-b1-r1.json").unlink()
        problems = self.br.validate_inputs(self.raw, self.sources)
        self.assertTrue(any("expected 20" in p for p in problems))

    def test_validate_rejects_warm_cache_and_non_finite(self):
        path = self.raw / "mtp" / "off-b1-r1.json"
        d = json.loads(path.read_text())
        d["timings"]["cache_n"] = 1
        d["timings"]["predicted_per_second"] = None
        path.write_text(json.dumps(d))
        problems = self.br.validate_inputs(self.raw, self.sources)
        self.assertTrue(any("not cold" in p for p in problems))
        self.assertTrue(any("not finite" in p for p in problems))

    def test_validate_rejects_missing_oom_evidence(self):
        path = self.raw / "p4" / "probe-results.txt"
        path.write_text(path.read_text().replace("OOM_cand17=allocating 128.00 MiB: out of memory\n", ""))
        problems = self.br.validate_inputs(self.raw, self.sources)
        self.assertTrue(any("OOM_cand17" in p for p in problems))

    def test_build_is_deterministic(self):
        first, second = self.root / "out1", self.root / "out2"
        self.assertEqual(self.run_build(first).returncode, 0)
        self.assertEqual(self.run_build(second).returncode, 0)
        self.assertEqual(digest(first), digest(second))
        self.assertEqual(len(digest(first)), 10)

    def test_bad_input_leaves_output_untouched(self):
        out = self.root / "out"
        self.assertEqual(self.run_build(out).returncode, 0)
        before = digest(out)
        (self.raw / "mtp" / "on-b1-r1.json").write_text("{broken")
        result = self.run_build(out)
        self.assertEqual(result.returncode, 2)
        self.assertIn("INPUT VALIDATION FAILED", result.stdout)
        self.assertEqual(digest(out), before)


if __name__ == "__main__":
    unittest.main()
