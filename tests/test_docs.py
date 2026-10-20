import json
import re
import subprocess
import sys
import unittest

from util import ROOT

LABELS = ("MEASURED", "REGENERATED", "REPRODUCIBLE", "ESTIMATED", "UNVERIFIED")


def text(name: str) -> str:
    return (ROOT / name).read_text()


class DocumentationTests(unittest.TestCase):
    def test_titles_are_specific(self):
        for name in ("ARTICLE.md", "README.md"):
            first = text(name).splitlines()[0]
            self.assertIn("RTX 3060", first, name)
            self.assertIn("64k", first, name)
        self.assertIn("RTX 3060", text("CITATION.cff"))

    def test_banned_absolutes_are_gone(self):
        body = text("ARTICLE.md").lower()
        for phrase in ("nobody runs", "measures nothing", "poor proxy"):
            self.assertNotIn(phrase, body)

    def test_every_figure_has_provenance(self):
        for name in ("ARTICLE.md", "README.md"):
            lines = text(name).splitlines()
            for index, line in enumerate(lines):
                if not re.search(r"!\[[^\]]*\]\(figures/", line):
                    continue
                nearby = "\n".join(lines[index + 1:index + 4])
                self.assertRegex(
                    nearby, r"\*Data:|\*Schematic",
                    f"{name}:{index + 1} has no provenance footer")

    def test_headline_rows_are_labelled(self):
        rows = [line for line in text("README.md").splitlines()
                if re.match(r"\|\s*[123]\s*\|", line)]
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertTrue(any(label in row for label in LABELS), row)

    def test_evidence_taxonomy_lives_in_evidence_doc(self):
        body = text("EVIDENCE.md")
        for label in LABELS:
            self.assertIn(f"**{label}**", body)
        self.assertIn("No run in this study is labelled REPRODUCIBLE", body)
        article = text("ARTICLE.md")
        for label in LABELS:
            self.assertNotIn(label, article)
        self.assertIn("# Appendix", article)
        self.assertIn("## A. Coding evaluation", article)
        self.assertIn("## B. Data, provenance, and reproduction", article)

    def test_public_reproduction_client_help(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "reproduce_mtp_ab.py"),
             "--help"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--off-url", result.stdout)
        self.assertIn("procedure", result.stdout.lower())

    def test_release_packager_help(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "package_release.py"),
             "--help"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--version", result.stdout)

    def test_claim_and_figure_maps_cover_headlines(self):
        claims = json.loads((ROOT / "data" / "claims.json").read_text())
        for cid in ("mtp_tg_off", "mtp_tg_on", "mtp_ci_half",
                    "cache_on", "depth0_tg", "coding_passes"):
            self.assertIn(cid, claims)
        figures = json.loads((ROOT / "data" / "figure-sources.json").read_text())
        for name in ("fig_mtp_ab.png", "fig_overview.png", "fig_setup.svg"):
            self.assertIn(name, figures)


if __name__ == "__main__":
    unittest.main()
