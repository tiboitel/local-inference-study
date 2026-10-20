import re
import unittest
from pathlib import Path

from util import ROOT

ARTICLE = ROOT / "ARTICLE.md"
IMAGE_RE = re.compile(r"!\[Figure (\d+) — [^\]]+\]\((figures/[^)]+)\)")


class MarkdownRenderingTests(unittest.TestCase):
    def setUp(self):
        self.text = ARTICLE.read_text()

    def test_fenced_blocks_are_balanced(self):
        fences = self.text.count("```")
        self.assertEqual(fences % 2, 0, "unbalanced triple-backtick fences")
        self.assertGreaterEqual(fences, 14)

    def test_no_tilde_after_emphasis_opener(self):
        # GitHub leaves '*~ ...*' literal in this context; avoid the ambiguity.
        # '**~3x**' (strong) is fine, so only reject a single emphasis opener.
        self.assertIsNone(re.search(r"(?<!\*)\*~(?!\*)", self.text))

    def test_conceptual_blocks_are_fenced_text(self):
        blocks = [
            "Large prompt\n    ↓\nPrefill\n    ↓\nLong-context state\n    ↓\nDecode",
            "off → on → on → off",
            "--spec-type draft-mtp\n--spec-draft-n-max 2",
            "turbo4\nturbo3",
            "-c 65536\n--n-cpu-moe 20\n--spec-type draft-mtp\n"
            "--spec-draft-n-max 2\n--moe-cache on",
        ]
        for block in blocks:
            self.assertIn(f"```text\n{block}\n```", self.text)

    def test_figures_are_numbered_and_exist(self):
        matches = IMAGE_RE.findall(self.text)
        self.assertEqual([int(number) for number, _ in matches],
                         list(range(1, 8)))
        self.assertEqual(self.text.count("!["), 7)
        for number, relative in matches:
            self.assertTrue((ROOT / relative).is_file(), relative)
            self.assertIn(f"**Figure {number} —", self.text)

    def test_every_figure_has_a_text_caption(self):
        lines = self.text.splitlines()
        for index, line in enumerate(lines):
            if not line.startswith("![Figure "):
                continue
            nearby = "\n".join(lines[index + 1:index + 4])
            self.assertRegex(
                nearby, r"\*\*Figure \d+ — .+\.\*\*",
                f"ARTICLE.md:{index + 1} has no numbered caption")
            self.assertRegex(
                nearby, r"\*Data:|\*Schematic:",
                f"ARTICLE.md:{index + 1} has no provenance footer")


if __name__ == "__main__":
    unittest.main()
