# v0.2.1 — editorial revision

This release revises the presentation of the study. **No measurements, result
tables, raw traces, or statistical claims changed.**

The title remains:

> **Long-Context Local LLM Inference on a 12 GB RTX 3060: Measuring MTP, Prefill, and VRAM at 64k**

## What changed

- `ARTICLE.md` now opens with an abstract and a direct test-configuration
  section instead of an audit preamble.
- The 29B primary track and 28B supporting track are separated structurally
  rather than labelled repeatedly in prose.
- Inline evidence labels were removed from the narrative. The taxonomy,
  evidence tiers, and claim ledger remain in `EVIDENCE.md`.
- The coding evaluation moved to **Appendix A**, keeping the main report focused
  on the inference-throughput question.
- Provenance and reproduction material moved to **Appendix B**.
- Mixed-configuration framing was reduced, and the MTP result is now the clear
  narrative centerpiece.
- Claim anchors in the article were preserved, so the traceability gate still
  checks each headline value against its table cell.
- Figures now carry explicit numbers in both alt text and visible captions, so a
  text-only export still identifies the figure even if the image is stripped.
- Conceptual blocks (request path, interleaved arms, flags) are normalized as
  fenced `text` blocks with blank lines around them.
- A Markdown regression test checks fence balance, expected blocks, image
  existence, figure numbering and provenance captions.

## Verification

```bash
python scripts/build_results.py --raw data/raw --out results
python scripts/make_figures.py --out /tmp/figures
python scripts/check_traceability.py --report results/traceability.txt
python scripts/manifest.py --check
python -m pytest -q
bash scripts/check_hygiene.sh
```

The test suite also checks that article code blocks stay balanced, the expected
conceptual blocks remain fenced, and every numbered figure has a visible caption
and provenance footer.

## Relationship to v0.2.0

`v0.2.0` remains the frozen evidence release. `v0.2.1` supersedes it for reading
and citation because the article is clearer, while the underlying data are
identical.
