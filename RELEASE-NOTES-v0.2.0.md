# v0.2.0 — evidence and reproducibility release

This release freezes the article, result tables, sanitized raw traces, and the
verification tooling. It does **not** publish the private benchmark runner,
coding harness, model weights, or server logs.

## What is included

- Sanitized per-request MTP traces (`data/raw/mtp/`) and coding traces
  (`data/raw/coding/`).
- Generated result tables, including the unpaired MTP comparison and block means.
- Deterministic traceability report (`results/traceability.txt`).
- Provenance manifests with SHA-256 hashes for inputs and outputs.
- Figures, the setup schematic, and a dependency-free procedure-reproduction
  client (`scripts/reproduce_mtp_ab.py`).

## Evidence status

The headline MTP result is **MEASURED + REGENERATED**. The controlled expert-cache
and depth-0 rows are **MEASURED** on the 28B bench sibling and regenerated from a
hand-transcribed, non-self-contained ledger. No measured run is **REPRODUCIBLE**:
the runner and coding harness are private. See `EVIDENCE.md` and
`PROMPT-PROVENANCE.md`.

## Verification

```bash
python scripts/build_results.py --raw data/raw --out results
python scripts/make_figures.py --out /tmp/figures
python scripts/check_traceability.py --report results/traceability.txt
python scripts/manifest.py --check
python -m pytest -q
bash scripts/check_hygiene.sh
```

## Known gaps

- The exact prompt body, tokenized representation, and prompt hash are not
  available. The original model artifact hash was not captured.
- Run-time GPU driver and power state were not logged; the published values are
  post-hoc captures.
- The 128k live-profile results are not validated; only the 28B sibling supports
  them as estimates.
