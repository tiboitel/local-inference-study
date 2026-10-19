# Traceability

Every number published in this repo resolves to a raw result. This table is the
map, and `scripts/check_traceability.py` enforces it (exit 0 is the release
gate). The deterministic gate output is committed as
[results/traceability.txt](../results/traceability.txt); evidence labels and
gaps are in [EVIDENCE.md](../EVIDENCE.md).

| Table | Rows | Raw source | Enforced by |
|---|---|---|---|
| `results/mtp_runs.csv` | 20 kept runs | `data/raw/mtp/{arm}-b{block}-r{run}.json` | filename existence check |
| `results/mtp_ab.csv` | 2 arms | aggregate of `mtp_runs.csv` | aggregate agreement check |
| `results/ncmoe_probe_29b.csv` | 3 configs | `data/raw/p4/probe-results.txt` | `PROBE_<config>=` presence |
| `results/ncmoe_sweep_28b.csv` | 5 configs | bench clip 2026-10-18 §04 | provenance section present |
| `results/moe_cache.csv` | 8 configs | bench clip 2026-10-18 §04/§06/§07 | provenance section present |
| `results/prefill_vs_ctx.csv` | 3 points | bench clip 2026-10-18 §06 | provenance section present |
| `results/deep_vs_depth0.csv` | 3 rows | bench clip 2026-10-18 §06/§07 | provenance section present |
| `results/coding_eval.csv` | 33 runs | `data/raw/coding/run-*.json` | filename existence check |
| `results/coding_summary.csv` | 11 tasks | aggregate of `coding_eval.csv` | aggregate agreement check |
| `results/coding_overall.csv` | 1 row | aggregate of `coding_eval.csv` | aggregate agreement check, incl. task-level counts and the seeded task-cluster bootstrap |

Legend:

- **Raw traces** (`data/raw/`, committed) are the per-run JSON files written by
  the harness. They are sanitized before commit (no host paths). Run logs and
  model weights are not committed, and the harness itself is not published — so
  the traces can be audited but not regenerated from this repository alone.
- **Hand-transcribed rows** are the bench-derived tables only. Their source is
  the dated benchmark clip, and `data/sources-bench.json` carries the section for
  each row so a reviewer can re-check it. This is the only hand-entered data in
  the repo; everything else is produced by `scripts/build_results.py`.
- **Task-clustered bootstrap.** The coding interval resamples the 11 tasks (not
  the 33 runs) with a fixed seed and draw count in `scripts/build_results.py`,
  and `scripts/check_traceability.py` re-derives it independently, so it is
  reproducible byte-for-byte like every other published number.

Adding a row to a hand-transcribed table requires naming its section in
`sources-bench.json`; the traceability check fails any row without one.
