# Evidence and release audit

This study separates **what was measured**, **what can be rebuilt**, and **what an
independent operator can repeat**. Those are different claims, and conflating
them is the easiest way to overstate the work.

## Evidence labels

A result may carry more than one label.

| Label | Meaning in this repository |
|---|---|
| **MEASURED** | Directly recorded observation with an identifiable raw trace or cited source row. |
| **REGENERATED** | The published table/figure can be rebuilt from committed inputs with the supplied scripts. |
| **REPRODUCIBLE** | An independent operator can repeat the experiment from the published procedure and artifacts. |
| **ESTIMATED** | Inferred from another measurement, a sibling configuration, or an incomplete observation. |
| **UNVERIFIED** | The evidence or procedure needed to check the claim is not available. |

**No run in this study is labelled REPRODUCIBLE.** The benchmark runner and the
coding harness are private. The raw traces are auditable and every table is
regenerated from them, but the original requests cannot be replayed from a fresh
clone. `scripts/reproduce_mtp_ab.py` publishes the *procedure* for the central
A/B comparison; it does not reproduce the exact prompt family or magnitudes.

## Label matrix

| Published result | Labels | Trace / source | Rebuild |
|---|---|---|---|
| MTP decode 14.79 → 38.56 t/s at 64k | MEASURED, REGENERATED | `data/raw/mtp/` | `results/mtp_ab.csv`, `results/mtp_comparison.csv` |
| MTP per-run distribution and block means | MEASURED, REGENERATED | `data/raw/mtp/` | `results/mtp_runs.csv` |
| Depth-0 46.0 t/s vs deep 15.3 t/s | MEASURED (28B bench sibling, two configurations) | `data/sources-bench.json` §06/§07 | `results/deep_vs_depth0.csv` |
| Expert cache 308.5 → 775.7 t/s at 1k | MEASURED (28B sibling), REGENERATED from a non-self-contained ledger | `data/sources-bench.json` §04 | `results/moe_cache.csv` |
| 64k prefill with/without cache | MEASURED (28B sibling), partially ESTIMATED (timeout upper bound) | `data/sources-bench.json` §06/§07 | `results/prefill_vs_ctx.csv`, `results/moe_cache.csv` |
| `ncmoe=17` OOM at 64k | MEASURED | `data/raw/p4/probe-results.txt` | `results/ncmoe_probe_29b.csv` |
| Coding 19/33 runs | MEASURED, REGENERATED | `data/raw/coding/` | `results/coding_overall.csv` |
| 128k on the live 29B MTP profile | ESTIMATED, not validated | 28B bench sibling only | not published as a live result |

The bench-derived rows are hand-transcribed from a dated source clip and carry a
section reference in `data/sources-bench.json`. Regenerating their CSV does not
make the underlying measurement self-contained; it re-derives the table from a
committed ledger.

## Claim and figure provenance

- **Claim registry.** [`data/claims.json`](data/claims.json) binds each marked
  headline value to a table, row selector, field, and display format. Markers in
  the article and README are checked against the recomputed tables; a number in
  the wrong sentence now fails the gate.
- **Figure map.** [`data/figure-sources.json`](data/figure-sources.json) lists the
  source table(s) for every rendered figure. The gate fails on a missing source,
  an unmapped image, or an unreadable PNG/SVG.
- **Visual limits.** Figure *pixels* are not compared across platforms. The gate
  verifies source tables, mapping, and file validity; it does not prove that a
  chart looks correct.
- **Comparison method.** The MTP difference is unpaired (independent nonces),
  uses a Welch standard error and conservative `t(df=9)=2.262`, and is generated
  in [`results/mtp_comparison.csv`](results/mtp_comparison.csv). Block means are
  in [`results/mtp_blocks.csv`](results/mtp_blocks.csv).

## Release checklist

Status values: **DONE**, **PARTIAL**, **NOT DONE**.

| # | Item | Status | Evidence / gap |
|---|---|---|---|
| 1 | Publish all retained measurements, including failures and outliers | **DONE** | 20 MTP runs, 33 coding runs (19 pass / 14 fail), 3 load probes; the slow MTP outlier is retained. |
| 2 | Document units, metric definitions, run IDs and configuration fingerprints | **PARTIAL** | Units and definitions are in `METHODOLOGY.md`; run IDs are filenames; model and prompt hashes are `null` because they were not captured at run time. |
| 3 | Link every headline value to its raw source and transformation | **DONE** | `data/traceability.md`, per-figure provenance footers, `results/traceability.txt`. |
| 4 | Publish the runner or a minimal equivalent reproduction harness | **PARTIAL** | `scripts/reproduce_mtp_ab.py` reproduces the client-side A/B procedure; the original runner and coding harness remain private. |
| 5 | Pin model revision, model-file hash, runtime commit, flags and dependencies | **PARTIAL** | Runtime commit, flags, Python and matplotlib are pinned; model hash is unavailable and the GPU driver record is post-hoc, not run-time. |
| 6 | Document GPU driver, OS, power settings and background-load controls | **PARTIAL** | OS and post-hoc driver/power capture are in `HARDWARE-AND-RUNTIME.md`; the run-time driver version and background-load controls were not recorded. |
| 7 | Provide a script that independently recalculates statistics and intervals | **DONE** | `scripts/check_traceability.py` re-derives every table, including the task-clustered bootstrap, independently of the builder. |
| 8 | Show individual measurements and the A/B difference | **DONE, design-limited** | Every run is in `results/mtp_runs.csv`; the unpaired difference interval and block means are generated in `results/mtp_comparison.csv` and `results/mtp_blocks.csv`. The arms are **not paired** (unique nonce per request), so a paired interval cannot be claimed. |
| 9 | Document the source of the noise floor and handling of exploratory sweeps | **PARTIAL** | The ~4% threshold is documented as an operational heuristic; the control series behind it was not retained. Exploratory sweeps are labelled as hand-transcribed bench rows. |
| 10 | Run all data-generation and traceability checks from a clean checkout | **DONE (local)** | A clean clone passed the documented build, figure, gate, manifest, test, hygiene, and no-drift checks. `.github/workflows/validate.yml` now has a matching `documented-reproduction` job; it has not yet run on GitHub. |
| 11 | Record exact commands, expected outputs and environment-specific differences | **PARTIAL** | `REPRODUCTION.md` lists commands; `results/traceability.txt` is the expected gate output. Figure bytes are environment-specific and are not compared across platforms. |
| 12 | Tag the article, results and raw data as a versioned release | **DONE** | Annotated tag `v0.2.0`; `scripts/package_release.py` writes a deterministic tarball and SHA-256 sidecar. No DOI. |

## Known evidence gaps

- **Model artifact hash:** not captured at run time; no artifact was available at
  publication time to hash post-hoc. It cannot be backfilled without retaining
  the exact file and is recorded as `null`.
- **Prompt hash:** the original prompt family is private; the traces record
  prompt token counts and a per-run nonce, not the request body.
- **Run-time driver:** the driver and power limit below were captured after the
  fact on the same host and are labelled as post-hoc.
- **Noise floor:** `~4%` is an operational threshold, not a confidence interval.
  The per-run SEM and Student-t intervals are computed from published runs.
- **Independent reproduction:** not available for the measured runs. The public
  client is a procedure reproduction and should be compared by direction and
  uncertainty, not by exact throughput.

## Commands

```bash
python scripts/build_results.py
python scripts/check_traceability.py --report results/traceability.txt
python scripts/manifest.py --check
python scripts/make_figures.py
python -m pytest -q
bash scripts/check_hygiene.sh
python scripts/package_release.py --version 0.2.0
```

The first four commands rebuild every published table and figure; the gate also
recomputes the tables independently and checks the claims in `ARTICLE.md`.
