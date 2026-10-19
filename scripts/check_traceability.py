#!/usr/bin/env python3
"""Verify that every figure in results/ traces to a raw source.

    python scripts/check_traceability.py   # exit 0 == release gate passed

Checks, in order:
  1. every expected results/ table exists;
  2. every row of a derived table names a raw trace that exists;
  3. hand-transcribed bench rows carry a provenance section;
  4. aggregate tables agree with their detail tables.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RAW = ROOT / "data" / "raw"

problems: list[str] = []


def load(name):
    p = RES / name
    if not p.exists():
        problems.append(f"missing result table: {name}")
        return []
    with p.open() as fh:
        return list(csv.DictReader(fh))


EXPECTED = [
    "mtp_runs.csv", "mtp_ab.csv", "ncmoe_probe_29b.csv", "ncmoe_sweep_28b.csv",
    "moe_cache.csv", "prefill_vs_ctx.csv", "deep_vs_depth0.csv",
    "coding_eval.csv", "coding_summary.csv", "coding_overall.csv",
]

for name in EXPECTED:
    if not (RES / name).exists():
        problems.append(f"missing result table: {name}")

# 2a. MTP per-run rows -> raw traces
for r in load("mtp_runs.csv"):
    fn = f"{r['arm']}-b{r['block']}-r{r['run']}.json"
    if not (RAW / "mtp" / fn).exists():
        problems.append(f"mtp_runs.csv row has no raw trace: {fn}")

# 2b. coding per-run rows -> raw traces
for r in load("coding_eval.csv"):
    if not (RAW / "coding" / r["file"]).exists():
        problems.append(f"coding_eval.csv row has no raw trace: {r['file']}")

# 2c. probe rows -> probe-results.txt
probe_txt = (RAW / "p4" / "probe-results.txt")
probe_body = probe_txt.read_text() if probe_txt.exists() else ""
for r in load("ncmoe_probe_29b.csv"):
    if f"PROBE_{r['config']}=" not in probe_body:
        problems.append(f"probe row {r['config']} not in probe-results.txt")

# 3. bench tables -> provenance section
for name in ("ncmoe_sweep_28b.csv", "moe_cache.csv", "prefill_vs_ctx.csv",
             "deep_vs_depth0.csv"):
    for r in load(name):
        if not (r.get("section") or "").strip():
            problems.append(f"{name} row lacks a provenance section: {r}")

# 4. aggregates agree with detail
overall = load("coding_overall.csv")
summary = load("coding_summary.csv")
runs = load("coding_eval.csv")
if overall:
    o = overall[0]
    s_passes = sum(int(r["passes"]) for r in summary)
    s_runs = sum(int(r["runs"]) for r in summary)
    if int(o["passes"]) != s_passes:
        problems.append(f"coding_overall passes {o['passes']} != summary {s_passes}")
    if int(o["runs"]) != s_runs or s_runs != len(runs):
        problems.append("coding run counts disagree across tables")
    k, n = int(o["passes"]), int(o["runs"])
    if abs(float(o["pass_rate"]) - round(k / n, 4)) > 1e-9:
        problems.append("coding_overall pass_rate does not match passes/runs")

ab = load("mtp_ab.csv")
mp = load("mtp_runs.csv")
if ab:
    for arm in ("off", "on"):
        arm_rows = [r for r in mp if r["arm"] == arm]
        agg = [r for r in ab if r["arm"] == arm]
        if agg and int(agg[0]["n_kept"]) != len(arm_rows):
            problems.append(f"mtp_ab {arm} n_kept != mtp_runs count")

# 5. the numbers written in the article match the result tables (not just that
#    the cited files exist)
ART = (ROOT / "ARTICLE.md").read_text()
overall = (load("coding_overall.csv") or [{}])[0]
ab_by_arm = {r["arm"]: r for r in ab}
dd = {r["mode"]: r for r in load("deep_vs_depth0.csv")}

expected: dict[str, str] = {}
if overall:
    k, n = int(overall["passes"]), int(overall["runs"])
    tp, tr = int(overall["tests_passed"]), int(overall["tests_run"])
    ck, cn = int(overall["classeval_passes"]), int(overall["classeval_runs"])
    hk, hn = int(overall["humaneval_passes"]), int(overall["humaneval_runs"])
    expected.update({
        f"{k}/{n}": "run-level completion",
        f"{tp}/{tr}": "test-method pass",
        f"{ck}/{cn}": "ClassEval-only passes (must be 13/27, not 15/27)",
        f"{hk}/{hn}": "single-function floor",
        f"{k / n * 100:.1f}": "run pass rate",
        f"{tp / tr * 100:.1f}": "test-method pass rate",
        f"{ck / cn * 100:.1f}": "ClassEval-only rate",
        f"{float(overall['wilson_lo']) * 100:.1f}": "Wilson lower bound",
        f"{float(overall['wilson_hi']) * 100:.1f}": "Wilson upper bound",
    })
if ab_by_arm:
    off_tg = float(ab_by_arm["off"]["tg_mean"])
    on_tg = float(ab_by_arm["on"]["tg_mean"])
    vram_delta = int(round(float(ab_by_arm["on"]["vram_peak_mib"])
                         - float(ab_by_arm["off"]["vram_peak_mib"])))
    expected.update({
        f"{off_tg:.2f}": "MTP-off deep TG",
        f"{on_tg:.2f}": "MTP-on deep TG",
        f"{(on_tg - off_tg) / off_tg * 100:.1f}": "MTP decode delta (%)",
        f"{float(ab_by_arm['on']['accept_mean']):.3f}": "MTP acceptance",
        str(vram_delta): "MTP VRAM cost (MiB)",
        f"{float(ab_by_arm['off']['pp_mean']):.2f}": "MTP-off prefill",
        f"{float(ab_by_arm['on']['pp_mean']):.2f}": "MTP-on prefill",
    })
if dd:
    expected[f"{float(dd['depth0_llama_bench']['tg_t_s']):.1f}"] = "depth-0 TG"
    expected[f"{float(dd['deep_server_measured']['tg_t_s']):.1f}"] = "deep-context TG"

# cache / prefill literals must exist in both the CSV and the article
cache_pp = {r["pp_t_s"] for r in load("moe_cache.csv")}
for v in ("308.5", "775.7"):
    if v not in cache_pp:
        problems.append(f"moe_cache.csv missing expected prefill {v}")
    if v not in ART:
        problems.append(f"ARTICLE.md missing cache prefill {v}")
ctx_pp = {r["pp_t_s"] for r in load("prefill_vs_ctx.csv")}
if "859" not in ctx_pp:
    problems.append("prefill_vs_ctx.csv missing 859 t/s at 1k")
for v in ("859", "109"):
    if v not in ART:
        problems.append(f"ARTICLE.md missing prefill-vs-context value {v}")

for s, label in expected.items():
    if s not in ART:
        problems.append(f"ARTICLE.md does not contain {s!r} from the tables ({label})")

if problems:
    print("TRACEABILITY FAILED")
    for p in problems:
        print("  -", p)
    sys.exit(1)

print(f"traceability OK: {len(EXPECTED)} tables, "
      f"{len(runs)} coding runs, {len(mp)} MTP runs")
