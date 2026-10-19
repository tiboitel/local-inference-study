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

if problems:
    print("TRACEABILITY FAILED")
    for p in problems:
        print("  -", p)
    sys.exit(1)

print(f"traceability OK: {len(EXPECTED)} tables, "
      f"{len(runs)} coding runs, {len(mp)} MTP runs")
