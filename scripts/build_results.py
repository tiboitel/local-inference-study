#!/usr/bin/env python3
"""Build the result tables from raw run traces.

This is the only place a number in results/ is produced. Do not hand-edit a CSV;
fix the raw trace or the parser and re-run.

    python scripts/build_results.py [--raw data/raw] [--out results]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import statistics
import tempfile
from pathlib import Path


def as_bool(x) -> int:
    return 1 if x else 0


# --- statistics -----------------------------------------------------------

def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def sem(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    return statistics.stdev(xs) / math.sqrt(len(xs))


def sd(xs):
    """Sample standard deviation; None below two observations."""
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    return statistics.stdev(xs)


def seconds(ms):
    """Server timings report milliseconds; publish seconds (3 dp)."""
    if not isinstance(ms, (int, float)):
        return None
    return round(ms / 1000.0, 3)


# two-sided 95% Student-t critical values by degrees of freedom (n - 1)
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
        19: 2.093, 20: 2.086}


def t95(n: int) -> float:
    """Two-sided 95% interval multiplier for n samples (z fallback off-table)."""
    return _T95.get(n - 1, 1.96)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cluster_bootstrap(groups, stat, seed: int = 0, draws: int = 10000):
    """Percentile bootstrap over clusters (tasks), not individual runs.

    `groups` is a list of per-cluster value lists. Resampling whole clusters
    keeps the within-task dependence that a run-level Wilson interval ignores.
    Fixed seed and draw count make the result reproducible byte-for-byte.
    """
    if not groups:
        return (None, None)
    rng = random.Random(seed)
    m = len(groups)
    out = sorted(stat([groups[rng.randrange(m)] for _ in range(m)])
                 for _ in range(draws))
    return (out[int(0.025 * draws)], out[int(0.975 * draws)])


def cluster_run_rate(groups) -> float:
    return sum(sum(g) for g in groups) / sum(len(g) for g in groups)


def cluster_all_pass_share(groups) -> float:
    return sum(1 for g in groups if all(g)) / len(groups)


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"  wrote {path.relative_to(path.parents[1])} ({len(rows)} rows)")


# --- MTP A/B --------------------------------------------------------------

def build_mtp(raw: Path, out: Path):
    run_dir = raw / "mtp"
    runs = []
    for p in sorted(run_dir.glob("*-b*-r*.json")):
        if p.name.startswith("summary-"):
            continue
        arm, block, run = p.stem.split("-")  # off-b1-r1
        d = json.loads(p.read_text())
        t = d["timings"]
        draft_n = t.get("draft_n") or 0
        accepted = t.get("draft_n_accepted")
        accept = (accepted / draft_n) if (draft_n and accepted is not None) else None
        runs.append({
            "arm": arm,
            "block": int(block[1:]),
            "run": int(run[1:]),
            "prompt_n": t.get("prompt_n"),
            "pp_t_s": t.get("prompt_per_second"),
            "tg_t_s": t.get("predicted_per_second"),
            "accept": accept,
            "cache_n": t.get("cache_n"),
            "vram_mib": (d.get("peaks") or {}).get("vram_mib"),
            "ram_mib": (d.get("peaks") or {}).get("ram_used_mib"),
            "finish_reason": d.get("finish_reason"),
            # request-level timing fields: one row is one HTTP request
            "predicted_n": t.get("predicted_n"),
            "prompt_s": seconds(t.get("prompt_ms")),
            "decode_s": seconds(t.get("predicted_ms")),
            "wall_s": d.get("wall_s"),
        })
    runs.sort(key=lambda r: (r["arm"], r["block"], r["run"]))

    write_csv(out / "mtp_runs.csv", runs,
              ["arm", "block", "run", "prompt_n", "pp_t_s", "tg_t_s", "accept",
               "cache_n", "vram_mib", "ram_mib", "finish_reason",
               "predicted_n", "prompt_s", "decode_s", "wall_s"])

    agg = []
    for arm in ("off", "on"):
        rs = [r for r in runs if r["arm"] == arm]
        tg_sem = sem([r["tg_t_s"] for r in rs])
        pp_sd, tg_sd, wall_sd = (sd([r[k] for r in rs])
                                 for k in ("pp_t_s", "tg_t_s", "wall_s"))
        agg.append({
            "arm": arm,
            "n_kept": len(rs),
            "prompt_n_mean": round(mean([r["prompt_n"] for r in rs]), 1),
            "pp_mean": round(mean([r["pp_t_s"] for r in rs]), 2),
            "pp_sd": round(pp_sd, 2) if pp_sd is not None else None,
            "pp_sem": round(sem([r["pp_t_s"] for r in rs]), 2),
            "tg_mean": round(mean([r["tg_t_s"] for r in rs]), 2),
            "tg_sd": round(tg_sd, 2) if tg_sd is not None else None,
            "tg_sem": round(tg_sem, 2),
            "tg_ci95": round(t95(len(rs)) * tg_sem, 2) if tg_sem else None,
            "accept_mean": (lambda m: round(m, 3) if m is not None else None)(
                mean([r["accept"] for r in rs])),
            "accept_sd": (round(statistics.stdev([r["accept"] for r in rs if r["accept"] is not None]), 3)
                          if len([r for r in rs if r["accept"] is not None]) > 1 else None),
            "vram_peak_mib": max(r["vram_mib"] for r in rs),
            "ram_peak_mib": round(max(r["ram_mib"] for r in rs), 0),
            # how the request time is spent: prefill, decode, total wall time
            "predicted_n_mean": round(mean([r["predicted_n"] for r in rs]), 1),
            "prompt_s_mean": round(mean([r["prompt_s"] for r in rs]), 2),
            "decode_s_mean": round(mean([r["decode_s"] for r in rs]), 2),
            "wall_s_mean": round(mean([r["wall_s"] for r in rs]), 2),
            "wall_s_sd": round(wall_sd, 2) if wall_sd is not None else None,
        })
    write_csv(out / "mtp_ab.csv", agg,
              ["arm", "n_kept", "prompt_n_mean", "pp_mean", "pp_sd", "pp_sem",
               "tg_mean", "tg_sd", "tg_sem", "tg_ci95", "accept_mean",
               "accept_sd", "vram_peak_mib", "ram_peak_mib",
               "predicted_n_mean", "prompt_s_mean", "decode_s_mean",
               "wall_s_mean", "wall_s_sd"])

    off = [r["tg_t_s"] for r in runs if r["arm"] == "off"]
    on = [r["tg_t_s"] for r in runs if r["arm"] == "on"]
    comparison = []
    if len(off) > 1 and len(on) > 1:
        n1, n2 = len(off), len(on)
        s1, s2 = statistics.stdev(off), statistics.stdev(on)
        se = math.sqrt(s1 * s1 / n1 + s2 * s2 / n2)
        df_welch = ((s1 * s1 / n1 + s2 * s2 / n2) ** 2 /
                    ((s1 * s1 / n1) ** 2 / (n1 - 1) +
                     (s2 * s2 / n2) ** 2 / (n2 - 1)))
        df = min(n1, n2) - 1  # conservative; no SciPy dependency
        tcrit = t95(df + 1)
        diff = mean(on) - mean(off)
        half = tcrit * se
        comparison.append({
            "comparison": "tg", "arm_a": "off", "arm_b": "on",
            "n_a": n1, "n_b": n2,
            "mean_a": round(mean(off), 2), "mean_b": round(mean(on), 2),
            "sd_a": round(s1, 2), "sd_b": round(s2, 2),
            "difference": round(diff, 2),
            "difference_pct": round(diff / mean(off) * 100, 1),
            "se_diff": round(se, 2), "df": df,
            "df_welch": round(df_welch, 2), "tcrit": round(tcrit, 3),
            "ci95_halfwidth": round(half, 2),
            "ci95_lo": round(diff - half, 2),
            "ci95_hi": round(diff + half, 2),
            "method": "unpaired Welch SE; conservative t(df=min(n)-1)",
        })
    write_csv(out / "mtp_comparison.csv", comparison,
              ["comparison", "arm_a", "arm_b", "n_a", "n_b", "mean_a",
               "mean_b", "sd_a", "sd_b", "difference", "difference_pct",
               "se_diff", "df", "df_welch", "tcrit", "ci95_halfwidth",
               "ci95_lo", "ci95_hi", "method"])

    blocks = []
    for block in (1, 2):
        a = [r["tg_t_s"] for r in runs
             if r["arm"] == "off" and r["block"] == block]
        b = [r["tg_t_s"] for r in runs
             if r["arm"] == "on" and r["block"] == block]
        sa, sb = sd(a), sd(b)
        blocks.append({
            "block": block, "n_off": len(a), "n_on": len(b),
            "off_mean": round(mean(a), 2) if a else None,
            "on_mean": round(mean(b), 2) if b else None,
            "off_sd": round(sa, 2) if sa is not None else None,
            "on_sd": round(sb, 2) if sb is not None else None,
            "difference": round(mean(b) - mean(a), 2) if a and b else None,
        })
    write_csv(out / "mtp_blocks.csv", blocks,
              ["block", "n_off", "n_on", "off_mean", "on_mean", "off_sd",
               "on_sd", "difference"])


# --- 29B ncmoe load probe -------------------------------------------------

def build_ncmoe_probe(raw: Path, out: Path):
    text = (raw / "p4" / "probe-results.txt").read_text()
    status = {}
    oom_field = {}
    for line in text.splitlines():
        if line.startswith("PROBE_") and "=" in line and line.split("=")[0].count("_") == 1:
            k, v = line.split("=", 1)
            status[k.replace("PROBE_", "")] = v
        elif line.startswith("OOM_") and "=" in line:
            # sanitized raw carries the OOM evidence directly, so the published
            # repo does not depend on the untracked *.log files
            k, v = line.split("=", 1)
            oom_field[k.replace("OOM_", "")] = v
    cfgmap = {
        "ref20": ("20", "on"),
        "cand17": ("17", "on"),
        "ctrl17off": ("17", "off"),
    }
    rows = []
    for cfg, (ncmoe, cache) in cfgmap.items():
        st = status.get(cfg, "?")
        oom = oom_field.get(cfg, "")
        if not oom and st.startswith("FAIL"):
            log = raw / "p4" / f"server-{cfg}-b0.log"
            if log.exists():
                for ln in log.read_text(errors="replace").splitlines():
                    if "cudaMalloc failed: out of memory" in ln:
                        oom = ln.split(" E ")[-1].strip()
                        break
        rows.append({"config": cfg, "ncmoe": ncmoe, "moe_cache": cache,
                     "probe": "OK" if st.startswith("OK") else "FAIL",
                     "oom_point": oom})
    write_csv(out / "ncmoe_probe_29b.csv",
              rows, ["config", "ncmoe", "moe_cache", "probe", "oom_point"])


# --- bench-derived tables -------------------------------------------------

def build_bench(sources: Path, out: Path):
    src = json.loads(sources.read_text())
    write_csv(out / "ncmoe_sweep_28b.csv", src["ncmoe_sweep_28b"],
              ["ncmoe", "pp_t_s", "tg_t_s", "vram_mib", "verdict", "section"])
    write_csv(out / "moe_cache.csv", src["moe_cache"],
              ["config", "ctx", "pp_t_s", "tg_t_s", "note", "section"])
    write_csv(out / "prefill_vs_ctx.csv", src["prefill_vs_ctx"],
              ["ctx", "pp_t_s", "note", "section"])
    write_csv(out / "deep_vs_depth0.csv", src["deep_vs_depth0"],
              ["mode", "tg_t_s", "note", "section"])


# --- coding evaluation ----------------------------------------------------

def build_coding(raw: Path, out: Path):
    run_dir = raw / "coding"
    runs = []
    for p in sorted(run_dir.glob("run-*.json")):
        d = json.loads(p.read_text())
        v = d["verdict"]
        turns = d["loop"]["turn_log"]
        first = turns[0] if turns else {}
        n_tool_calls = sum(len(t.get("tool_calls") or []) for t in turns)
        tests_run = v["tests_run"]
        failures = v["failures"]
        runs.append({
            "file": p.name,
            "task_id": d["task_id"],
            "class_name": d["class_name"],
            "source": d["source"],
            "run": d["run"],
            "passed": as_bool(v["passed"]),
            "tests_run": tests_run,
            "tests_failed": failures,
            "tests_passed": tests_run - failures,
            "wall_s": round(d["wall_s"], 2),
            "predicted_n": sum(t.get("predicted_n") or 0 for t in turns),
            "turns": d["loop"]["turns"],
            "tool_calls": n_tool_calls,
            "finish_reason": first.get("finish_reason"),
            "extract_ok": as_bool(d["extract_ok"]),
        })
    runs.sort(key=lambda r: (r["task_id"], r["run"]))
    write_csv(out / "coding_eval.csv", runs,
              ["file", "task_id", "class_name", "source", "run", "passed",
               "tests_run", "tests_failed", "tests_passed", "wall_s",
               "predicted_n", "turns", "tool_calls", "finish_reason", "extract_ok"])

    # per-task aggregates
    tasks = []
    for tid in dict.fromkeys(r["task_id"] for r in runs):
        rs = [r for r in runs if r["task_id"] == tid]
        walls = sorted(r["wall_s"] for r in rs)
        tasks.append({
            "task_id": tid,
            "class_name": rs[0]["class_name"],
            "source": rs[0]["source"],
            "runs": len(rs),
            "passes": sum(r["passed"] for r in rs),
            "tests_run": sum(r["tests_run"] for r in rs),
            "tests_passed": sum(r["tests_passed"] for r in rs),
            "wall_p50_s": round(statistics.median(walls), 1),
            "predicted_n_mean": round(mean([r["predicted_n"] for r in rs]), 0),
        })
    write_csv(out / "coding_summary.csv", tasks,
              ["task_id", "class_name", "source", "runs", "passes", "tests_run",
               "tests_passed", "wall_p50_s", "predicted_n_mean"])

    # overall
    n = len(runs)
    k = sum(r["passed"] for r in runs)
    lo, hi = wilson(k, n)
    groups = [[r["passed"] for r in runs if r["task_id"] == tid]
              for tid in dict.fromkeys(r["task_id"] for r in runs)]
    clus_lo, clus_hi = cluster_bootstrap(groups, cluster_run_rate)
    clus_all_lo, clus_all_hi = cluster_bootstrap(groups,
                                                 cluster_all_pass_share)
    ce = [r for r in runs if r["source"] == "classeval"]
    he = [r for r in runs if r["source"] != "classeval"]
    tests_run = sum(r["tests_run"] for r in runs)
    tests_passed = sum(r["tests_passed"] for r in runs)
    overall = [{
        "runs": n,
        "passes": k,
        "pass_rate": round(k / n, 4),
        "wilson_lo": round(lo, 4),
        "wilson_hi": round(hi, 4),
        "tasks_total": len(groups),
        "tasks_any_pass": sum(1 for g in groups if any(g)),
        "tasks_all_pass": sum(1 for g in groups if all(g)),
        "cluster_lo": round(clus_lo, 4),
        "cluster_hi": round(clus_hi, 4),
        "cluster_all_lo": round(clus_all_lo, 4),
        "cluster_all_hi": round(clus_all_hi, 4),
        "tests_run": tests_run,
        "tests_passed": tests_passed,
        "test_pass_rate": round(tests_passed / tests_run, 4),
        "classeval_runs": len(ce),
        "classeval_passes": sum(r["passed"] for r in ce),
        "humaneval_runs": len(he),
        "humaneval_passes": sum(r["passed"] for r in he),
        "total_predicted_n": sum(r["predicted_n"] for r in runs),
        "total_tool_calls": sum(r["tool_calls"] for r in runs),
    }]
    write_csv(out / "coding_overall.csv", overall, list(overall[0].keys()))


def validate_inputs(raw: Path, sources: Path) -> list[str]:
    """Reject missing, malformed, or incomplete inputs before writing anything."""
    problems: list[str] = []
    mtp_dir, coding_dir, p4 = raw / "mtp", raw / "coding", raw / "p4"
    for label, p in (("mtp dir", mtp_dir), ("coding dir", coding_dir),
                     ("p4 dir", p4), ("sources", sources)):
        if not p.exists():
            problems.append(f"missing {label}: {p}")

    mtp_files = sorted(mtp_dir.glob("*-b*-r*.json")) if mtp_dir.exists() else []
    if len(mtp_files) != 20:
        problems.append(f"mtp: expected 20 run traces, found {len(mtp_files)}")
    seen, per_arm, blocks = set(), {"off": 0, "on": 0}, set()
    per_block = {(arm, block): 0 for arm in ("off", "on") for block in (1, 2)}
    prompts: dict[tuple[str, int], list[int]] = {}
    for p in mtp_files:
        parts = p.stem.split("-")
        if (len(parts) != 3 or parts[0] not in per_arm
                or not parts[1].startswith("b") or not parts[2].startswith("r")):
            problems.append(f"mtp: bad trace filename {p.name}")
            continue
        arm, block, run = parts
        if (arm, block, run) in seen:
            problems.append(f"mtp: duplicate run {arm}-{block}-{run}")
        seen.add((arm, block, run))
        per_arm[arm] += 1
        per_block[(arm, int(block[1:]))] += 1
        blocks.add(block)
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"mtp: {p.name} is not valid JSON ({exc})")
            continue
        t = d.get("timings") or {}
        for key in ("prompt_n", "prompt_per_second", "predicted_per_second",
                    "cache_n", "predicted_n", "prompt_ms", "predicted_ms"):
            if key not in t:
                problems.append(f"mtp: {p.name} missing timings.{key}")
        for key in ("prompt_per_second", "predicted_per_second",
                    "prompt_ms", "predicted_ms", "predicted_n"):
            v = t.get(key)
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                problems.append(f"mtp: {p.name} timings.{key} is not finite")
        wall = d.get("wall_s")
        if not isinstance(wall, (int, float)) or not math.isfinite(wall):
            problems.append(f"mtp: {p.name} wall_s is missing or not finite")
        if isinstance(t.get("prompt_n"), int):
            prompts.setdefault((arm, int(block[1:])), []).append(t["prompt_n"])
        if t.get("cache_n") != 0:
            problems.append(f"mtp: {p.name} cache_n={t.get('cache_n')} (not cold)")
        if d.get("finish_reason") not in ("stop", "length"):
            problems.append(f"mtp: {p.name} finish_reason={d.get('finish_reason')!r}")
        peaks = d.get("peaks") or {}
        for key in ("vram_mib", "ram_used_mib"):
            v = peaks.get(key)
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                problems.append(f"mtp: {p.name} peaks.{key} is not finite")
    for arm, n in per_arm.items():
        if n != 10:
            problems.append(f"mtp: arm {arm} has {n} runs (need exactly 10)")
    for (arm, block), n in per_block.items():
        if n != 5:
            problems.append(
                f"mtp: arm {arm} block {block} has {n} runs (need exactly 5)")
    if blocks and blocks != {"b1", "b2"}:
        problems.append(f"mtp: blocks {sorted(blocks)} != ['b1', 'b2']")
    for (arm, block), vals in prompts.items():
        if vals and max(vals) - min(vals) > 16:
            problems.append(
                f"mtp: arm {arm} block {block} prompt spread "
                f"{max(vals) - min(vals)} tokens (need <=16)")
    for arm in ("off", "on"):
        for block in (1, 2):
            s = mtp_dir / f"summary-{arm}-b{block}.json"
            if not s.exists():
                problems.append(f"mtp: missing {s.name}")
                continue
            try:
                d = json.loads(s.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                problems.append(f"mtp: {s.name} is not valid JSON ({exc})")
                continue
            if d.get("arm") != arm or d.get("block") != block:
                problems.append(f"mtp: {s.name} arm/block mismatch")
            kept, attempts = d.get("kept"), d.get("attempts")
            failures = d.get("failures")
            if kept != 5:
                problems.append(f"mtp: {s.name} kept={kept} != 5")
            if not isinstance(failures, list):
                problems.append(f"mtp: {s.name} failures is not a list")
            elif attempts != kept + len(failures):
                problems.append(
                    f"mtp: {s.name} attempts={attempts} != kept+failures")
            if not d.get("finished_at"):
                problems.append(f"mtp: {s.name} missing finished_at")

    coding_files = sorted(coding_dir.glob("run-*.json")) if coding_dir.exists() else []
    if len(coding_files) != 33:
        problems.append(f"coding: expected 33 run traces, found {len(coding_files)}")
    task_runs: dict[str, list] = {}
    for p in coding_files:
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"coding: {p.name} is not valid JSON ({exc})")
            continue
        for key in ("task_id", "run", "class_name", "source", "wall_s",
                    "extract_ok", "loop", "verdict"):
            if key not in d:
                problems.append(f"coding: {p.name} missing {key}")
        v = d.get("verdict") or {}
        for key in ("tests_run", "failures", "passed"):
            if key not in v:
                problems.append(f"coding: {p.name} verdict missing {key}")
        if v.get("tests_run", 0) < v.get("failures", 0):
            problems.append(f"coding: {p.name} failures > tests_run")
        if bool(v.get("passed")) != (v.get("failures") == 0):
            problems.append(f"coding: {p.name} passed/failures inconsistent")
        if not d.get("extract_ok"):
            problems.append(f"coding: {p.name} extract_ok is false")
        task_runs.setdefault(str(d.get("task_id")), []).append(d.get("run"))
    for tid, runs in task_runs.items():
        if len(runs) != 3 or len(set(runs)) != 3:
            problems.append(f"coding: task {tid} has runs {runs} (need 3 distinct)")
    if len(task_runs) != 11:
        problems.append(f"coding: expected 11 tasks, found {len(task_runs)}")

    if (p4 / "probe-results.txt").exists():
        probe = (p4 / "probe-results.txt").read_text()
        for cfg in ("ref20", "cand17", "ctrl17off"):
            if f"PROBE_{cfg}=" not in probe:
                problems.append(f"p4: missing PROBE_{cfg}=")
        for cfg in ("cand17", "ctrl17off"):
            if f"OOM_{cfg}=" not in probe:
                problems.append(f"p4: missing OOM_{cfg}= evidence")

    if sources.exists():
        try:
            src = json.loads(sources.read_text())
        except json.JSONDecodeError as exc:
            problems.append(f"sources: not valid JSON ({exc})")
        else:
            for key in ("ncmoe_sweep_28b", "moe_cache", "prefill_vs_ctx",
                        "deep_vs_depth0"):
                if not src.get(key):
                    problems.append(f"sources: missing {key}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--sources", default="data/sources-bench.json")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    raw = Path(args.raw)
    sources = Path(args.sources)
    out = Path(args.out)

    problems = validate_inputs(raw, sources)
    if problems:
        print("INPUT VALIDATION FAILED")
        for problem in problems:
            print("  -", problem)
        raise SystemExit(2)

    print("building results from", raw)
    out.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=out.parent, prefix=".build-"))
    try:
        build_mtp(raw, staging)
        build_ncmoe_probe(raw, staging)
        build_bench(sources, staging)
        build_coding(raw, staging)
        for path in sorted(staging.glob("*.csv")):
            os.replace(path, out / path.name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print("done")


if __name__ == "__main__":
    main()
