#!/usr/bin/env python3
"""Release gate: prove the published numbers from the committed evidence.

Stages, each independent of scripts/build_results.py:

  1. schema      every raw trace and provenance sidecar is present and well-typed
  2. integrity   every raw run is consumed exactly once; manifest hashes match
  3. recompute   result tables are re-derived from the raw traces and compared
                 field-by-field within a tolerance
  4. invariants  experimental-design properties hold (cold cache, arm order,
                 >=5 kept runs/arm, 3 runs/task, no duplicates)
  5. claims      numbers quoted in ARTICLE.md occur in the recomputed tables
  6. staleness   regenerating results/ from raw reproduces the committed CSVs

    python scripts/check_traceability.py   # exit 0 == release gate passed
    python scripts/check_traceability.py --report results/traceability.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RAW = ROOT / "data" / "raw"
DATA = ROOT / "data"
RAW_MTP = RAW / "mtp"
RAW_CODING = RAW / "coding"
RAW_P4 = RAW / "p4"

# per-field tolerance: defaults to two-decimal rounding; wider for rounded ints
TOL = {"ram_peak_mib": 0.51, "predicted_n_mean": 0.51,
       "pass_rate": 1e-4, "wilson_lo": 1e-4, "wilson_hi": 1e-4,
       "cluster_lo": 1e-4, "cluster_hi": 1e-4,
       "cluster_all_lo": 1e-4, "cluster_all_hi": 1e-4,
       "test_pass_rate": 1e-4}

problems: list[str] = []


def fail(msg: str) -> None:
    problems.append(msg)


def load(name: str):
    p = RES / name
    if not p.exists():
        fail(f"missing result table: {name}")
        return []
    with p.open() as fh:
        return list(csv.DictReader(fh))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def close(a, b, tol: float) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


def t95(n: int) -> float:
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
             13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
             18: 2.101, 19: 2.093, 20: 2.086}
    return table.get(n - 1, 1.96)


def sem(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None


def wilson(k: int, n: int, z: float = 1.96):
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def cluster_bootstrap(groups, stat, seed: int = 0, draws: int = 10000):
    """Independent re-implementation of the builder's task-cluster bootstrap."""
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


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def compare(name: str, want, keys) -> None:
    got = load(name)
    index = {tuple(str(r[k]) for k in keys): r for r in got}
    if len(got) != len(want):
        fail(f"{name}: {len(got)} rows != recomputed {len(want)}")
    for w in want:
        k = tuple(str(w[k]) for k in keys)
        g = index.get(k)
        if g is None:
            fail(f"{name}: recomputed row missing from table: {k}")
            continue
        for field, wv in w.items():
            gv = g.get(field, "")
            if wv is None:
                if gv not in ("", "None"):
                    fail(f"{name} {k} {field}: expected empty, got {gv!r}")
            elif isinstance(wv, (int, float)):
                if not close(gv, wv, TOL.get(field, 0.011)):
                    fail(f"{name} {k} {field}: {gv} != {wv}")
            elif str(wv) != str(gv):
                fail(f"{name} {k} {field}: {gv!r} != {wv!r}")


# --- raw parsing (independent of the builder) -----------------------------

def mtp_raw() -> dict:
    return {p.name: json.loads(p.read_text())
            for p in sorted(RAW_MTP.glob("*-b*-r*.json"))}


def coding_raw() -> dict:
    return {p.name: json.loads(p.read_text())
            for p in sorted(RAW_CODING.glob("run-*.json"))}


# --- stage 1: schema -------------------------------------------------------

MTP_KEYS = {"final_text", "finish_reason", "nonce", "timings", "peaks",
            "usage", "wall_s"}
TIMING_KEYS = {"cache_n", "prompt_n", "prompt_per_second",
               "predicted_per_second"}
CODING_KEYS = {"task_id", "run", "class_name", "source", "wall_s",
               "extract_ok", "loop", "verdict"}


def stage_schema(mtp, coding) -> None:
    if not mtp:
        fail("schema: no MTP raw traces")
    for name, d in mtp.items():
        if not MTP_KEYS <= set(d):
            fail(f"schema: {name} missing keys {sorted(MTP_KEYS - set(d))}")
            continue
        t = d["timings"] or {}
        if not TIMING_KEYS <= set(t):
            fail(f"schema: {name} timings missing {sorted(TIMING_KEYS - set(t))}")
        for field in ("prompt_per_second", "predicted_per_second", "wall_s"):
            v = num(d.get(field, t.get(field)))
            if v is None or not math.isfinite(v):
                fail(f"schema: {name} {field} is not finite")
        if num((d.get("peaks") or {}).get("vram_mib")) is None:
            fail(f"schema: {name} peaks.vram_mib missing")
    for name, d in coding.items():
        if not CODING_KEYS <= set(d):
            fail(f"schema: {name} missing keys {sorted(CODING_KEYS - set(d))}")
            continue
        v = d.get("verdict") or {}
        for field in ("tests_run", "failures", "passed"):
            if field not in v:
                fail(f"schema: {name} verdict missing {field}")
        if num(d.get("wall_s")) is None:
            fail(f"schema: {name} wall_s not finite")
        if not (d.get("loop") or {}).get("turn_log") and \
                not (d.get("loop") or {}).get("turns"):
            fail(f"schema: {name} loop has no turns")
    for cfg in ("ref20", "cand17", "ctrl17off"):
        if f"PROBE_{cfg}=" not in (RAW_P4 / "probe-results.txt").read_text():
            fail(f"schema: probe-results.txt missing PROBE_{cfg}=")
    for rel in ("mtp/config.json", "coding/config.json", "p4/config.json",
                "README.md"):
        if not (RAW / rel).exists():
            fail(f"schema: missing data/raw/{rel}")


# --- stage 2: integrity ----------------------------------------------------

def stage_integrity(mtp, coding) -> None:
    rows = {f"{r['arm']}-b{r['block']}-r{r['run']}.json"
            for r in load("mtp_runs.csv")}
    if rows != set(mtp):
        fail(f"integrity: MTP raw/runs mismatch: only-raw={set(mtp) - rows} "
             f"only-table={rows - set(mtp)}")
    crows = {r["file"] for r in load("coding_eval.csv")}
    if crows != set(coding):
        fail("integrity: coding raw/table filename mismatch")
    for arm in ("off", "on"):
        for block in (1, 2):
            s = RAW_MTP / f"summary-{arm}-b{block}.json"
            if not s.exists():
                fail(f"integrity: missing {s.name}")
                continue
            d = json.loads(s.read_text())
            n = len([k for k in mtp
                     if k.startswith(f"{arm}-b{block}-")])
            if int(d.get("kept", -1)) != n:
                fail(f"integrity: {s.name} kept={d.get('kept')} but {n} traces")
    r = subprocess.run([sys.executable, "scripts/manifest.py", "--check"],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        fail("integrity: manifest check failed: " +
             (r.stdout.strip().splitlines() or ["?"])[-1])


# --- stage 3: recompute ----------------------------------------------------

def recompute_mtp(mtp) -> tuple[list, list]:
    runs = []
    for name, d in mtp.items():
        arm, block, run = name[:-5].split("-")
        t = d["timings"] or {}
        dn = t.get("draft_n") or 0
        acc = t.get("draft_n_accepted")
        runs.append({
            "arm": arm, "block": int(block[1:]), "run": int(run[1:]),
            "prompt_n": t.get("prompt_n"),
            "pp_t_s": t.get("prompt_per_second"),
            "tg_t_s": t.get("predicted_per_second"),
            "accept": (acc / dn) if (dn and acc is not None) else None,
            "cache_n": t.get("cache_n"),
            "vram_mib": (d.get("peaks") or {}).get("vram_mib"),
            "ram_mib": (d.get("peaks") or {}).get("ram_used_mib"),
            "finish_reason": d.get("finish_reason"),
        })
    runs.sort(key=lambda r: (r["arm"], r["block"], r["run"]))
    agg = []
    for arm in ("off", "on"):
        rs = [r for r in runs if r["arm"] == arm]
        tg_sem = sem([r["tg_t_s"] for r in rs])
        accs = [r["accept"] for r in rs if r["accept"] is not None]
        agg.append({
            "arm": arm, "n_kept": len(rs),
            "prompt_n_mean": round(mean([r["prompt_n"] for r in rs]), 1),
            "pp_mean": round(mean([r["pp_t_s"] for r in rs]), 2),
            "pp_sem": round(sem([r["pp_t_s"] for r in rs]), 2),
            "tg_mean": round(mean([r["tg_t_s"] for r in rs]), 2),
            "tg_sem": round(tg_sem, 2),
            "tg_ci95": round(t95(len(rs)) * tg_sem, 2) if tg_sem else None,
            "accept_mean": round(mean(accs), 3) if accs else None,
            "accept_sd": round(statistics.stdev(accs), 3) if len(accs) > 1 else None,
            "vram_peak_mib": max(r["vram_mib"] for r in rs),
            "ram_peak_mib": round(max(r["ram_mib"] for r in rs), 0),
        })
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
        df = min(n1, n2) - 1
        tcrit = t95(df + 1)
        diff = mean(on) - mean(off)
        half = tcrit * se
        comparison.append({
            "comparison": "tg", "arm_a": "off", "arm_b": "on",
            "n_a": n1, "n_b": n2,
            "mean_a": round(mean(off), 2), "mean_b": round(mean(on), 2),
            "difference": round(diff, 2),
            "difference_pct": round(diff / mean(off) * 100, 1),
            "se_diff": round(se, 2), "df": df,
            "df_welch": round(df_welch, 2), "tcrit": round(tcrit, 3),
            "ci95_halfwidth": round(half, 2),
            "ci95_lo": round(diff - half, 2),
            "ci95_hi": round(diff + half, 2),
            "method": "unpaired Welch SE; conservative t(df=min(n)-1)",
        })
    blocks = []
    for block in (1, 2):
        a = [r["tg_t_s"] for r in runs
             if r["arm"] == "off" and r["block"] == block]
        b = [r["tg_t_s"] for r in runs
             if r["arm"] == "on" and r["block"] == block]
        blocks.append({
            "block": block, "n_off": len(a), "n_on": len(b),
            "off_mean": round(mean(a), 2) if a else None,
            "on_mean": round(mean(b), 2) if b else None,
            "difference": round(mean(b) - mean(a), 2) if a and b else None,
        })
    return runs, agg, comparison, blocks


def recompute_coding(coding) -> tuple[list, list, dict]:
    runs = []
    for name, d in coding.items():
        v = d["verdict"]
        turns = d["loop"]["turn_log"]
        first = turns[0] if turns else {}
        runs.append({
            "file": name, "task_id": d["task_id"], "class_name": d["class_name"],
            "source": d["source"], "run": d["run"],
            "passed": 1 if v["passed"] else 0,
            "tests_run": v["tests_run"], "tests_failed": v["failures"],
            "tests_passed": v["tests_run"] - v["failures"],
            "wall_s": round(d["wall_s"], 2),
            "predicted_n": sum(t.get("predicted_n") or 0 for t in turns),
            "turns": d["loop"]["turns"],
            "tool_calls": sum(len(t.get("tool_calls") or []) for t in turns),
            "finish_reason": first.get("finish_reason"),
            "extract_ok": 1 if d["extract_ok"] else 0,
        })
    runs.sort(key=lambda r: (r["task_id"], r["run"]))
    tasks = []
    for tid in dict.fromkeys(r["task_id"] for r in runs):
        rs = [r for r in runs if r["task_id"] == tid]
        tasks.append({
            "task_id": tid, "class_name": rs[0]["class_name"],
            "source": rs[0]["source"], "runs": len(rs),
            "passes": sum(r["passed"] for r in rs),
            "tests_run": sum(r["tests_run"] for r in rs),
            "tests_passed": sum(r["tests_passed"] for r in rs),
            "wall_p50_s": round(statistics.median([r["wall_s"] for r in rs]), 1),
            "predicted_n_mean": round(mean([r["predicted_n"] for r in rs]), 0),
        })
    n, k = len(runs), sum(r["passed"] for r in runs)
    lo, hi = wilson(k, n)
    groups = [[r["passed"] for r in runs if r["task_id"] == tid]
              for tid in dict.fromkeys(r["task_id"] for r in runs)]
    clus_lo, clus_hi = cluster_bootstrap(groups, cluster_run_rate)
    clus_all_lo, clus_all_hi = cluster_bootstrap(groups,
                                                 cluster_all_pass_share)
    ce = [r for r in runs if r["source"] == "classeval"]
    he = [r for r in runs if r["source"] != "classeval"]
    tr = sum(r["tests_run"] for r in runs)
    tp = sum(r["tests_passed"] for r in runs)
    overall = {
        "runs": n, "passes": k, "pass_rate": round(k / n, 4),
        "wilson_lo": round(lo, 4), "wilson_hi": round(hi, 4),
        "tasks_total": len(groups),
        "tasks_any_pass": sum(1 for g in groups if any(g)),
        "tasks_all_pass": sum(1 for g in groups if all(g)),
        "cluster_lo": round(clus_lo, 4), "cluster_hi": round(clus_hi, 4),
        "cluster_all_lo": round(clus_all_lo, 4),
        "cluster_all_hi": round(clus_all_hi, 4),
        "tests_run": tr, "tests_passed": tp,
        "test_pass_rate": round(tp / tr, 4),
        "classeval_runs": len(ce), "classeval_passes": sum(r["passed"] for r in ce),
        "humaneval_runs": len(he), "humaneval_passes": sum(r["passed"] for r in he),
        "total_predicted_n": sum(r["predicted_n"] for r in runs),
        "total_tool_calls": sum(r["tool_calls"] for r in runs),
    }
    return runs, tasks, overall


def stage_recompute(mtp, coding) -> tuple[list, list, list]:
    mruns, magg, mcomp, mblocks = recompute_mtp(mtp)
    compare("mtp_runs.csv", mruns, ("arm", "block", "run"))
    compare("mtp_ab.csv", magg, ("arm",))
    compare("mtp_comparison.csv", mcomp, ("comparison",))
    compare("mtp_blocks.csv", mblocks, ("block",))
    cruns, ctasks, coverall = recompute_coding(coding)
    compare("coding_eval.csv", cruns, ("file",))
    compare("coding_summary.csv", ctasks, ("task_id",))
    compare("coding_overall.csv", [coverall], ("runs",))
    return magg, mcomp, [coverall]


# --- stage 4: invariants ---------------------------------------------------

def stage_invariants(mtp, coding) -> None:
    for name, d in mtp.items():
        if (d["timings"] or {}).get("cache_n") != 0:
            fail(f"invariant: {name} cache_n != 0 (not a cold prefill)")
        if d.get("finish_reason") not in ("stop", "length"):
            fail(f"invariant: {name} unexpected finish_reason "
                 f"{d.get('finish_reason')!r}")
    order = []
    for arm in ("off", "on"):
        for block in (1, 2):
            s = RAW_MTP / f"summary-{arm}-b{block}.json"
            if s.exists():
                order.append((json.loads(s.read_text()).get("finished_at"), arm))
    if [a for _, a in sorted(order)] != ["off", "on", "on", "off"]:
        fail(f"invariant: MTP arm order is {[a for _, a in sorted(order)]}, "
             "expected off,on,on,off")
    for arm in ("off", "on"):
        n = len([k for k in mtp if k.startswith(f"{arm}-")])
        if n != 10:
            fail(f"invariant: {n} kept runs for arm {arm} (need exactly 10)")
        for block in (1, 2):
            nb = len([k for k in mtp if k.startswith(f"{arm}-b{block}-")])
            if nb != 5:
                fail(f"invariant: {nb} runs for {arm} block {block} "
                     "(need exactly 5)")
            vals = [d["timings"].get("prompt_n") for name, d in mtp.items()
                    if name.startswith(f"{arm}-b{block}-")]
            vals = [v for v in vals if isinstance(v, int)]
            if vals and max(vals) - min(vals) > 16:
                fail(f"invariant: {arm} block {block} prompt spread "
                     f"{max(vals) - min(vals)} (need <=16)")
            s = RAW_MTP / f"summary-{arm}-b{block}.json"
            if not s.exists():
                fail(f"invariant: missing {s.name}")
                continue
            sd = json.loads(s.read_text())
            kept, attempts = sd.get("kept"), sd.get("attempts")
            failures = sd.get("failures")
            if sd.get("arm") != arm or sd.get("block") != block:
                fail(f"invariant: {s.name} arm/block mismatch")
            if kept != 5:
                fail(f"invariant: {s.name} kept={kept} != 5")
            if not isinstance(failures, list):
                fail(f"invariant: {s.name} failures is not a list")
            elif attempts != kept + len(failures):
                fail(f"invariant: {s.name} attempts={attempts} != kept+failures")

    per_task: dict[str, int] = {}
    for name, d in coding.items():
        v = d["verdict"]
        if v["tests_run"] < v["failures"]:
            fail(f"invariant: {name} failures > tests_run")
        if bool(v["passed"]) != (v["failures"] == 0):
            fail(f"invariant: {name} passed={v['passed']} but failures={v['failures']}")
        if not d.get("extract_ok"):
            fail(f"invariant: {name} extract_ok is false")
        if not (d["loop"]["turn_log"][0].get("finish_reason")
                if d["loop"]["turn_log"] else None):
            fail(f"invariant: {name} missing finish_reason")
        key = f"{d['task_id']}:{d['run']}"
        per_task[key] = per_task.get(key, 0) + 1
    dupes = [k for k, c in per_task.items() if c != 1]
    if dupes:
        fail(f"invariant: duplicate task/run keys: {dupes}")
    tasks = {d["task_id"] for d in coding.values()}
    for tid in tasks:
        n = len([d for d in coding.values() if d["task_id"] == tid])
        if n != 3:
            fail(f"invariant: task {tid} has {n} runs, expected 3")

    probe = load("ncmoe_probe_29b.csv")
    if {r["config"] for r in probe} != {"ref20", "cand17", "ctrl17off"}:
        fail("invariant: probe config set changed")
    for r in probe:
        if r["probe"] == "FAIL" and not r.get("oom_point"):
            fail(f"invariant: probe {r['config']} FAIL without oom evidence")


# --- stage 5: claims -------------------------------------------------------

def stage_claims(magg, mcomp, coverall) -> None:
    art = (ROOT / "ARTICLE.md").read_text()
    o = coverall[0]
    k, n = o["passes"], o["runs"]
    tp, tr = o["tests_passed"], o["tests_run"]
    expected = {
        f"{tp}/{tr}": "test-method pass",
        f"{o['classeval_passes']}/{o['classeval_runs']}": "ClassEval-only passes",
        f"{o['humaneval_passes']}/{o['humaneval_runs']}": "single-function floor",
        f"{o['tasks_all_pass']}/{o['tasks_total']}": "tasks passing all runs",
        f"{o['tasks_any_pass']}/{o['tasks_total']}": "tasks passing any run",
        f"{k / n * 100:.1f}": "run pass rate",
        f"{tp / tr * 100:.1f}": "test-method pass rate",
        f"{o['wilson_lo'] * 100:.1f}": "Wilson lower bound",
        f"{o['wilson_hi'] * 100:.1f}": "Wilson upper bound",
        f"{o['cluster_lo'] * 100:.1f}": "task-clustered lower bound",
        f"{o['cluster_hi'] * 100:.1f}": "task-clustered upper bound",
    }
    arms = {r["arm"]: r for r in magg}
    off_tg, on_tg = float(arms["off"]["tg_mean"]), float(arms["on"]["tg_mean"])
    vram_delta = int(round(float(arms["on"]["vram_peak_mib"])
                           - float(arms["off"]["vram_peak_mib"])))
    expected.update({
        f"{off_tg:.2f}": "MTP-off deep TG",
        f"{on_tg:.2f}": "MTP-on deep TG",
        f"{(on_tg - off_tg) / off_tg * 100:.1f}": "MTP decode delta (%)",
        f"{float(arms['on']['accept_mean']):.3f}": "MTP acceptance",
        str(vram_delta): "MTP VRAM cost (MiB)",
        f"{float(arms['off']['pp_mean']):.2f}": "MTP-off prefill",
        f"{float(arms['on']['pp_mean']):.2f}": "MTP-on prefill",
    })
    if mcomp:
        c = mcomp[0]
        expected.update({
            f"{float(c['difference']):.2f}": "MTP decode difference",
            f"{float(c['difference_pct']):.1f}": "MTP decode delta (%)",
            f"{float(c['ci95_halfwidth']):.2f}": "MTP difference CI half-width",
        })
    dd = {r["mode"]: r for r in load("deep_vs_depth0.csv")}
    if dd:
        expected[f"{float(dd['depth0_llama_bench']['tg_t_s']):.1f}"] = "depth-0 TG"
        expected[f"{float(dd['deep_server_measured']['tg_t_s']):.1f}"] = "deep TG"
    for s, label in expected.items():
        if s not in art:
            fail(f"claims: ARTICLE.md does not contain {s!r} ({label})")
    cache_pp = {r["pp_t_s"] for r in load("moe_cache.csv")}
    for v in ("308.5", "775.7"):
        if v not in cache_pp or v not in art:
            fail(f"claims: expert-cache prefill {v} missing from CSV or article")
    ctx_pp = {r["pp_t_s"] for r in load("prefill_vs_ctx.csv")}
    if "859" not in ctx_pp:
        fail("claims: prefill_vs_ctx.csv missing 859 t/s at 1k")
    for v in ("859", "109"):
        if v not in art:
            fail(f"claims: ARTICLE.md missing prefill-vs-context value {v}")


def stage_claim_registry() -> None:
    """Every marked value must match the structured claim registry exactly."""
    path = DATA / "claims.json"
    if not path.exists():
        fail("claims: data/claims.json missing")
        return
    try:
        registry = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        fail(f"claims: data/claims.json invalid ({exc})")
        return
    docs = {name: (ROOT / name).read_text()
            for name in ("ARTICLE.md", "README.md")}
    for cid, spec in registry.items():
        rows = load(spec["table"])
        hits = [r for r in rows
                if all(str(r.get(k)) == str(v)
                       for k, v in (spec.get("where") or {}).items())]
        if len(hits) != 1:
            fail(f"claims: {cid} matched {len(hits)} rows in {spec['table']}")
            continue
        raw = hits[0].get(spec["field"])
        if raw in (None, ""):
            fail(f"claims: {cid} has no {spec['field']}")
            continue
        try:
            rendered = (str(int(round(float(raw))))
                        if spec["format"].endswith("d")
                        else format(float(raw), spec["format"]))
        except (TypeError, ValueError, KeyError) as exc:
            fail(f"claims: {cid} cannot format {raw!r} ({exc})")
            continue
        found = 0
        pattern = (r"<!--\s*claim:" + re.escape(cid) +
                   r"\s*-->(.*?)<!--\s*/claim\s*-->")
        for name, body in docs.items():
            for value in re.findall(pattern, body, re.S):
                found += 1
                if value.strip() != rendered:
                    fail(f"claims: {name} marker {cid} shows "
                         f"{value.strip()!r}, expected {rendered!r}")
        if found == 0:
            fail(f"claims: no marker for {cid} in ARTICLE.md/README.md")


def stage_figures() -> None:
    """Every rendered figure must declare its source tables and be readable."""
    path = DATA / "figure-sources.json"
    if not path.exists():
        fail("figures: data/figure-sources.json missing")
        return
    try:
        sources = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        fail(f"figures: data/figure-sources.json invalid ({exc})")
        return
    files = {p.name for p in (ROOT / "figures").glob("*")
             if p.suffix in (".png", ".svg")}
    if files != set(sources):
        fail(f"figures: source map mismatch: only-files={sorted(files - set(sources))} "
             f"only-map={sorted(set(sources) - files)}")
    for name, spec in sources.items():
        for table in spec.get("sources", []):
            if not (RES / table).exists():
                fail(f"figures: {name} references missing results/{table}")
        p = ROOT / "figures" / name
        if not p.exists():
            continue
        if name.endswith(".png"):
            head = p.read_bytes()[:24]
            if (len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n"
                    or int.from_bytes(head[16:20], "big") == 0
                    or int.from_bytes(head[20:24], "big") == 0):
                fail(f"figures: {name} is not a valid non-empty PNG")
        if name.endswith(".svg"):
            if "<svg" not in p.read_text()[:500]:
                fail(f"figures: {name} is not a readable SVG")


def stage_bench_provenance() -> None:
    path = DATA / "sources-bench.json"
    if not path.exists():
        fail("bench: data/sources-bench.json missing")
        return
    try:
        src = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        fail(f"bench: data/sources-bench.json invalid ({exc})")
        return
    status = src.get("_provenance_status", "")
    if not status.startswith("not self-contained"):
        fail("bench: provenance status must state that the source clip is "
             "not self-contained")
    for key in ("ncmoe_sweep_28b", "moe_cache", "prefill_vs_ctx",
                "deep_vs_depth0"):
        for index, row in enumerate(src.get(key, [])):
            if not row.get("section"):
                fail(f"bench: {key}[{index}] has no source section")
    evidence = (ROOT / "EVIDENCE.md").read_text()
    if "self-contained" not in evidence:
        fail("bench: EVIDENCE.md does not disclose the non-self-contained ledger")


# --- stage 6: staleness ----------------------------------------------------

def stage_staleness() -> None:
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([sys.executable, "scripts/build_results.py",
                            "--raw", "data/raw", "--out", td],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            fail("staleness: build_results.py failed: " +
                 (r.stderr.strip().splitlines() or ["?"])[-1])
            return
        for csv_path in sorted(RES.glob("*.csv")):
            fresh = Path(td) / csv_path.name
            if not fresh.exists():
                fail(f"staleness: rebuild did not produce {csv_path.name}")
            elif fresh.read_bytes() != csv_path.read_bytes():
                fail(f"staleness: results/{csv_path.name} differs from a "
                     "fresh rebuild of data/raw")


def report_text() -> str:
    """Deterministic, timestamp-free evidence report for release auditing."""
    if problems:
        lines = ["TRACEABILITY FAILED", ""]
        lines += [f"- {p}" for p in problems]
        lines += ["", "Reproduction: not claimed (private runner and coding harness)."]
        return "\n".join(lines) + "\n"
    tables = sorted(p.name for p in RES.glob("*.csv"))
    lines = [
        "TRACEABILITY REPORT",
        "",
        "result: OK",
        "stage_schema: OK",
        "stage_integrity: OK",
        "stage_recompute: OK",
        "stage_invariants: OK",
        "stage_claims: OK",
        "stage_claim_registry: OK",
        "stage_figures: OK",
        "stage_bench_provenance: OK",
        "stage_staleness: OK",
        f"mtp_runs: {len(load('mtp_runs.csv'))}",
        f"coding_runs: {len(load('coding_eval.csv'))}",
        f"result_tables: {len(tables)}",
        "",
        "tables:",
    ]
    lines += [f"- results/{name}" for name in tables]
    lines += [
        "",
        "reproducible: no",
        "reason: the benchmark runner and coding harness are private; the raw",
        "traces are committed, and the tables/figures are regenerated from them.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", help="also write the deterministic release report")
    args = ap.parse_args()

    mtp, coding = mtp_raw(), coding_raw()
    stage_schema(mtp, coding)
    stage_integrity(mtp, coding)
    magg, mcomp, coverall = stage_recompute(mtp, coding)
    stage_invariants(mtp, coding)
    stage_claims(magg, mcomp, coverall)
    stage_claim_registry()
    stage_figures()
    stage_bench_provenance()
    stage_staleness()

    if problems:
        print("TRACEABILITY FAILED")
        for p in problems:
            print("  -", p)
        text = report_text()
        if args.report:
            path = Path(args.report)
            path = path if path.is_absolute() else ROOT / path
            path.write_text(text)
        sys.exit(1)

    print(f"traceability OK: {len(load('mtp_runs.csv'))} MTP runs, "
          f"{len(load('coding_eval.csv'))} coding runs, "
          f"{len(load('mtp_ab.csv')) + len(load('coding_summary.csv'))} tables "
          "recomputed and matched")
    if args.report:
        path = Path(args.report)
        path = path if path.is_absolute() else ROOT / path
        path.write_text(report_text())
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
