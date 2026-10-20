"""Build a small synthetic experiment tree so the pipeline can be tested on a
machine with no GPU, no model, and no private harness."""
from __future__ import annotations

import json
from pathlib import Path

TASKS = [f"ClassEval_{i}" for i in (6, 7, 47, 55, 67, 75, 86, 27, 46)]
TASKS += ["HumanEval_style_count_vowels", "HumanEval_style_merge_intervals"]
BLOCK_ORDER = {("off", 1): "01", ("on", 1): "02", ("on", 2): "03", ("off", 2): "04"}


def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj))


def make_repo(root: Path) -> Path:
    raw = root / "data" / "raw"
    mtp, coding, p4 = raw / "mtp", raw / "coding", raw / "p4"
    for d in (mtp, coding, p4, root / "results"):
        d.mkdir(parents=True, exist_ok=True)

    for arm, base in (("off", 10.0), ("on", 20.0)):
        for block in (1, 2):
            for run in range(1, 6):
                pp_rate = 600.0 + run
                tg_rate = base + 0.1 * run
                timings = {"cache_n": 0, "prompt_n": 1000,
                           "prompt_per_second": pp_rate,
                           "predicted_per_second": tg_rate,
                           # request-level timings: ms are converted to s by
                           # the builder (prompt_s / decode_s)
                           "prompt_ms": 1000.0 / pp_rate * 1000.0,
                           "predicted_n": 50,
                           "predicted_ms": 50.0 / tg_rate * 1000.0}
                if arm == "on":
                    timings.update({"draft_n": 10, "draft_n_accepted": 8})
                _write(mtp / f"{arm}-b{block}-r{run}.json", {
                    "final_text": "x", "finish_reason": "stop",
                    "nonce": f"{arm}{block}{run}", "timings": timings,
                    "peaks": {"vram_mib": 1000.0 + base,
                              "ram_used_mib": 2000.0},
                    "usage": {}, "wall_s": 1.0 + 1000.0 / pp_rate})
            _write(mtp / f"summary-{arm}-b{block}.json", {
                "arm": arm, "block": block, "kept": 5, "attempts": 5,
                "failures": [],
                "finished_at": f"2026-10-19T{BLOCK_ORDER[(arm, block)]}:00:00Z"})

    for index, task in enumerate(TASKS):
        source = "classeval" if task.startswith("ClassEval") else "humaneval-style"
        for run in (1, 2, 3):
            passed = index != 0  # the first task never passes
            _write(coding / f"run-{run:02d}-{task}.json", {
                "task_id": task, "run": run, "class_name": task,
                "source": source, "wall_s": 10.0 + run, "extract_ok": True,
                "loop": {"answer": "x", "turns": 1,
                         "turn_log": [{"tool_calls": [], "predicted_n": 10,
                                       "finish_reason": "stop"}]},
                "verdict": {"passed": passed, "tests_run": 4,
                            "failures": 0 if passed else 1}})

    (p4 / "probe-results.txt").write_text(
        "PROBE_ref20=OK\n"
        "PROBE_cand17=FAIL rc=1\n"
        "OOM_cand17=allocating 128.00 MiB: out of memory\n"
        "PROBE_ctrl17off=FAIL rc=1\n"
        "OOM_ctrl17off=allocating 208.03 MiB: out of memory\n")

    for rel in ("mtp", "coding", "p4"):
        (raw / rel / "config.json").write_text("{}")
    (raw / "README.md").write_text("# fixture\n")

    _write(root / "data" / "sources-bench.json", {
        "ncmoe_sweep_28b": [{"ncmoe": 20, "pp_t_s": 300.0, "tg_t_s": 40.0,
                             "vram_mib": 8000, "verdict": "ok", "section": "04"}],
        "moe_cache": [{"config": "c", "ctx": 1024, "pp_t_s": 300.0,
                       "tg_t_s": 40.0, "note": "", "section": "04"}],
        "prefill_vs_ctx": [{"ctx": 1024, "pp_t_s": 300.0, "section": "06"}],
        "deep_vs_depth0": [{"mode": "m", "tg_t_s": 40.0, "section": "06"}],
    })
    return root
