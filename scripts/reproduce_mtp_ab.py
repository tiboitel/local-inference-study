#!/usr/bin/env python3
"""Minimal public client for the MTP on/off A/B procedure.

    python scripts/reproduce_mtp_ab.py \
        --off-url http://127.0.0.1:8080/v1 \
        --on-url  http://127.0.0.1:8081/v1 \
        --model local-model --out repro-mtp

This reproduces the *procedure* on two already-running OpenAI-compatible
llama.cpp servers (MTP off and on), not the exact published magnitudes: the
original prompt family and private runner are not published. It sends cold
prompts with a unique nonce, asserts ``cache_n == 0``, interleaves the arms, and
writes one JSON trace per request plus ``runs.csv`` and ``ab.csv``.

The server must expose llama.cpp's ``timings`` block. If it does not, the script
stops rather than substituting a wall-clock estimate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SEED = (
    "A local inference server holds weights, a KV cache, and compute buffers. "
    "The KV cache grows with context while the weights and buffers stay roughly "
    "fixed. When experts are offloaded to host memory, prompt processing also "
    "fetches them over the interconnect. This synthetic passage exists only to "
    "give the reproduction request a long, deterministic body. "
)


def t95(n: int) -> float:
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
             13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
             18: 2.101, 19: 2.093, 20: 2.086}
    return table.get(n - 1, 1.96)


def post(url: str, model: str, prompt: str, token: str | None,
         timeout: float) -> tuple[dict, float]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "cache_prompt": False,
        "timings_per_token": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
    }
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(), headers=headers, method="POST")
    started = time.perf_counter()
    text: list[str] = []
    timings = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("timings"):
                    timings = obj["timings"]
                for choice in obj.get("choices") or []:
                    text.append((choice.get("delta") or {}).get("content") or "")
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"request failed ({exc.code}): {exc.read()[:300]!r}")
    elapsed = time.perf_counter() - started
    if timings is None:
        raise SystemExit(
            "server did not return a timings block; this client refuses to "
            "substitute a wall-clock estimate")
    if timings.get("cache_n") != 0:
        raise SystemExit(
            f"cache_n={timings.get('cache_n')!r}: not a cold prefill; rerun "
            "with a fresh nonce and cache_prompt false")
    return {"text": "".join(text), "timings": timings}, elapsed


def t_value(n: int) -> float:
    return t95(n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--off-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--on-url", default="http://127.0.0.1:8081/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=5,
                    help="kept runs per arm per block (default: 5 => 10/arm)")
    ap.add_argument("--warmups", type=int, default=1,
                    help="discarded requests per arm before measurement")
    ap.add_argument("--target-tokens", type=int, default=32000)
    ap.add_argument("--prompt-file",
                    help="optional fixed prompt; a nonce is still appended")
    ap.add_argument("--auth-env", default="",
                    help="environment variable holding a bearer token, if needed")
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()
    if args.runs < 5:
        raise SystemExit("use at least 5 kept runs per arm per block")

    out = Path(args.out)
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get(args.auth_env) if args.auth_env else None
    if args.prompt_file:
        base_body = Path(args.prompt_file).read_text()
    else:
        target_chars = max(4000, args.target_tokens * 4)
        base_body = (SEED * (target_chars // len(SEED) + 1))[:target_chars]
    urls = {"off": args.off_url, "on": args.on_url}

    def send(arm: str, block: int, run: int, keep: bool):
        nonce = f"repro-{os.getpid()}-{time.time_ns()}-{arm}-b{block}-r{run}"
        body = (base_body + f"\n\nNonce: {nonce}\n\n"
                "Summarize the passage above in one sentence.")
        result, elapsed = post(urls[arm], args.model, body, token, args.timeout)
        t = result["timings"]
        dn = t.get("draft_n") or 0
        accepted = t.get("draft_n_accepted")
        record = {
            "arm": arm, "block": block, "run": run, "nonce": nonce,
            "wall_s": round(elapsed, 3), "final_text": result["text"],
            "timings": t, "peaks": {},
        }
        if keep:
            (raw_dir / f"{arm}-b{block}-r{run}.json").write_text(
                json.dumps(record, indent=2) + "\n")
        return {
            "arm": arm, "block": block, "run": run,
            "prompt_n": t.get("prompt_n"),
            "pp_t_s": t.get("prompt_per_second"),
            "tg_t_s": t.get("predicted_per_second"),
            "accept": (accepted / dn) if dn and accepted is not None else "",
            "cache_n": t.get("cache_n"),
        }

    for arm in ("off", "on"):
        for _ in range(args.warmups):
            send(arm, 0, 0, keep=False)

    rows = []
    for block in (1, 2):
        order = ("off", "on") if block == 1 else ("on", "off")
        for arm in order:
            for run in range(1, args.runs + 1):
                rows.append(send(arm, block, run, keep=True))
                print(f"{arm} block {block} run {run}: "
                      f"{rows[-1]['tg_t_s']} t/s")

    with (out / "runs.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    agg = []
    for arm in ("off", "on"):
        vals = [float(r["tg_t_s"]) for r in rows if r["arm"] == arm]
        sd = statistics.stdev(vals) if len(vals) > 1 else None
        sem = sd / math.sqrt(len(vals)) if sd is not None else None
        agg.append({
            "arm": arm, "n_kept": len(vals),
            "tg_mean": round(statistics.fmean(vals), 3),
            "tg_sem": round(sem, 3) if sem is not None else "",
            "tg_ci95": round(t_value(len(vals)) * sem, 3) if sem else "",
            "vram_note": "not sampled by this minimal client",
        })
    with (out / "ab.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(agg[0]))
        writer.writeheader(); writer.writerows(agg)
    print(f"wrote {out}/runs.csv and {out}/ab.csv")
    print("This is a procedure reproduction; compare directions and intervals, "
          "not the published magnitudes.")


if __name__ == "__main__":
    main()
