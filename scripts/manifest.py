#!/usr/bin/env python3
"""Write or check the provenance manifest for the published study.

    python scripts/manifest.py          # write data/manifest.json + results/manifest.json
    python scripts/manifest.py --check  # exit 1 if an input or output has drifted

The manifest is deterministic: no timestamps, sorted keys. It records the git
commit, the dirty state, the Python/matplotlib versions, the run counts, and a
SHA-256 for every committed input and every generated output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
RES = ROOT / "results"
FIG = ROOT / "figures"
MANIFESTS = {DATA / "manifest.json", RES / "manifest.json"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def input_files() -> list[Path]:
    files = [p for p in (DATA / "sources-bench.json",
                         DATA / "claims.json",
                         DATA / "figure-sources.json") if p.exists()]
    for p in sorted(RAW.rglob("*")):
        if p.is_file() and p not in MANIFESTS and p.suffix != ".log":
            files.append(p)
    return files


def output_files() -> list[Path]:
    return sorted(list(RES.glob("*.csv")) + list(FIG.glob("*.png"))
                  + list(FIG.glob("*.svg")))


def git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, text=True,
                              capture_output=True).stdout.strip()

    dirty = [line for line in run("status", "--porcelain",
                                 "--untracked-files=all").splitlines()
             if not (line.endswith("data/manifest.json")
                     or line.endswith("results/manifest.json"))]
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(dirty),
            "dirty_paths": dirty}


def environment() -> dict:
    env = {"python": platform.python_version()}
    try:
        import matplotlib
        env["matplotlib"] = matplotlib.__version__
    except ImportError:
        env["matplotlib"] = None
    return env


def counts() -> dict:
    mtp = [p for p in (RAW / "mtp").glob("*-b*-r*.json")]
    coding = list((RAW / "coding").glob("run-*.json"))
    probes = [line for line in (RAW / "p4" / "probe-results.txt").read_text().splitlines()
              if line.startswith("PROBE_") and "=" in line]
    return {"mtp_runs": len(mtp), "coding_runs": len(coding),
            "probe_configs": len(probes)}


def build() -> dict:
    return {
        "schema": 1,
        "git": git_state(),
        "environment": environment(),
        "counts": counts(),
        "inputs": {rel(p): sha256(p) for p in input_files()},
        "outputs": {rel(p): sha256(p) for p in output_files()},
    }


def write() -> int:
    m = build()
    (DATA / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
    rm = {
        "schema": 1,
        "source_manifest": "data/manifest.json",
        "source_manifest_sha256": sha256(DATA / "manifest.json"),
        "git": m["git"],
        "outputs": m["outputs"],
    }
    (RES / "manifest.json").write_text(json.dumps(rm, indent=2, sort_keys=True) + "\n")
    print(f"wrote data/manifest.json ({len(m['inputs'])} inputs, {len(m['outputs'])} outputs)")
    print("wrote results/manifest.json")
    return 0


def check() -> int:
    path = DATA / "manifest.json"
    if not path.exists():
        print("manifest missing: run python scripts/manifest.py", file=sys.stderr)
        return 1
    old = json.loads(path.read_text())
    new = build()
    problems = []
    for section in ("inputs", "outputs"):
        for key in sorted(set(old[section]) | set(new[section])):
            if old[section].get(key) != new[section].get(key):
                problems.append(f"{section}: {key} changed or missing")
    if old.get("counts") != new.get("counts"):
        problems.append("run counts changed")
    rpath = RES / "manifest.json"
    if not rpath.exists():
        problems.append("results/manifest.json missing")
    elif json.loads(rpath.read_text()).get("source_manifest_sha256") != sha256(path):
        problems.append("results/manifest.json does not match data/manifest.json")
    if problems:
        print("MANIFEST CHECK FAILED")
        for p in problems:
            print("  -", p)
        return 1
    print(f"manifest OK ({len(new['inputs'])} inputs, {len(new['outputs'])} outputs)")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    sys.exit(check() if args.check else write())


if __name__ == "__main__":
    main()
