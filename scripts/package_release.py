#!/usr/bin/env python3
"""Build a deterministic release tarball of the public evidence.

    python scripts/package_release.py --version 0.2.0

The archive contains the sanitized raw traces, result tables, figures, manifests,
the traceability report, and the public scripts needed to rebuild them. It does
not contain model weights, server logs, or the private runner.
"""
from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILES = [
    "ARTICLE.md", "CITATION.cff", "DATA-HYGIENE.md", "EVIDENCE.md",
    "HARDWARE-AND-RUNTIME.md", "LICENSE", "LIMITATIONS.md", "METHODOLOGY.md",
    "PROMPT-PROVENANCE.md", "README.md", "RELEASE-NOTES-v0.2.0.md",
    "REPRODUCTION.md",
    "data/claims.json", "data/figure-sources.json", "data/manifest.json",
    "data/sources-bench.json", "results/manifest.json",
    "results/traceability.txt",
]
DIRS = ["data/raw", "figures", "results", "scripts", "tests"]
EXCLUDE_SUFFIXES = {".pyc"}
EXCLUDE_NAMES = {"__pycache__"}


def wanted(path: Path) -> bool:
    if any(part in EXCLUDE_NAMES for part in path.parts):
        return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    if path.name.endswith(".log") or path.suffix == ".gguf":
        return False
    return path.is_file()


def release_files() -> list[Path]:
    files = {ROOT / name for name in FILES if (ROOT / name).exists()}
    for name in DIRS:
        base = ROOT / name
        if base.exists():
            files.update(p for p in base.rglob("*") if wanted(p))
    return sorted(files, key=lambda p: str(p.relative_to(ROOT)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", required=True)
    ap.add_argument("--out", default=str(ROOT / "dist"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"local-inference-study-v{args.version}.tar.gz"
    target = out / name
    with tarfile.open(target, "w:gz", format=tarfile.GNU_FORMAT) as tar:
        for path in release_files():
            info = tar.gettarinfo(str(path), arcname=str(path.relative_to(ROOT)))
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as fh:
                tar.addfile(info, fh)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (out / f"{name}.sha256").write_text(f"{digest}  {name}\n")
    print(f"wrote {target.relative_to(ROOT)}")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
