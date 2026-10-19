# Reproduction

This repository supports three different activities, and it is worth naming them
separately:

- **Audit:** read the committed raw traces and check that every table follows.
- **Regeneration:** rebuild `results/*.csv` and `figures/*.png` from the
  committed inputs.
- **Independent reproduction:** repeat the original experiment from the published
  procedure. This is **not** possible for the measured runs, because the runner
  and coding harness are private.

The aim is that a reader with similar hardware can rebuild every figure from raw
traces, and that no figure is a hand-typed number. A fresh clone contains
`data/raw/`; steps 3, 4 and 6 rebuild every table and figure from those committed
traces. Step 1 needs the model artifact, step 5 needs the unpublished harness, and
step 7 is a separate procedure-only client. The original prompt body is not
published; see [PROMPT-PROVENANCE.md](PROMPT-PROVENANCE.md).

## 1. Serve the model

Use the flags in [HARDWARE-AND-RUNTIME.md](HARDWARE-AND-RUNTIME.md). The critical
ones:

```
llama-server -m <model>.gguf \
  -c 65536 --n-gpu-layers 99 --n-cpu-moe 20 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --cache-type-k turbo4 --cache-type-v turbo3 \
  -b 2048 -ub 1024 -np 1 --threads 8
```

Follow the methodology protocol literally: `stream: true`, `cache_prompt: false`,
unique nonce per request, interleaved arms, warmups discarded, ≥5 kept runs.

The model artifact SHA-256 and the GPU driver version were not captured at run
time and are recorded as `null` / post-hoc respectively. They cannot be
backfilled honestly without retaining the exact artifact and rerunning; see
[EVIDENCE.md](EVIDENCE.md).

## 2. Raw traces

Each run writes one JSON trace: the response, the `timings` block reported by
the server, and the per-run peak resource samples (VRAM, RAM). The traces are
committed under `data/raw/` (sanitized: no host paths); `*.log` files and model
weights are not. They are the source of truth for [results/](results/).

The runner and coding harness are **not published** with this repository, so the
*experiment* cannot be rerun from a fresh clone. The committed traces themselves
are present and sufficient to rebuild every published table. Steps that need the
harness are marked **[private runner]**.

## 3. Build the result tables

```bash
pip install -r scripts/requirements.txt
python scripts/build_results.py --raw data/raw --out results
```

`build_results.py` reads the raw traces and emits one CSV per experiment. It is
the *only* place a number in `results/` is produced; hand-editing a CSV breaks
the traceability chain and will fail `scripts/check_traceability.py`. The MTP
summary includes `mtp_ab.csv` (per-arm means, SEM, t-interval),
`mtp_comparison.csv` (unpaired difference, Welch SE, conservative df, CI), and
`mtp_blocks.csv` (block-level means). The claim registry in
[`data/claims.json`](data/claims.json) binds marked article values to table
cells.

Record or verify the provenance of those inputs and outputs:

```bash
python scripts/manifest.py          # write data/manifest.json + results/manifest.json
python scripts/manifest.py --check  # fail if a trace or result has drifted
```

## 4. Regenerate figures

```bash
python scripts/make_figures.py   # results/*.csv -> figures/*.png
```

## 5. Coding suite

The coding result comes from a separate harness that drives the same endpoint:
each task is one turn, the generated code is extracted mechanically and executed
against hidden `unittest` tests, and the harness self-tests 11/11 on canonical
solutions before any model call. The harness is **not published**; the committed
per-run traces are the auditable artifact. See
[results/coding_eval.csv](results/).

## 6. Verify

```bash
python scripts/check_traceability.py --report results/traceability.txt
python scripts/manifest.py --check     # inputs/outputs match the committed manifest
python -m pytest -q
bash scripts/check_hygiene.sh
```

A green run of `check_traceability.py` is the release gate: no untraceable figure
leaves the repo. The report is deterministic and is committed as
[results/traceability.txt](results/traceability.txt); CI regenerates it and
fails if it drifts.

The CI `documented-reproduction` job runs the public workflow from a clean
checkout, writes figures to a temporary directory, and then requires

```bash
git diff --exit-code -- results data
```

so a missing raw input or a stale result table fails the release, not just the
prose checks.

## 7. Procedure-only public A/B client

`scripts/reproduce_mtp_ab.py` is a minimal, dependency-free client for the
central MTP comparison. Start two OpenAI-compatible llama.cpp servers with the
same model and flags, one with `--spec-type draft-mtp --spec-draft-n-max 2` and
one without, then:

```bash
python scripts/reproduce_mtp_ab.py \
  --off-url http://127.0.0.1:8080/v1 \
  --on-url  http://127.0.0.1:8081/v1 \
  --model <served-model-name> --out repro-mtp
```

It sends a unique nonce, refuses to continue unless the server reports
`cache_n == 0`, interleaves the arms in `off → on → on → off` order, and writes
raw JSON plus `runs.csv` and `ab.csv`. It does **not** reproduce the original
prompt family or the exact published magnitudes. Compare directions and
uncertainty, and label the output **procedure reproduction**, not result
reproduction.

## 8. Release artifact

```bash
python scripts/package_release.py --version 0.2.0
```

This writes a deterministic `dist/local-inference-study-v0.2.0.tar.gz` and a
SHA-256 sidecar. The archive contains the sanitized traces, result tables,
figures, manifests, report, and public scripts; it excludes model weights,
server logs, and the private harness. The version tag is the immutable
reference for the article, results, and raw data.

## What is not published

| Artifact | Why it is not committed | Public substitute |
|---|---|---|
| Benchmark runner | Private orchestration code; contains host-specific paths and operational detail | `scripts/reproduce_mtp_ab.py` (client procedure) + committed raw traces |
| Coding harness and task prompts | Private harness; redistributing task text/hidden tests may violate the benchmark license | 33 committed per-run traces including verdicts; ClassEval revision citation |
| Model weights (`.gguf`) | Multi-GB artifact, not redistributable here | Model name and quantization; **no hash** (not captured) |
| Full long prompt / prompt corpus | Private fixed prompt family | Prompt token counts, per-run nonces, and the public client's synthetic prompt |
| Server logs (`*.log`) | Host paths and operational noise | Structured `PROBE_*` / `OOM_*` lines in `probe-results.txt` |
| GPU driver / power at run time | Not recorded | Post-hoc capture, explicitly labelled; see `EVIDENCE.md` |
