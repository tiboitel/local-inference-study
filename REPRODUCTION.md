# Reproduction

The aim is that a reader with similar hardware can rebuild every figure from raw
traces, and that no figure is a hand-typed number.

From a fresh clone, steps 3, 4 and 6 work using the committed traces. Step 1
needs the model artifact and step 5 needs the unpublished harness.

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

## 2. Raw traces

Each run writes one JSON trace: the response, the `timings` block reported by
the server, and the per-run peak resource samples (VRAM, RAM). The traces are
committed under `data/raw/` (sanitized: no host paths); `*.log` files and model
weights are not. They are the source of truth for [results/](results/).

The runner and coding harness are **not published** with this repository, so the
committed traces can be audited but cannot be regenerated from a fresh clone.
Steps that need the harness are marked **[private runner]**.

## 3. Build the result tables

```bash
pip install -r scripts/requirements.txt
python scripts/build_results.py --raw data/raw --out results
```

`build_results.py` reads the raw traces and emits one CSV per experiment. It is
the *only* place a number in `results/` is produced; hand-editing a CSV breaks
the traceability chain and will fail `scripts/check_traceability.py`.

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
python scripts/check_traceability.py   # every figure cell -> a raw trace or a ledger row
python scripts/manifest.py --check     # inputs/outputs match the committed manifest
```

A green run of `check_traceability.py` is the release gate: no untraceable figure
leaves the repo.
