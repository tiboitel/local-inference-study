# Reproduction

The aim is that a reader with similar hardware can rebuild every figure from raw
traces, and that no figure is a hand-typed number.

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

## 2. Capture raw traces

Each run writes one JSON trace: the request, the response, the `timings` block
reported by the server, and the per-second resource samples (VRAM, RAM, GPU%,
watts, swap). Raw traces are the source of truth for [results/](results/); they
are kept out of git by default (see `.gitignore`) and can be re-generated.

## 3. Build the result tables

```bash
pip install -r scripts/requirements.txt
python scripts/build_results.py --raw data/raw --out results
```

`build_results.py` reads the raw traces and emits one CSV per experiment. It is
the *only* place a number in `results/` is produced; hand-editing a CSV breaks
the traceability chain and will fail `scripts/check_traceability.py`.

## 4. Regenerate figures

```bash
python scripts/make_figures.py   # results/*.csv -> figures/*.png
```

## 5. Coding suite

The coding result comes from a separate harness that drives the same endpoint:
each task is one turn, the generated code is extracted mechanically and executed
against hidden `unittest` tests, and the harness self-tests 11/11 on canonical
solutions before any model call. See [results/coding_eval.csv](results/).

## 6. Verify

```bash
python scripts/check_traceability.py   # every figure cell -> a raw trace or a ledger row
```

A green run of `check_traceability.py` is the release gate: no untraceable figure
leaves the repo.
