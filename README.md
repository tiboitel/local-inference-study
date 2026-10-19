# Long-Context Local LLM Inference on 12 GB VRAM: What Actually Moves Performance?

A measurement study of running a 29B-class Mixture-of-Experts model with
64k context on a single consumer GPU (RTX 3060, 12 GB). The live measurements
come from the real request path, not a synthetic depth-0 benchmark; imported
`llama-bench` rows are labelled as such in each figure's provenance, and every
generated table traces to a raw run or a sourced row.

**Headline findings**

| # | Finding | Evidence |
|---|---|---|
| 1 | Depth-0 benchmark numbers are a poor proxy for agentic performance: real deep-context decode is ~15 t/s (28B bench sibling; 14.8 t/s MTP-off on the live 29B profile), not the 46 t/s depth-0 figure. | [results/deep_vs_depth0.csv](results/deep_vs_depth0.csv) |
| 2 | MTP speculative decoding more than doubles deep-context decode at 64k: **14.79 → 38.56 t/s (+160.7%)** for −1.19% prefill (inside noise). The best validated result for this exact configuration, not a universal MTP speedup. | [results/mtp_ab.csv](results/mtp_ab.csv) |
| 3 | MoE expert-cache `on` multiplies long-context prefill roughly 2.5–3× (308 → 776 t/s). | [results/moe_cache.csv](results/moe_cache.csv) |
| 4 | The 28B 64k memory optimum does **not** transfer to the 29B MTP profile: `ncmoe=17` OOMs; `ncmoe=20` is the fitting point. | [results/ncmoe_sweep_28b.csv](results/ncmoe_sweep_28b.csv), [results/ncmoe_probe_29b.csv](results/ncmoe_probe_29b.csv) |
| 5 | End-to-end coding completion on the shipped sampling: **57.6%** of class-level runs (Wilson 95% CI 40.8–72.8), 84.0% of individual test methods. All 14 failures were implementation failures under the test contracts — no truncation, extraction, or tool errors. | [results/coding_eval.csv](results/coding_eval.csv) |

## Why this exists

Most local-LLM performance writing reports `llama-bench` prompt-eval and
token-generation at depth 0. That measures a workload nobody runs. An agent
serving real requests sees a large prompt (tens of thousands of tokens) and then
generates in its shadow. This study measures that path.

## Layout

| Path | Purpose |
|---|---|
| [ARTICLE.md](ARTICLE.md) | The flagship article (narrative + figures). |
| [METHODOLOGY.md](METHODOLOGY.md) | Measurement protocol, what makes a result real, rejected experiments. |
| [HARDWARE-AND-RUNTIME.md](HARDWARE-AND-RUNTIME.md) | Host, GPU, model, build, exact serving flags. |
| [LIMITATIONS.md](LIMITATIONS.md) | What this study does not establish. |
| [REPRODUCTION.md](REPRODUCTION.md) | How to reproduce each result. |
| [DATA-HYGIENE.md](DATA-HYGIENE.md) | The public-release checklist every figure must pass. |
| `results/` | One CSV per experiment, each row traceable to a raw run. |
| `figures/` | Generated plots. |
| `scripts/` | Regenerate every CSV and figure from raw inputs. |
| `data/` | Figure-to-source traceability table. |

## Reproduce

```bash
# raw traces are committed under data/raw/ (see REPRODUCTION.md)
pip install -r scripts/requirements.txt
python scripts/build_results.py        # raw traces -> results/*.csv
python scripts/make_figures.py         # results/*.csv -> figures/*.png
python scripts/check_traceability.py   # release gate: figure cells -> raw sources
```

## License

Text and figures: CC BY 4.0. Scripts: MIT. See [LICENSE](LICENSE).
