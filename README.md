# Long-Context Local LLM Inference on 12 GB VRAM: What Actually Moves Performance?

**Research question.** How do prefill, decode, and memory behave when serving a
29B-class Mixture-of-Experts model at 64k context on a single 12 GB consumer GPU
— and which knobs actually move each one?

Every number below comes from a real request path and links to the table or raw
trace behind it.

## Headline findings

| # | Finding | Evidence |
|---|---|---|
| 1 | **MTP speculative decoding more than doubles deep-context decode at 64k: 14.79 → 38.56 t/s (+160.7%)**, while prefill is unchanged (−1.19%, inside the ~4% noise floor). | [results/mtp_ab.csv](results/mtp_ab.csv) |
| 2 | **Depth-0 `llama-bench` overstates real decode by roughly 3×: 46.0 t/s at depth 0 vs ~15 t/s once a full 64k prompt is in front of it.** A configuration tuned on depth-0 measures a workload nobody runs. | [results/deep_vs_depth0.csv](results/deep_vs_depth0.csv) |
| 3 | **The MoE expert cache is the prefill lever: 308 → 776 t/s (~2.5×) in the controlled short-context A/B.** At 64k, cache-off prefill collapses below 73 t/s, versus ~677–712 t/s with the cache on. | [results/moe_cache.csv](results/moe_cache.csv) |

*Also measured:* the 28B `ncmoe` optimum does **not** transfer to the 29B MTP
profile (`ncmoe=17` OOMs at load, `ncmoe=20` fits) —
[results/ncmoe_probe_29b.csv](results/ncmoe_probe_29b.csv); and end-to-end
coding on the shipped sampling is 19/33 runs (57.6%, Wilson 95% CI 40.8–72.8)
with 84.0% of individual test methods passing —
[results/coding_overall.csv](results/coding_overall.csv).

## MTP comparison at 64k

![MTP on/off at 64k](figures/fig_mtp_ab.png)

*Live 29B MTP profile, 10 kept runs per arm, interleaved `off → on → on → off`,
cold prefill (`cache_n == 0`), ~57.9k-token prompt. Decode 14.79 → 38.56 t/s
(+160.7%, combined 95% CI ±4.58). Prefill 645 → 637 t/s (−1.19%). Cost: +993 MiB
peak VRAM and ~1.7 GiB host RAM. Acceptance 0.838 (sd 0.070).*

## Tested configuration

| Component | Value |
|---|---|
| GPU | NVIDIA RTX 3060, **12 GB** (12,288 MiB) |
| CPU / RAM | Intel i7-9700K, 8 cores / 8 threads / 32 GB |
| OS | Arch Linux, kernel 7.2.8 |
| Model | `Qwen3.6-29B-REAP-Opus-Reasoning-Distill-MTP`, Q4_K_M |
| Architecture | sparse Mixture-of-Experts, ~29B total |
| Context | 64k (`-c 65536`) |
| Expert offload | `--n-cpu-moe 20` |
| Speculative decoding | `--spec-type draft-mtp --spec-draft-n-max 2` |
| KV cache | `turbo4` (K) / `turbo3` (V) |
| Expert cache | `--moe-cache on` |
| Build | `llama.cpp` (turboquant KV fork) @ `bcb85fc3ae85efa0f5f392c6c880dfc524923860` |

**VRAM, not host RAM, is the binding constraint** — the KV cache is
VRAM-resident, so host RAM does not scale with context length. If your GPU is
not a 12 GB Ampere-class card, or your model is not this 29B MoE, treat the
magnitudes as a starting hypothesis rather than a prediction. Full host, model,
and flag detail: [HARDWARE-AND-RUNTIME.md](HARDWARE-AND-RUNTIME.md).

## What is and is not established

**Established here**

- The *direction* of the effects: deep context is much slower than depth-0;
  MTP helps deep decode; the expert cache helps prefill; memory optima do not
  transfer across models.
- The *magnitudes* for this exact configuration: 64k context, `ncmoe=20`, MTP
  `n-max=2`, `turbo4`/`turbo3`, this build, this prompt family.
- End-to-end coding completion under the shipped sampling, with a Wilson
  interval rather than a point estimate.

**Not established here**

- Other GPUs, VRAM tiers, or multi-GPU setups; other stacks (vLLM,
  TensorRT-LLM, MLX); other quantizations or model families.
- 128k on the live profile — the 128k figures are bench-sibling estimates, not
  live 29B measurements.
- That MTP `n-max=2` is the optimum draft count, or that these magnitudes
  generalize to another corpus. Acceptance is content-driven.
- That throughput predicts correctness. The coding suite measures completion
  under hidden tests, not general code quality.

Full boundary of the claims: [LIMITATIONS.md](LIMITATIONS.md).

## Reproduce

```bash
# raw traces are committed under data/raw/ (see REPRODUCTION.md)
pip install -r scripts/requirements.txt
python scripts/build_results.py --raw data/raw --out results   # raw traces -> results/*.csv
python scripts/make_figures.py                                 # results/*.csv -> figures/*.png
python scripts/check_traceability.py                           # release gate: figure cells -> raw sources
```

The traces can be audited from a fresh clone, but the runner and coding harness
are **not published**, so they cannot be regenerated from this repository alone.
Step-by-step detail and the private-runner boundaries:
[REPRODUCTION.md](REPRODUCTION.md).

## Links

| Resource | Where |
|---|---|
| Full article | [ARTICLE.md](ARTICLE.md) |
| Methodology and protocol | [METHODOLOGY.md](METHODOLOGY.md) |
| Hardware, model, and serving flags | [HARDWARE-AND-RUNTIME.md](HARDWARE-AND-RUNTIME.md) |
| Raw traces and traceability map | [data/raw/](data/raw/) · [data/traceability.md](data/traceability.md) |
| Result tables | [results/](results/) |
| Figures | [figures/](figures/) |
| Limitations | [LIMITATIONS.md](LIMITATIONS.md) |
| Data hygiene checklist | [DATA-HYGIENE.md](DATA-HYGIENE.md) |
| Citation metadata | [CITATION.cff](CITATION.cff) |
| License | [LICENSE](LICENSE) |

## The study in detail

Most local-LLM performance writing reports `llama-bench` prompt-eval and
token-generation at depth 0. That measures a workload nobody runs. An agent
serving real requests sees a large prompt (tens of thousands of tokens) and then
generates in its shadow. This study measures that path.

The live measurements come from the real request path, not a synthetic depth-0
benchmark; imported `llama-bench` rows are labelled as such in each figure's
provenance, and every generated table traces to a raw run or a sourced row.
The protocol in brief: `POST /v1/chat/completions` with `stream: true`,
`cache_prompt: false`, and a unique nonce per request (guaranteeing cold prefill
via `cache_n == 0`); arms interleaved in alternated then reversed block order;
warmups discarded and ≥5 kept runs per arm; mean and SEM reported, with a ~4%
noise floor; and VRAM, host RAM, GPU utilization, watts, and swap sampled per
request. Full protocol and rejected framings: [METHODOLOGY.md](METHODOLOGY.md).

| Path | Purpose |
|---|---|
| [ARTICLE.md](ARTICLE.md) | The flagship article (narrative + figures). |
| [METHODOLOGY.md](METHODOLOGY.md) | Measurement protocol, what makes a result real, rejected experiments. |
| [HARDWARE-AND-RUNTIME.md](HARDWARE-AND-RUNTIME.md) | Host, GPU, model, build, exact serving flags. |
| [LIMITATIONS.md](LIMITATIONS.md) | What this study does not establish. |
| [REPRODUCTION.md](REPRODUCTION.md) | How to reproduce each result. |
| [DATA-HYGIENE.md](DATA-HYGIENE.md) | The public-release checklist every figure must pass. |
| [data/traceability.md](data/traceability.md) | The map from every published number to its raw source. |
| `results/` | One CSV per experiment, each row traceable to a raw run. |
| `figures/` | Generated plots. |
| `scripts/` | Regenerate every CSV and figure from raw inputs. |

## License

Text and figures: CC BY 4.0. Scripts: MIT. See [LICENSE](LICENSE).
