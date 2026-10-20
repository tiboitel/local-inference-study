# Long-Context Local LLM Inference on a 12 GB RTX 3060: Measuring MTP, Prefill, and VRAM at 64k

*An engineering study of a 29B-class Mixture-of-Experts model on a single consumer GPU*

## Abstract

This study examines long-context inference for a **29B Mixture-of-Experts** model (*Qwen 29B REAP*) on a single **RTX 3060** with **12 GB** of **VRAM**. In the context of agentic use, the practical question is not how fast the model generates with an empty context, but what happens when a request arrives with tens of thousands of tokens already in the prompt. The experiments therefore measure the serving path: long-context prefill, subsequent decode, memory pressure, and the serving parameters that affect each stage.

In the primary **64k-context** configuration, using approximately **57.9k-token prompts**, the MTP-on arm raised measured deep-context decode from **14.79** to **38.56 tokens/s** (*+160.7%*) under the documented shared serving configuration, with an estimated difference of **+23.77 tokens/s** (*request-level unpaired nominal 95% CI: [19.19, 28.34] tokens/s; n = 10 per arm; independence assumed, see Section V.B*). The arms used identical shared serving settings; their speculative-decoding configuration differed (`--spec-type none` vs `--spec-type draft-mtp --spec-draft-n-max 2`). Prefill was **645.1 tokens/s** vs **637.4 tokens/s** (*-1.19%*), below the study's approximately **4%** operational noise threshold, which does not establish statistical equivalence. The MTP arm used an additional **+993 MiB of peak VRAM** and about **+1.7 GiB of peak host RAM**; both are differences between the arms' observed per-request peaks.

Additional experiments on a related **28B MoE** model (*Qwen 28B REAP*) show a different pattern: an MoE expert cache raised short-context prefill from **308.5** to **775.7 tokens/s** while deep decode stayed around **14–15 tokens/s**. A separate depth-0 benchmark reported **46.0 tokens/s** versus **15.3 tokens/s** after a long prompt in a **64k-context configuration**. The **46.0 tokens/s** and **15.3 tokens/s** are not directly comparable; the **~3x** is a measured illustration on a related model, not a controlled ratio.

These measurements characterize **one model**, **one host**, **one runtime**, **one context length** and **one prompt family**.

They are not universal performance estimates.

## I. Introduction

Serving a 29B-class Mixture-of-Experts (MoE) model at a 64k context length on a 12 GB GPU creates a coupled memory and throughput problem. The KV cache grows with sequence length, while the model's expert weights compete for the same limited GPU memory as the cache and other runtime allocations. CPU expert offloading can make the configuration fit, but it introduces additional execution and data-transfer costs. Speculative decoding may improve generation throughput, although its benefit and memory overhead depend on the workload and serving configuration. The central question is which bottlenecks dominate the request path and which optimizations improve performance within the available memory budget.

This study measures that path for a long-context agent request: prefill an approximately 57.9k-token history, then decode from it. *Performance* here means prefill throughput, decode throughput, and the peak VRAM and host RAM required to sustain them. Task quality is checked separately by a small end-to-end coding suite; it is not inferred from throughput.

The experiments address four questions:

1. How does decode throughput measured without conversation history (*depth-0*) compare with throughput after processing a long prompt?
2. How does enabling the expert cache affect prefill and decode throughput under the tested configurations?
3. What throughput and memory differences are observed between the multi-token prediction (MTP) on and off arms at deep context under the documented shared serving configuration?
4. Which results obtained with the supporting benchmark model transfer to the primary served checkpoint?

The study has two tracks, reported separately and never combined into a single estimate:

| Track | Model | Experiments |
|---|---|---|
| **Primary** | `Qwen3.6-29B-REAP-Opus-Reasoning-Distill-MTP`, Q4_K_M | MTP on/off A/B, `ncmoe` load probes, coding evaluation |
| **Supporting** | 28B benchmark related model | Depth-0 baseline, expert-cache comparison, context sweep |

The primary model is a sparse MoE with approximately 29B total parameters, of which a subset is active per token, quantized to Q4_K_M. The supporting track uses a 28B benchmark related model (*Qwen 28B REAP, Q4_K_M*) for comparison points that the live configuration does not cover: a depth-0 baseline, a recorded cache on/off comparison, and a context sweep. It is not a measurement of the live model.

Each section states which track it uses. The primary configuration is summarized below; the full host specification, OS, driver, power limit, and serving flags are documented in [HARDWARE-AND-RUNTIME.md](HARDWARE-AND-RUNTIME.md).

| Component | Configuration |
|---|---|
| GPU | NVIDIA RTX 3060, 12 GB (12,288 MiB) |
| Host | Intel i7-9700K, 8 cores / 8 threads; 32 GB RAM |
| Model | `Qwen3.6-29B-REAP-Opus-Reasoning-Distill-MTP`, Q4_K_M; sparse MoE, ~29B total parameters |
| Context limit / prompt length | 65,536 tokens; ~57.9k-token prompts in the MTP A/B |
| Expert offload | `--n-cpu-moe 20` (experts for 20 layers on CPU) |
| MTP arms | `--spec-type none` vs `--spec-type draft-mtp --spec-draft-n-max 2`; all other serving settings shared |
| KV cache | `turbo4` (K) / `turbo3` (V), quantized |
| Expert cache | `--moe-cache on` in both arms |
| Runtime | `llama.cpp` KV-cache fork, revision `bcb85fc` (full hash in the linked document) |

`64k` denotes the 65,536-token context limit. `--n-cpu-moe 20` keeps the experts of 20 transformer layers on the CPU. `turbo4` (K) and `turbo3` (V) are the serving build's quantized KV-cache types. MTP uses the model's draft head to propose tokens that the main model verifies.

In the primary 64k experiment, the MTP-on arm reached **38.56 t/s** deep-context decode against **14.79 t/s** for the MTP-off arm (**+160.7%**) under the documented shared serving configuration. The estimated difference was **+23.77 t/s** (*request-level unpaired nominal 95% CI [19.19, 28.34] t/s; 10 retained runs per arm; independence assumed, see Section V.B*). The documented arms used identical shared serving settings and differed in their speculative-decoding configuration; per-run flags were not captured in the traces, so this matching is documented rather than trace-verifiable. Prefill was **637.4 vs 645.1 t/s** (**−1.19%**), below the study's approximately 4% operational noise threshold. That threshold is used to avoid over-interpreting small differences; it does not establish statistical equivalence. The MTP arm used **+993 MiB of peak VRAM** (*11,588 of 12,288 MiB*) and approximately **+1.7 GiB of peak host RAM**.

On the supporting 28B track, enabling the expert cache at 1k context with `ncmoe=20` raised prefill from **308.5 to 775.7 t/s** (*about 2.5×; the `soft` mode reached 313.2 t/s*), while deep decode stayed around **14–15 t/s** across the tested cache and `ncmoe` configurations. A depth-0 benchmark measured **46.0 t/s**, against **15.3 t/s** after a long prompt in a 64k-context configuration. Those two rows come from different sibling configurations, so the roughly 3× gap illustrates context sensitivity rather than a controlled estimate of the context-length effect.

On the live profile, `ncmoe=17` failed during initialization both with the cache on (*while allocating 128 MiB of KV cache*) and with it off (*while allocating 208 MiB of compute buffers*), whereas `ncmoe=20` loaded. That is an infeasibility result for the tested configuration, not a throughput comparison, and it does not locate the boundary between 17 and 20. Separately, 19 of 33 coding runs (*11 tasks × 3 attempts*) passed hidden tests (*57.6%; [Appendix A](#a-coding-evaluation)*), a bounded task-quality check rather than a throughput result.

These findings are specific to one model, one host, one runtime, one context length, and one prompt family. [Section VIII](#viii-limitations) and [LIMITATIONS.md](LIMITATIONS.md) bound the claims.

![Diagram of the tested request path and GPU memory split](figures/fig_setup.svg)

**Figure 1 — Tested serving path and memory split.**
*Schematic: request path and measured GPU memory split. Peak VRAM was 10,595 MiB with MTP off and 11,588 MiB with MTP on under the tested configurations. Full flags: [HARDWARE-AND-RUNTIME.md](HARDWARE-AND-RUNTIME.md).*

Raw traces, data provenance, and supporting evidence for individual claims are documented in [EVIDENCE.md](EVIDENCE.md). Every table in this article is regenerated from committed inputs by the supplied scripts.

## II. Depth-0 throughput and the long-context gap

`llama-bench` is useful for measuring baseline generation throughput, but its token-generation measurement starts at depth 0. In `llama-bench`, depth is the number of prompt tokens used to prefill the KV cache before generation is timed; depth 0 means no prompt tokens are prefilled. That measurement answers a specific question:

> How quickly can this model generate with no preceding prompt tokens?

It is a different question from:

> How quickly can this serving configuration generate after a long prompt has populated the KV cache?

In the available 28B benchmark-sibling comparison:

| Measurement | Path and configuration | Decode throughput |
|---|---|---:|
| Depth-0 baseline | `llama-bench`, `ncmoe=18`, `--moe-cache on` | **<!-- claim:depth0_tg -->46.0<!-- /claim --> t/s** |
| Deep-context, steady state | server request, `ncmoe=17`, `--moe-cache on` | **<!-- claim:deep_tg -->15.3<!-- /claim --> t/s** |
| Deep-context, first generation | server request, `ncmoe=17`, `--moe-cache on` | 15.7 t/s |

The two server rows come from a long prompt in a 64k-context configuration. The 15.7 t/s row is the first generation after prefill; the 15.3 t/s row is steady-state decode with the prompt cached.

![Bar chart comparing depth-0 and deep-context decode throughput](figures/fig_deep_vs_depth0.png)

**Figure 2 — Depth-0 vs deep-context decode.**
*Data: [deep_vs_depth0.csv](results/deep_vs_depth0.csv) · source rows: [data/sources-bench.json](data/sources-bench.json) §06/§07 · supporting track: 28B benchmark sibling (`Qwen3.6-28B-REAP20-A3B-Q4_K_M`, RTX 3060 12 GB). The three bars are the table rows. The rows do not carry matched VRAM readings.*

The depth-0 row differs from the deep-context rows in tool and offload setting as well as context state. In this comparison, the depth-0 result is about **3.0×** the steady-state deep-context decode rate (46.0 ÷ 15.3). That illustrates the size of the gap in these configurations; it is not an estimate of the context-length effect alone.

That does not make depth-0 benchmarking useless. It means that it should not be used as a proxy for the full request path when the workload routinely carries a large context. For an agentic workload, the relevant sequence is closer to:

```text
Large prompt
    ↓
Prefill: process prompt, populate KV cache
    ↓
Decode: generate with populated KV cache
```

A benchmark that times only the last step from an empty cache omits prompt processing and measures decode in a different runtime state. (`llama-bench` also excludes tokenization and sampling, so its figure is not an end-to-end request rate.)

For these reasons, the measurements below use the actual server request path wherever possible.

## III. Measurement protocol

The primary measurements use the server's OpenAI-compatible request path rather than a synthetic token-generation loop.

**The experimental unit is one request.** A retained run is a single `POST /v1/chat/completions` call with `stream: true` and the tool schemas offered to the server. It is not an aggregate over several requests: every value in [Section V](#v-mtp-at-64k-live-29b) is computed from the individual kept requests in [results/mtp_runs.csv](results/mtp_runs.csv).

The protocol applied the following controls:

- `cache_prompt: false` and a unique nonce per request.
- `cache_n == 0` on every retained run. `cache_n` is the number of prompt tokens the server was able to reuse from its prompt cache for that request, so zero means no prompt-prefix reuse and every measured prefill was cold *with respect to the prompt cache*. It does not mean that kernel, page, filesystem or expert-weight caches were cold: the shared `--moe-cache on` setting is an expert-weights cache and is part of both arms by design.
- The two arms run in blocked, reversed order, five consecutive kept requests per arm phase (below).
- Two warmups discarded at every server start, then five kept requests per arm phase — **10 kept requests per arm**.
- All kept requests reported, including outliers; per arm the mean, standard deviation (SD), standard error of the mean (SEM), and the difference interval.
- Peak VRAM and host RAM recorded per request; GPU utilization, power and swap were **not** sampled per request in this experiment; the utilization figure in [Section VI](#vi-kv-cache-and-configuration-limits) comes from a separate observation.

**How the reported numbers are defined.**

- **Deep decode (`tg_t_s`)** is the server's own `timings.predicted_per_second`. On every retained run it equals `predicted_n / predicted_ms × 1000`: generated tokens over the decode window, which opens after prompt processing, so generation of the first token is inside the window while prompt processing, tokenization and HTTP overhead are outside. It is a decode rate, not a request latency. The traces' `usage` block is empty, so the generated-token count has no second source to check against.
- **Prefill (`pp_t_s`)** is `timings.prompt_per_second`.
- **Request wall time (`wall_s`)** is the harness's end-to-end timing of one request, from the HTTP call to the last streamed chunk. It is measured client-side and includes tokenization, HTTP, sampling and scheduling overhead that the server's `prompt_s` and `decode_s` timings exclude; those two phase means sum to about 0.5 s less than `wall_s` in both arms.
- **MTP acceptance (`accept`)** is `timings.draft_n_accepted / timings.draft_n` for a request: the share of all draft tokens proposed in that request that the verifier accepted, using the server's totals across its speculative rounds. It excludes the verifier's own token. The arm statistic is the unweighted mean of the ten per-request rates on the MTP-on arm; the SD is across those ten requests. It measures draft quality, not a throughput multiplier.
- **Peak memory (`peaks.vram_mib`, `peaks.ram_used_mib`)** is the maximum observed by the ~1 Hz sampler during a request; a shorter spike would be missed. The reported host-RAM cost is the difference between the two arms' highest observed readings, not a mean, and the sampler's scope (process vs system) was not recorded.

For the primary multi-token prediction (MTP) comparison, the arms were interleaved in blocks:

```text
off → on → on → off
```

| Block | Execution order | Kept requests |
|---|---|---:|
| 1 | MTP off → MTP on | 5 + 5 |
| 2 | MTP on → MTP off | 5 + 5 |

Each block therefore contributes five kept requests per arm, 10 per arm in total. The arrows are arm phases, not individual requests: a phase is five consecutive requests issued against one server start, after its two warmups. The arm flags are set when the server starts, so each phase required its own server start, with the live service stopped while a block ran and restarted afterwards. The order shown above is reconstructed from the block summaries' `finished_at` stamps; the per-run traces carry no timestamps, so the ordering *within* a phase cannot be re-derived from them. Warmups and start counts are procedure-level; the traces carry no start markers, so they cannot be recounted from the published artifacts.

With five requests per phase and the two phases of each arm placed symmetrically in block order (off in positions 1 and 4, on in 2 and 3), a linear trend in block order contributes the same total to both arm means and cancels in the pooled difference to first order; both block differences are positive (Section V.B). This does not establish the same for a trend in elapsed time: phase start times and durations were not logged, so unequal gaps or block durations could break the symmetry. The reversal also does not remove within-phase changes, non-linear drift, thermal or background-load effects, or carryover from the preceding request. Nor are the ten kept requests of an arm ten independent draws: they are two groups of five, each from its own server start. The interval in Section V.B is therefore a request-level interval under this design, not a block-level or population interval.

The documented arms used identical shared serving settings and differed in their speculative-decoding configuration: the MTP-off arm used `--spec-type none`, while the MTP-on arm used `--spec-type draft-mtp --spec-draft-n-max 2`. All other flags, including `--moe-cache on`, `--n-cpu-moe 20`, and the KV-cache types, were shared. Arm flags were not captured in the per-run traces, so this matching rests on the documented run configuration rather than on trace metadata.

One initial on-block was discarded because a stale server from the off arm still held port 8097 when the on-block tried to start; the runner was switched to process groups and the block was rerun after cleanup. The exclusion is operational rather than performance-based: no kept request was excluded because of its measured result, and the rerun start followed the same two-warmup procedure. The discarded attempt is not in the published trace set, so this exclusion is documented in the run record rather than trace-verifiable. The kept set includes the slowest on-arm run (21.4 t/s).

A small-prompt smoke test and a depth-0 benchmark were not used as substitutes for the long-context measurement. Single-point measurements were also not used as the basis for the primary claims.

The complete protocol, including the comparison interval, is documented in
[METHODOLOGY.md](METHODOLOGY.md).

## IV. Expert cache and prefill

### A. Optimization of prefill

The first optimization target is prefill. This section uses the **28B benchmark sibling**, not the live 29B MTP profile.

When expert layers are kept on the CPU, the serving path has to access those experts during processing of the prompt. At long context, repeated expert access can become a substantial part of the prefill cost. The tested runtime provides an expert cache that uses spare VRAM to retain CPU-resident experts.

**Example at 1k context:**

| Configuration | Prefill |
|---|---:|
| `ncmoe=20`, cache off | **<!-- claim:cache_off -->308.5<!-- /claim --> t/s** |
| `ncmoe=20`, cache soft | 313.2 t/s |
| `ncmoe=20`, cache on | **<!-- claim:cache_on -->775.7<!-- /claim --> t/s** |

The three 1k rows share the context length and `ncmoe=20`; the bench ledger does not record the prompt text or sampling settings, so this is a configuration comparison rather than a fully traced A/B.

![Bar chart of the expert-cache effect on prefill](figures/fig_moe_cache.png)

**Figure 3 — Expert-cache effect on prefill.**
*Data: [moe_cache.csv](results/moe_cache.csv) · source rows: [data/sources-bench.json](data/sources-bench.json) §04 · supporting track: 28B benchmark sibling at 1k context. The source ledger is hand-transcribed; its provenance is documented in [EVIDENCE.md](EVIDENCE.md).*

The cache increased measured prefill by approximately **2.5×** in this comparison. The effect is much less useful for decode. Across the tested cache and `ncmoe` configurations, deep decode remained around **14–15 t/s**.

The distinction is important:

> **The expert cache is a prefill optimization in this study, not a deep-decode optimization.**

### B. Context length

Prefill also decreases as context grows, even when the configuration still fits in memory.

On the 28B benchmark sibling:

| Context | Prefill |
|---:|---:|
| 1k | 859 t/s |
| 64k | 712 t/s |
| 128k | <109 t/s |

![Line chart of prefill throughput versus context length](figures/fig_prefill_vs_ctx.png)

**Figure 4 — Prefill throughput vs context.**
*Data: [prefill_vs_ctx.csv](results/prefill_vs_ctx.csv) · source rows: [data/sources-bench.json](data/sources-bench.json) §06 · supporting track: 28B benchmark sibling. The 64k no-cache point is a timeout upper bound rather than a completed measurement.*

Two constraints become visible here.

First, the cache needs spare VRAM. It reserves memory adaptively, so a configuration with insufficient headroom cannot simply enable it.

Second, context length itself becomes increasingly expensive during prefill. The 128k measurement is particularly slow even though the sibling configuration can fit the context.

The practical consequence is that **memory capacity and prefill throughput have to be considered together**.

## V. MTP at 64k (live 29B)

For deep-context decode, the tested optimization is **multi-token prediction (MTP) speculative decoding**.

The model provides a draft head. The runtime can use:

```text
--spec-type draft-mtp
--spec-draft-n-max 2
```

to propose multiple tokens before the main model verifies them.

The primary experiment compares the MTP-off and MTP-on arms using the same live 29B configuration. The documented arms used identical shared serving settings and differed in their speculative-decoding configuration; all other flags, including `--moe-cache on`, were shared.

### A. Results

| Metric | MTP off | MTP on (draft budget 2) | Difference |
|---|---:|---:|---:|
| Retained runs | 10 | 10 | — |
| Prompt tokens, mean | 57,881 | 57,880 | — |
| Deep decode | **<!-- claim:mtp_tg_off -->14.79<!-- /claim --> t/s** | **<!-- claim:mtp_tg_on -->38.56<!-- /claim --> t/s** | **+<!-- claim:mtp_delta -->23.77<!-- /claim --> t/s (+<!-- claim:mtp_delta_pct -->160.7<!-- /claim -->%)** |
| Deep decode, SD | <!-- claim:mtp_tg_sd_off -->1.20<!-- /claim --> t/s | <!-- claim:mtp_tg_sd_on -->6.28<!-- /claim --> t/s | — |
| Prefill | 645.10 t/s | 637.39 t/s | −1.19% |
| Prefill, SD | <!-- claim:mtp_pp_sd_off -->6.34<!-- /claim --> t/s | <!-- claim:mtp_pp_sd_on -->6.21<!-- /claim --> t/s | — |
| MTP acceptance (draft tokens) | — | **<!-- claim:mtp_accept -->0.838<!-- /claim -->** (83.8%) | SD 0.070 |
| Peak VRAM | 10,595 MiB | 11,588 MiB | **+993 MiB** |
| Peak host RAM | 15,334 MiB | 17,030 MiB | **+1,696 MiB (~1.7 GiB)** |

Memory values are per-arm maxima of per-request observed peaks (~1 Hz); the host-RAM reading's scope was not recorded. Acceptance is the unweighted mean of the ten per-request draft-token rates on the MTP-on arm, with the SD across requests; it is not a share of generated tokens and not a throughput multiplier.

![Bar charts of mean prefill and deep-context decode for MTP off and on](figures/fig_mtp_ab.png)

**Figure 5 — MTP on/off at 64k.**
*Data: [mtp_ab.csv](results/mtp_ab.csv); difference interval: [mtp_comparison.csv](results/mtp_comparison.csv) · live 29B configuration, 10 retained runs per arm, approximately 57.9k-token prompts, cold prefill on every retained run. Bars are arm means with SEM error bars.*

The run-level distribution is shown below.

![Scatter plot of decode throughput for every kept MTP run](figures/fig_mtp_runs.png)

**Figure 6 — Every kept MTP run.**
*Data: [mtp_runs.csv](results/mtp_runs.csv), arm means: [mtp_ab.csv](results/mtp_ab.csv) · live 29B configuration. Each request used a unique nonce, so no off/on requests share an identical prompt; the comparison is unpaired. Points are individual requests grouped by block; short lines mark each arm's overall mean, not per-block means.*

The measured decode rate increased from **14.79 to 38.56 t/s**, a gain of **160.7%**, under the documented shared serving configuration.

Prefill changed by only **−1.19%**, below the study's approximately 4% operational noise threshold. That threshold is a guard against over-reading small deltas, not an equivalence test; the experiment does not establish that MTP leaves prefill unchanged.

The additional observed peak memory was:

- **993 MiB of VRAM** (10,595 → 11,588 MiB)
- **1,696 MiB of host RAM** (15,334 → 17,030 MiB, about 1.7 GiB)

The MTP-on configuration used 94.3% of the GPU's nominal 12,288 MiB VRAM, leaving 700 MiB (5.7%). That is limited headroom, an operational caution rather than demonstrated instability; validate the configuration under the intended workload.

**Where the request time goes.** Decode is only a small part of a request this large. Mean prefill was <!-- claim:mtp_prompt_s_off -->89.7<!-- /claim --> s (MTP-off arm) and <!-- claim:mtp_prompt_s_on -->90.8<!-- /claim --> s (MTP-on arm), against mean decode of <!-- claim:mtp_decode_s_off -->7.1<!-- /claim --> s and <!-- claim:mtp_decode_s_on -->3.2<!-- /claim --> s for a slightly longer MTP-on output (<!-- claim:mtp_pred_n_off -->104.9<!-- /claim --> vs <!-- claim:mtp_pred_n_on -->123.2<!-- /claim --> generated tokens per request). End-to-end wall time per request was <!-- claim:mtp_wall_off -->97.4<!-- /claim --> s and <!-- claim:mtp_wall_on -->94.5<!-- /claim --> s (SD <!-- claim:mtp_wall_sd_off -->4.13<!-- /claim --> s and <!-- claim:mtp_wall_sd_on -->2.46<!-- /claim --> s). The server's prefill and decode timings are independent phases; their means sum to about 0.5 s less than wall time in both arms, because `wall_s` also includes tokenization, HTTP, sampling and scheduling overhead. The mean wall times differ by about 2.9 s (about −3.0%), which is small relative to the spread of individual requests, and the MTP-on arm generated more tokens on average. Those means are reported descriptively: this study does not test a wall-time effect and makes no equivalence claim. The observed decode-throughput increase is large under the tested configuration, but it is a decode-throughput gain, not a proportionally faster request.

### B. Difference interval

The difference interval is **unpaired**: each request used a different nonce, so no off/on requests share an identical prompt; a paired analysis does not apply even though the experiment uses interleaved blocks.

For the two arms:

- MTP-off arm: SD <!-- claim:mtp_tg_sd_off -->1.20<!-- /claim --> t/s, SEM 0.38 t/s
- MTP-on arm: SD <!-- claim:mtp_tg_sd_on -->6.28<!-- /claim --> t/s, SEM 1.99 t/s
- combined nominal request-level difference interval: approximately **±<!-- claim:mtp_ci_half -->4.58<!-- /claim --> t/s**, i.e. **[19.19, 28.34] t/s**

The interval is a **nominal 95% Welch interval** computed from the individual request measurements under an assumption that those observations are independent. It uses a Welch standard error and the conservative multiplier `t(df=9)=2.262`; the Welch–Satterthwaite degrees of freedom are approximately 9.66. It is not a population interval for the effect of MTP. The half-width and endpoints are computed from unrounded run values and rounded independently; 23.77 + 4.58 = 28.35 while the reported upper endpoint is 28.34 for that reason. Ten runs per arm were retained; two warmups per server start were discarded; all retained runs were reported, including the 21.4 t/s outlier. One initial on-block was discarded for a port conflict with a stale server and rerun; it was not excluded for its result.

The independence assumption is the weak point rather than the arithmetic. The ten kept requests of an arm are two groups of five, each issued against its own server start ([Section III](#iii-measurement-protocol)), so within-phase correlation would make the interval narrower than the design warrants. With two server starts (one per block) per arm, a block-clustered interval is not estimable. The interval therefore describes request-level variation under this run design; it is not a block-level or population-level uncertainty interval, and it does not cover uncertainty from undocumented per-run configuration. Strengthening it requires independent server starts with randomized or counterbalanced arm order across starts; more requests inside one phase would not address the clustering.

The two experimental blocks moved in the same direction:

| Block | MTP off | MTP on | SD (off / on) | Difference |
|---|---:|---:|---:|---:|
| Block 1 | 15.04 t/s | 40.69 t/s | 0.36 / <!-- claim:mtp_on_sd_b1 -->2.42<!-- /claim --> t/s | +25.65 |
| Block 2 | 14.53 t/s | 36.42 t/s | 1.72 / <!-- claim:mtp_on_sd_b2 -->8.46<!-- /claim --> t/s | +21.89 |

These block means are descriptive; they are not used as a separate block-level confidence interval. The spread within a block is not constant: the on-arm SD is <!-- claim:mtp_on_sd_b1 -->2.42<!-- /claim --> t/s in block 1 and <!-- claim:mtp_on_sd_b2 -->8.46<!-- /claim --> t/s in block 2, and the retained 21.4 t/s run accounts for most of the block-2 spread — the other four on-arm runs in that block are tightly grouped. Because the two phases of each arm sit symmetrically in block order, a linear trend in block order cancels in the mean of the two differences to first order; elapsed-time symmetry is not established because phase start times and durations were not logged. The block-level table shows how much variation remains beyond that.

### C. What the result means

The strongest conclusion is deliberately narrow:

> **On this exact 29B / RTX 3060 / 64k / runtime configuration, the MTP-on arm with a draft budget of two tokens produced a large observed increase in deep-context decode throughput.**

The result is a measurement of this configuration: one model, one GPU, one context length, one prompt family, and one draft budget. It does not establish that the draft budget is optimal, that the same percentage will occur on another GPU or model, or that another prompt will produce the same acceptance rate. Acceptance is content-dependent, so the prompt is part of the experimental configuration.

Practically, the trade is 993 MiB of additional observed peak VRAM for a large decode-throughput gain, with about 700 MiB of headroom remaining and little change in end-to-end wall time because prefill dominates. The next step is replication across independent server starts with randomized or counterbalanced arm order, per-run configuration capture, and an analysis that accounts for phase-level clustering (Section V.B). The broader boundary conditions are in [LIMITATIONS.md](LIMITATIONS.md).

An earlier single-point observation put MTP-on decode near 30 t/s at approximately 54.6k prompt depth. The controlled A/B is the stronger evidence and supersedes that estimate as the principal measurement.

## VI. KV cache and configuration limits

### A. KV-cache quantization

The serving build used in this study supports quantized KV-cache types:

```text
turbo4
turbo3
```

The startup log also reports:

```text
TurboQuant rotation matrices initialized (128x128)
```

This establishes that the tested build contains the corresponding runtime implementation. KV-cache quantization is important to the configuration because reducing KV memory pressure helps make a 64k context possible within 12 GB of VRAM.

It was **not**, however, isolated as an independent throughput experiment. The study therefore does not establish a decode-throughput improvement attributable specifically to the KV-cache quantization settings. The appropriate interpretation is:

> **KV-cache quantization is part of the memory configuration that makes the workload fit; this study does not isolate its performance contribution.**

Startup evidence is retained in [data/raw/p4/probe-results.txt](data/raw/p4/probe-results.txt). Claims about the broader performance characteristics of the quantization method are outside the scope of this experiment.

### B. Memory limits and `ncmoe`

The serving parameters interact. An optimization measured on one configuration cannot automatically be transferred to another.

A smaller, non-MTP sibling configuration indicated that `ncmoe=17` was a useful operating point. On the live 29B MTP configuration, the same setting does not fit.

| Live 29B MTP load probe | `ncmoe` | Result |
|---|---:|---|
| Reference | 20 | **OK** |
| Candidate, cache on | 17 | **FAIL — OOM** |
| Control, cache off | 17 | **FAIL — OOM** |

Both failures occurred during initialization; the run record places them at MTP-context creation:

- cache-on failed while allocating a 128 MiB KV-cache allocation;
- cache-off failed while allocating 208 MiB of compute buffers.

These are load probes, not throughput comparisons. The OOM traces are retained in [data/raw/p4/probe-results.txt](data/raw/p4/probe-results.txt).

The result illustrates an important practical property of constrained inference:

> **The best memory split is a property of the complete model/runtime configuration, not simply of the model.**

Adding MTP changes the memory requirements enough to invalidate a previously measured `ncmoe` setting.

### C. GPU utilization is not enough to identify the bottleneck

During deep generation, the GPU was observed at approximately:

- **82% utilization**
- **107 W**
- against a **170 W** power limit.

These figures come from the deep-context measurement in [Section II](#ii-depth-0-throughput-and-the-long-context-gap) on the supporting 28B track. They were recorded during that generation, not sampled per request in the MTP A/B of [Section V](#v-mtp-at-64k-live-29b), where only VRAM and host RAM were sampled; the sampling tool used for the utilization and power readings was not recorded.

The GPU was therefore not fully saturated during the measured decode path.

This does not identify the remaining bottleneck. It only shows that utilization and power draw alone are insufficient to select the next optimization. A different kernel, runtime implementation, memory-access pattern, or architecture change could still affect decode performance.

The practical approach is consequently simple:

> **Measure the real request path, then optimize the resource that the measurement identifies.**

Configurations close to the VRAM limit deserve particular caution. A configuration that passes a smoke test with approximately 150 MiB of headroom may still be fragile. The `ncmoe=17` probe demonstrates the more important failure mode: infeasibility can appear during initialization, before throughput can be measured at all.

## VII. Recommended configurations

The measurements suggest three practical profiles.

| Profile | Context | Key settings | Measured result | Track |
|---|---:|---|---|---|
| **Agent default** | 64k | MTP on, expert cache on, `ncmoe=20` | 38.6 t/s decode, 637 t/s prefill, 11,588 MiB peak | Primary 29B |
| **Prefill-first** | 64k | Expert cache on | 308.5 → 775.7 t/s at 1k | Supporting 28B |
| **Maximum context** | 128k | Deeper CPU offload | Prefill below 109 t/s | Supporting 28B; not validated on live 29B |

### A. Agent default

**64k context · MTP · expert cache on**

```text
-c 65536
--n-cpu-moe 20
--spec-type draft-mtp
--spec-draft-n-max 2
--moe-cache on
```

This is the best-fitting configuration found for the live 29B profile:

- deep decode: **38.6 t/s**
- prefill: **637 t/s**
- peak VRAM: **11,588 MiB**
- additional peak host RAM for the draft context: approximately **1.7 GiB** (difference of observed maxima)

It is most appropriate for workloads that process long context and generate comparatively short responses.

### B. Prefill-first

**Keep the expert cache enabled.**

The recorded sibling measurement showed approximately **2.5×** higher prefill with the cache enabled at 1k context:

```text
308.5 t/s → 775.7 t/s
```

At 64k, the cache-off sibling configuration timed out with a prefill rate below 73 t/s, while cache-enabled configurations reached approximately 677–712 t/s depending on the tested configuration.

The cache comparison is on the 28B sibling, so this should be treated as supporting evidence rather than a 29B A/B result.

### C. Maximum context

**128k requires validation before adoption.**

The 28B sibling measurements indicate that a 128k context can fit with deeper CPU offload (~9.9 GiB), but prefill falls below 109 t/s.

That may be acceptable for a workload that reads very deep context and generates little output. It is a poor trade for workloads that repeatedly prefill large documents.

The 128k result was **not validated on the live 29B MTP profile**.

## VIII. Limitations

This is a single-machine engineering study.

The primary experiment covers one 29B-class model, one RTX 3060 12 GB, one host CPU and memory configuration, one runtime build, one 64k context, one long-prompt family, and one tested MTP draft budget (two tokens). The supporting cache, depth-0 and context experiments use a 28B benchmark sibling.

The study therefore establishes **configuration-specific measurements**, not general performance laws. In particular, it does not establish how these results transfer to other GPUs, other MoE models, other runtimes or other prompt families; whether the two-token draft budget is optimal; whether 128k is practical on the live 29B configuration; or whether throughput predicts coding-task correctness.

The full boundary of the claims is documented in [LIMITATIONS.md](LIMITATIONS.md).

## IX. Conclusion

The main result of this study is not that one particular optimization is universally better than another. It is that **long-context local inference has several distinct bottlenecks, and the useful optimization depends on which part of the request path is limiting.**

On the tested RTX 3060 configuration:

- depth-0 generation reached **46.0 t/s** on the 28B benchmark sibling, while decode after a long prompt in a 64k-context configuration was approximately **15.3 t/s**;
- the expert cache increased sibling prefill from **308.5 to 775.7 t/s** at 1k context;
- the MTP-on arm increased live 29B deep-context decode from **14.79 to 38.56 t/s** at a 64k context setting;
- the MTP arm cost approximately **993 MiB of additional peak VRAM** and **1.7 GiB of peak host RAM**;
- a memory split that worked on the sibling model did not automatically work on the live 29B MTP configuration.

The practical lesson is straightforward:

> **Benchmark the request path you actually care about.**

For long-context inference, that means measuring prefill and decode separately, at realistic context lengths, while tracking the memory required to make the configuration fit. For constrained hardware, an apparent optimization can simply move the bottleneck somewhere else. A cache can make prefill faster without changing decode. Increasing GPU residency can improve one phase while making the context impossible to load. MTP can materially improve decode while consuming the memory needed by other parts of the serving configuration.

The useful configuration is therefore not the one with the best isolated benchmark number. It is the one that fits, processes realistic context at an acceptable rate, generates fast enough for the workload, and produces useful results.

# Appendix

## A. Coding evaluation

This is a separate end-to-end utility check, not evidence for the throughput thesis. It uses a fixed 11-task suite and hidden test contracts; its intervals do not project to unseen coding tasks.

Throughput is a means, not an end. The end-to-end question: can this configuration do real work? We ran a **ClassEval-derived coding suite** — 9 class-level tasks plus 2 single-function tasks as a labelled lower bound — on the shipped sampling (temp 0.6), 3 runs per task, with the generated code executed against hidden `unittest` tests.

The verifier passed **11/11 canonical-solution self-tests** before model evaluation. All 14 failed runs executed successfully and produced code that failed one or more hidden tests. No truncation, extraction error, or tool error was observed. The results therefore establish **implementation failures under the benchmark's test contracts**; attributing individual failures to model capability versus prompt-contract ambiguity remains a limitation.

The results were:

| Metric | Result |
|---|---:|
| Run-level completion | **<!-- claim:coding_passes -->19<!-- /claim -->/<!-- claim:coding_runs -->33<!-- /claim --> = 57.6%** |
| Run-level Wilson 95% CI | 40.8–72.8% |
| Task-clustered bootstrap 95% CI | 27.3–84.9% |
| Tasks passing all 3 runs | **6/11 = 54.5%** |
| Tasks passing at least 1 run | **7/11 = 63.6%** |
| Test methods passed | **388/462 = 84.0%** |
| Class-level tasks | 13/27 = 48.1% |
| Single-function tasks | 6/6 = 100% |
| Median wall-clock time / task | 16.3 s |

![Bar chart of coding completion by task](figures/fig_coding_passrate.png)

**Figure 7 — Coding completion by task.**
*Data: [coding_summary.csv](results/coding_summary.csv) · raw traces: [data/raw/coding/](data/raw/coding/).*

The run-level Wilson interval treats all 33 runs as independent observations. They are not: the experiment contains 11 tasks × 3 repetitions. The task-clustered bootstrap therefore gives a more conservative view of uncertainty: **27.3–84.9%**. Neither interval estimates performance on unseen coding tasks. The suite contains only 11 selected tasks with hidden test contracts.

### What the coding results show

The distribution across task types is more informative than the overall percentage.

- **Single-function tasks:** 6/6. This provides a lower-bound sanity check for the pipeline and simple tasks; it is not a representative coding benchmark.
- **Class-level tasks passing all three runs:** 4/9.
- **Class-level tasks failing all three runs:** 4/9.
- **Test-method pass rate:** 388/462 = 84.0%.

The difference between whole-task completion and individual test-method success is notable. Many generated solutions were close to correct while still failing one or more requirements.

The failed class-level tasks included both contract-specific requirements and more demanding algorithmic requirements such as Manacher's algorithm and 2D interpolation.

The practical conclusion is therefore modest:

> **This configuration appears useful for bounded, testable coding tasks where generated code can be executed against a verifier.**

The experiment does not establish general coding ability, nor does it establish that higher inference throughput produces higher task success.

## B. Data, provenance, and reproduction

The repository keeps a separate evidence ledger so that the article does not need to carry provenance metadata through every paragraph. The evidence status of each published result, its source artifact, regeneration path, and remaining limitations are documented in [EVIDENCE.md](EVIDENCE.md).

The repository can audit and regenerate the published tables and figures from committed raw traces. The original inference runner and coding harness are private, so the experiment itself cannot be independently repeated from a fresh clone. That boundary is intentional and documented rather than hidden.

Every figure maps to its source table in [data/figure-sources.json](data/figure-sources.json). Headline values are bound to table cells by [data/claims.json](data/claims.json), and the release gate in `scripts/check_traceability.py` recomputes the tables and checks the article markers. Reproduction and regeneration instructions are in [REPRODUCTION.md](REPRODUCTION.md).
