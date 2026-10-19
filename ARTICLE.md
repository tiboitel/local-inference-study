# Long-Context Local LLM Inference on 12 GB VRAM: What Actually Moves Performance?

*A measurement study of a 29B-class Mixture-of-Experts model at 64k context on a
single consumer GPU. Every number traces to a raw run — see [data/traceability.md](data/traceability.md).*

---

## 1. The constraint: 12 GB, a 29B model, and a 64k context

The pitch for local LLMs is that you can run a serious model on hardware you
already own. The catch is arithmetic. A 12 GB GPU has to hold three things at
once:

1. the **weights** that live on the GPU,
2. the **KV cache**, which grows linearly with context, and
3. the **compute buffers** for the current batch.

For a Mixture-of-Experts model you get one lever: expert layers that do not fit
can live in host RAM and be streamed over PCIe per token. That trades VRAM for
PCIe bandwidth, and it is the hinge the whole configuration turns on. On this
box the split is set by `--n-cpu-moe N`: keep N transformer layers' experts on
the CPU, put the rest on the GPU.

The study below is a 29B MoE model (Q4_K_M) served at **64k context** with a
multi-token-prediction (MTP) draft head. Full flags are in
[HARDWARE-AND-RUNTIME.md](HARDWARE-AND-RUNTIME.md).

## 2. Why depth-0 is a poor proxy

The standard local-LLM benchmark, `llama-bench`, measures token generation
**at depth 0** — as if the model had no history. That is not the workload an
agent runs. An agent sends a large prompt (tens of thousands of tokens of
context, tools, and history) and then generates in its shadow.

The gap is not subtle:

| Measurement | Decode throughput |
|---|---|
| `llama-bench` token generation, depth 0 | **46.0 t/s** |
| After a full 64k prompt, measured on the server | **15.3 t/s** |

![Depth-0 vs deep-context decode](figures/fig_deep_vs_depth0.png)

The depth-0 number overstates real decode throughput by roughly **3×**. A
configuration chosen on depth-0 figures will be tuned for a workload nobody
runs. Small prompts are worse still: a 25-token request is about **97% fixed
per-request overhead** (~62 ms), so its tokens-per-second number says nothing
about prefill at all.

This is the study's premise: measure the request path, or do not claim a result.

## 3. The measurement protocol (and the experiments we rejected)

The protocol is deliberately boring, because the interesting failures come from
skipping it. Full detail in [METHODOLOGY.md](METHODOLOGY.md); the short version:

- **Real path.** `POST /v1/chat/completions`, `stream: true`, tool schemas
  offered, `cache_prompt: false`.
- **Cold prefill.** A unique nonce per request guarantees `cache_n == 0`.
- **Interleaved arms.** Blocks run `off → on → on → off`, so drift cannot look
  like a treatment effect.
- **Warmups discarded; ≥5 kept runs per arm**, all reported, outliers included.
- **Mean, SEM, and a ~4% noise floor.** A smaller delta is not a result.
- **Sample the resource side per request** — VRAM, RAM, GPU%, watts, swap.

Three framings were tested and rejected:

- **Depth-0 token generation as a headline** (see §2): it measures nothing the
  agent does.
- **Small-prompt smoke tests as a "baseline"**: ~97% overhead.
- **Single-point measurements**: every claim here has ≥5 kept runs and a
  reported uncertainty.

## 4. MoE expert cache: the prefill multiplier, and where it stops

When expert layers live on the CPU, every token's prefill has to fetch them over
PCIe. At long context that fetch dominates. The fix is an **expert cache** — a
spare-VRAM region that holds the hottest CPU-resident experts, eliminating
repeat round-trips during prefill. Measured on the benchmark-model sibling at
small context:

| Configuration (small ctx) | Prefill PP |
|---|---:|
| `ncmoe=20`, cache **off** | 308.5 t/s |
| `ncmoe=20`, cache **soft** | 313.2 t/s |
| `ncmoe=20`, cache **on** | **775.7 t/s** |

![Expert-cache effect on prefill](figures/fig_moe_cache.png)

The cache multiplies prefill by about **2.5×** (and 2× over the best
non-cache configuration). At 64k the effect is the difference between usable and
not: with the cache off, prefill collapses to **<73 t/s** (it timed out); with it
on, a well-fitting configuration prefills at **712 t/s**.

Prefill also falls with context even when it fits: it is attention-bound as
context grows, from 859 t/s at 1k to 712 t/s at 64k to under 109 t/s at 128k.

![Prefill throughput vs context](figures/fig_prefill_vs_ctx.png)

Two limits matter. First, the cache needs *spare* VRAM to reserve, and it
reserves adaptively (free-minus-reserve), so a configuration with little
headroom cannot use it. Second, the cache helps **prefill**, not decode: deep
decode is essentially flat (~14–15 t/s) across every cache and `ncmoe`
configuration. If you want decode, the cache is not your lever.

## 5. Speculative decoding: the measured optimum

The decode lever is **MTP speculative decoding**. The model ships a draft head;
`--spec-type draft-mtp --spec-draft-n-max 2` lets it propose tokens that the main
model verifies. A protocol-faithful on/off A/B at 64k:

| Metric | MTP off | MTP on (n-max 2) | Delta |
|---|---:|---:|---:|
| Deep decode (TG) | 14.79 t/s | **38.56 t/s** | **+23.77 (+160.7%)** |
| Prefill (PP) | 645.10 t/s | 637.39 t/s | −1.19% (inside noise) |
| MTP acceptance | — | 0.838 | — |
| Peak VRAM | 10,595 MiB | 11,588 MiB | **+993 MiB** |

![MTP on/off at 64k](figures/fig_mtp_ab.png)

Decode **more than doubles**. The prefill cost is inside the noise floor, so MTP
is not a prefill tax at this context. It is not free, though: it costs about a
gigabyte of VRAM (a draft head plus a draft context) and ~1.7 GiB of host RAM.

The optimum is `n-max=2`. Higher draft counts did not help and lower ones leave
throughput on the table; the sweep is in the raw evidence.

## 6. TurboQuant: theory vs implementation

The serving build is a fork that supports **quantized KV cache types**
(`turbo4` / `turbo3`). That quantization is part of why a 64k context fits in
12 GB at all — it shrinks the KV cache that would otherwise crowd out expert
weights.

But it is a **constraint to design within, not a throughput lever to pull**.
Nothing in these measurements shows KV quantization moving decode throughput;
deep decode is flat across configurations, and the only thing that moved it was
MTP. A source-level grep for the "TurboQuant" feature/kernel in the build
returns no hits, and the headline performance claims published around it are
contested. Treat it as the thing that makes the context fit, and measure
everything else.

## 7. Interactions and stability

The knobs are not independent, and that is where naive tuning breaks.

**MTP forces the memory split up.** The draft head and draft context live on the
GPU. On the live 29B MTP profile, `--n-cpu-moe 17` — the measured optimum on a
smaller, non-MTP sibling model — **fails at load**: it OOMs while creating the
MTP context, with the cache on *and* off. The fitting point is `ncmoe=20`. In
other words, an optimum measured on one model did not transfer to a sibling with
speculative decoding added.

| Load probe (live 29B MTP, 64k) | `ncmoe` | Result |
|---|---:|---|
| `ref20` | 20 | **OK** |
| `cand17` (cache on) | 17 | FAIL — OOM allocating KV cache (128 MiB) while creating the MTP context |
| `ctrl17off` (cache off) | 17 | FAIL — OOM allocating compute buffers (208 MiB) while creating the MTP context |

**The bottleneck is not always choosable.** During deep generation the GPU sits
at ~82% utilization and ~107 W of a 170 W budget — it is not saturated. The
remaining decode limit is not addressable through `ncmoe` or the expert cache;
those move prefill. This is why the honest recommendation is "measure, then pick
the lever," not "turn everything on."

**Stability at the edge.** A configuration can be within 150 MiB of OOM and
still "work" in a smoke test. The `ncmoe=17` result is the reminder: infeasibility
is discovered at load, before any throughput is possible, and a config that loads
today can fail on the next reboot or driver change.

## 8. Is it actually useful? A coding-task evaluation

Throughput is a means, not an end. The end-to-end question: can this
configuration do real work? We ran a **ClassEval-derived coding suite** — 9
class-level tasks plus 2 single-function tasks as a floor — on the shipped
sampling (temp 0.6), 3 runs per task, code executed against hidden `unittest`
tests. The harness self-tests 11/11 on canonical solutions before any model call,
so a failure is the model's, not the harness's.

| Metric | Result |
|---|---|
| Run-level completion | **19/33 = 57.6%** (Wilson 95% CI 40.8–72.8) |
| Test-method pass | **388/462 = 84.0%** |
| Class-level tasks only | 13/27 = 48.1% |
| Single-function floor | 6/6 = 100% |
| Median wall clock / task | 16.3 s |

![Coding completion by task](figures/fig_coding_passrate.png)

The shape of the result is the useful part. The model is **strong on
single-function and mid-size data-structure tasks** and **weak on multi-method
contracts with edge-case rules**: four of eleven classes never passed. When it
fails, it fails *partially* — 84% of individual methods pass even as whole
classes fail — which is the signature of almost-right code that misses a
contract detail (a remainder rule, a non-raising error path), not of a model that
cannot code.

Every one of the 14 failing runs was **behavioural**: no truncation, extraction
succeeded, no tool errors, and all 33 runs answered in a single turn. Zero
failures were mechanical. That is the claim a benchmark has to earn before it can
say anything about the model rather than the harness.

## 9. Recommended configurations

Three profiles, each measured or explicitly marked as unvalidated:

**Agent default — 64k, MTP, expert cache on (measured, shipped).**
`-c 65536 --n-cpu-moe 20 --spec-type draft-mtp --spec-draft-n-max 2 --moe-cache on`.
This is the best fit found: deep decode 38.6 t/s, prefill 637 t/s, 11,588 MiB
peak. Use it for agentic workloads that read long context and generate short
answers.

**Prefill-first — expert cache is mandatory (measured).** If your workload is
long prompts and short outputs, the expert cache is the whole ballgame: without
it, 64k prefill is unusable (<73 t/s). Keep the cache on and spend VRAM on it
before anything else.

**Max context — validate before adopting (bench-model evidence, not live).** A
128k context fits with a deeper CPU offload (~9.9 GiB), but prefill becomes
attention-bound (<109 t/s). If your workload only *reads* deep context and writes
a little, this is viable; if it prefills long documents, 64k is the honest
ceiling on this hardware.

## 10. Limitations, reproduction, and what's next

Everything here is **one model, one host, one GPU, one build, one context, one
prompt family** (except the coding suite). The *directions* — depth matters, MTP
helps decode, the expert cache helps prefill, memory optima do not transfer
across models — should generalize to MoE models on constrained VRAM. The
*magnitudes* will not. The full list is in [LIMITATIONS.md](LIMITATIONS.md).

Every figure traces to a raw result, and the trail is enforced by
`scripts/check_traceability.py` (exit 0 is the release gate). Reproduction steps
are in [REPRODUCTION.md](REPRODUCTION.md). The takeaway for anyone tuning local
inference: **measure the real path, report the uncertainty, and treat every
"optimum" as local to the exact model it was measured on.**
