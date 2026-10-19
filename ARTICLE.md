# Long-Context Local LLM Inference on a 12 GB RTX 3060: Measuring MTP, Prefill, and VRAM at 64k

*A measurement study of a 29B-class Mixture-of-Experts model at 64k context on a
single consumer GPU, with published raw traces and regenerated result tables.*

> **Read this first — scope and cost.** One model
> (`Qwen3.6-29B-REAP-Opus-Reasoning-Distill-MTP`, Q4_K_M), one host (RTX 3060
> 12 GB, i7-9700K, 32 GB RAM), one serving build (`llama.cpp` KV-cache fork at
> `bcb85fc`), one 64k context, and one fixed long-prompt family. The MTP result
> buys **+23.77 t/s of decode for +993 MiB of peak VRAM** and ~1.7 GiB of host
> RAM. The expert-cache result is measured on the 28B bench sibling, not the live
> 29B profile. These are configuration-specific findings, not universal speedup
> estimates.
>
> **Evidence labels:** **MEASURED**, **REGENERATED**, **REPRODUCIBLE**,
> **ESTIMATED**, **UNVERIFIED**. No measured run here is REPRODUCIBLE: the
> runner and coding harness are private, so tables can be rebuilt but the
> experiment cannot be independently repeated from a fresh clone. Definitions
> and per-result labels: [EVIDENCE.md](EVIDENCE.md).

## At a glance

**Executive summary.** This study measures one long-context serving configuration
on one consumer GPU, not a family of models or workloads. In the tested 64k
configuration, MTP speculative decoding roughly doubled deep decode while adding
about a gigabyte of peak VRAM; the MoE expert cache improved prefill by about
2.5× on the 28B bench sibling without moving deep decode; and a depth-0 benchmark
overstated deep decode by about 3× in the one comparison available. Those are
measured, configuration-specific effects. The tables can be regenerated from
committed raw traces, but the experiment itself is not independently
reproducible because the runner and coding harness are private.

- **MTP roughly doubles deep decode at 64k:** 14.79 → 38.56 t/s (+160.7%),
  while prefill is unchanged within noise. Cost: +993 MiB peak VRAM, ~1.7 GiB
  host RAM. *(MEASURED + REGENERATED; live 29B profile.)*
- **Depth-0 overstates long-context decode in this comparison:** 46.0 t/s at
  depth 0 versus 15.3 t/s behind a full 64k prompt, about 3×. *(MEASURED;
  both rows are 28B bench-sibling measurements, not the live 29B profile.)*
- **The expert cache is a prefill lever, not a decode lever:** 308.5 → 775.7
  t/s at 1k on the 28B bench sibling (~2.5×); deep decode stays ~14–15 t/s.
  *(MEASURED on the sibling; REGENERATED from a non-self-contained ledger.)*

![Three measured levers](figures/fig_overview.png)

*Data: [mtp_ab.csv](results/mtp_ab.csv),
[deep_vs_depth0.csv](results/deep_vs_depth0.csv),
[moe_cache.csv](results/moe_cache.csv) · raw traces:
[data/raw/mtp/](data/raw/mtp/) · rebuild: `python scripts/build_results.py &&
python scripts/make_figures.py`.*

The three panels are deliberately different experiments. The depth-0 and cache
panels come from the 28B bench sibling; only the MTP panel is the live 29B
profile. Read every panel with its configuration, not as one uniform benchmark.

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

![Tested serving path and memory split](figures/fig_setup.svg)

*Schematic, not a data plot: the request path and the measured memory split.
On the tested host the MTP-on peak was 11,588 MiB of 12,288 MiB VRAM, versus
10,595 MiB with MTP off. Exact flags: [HARDWARE-AND-RUNTIME.md](HARDWARE-AND-RUNTIME.md).*

## 2. Depth-0 benchmarks and the long-context gap

The standard local-LLM benchmark, `llama-bench`, measures token generation
**at depth 0** — with no history in front of the model. Depth-0 throughput is a
useful characterisation of a model's baseline generation rate, and it remains
the right reference when that is the question. It does not capture the full
effect of serving a long context.

An agent workload sends a large prompt (tens of thousands of tokens of context,
tools, and history) and then generates in its shadow. In the comparison
available here, that changes the decode rate substantially:

| Measurement | Decode throughput |
|---|---|
| `llama-bench` token generation, depth 0 | **46.0 t/s** |
| After a full 64k prompt, measured on the server | **15.3 t/s** |

*Source: 28B bench sibling. The depth-0 row is `llama-bench`; the deep row is a
server-measured request. These two rows do not carry a matched VRAM reading; the
live memory cost is reported with the MTP A/B in §5.*

![Depth-0 vs deep-context decode](figures/fig_deep_vs_depth0.png)

*Data: [deep_vs_depth0.csv](results/deep_vs_depth0.csv) · source rows:
[data/sources-bench.json](data/sources-bench.json) §06/§07 · rebuild:
`python scripts/build_results.py && python scripts/make_figures.py`.*

In this comparison the depth-0 number overstates deep-context decode by roughly
**3×**. That is a measured gap in one configuration, not a claim that depth-0 is
useless: it is the appropriate baseline for model-only generation throughput.
For workloads with large prompts, benchmark the actual request path at
representative context lengths. Small prompts are a separate trap: a 25-token
request is about **97% fixed per-request overhead** (~62 ms), so its
tokens-per-second number says nothing about prefill at all.

This study therefore treats depth-0 as a baseline, not as a proxy for the full
request path.

## 3. The measurement protocol (and the experiments we rejected)

The protocol is deliberately boring, because the interesting failures come from
skipping it. Full detail in [METHODOLOGY.md](METHODOLOGY.md); the short version:

- **Real path.** `POST /v1/chat/completions`, `stream: true`, tool schemas
  offered, `cache_prompt: false`.
- **Cold prefill.** A unique nonce per request guarantees `cache_n == 0`.
- **Interleaved arms.** Blocks run `off → on → on → off`, so drift cannot look
  like a treatment effect.
- **Warmups discarded; ≥5 kept runs per arm**, all reported, outliers included.
- **Mean, SEM, and a ~4% operational noise threshold.** A smaller delta is not
  treated as a result. The threshold is a lab heuristic, not a confidence
  interval; its control series was not retained.
- **Sample the resource side per request** — VRAM, RAM, GPU%, watts, swap.

Three framings were tested and rejected:

- **Depth-0 token generation as a headline** (see §2): it can substantially
  overestimate decode at long context and says nothing about the request path
  an agent runs.
- **Small-prompt smoke tests as a "baseline"**: ~97% overhead.
- **Single-point measurements**: every claim here has ≥5 kept runs and a
  reported uncertainty.

**Evidence labels.** Every result carries a label so observations, rebuilds,
and hypotheses stay distinct:

- **MEASURED** — a retained raw trace or cited source row exists.
- **REGENERATED** — the published table or figure is rebuilt from committed
  inputs by the supplied scripts.
- **REPRODUCIBLE** — an independent operator can repeat the experiment from the
  published procedure and artifacts. **No measured run qualifies today**: the
  runner and coding harness are private.
- **ESTIMATED** — inferred from a sibling model, another measurement, or an
  incomplete observation.
- **UNVERIFIED** — the evidence or procedure needed to check the claim is not
  available.

Mechanistic explanations — why the cache helps (fewer repeated CPU-expert fetches
during prefill) and why MTP helps (verified draft tokens) — are **hypotheses
consistent with the measurements**, not additional measurements. Statements
about other GPUs, models, corpora, or draft counts are labelled **ESTIMATED** or
**UNVERIFIED** rather than presented as findings.

Each figure and table also carries its provenance: **live 29B MTP**,
**28B bench sibling**, or **depth-0 bench**. Full definitions and the release
checklist: [EVIDENCE.md](EVIDENCE.md).

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

*MEASURED on the 28B bench sibling at 1k context, not on the live 29B MTP
profile; REGENERATED from a non-self-contained source ledger. The cache spends
spare VRAM, so the cost is memory headroom, not a separate throughput tax.*

![Expert-cache effect on prefill](figures/fig_moe_cache.png)

*Data: [moe_cache.csv](results/moe_cache.csv) · source rows:
[data/sources-bench.json](data/sources-bench.json) §04 · rebuild:
`python scripts/build_results.py && python scripts/make_figures.py`.*

The cache multiplies prefill by about **2.5×** (and 2× over the best
non-cache configuration). At 64k the effect is the difference between usable and
not: with the cache off, prefill collapses to **<73 t/s** (it timed out); with it
on, a well-fitting configuration prefills at **712 t/s**.

Prefill also falls with context even when it fits: it is attention-bound as
context grows, from 859 t/s at 1k to 712 t/s at 64k to under 109 t/s at 128k.

![Prefill throughput vs context](figures/fig_prefill_vs_ctx.png)

*Data: [prefill_vs_ctx.csv](results/prefill_vs_ctx.csv) · source rows:
[data/sources-bench.json](data/sources-bench.json) §06 · rebuild:
`python scripts/build_results.py && python scripts/make_figures.py`. Source: 28B
bench sibling; the 64k no-cache point is a timeout upper bound (ESTIMATED).*

Two limits matter. First, the cache needs *spare* VRAM to reserve, and it
reserves adaptively (free-minus-reserve), so a configuration with little
headroom cannot use it. Second, the cache helps **prefill**, not decode: deep
decode is essentially flat (~14–15 t/s) across every cache and `ncmoe`
configuration. If you want decode, the cache is not your lever.

## 5. Speculative decoding: the tested long-context decode lever

The decode lever is **MTP speculative decoding**. The model ships a draft head;
`--spec-type draft-mtp --spec-draft-n-max 2` lets it propose tokens that the main
model verifies. A protocol-faithful on/off A/B at 64k (*live 29B MTP profile*):

| Metric | MTP off | MTP on (n-max 2) | Delta |
|---|---:|---:|---:|
| Kept runs | 10 | 10 | — |
| Prompt tokens (mean) | 57,881 | 57,880 | — |
| Deep decode (TG) | 14.79 t/s | **38.56 t/s** | **+23.77 (+160.7%)** |
| TG SEM / 95% CI (t, n=10) | 0.38 / ±0.86 | 1.99 / ±4.50 | ±4.58 |
| Prefill (PP) | 645.10 t/s | 637.39 t/s | −1.19% (inside noise) |
| MTP acceptance | — | 0.838 (sd 0.070) | — |
| Peak VRAM | 10,595 MiB | 11,588 MiB | **+993 MiB** |

*MEASURED + REGENERATED. Live 29B MTP profile, 10 kept runs per arm, ~57.9k
prompt, interleaved `off → on → on → off`, cold prefill on every kept run.
Cost: +993 MiB peak VRAM and ~1.7 GiB host RAM. Not REPRODUCIBLE: the runner is
private.*

![MTP on/off at 64k](figures/fig_mtp_ab.png)

*Data: [mtp_ab.csv](results/mtp_ab.csv) · raw traces:
[data/raw/mtp/](data/raw/mtp/) · rebuild: `python scripts/build_results.py &&
python scripts/make_figures.py`.*

![Every kept MTP run](figures/fig_mtp_runs.png)

*Data: [mtp_runs.csv](results/mtp_runs.csv) · raw traces:
[data/raw/mtp/](data/raw/mtp/). Each request has a unique nonce, so the off and
on arms are **not paired**; the points show the full spread, and the short lines
are arm means. No paired-difference interval is claimed.*

Decode **more than doubles**, and the delta is well outside the combined 95% CI.
The prefill cost is inside the operational noise threshold, so MTP is not a
prefill tax at this context. It is not free: about a gigabyte of VRAM (a draft head plus a draft
context) and ~1.7 GiB of host RAM.

**This supersedes an earlier single-point estimate, and the two are
comparable.** An earlier estimate put MTP-on decode near 30 t/s at ~54.6k depth;
the A/B above measures 38.56 t/s at 57.9k. They are in the same depth regime; the
difference is that the new result is the rigorous one:

| | Earlier estimate | This A/B |
|---|---|---|
| Prompt depth | ~54.6k | 57.9k |
| Kept runs | 1 (single point) | 10 per arm |
| Arm order | none | interleaved `off→on→on→off` |
| Uncertainty | not reported | SEM + 95% CI |
| Cache state | not verified | `cache_n == 0` on every kept run |
| Acceptance | ~0.72 | 0.838 (sd 0.070) |

Both arms used the same model file, the same fixed prompt, the same server build
and flags, the same 64k context, and llama.cpp's own `timings` block; only the
speculation flag differed. So **+160.7% is the best validated result for this
exact configuration, not a universal MTP speedup** — acceptance is content-driven,
and a different corpus will move the mean and the variance.

The tested point is `n-max=2`. The published evidence covers only that value, so
this study does not claim it is the optimum over other draft counts.

## 6. KV-cache quantization: what this build did and did not show

The serving build is a fork that supports **quantized KV cache types**
(`turbo4` / `turbo3`), sometimes described as TurboQuant-related. The
quantization is part of why a 64k context fits in 12 GB at all: it shrinks the
KV cache that would otherwise crowd out expert weights.

The tested build's start-up log reports `TurboQuant rotation matrices
initialized (128x128)`, so a TurboQuant-related KV implementation **is present in
this build**. The absence of a particular source symbol would not, by itself,
prove that no equivalent implementation exists elsewhere in the source, in a
dependency, or in a different revision; the runtime evidence is the stronger
observation here.

Within this study, quantization is a **constraint to design within, not a
throughput lever that was isolated and measured**. Deep decode is flat across
every configuration tested, and the only tested configuration change that
materially moved it was MTP speculative decoding. The defensible claim is
therefore narrower than a verdict on the method: **the tested build did not
establish a measurable decode-throughput improvement attributable to the tested
KV-cache quantization settings.** The broader performance claims published around
the method are contested and were not evaluated here. Treat quantization as the
thing that makes the context fit, and measure the rest of the path separately.
*(Evidence: MEASURED for the configurations tested; UNVERIFIED for any claim
about the method beyond this build. Start-up evidence:
[probe-results.txt](data/raw/p4/probe-results.txt).)*

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

*Source: live 29B MTP profile; load probes, not a throughput comparison.
Evidence: MEASURED + REGENERATED; the OOM lines are in
[probe-results.txt](data/raw/p4/probe-results.txt).*

**The remaining bottleneck is not identified by utilization alone.** During deep
generation the GPU sits at ~82% utilization and ~107 W of a 170 W budget — it is
not saturated. That rules out `ncmoe` and the expert cache as decode levers (they
move prefill), and it establishes that utilization and power draw alone do not
point to a useful next optimization. It does **not** prove that no other
software, kernel, or architecture change could improve decode. This is why the
honest recommendation is "measure, then pick the lever," not "turn everything
on."

**Stability at the edge.** A configuration can be within 150 MiB of OOM and
still "work" in a smoke test. The `ncmoe=17` result is the reminder: infeasibility
is discovered at load, before any throughput is possible, and a config that loads
today can fail on the next reboot or driver change.

## 8. A separate question: coding-task evaluation

> **Scope note.** This section is an end-to-end utility check, not evidence for
> the throughput thesis. It uses a fixed 11-task suite and hidden test
> contracts; its intervals do not project to unseen coding tasks. It is kept
> separate so the inference-performance findings can be read on their own.

Throughput is a means, not an end. The end-to-end question: can this
configuration do real work? We ran a **ClassEval-derived coding suite** — 9
class-level tasks plus 2 single-function tasks as a labelled lower bound — on the
shipped sampling (temp 0.6), 3 runs per task, with the generated code executed
against hidden `unittest` tests. *Source: live 29B MTP profile. Evidence:
MEASURED + REGENERATED; not REPRODUCIBLE because the harness is private.*

The verifier passed **11/11 canonical-solution self-tests** before model
evaluation. All 14 failed runs executed successfully and produced code that
failed one or more hidden tests. No truncation, extraction error, or tool error
was observed. The results therefore establish **implementation failures under the
benchmark's test contracts**; attributing individual failures to model capability
versus prompt-contract ambiguity remains a limitation.

| Metric | Result |
|---|---|
| Run-level completion | **19/33 = 57.6%** (run-level Wilson 95% CI 40.8–72.8) |
| Task-clustered bootstrap, run rate | **95% CI 27.3–84.9%** (resamples the 11 tasks, not the 33 runs) |
| Tasks passing all 3 runs | **6/11 = 54.5%** |
| Tasks passing ≥1 run | **7/11 = 63.6%** |
| Test-method pass | **388/462 = 84.0%** |
| Class-level tasks only | 13/27 = 48.1% |
| Single-function floor | 6/6 = 100% |
| Median wall clock / task | 16.3 s |

*The run-level Wilson interval assumes 33 independent trials; the runs are 11
tasks × 3, so repeated runs of the same task are correlated and that interval is
too narrow. The task-clustered bootstrap respects the task structure and is
correspondingly wide. Neither interval is an estimate of performance on unseen
coding tasks — the suite is 11 selected tasks under hidden-test contracts.*

![Coding completion by task](figures/fig_coding_passrate.png)

*Data: [coding_summary.csv](results/coding_summary.csv) · raw traces:
[data/raw/coding/](data/raw/coding/) · rebuild: `python scripts/build_results.py
&& python scripts/make_figures.py`.*

The useful result is the **shape across complexity tiers**, not the single
headline:

- **Single-function tasks (lower bound):** 6/6. A sanity check that the pipeline
  and a simple task work — not a representative coding benchmark.
- **Class-level tasks that pass consistently:** 4 of 9 classes passed all three
  runs (string parsing, address handling, a small state machine).
- **Class-level tasks that fail repeatedly:** 4 of 9 never passed, on contract
  details (a remainder-distribution rule; a non-raising error path) or on
  algorithmic requirements (Manacher, 2D interpolation).
- **Partial credit is common:** 84% of individual test methods pass even as whole
  classes fail — the signature of almost-right code that misses one specific rule.

That 84% is a **descriptive statistic over the selected test methods**, which vary
in importance and difficulty. It is not a general measure of code quality and
should not be read as one.

**Practical recommendation.** This configuration is promising for **bounded,
testable coding tasks** — ones that ship with a verifier you can run. Broader
deployment would need task-specific evaluation and a measured verification
workflow; this benchmark shows that plenty of failures would be caught by one, but
it does not yet establish the benefit of adding it.

## 9. Recommended configurations

Three profiles, each measured or explicitly marked as unvalidated:

**Agent default — 64k, MTP, expert cache on (MEASURED, shipped).**
`-c 65536 --n-cpu-moe 20 --spec-type draft-mtp --spec-draft-n-max 2 --moe-cache on`.
This is the best fit found: deep decode 38.6 t/s, prefill 637 t/s, 11,588 MiB
peak, ~1.7 GiB host RAM for the draft context. Use it for agentic workloads that
read long context and generate short answers. *(MEASURED + REGENERATED; live 29B
MTP profile.)*

**Prefill-first — the expert cache is mandatory (MEASURED on the sibling).** If
your workload is long prompts and short outputs, the expert cache is the whole
ballgame: without it, 64k prefill is unusable (<73 t/s, a timeout upper bound).
Keep the cache on and spend VRAM on it before anything else. *(MEASURED on the
28B bench sibling; the live 29B profile also runs cache-on, but the controlled
cache A/B is sibling evidence.)*

**Max context — validate before adopting (ESTIMATED, not live).** A 128k context
fits with a deeper CPU offload (~9.9 GiB), but prefill becomes attention-bound
(<109 t/s). If your workload only *reads* deep context and writes a little, this
is viable; if it prefills long documents, 64k is the honest ceiling on this
hardware. *(Source: 28B bench sibling; not validated on the live 29B MTP
profile. Evidence: ESTIMATED/UNVERIFIED for the live profile.)*

## 10. Limitations, reproduction, and what's next

Everything here is **one model, one host, one GPU, one build, one context, one
prompt family** (except the coding suite). The *directions* — depth matters, MTP
helps decode, the expert cache helps prefill, memory optima do not transfer
across models — are **hypotheses** about other MoE deployments, not findings
established by this study. The *magnitudes* are local to this configuration. The
full list is in [LIMITATIONS.md](LIMITATIONS.md); per-result labels are in
[EVIDENCE.md](EVIDENCE.md).

Every figure traces to a raw result, and the trail is enforced by
`scripts/check_traceability.py` (exit 0 is the release gate). Reproduction steps
are in [REPRODUCTION.md](REPRODUCTION.md). The takeaway for anyone tuning local
inference: **measure the real path, report the uncertainty, and treat every
"optimum" as local to the exact model it was measured on.**

## Conclusion

### Observed in this configuration *(MEASURED + REGENERATED)*

On a single RTX 3060 12 GB system, long-context local inference is governed by
distinct prefill, decode, and memory constraints. At 64k with the live 29B MTP
profile, MTP speculative decoding produced the largest measured decode
improvement (14.79 → 38.56 t/s) for +993 MiB peak VRAM. The controlled
expert-cache A/B on the 28B bench sibling produced a ~2.5× prefill improvement at
1k, while deep decode stayed flat. Memory headroom determined which
configurations could load, and an optimum measured on the sibling did not
transfer automatically to the live MTP profile.

The end-to-end coding evaluation adds a separate measure of practical utility:
the model completed **19 of 33** single-turn runs, passing every hidden test in
those runs, while passing **84%** of the evaluated test methods overall. The gap
between those two numbers is why plausible code and successful completion are not
interchangeable.

### Hypotheses, not findings *(ESTIMATED / UNVERIFIED)*

The expectation that these directions hold on other GPUs, other MoE models,
other quantizations, or other prompt families is a reasonable hypothesis, not a
result of this study. The same applies to any suggestion that 128k is viable on
the live profile, that MTP `n-max=2` is the optimum draft count, or that
throughput predicts task correctness. None of those was tested here.

These findings do not show that throughput predicts correctness. They show why
both must be measured. A useful local inference system has to fit in memory,
process realistic context efficiently, generate at an acceptable rate, and
satisfy the requirements of the task it is given.

The resulting engineering principle is simple: **measure the real request path,
evaluate actual task outcomes, preserve uncertainty, and treat every
configuration optimum as specific to the model, hardware, runtime, and workload
under which it was established.**
