# Methodology

The single rule this study follows: **measure the request path a real caller
uses, and report the uncertainty.** Synthetic micro-benchmarks are cheap and
misleading; we use them only as a negative control.

## Protocol (all throughput experiments)

1. **Real path; one run is one request.** Every throughput number comes from a
   single `POST /v1/chat/completions` call with `stream: true`, tool schemas
   offered from `GET /tools`, and `cache_prompt: false`. A run is one HTTP
   request, not an aggregate over several requests.
2. **Cold prefill.** A unique nonce per request guarantees `cache_n == 0`, so
   reported prompt-eval is a true cold prefill and not a cache hit. `cache_n` is
   the number of prompt tokens the server was able to reuse from its prompt
   cache for that request. Zero therefore rules out prompt-prefix reuse; it does
   not imply that kernel, page or filesystem caches were cold, and it does not
   touch the expert-weight cache behind `--moe-cache on`, which is shared by
   both arms by design.
3. **Arm discipline: blocked and reversed.** Arms run in `off → on → on → off`
   block order, where each arrow is a phase of five consecutive kept requests
   against one server start, with two warmups discarded before that start. The
   arm flags are startup flags, so every phase is its own server start, and the
   live service was stopped while a block ran and restarted afterwards. With
   five requests per phase and the two phases of each arm placed symmetrically
   in the run order, a linear trend across the run contributes the same total to
   both arm means and cancels in the pooled difference to first order. The
   reversal does not remove within-phase changes, non-linear drift, thermal or
   background-load effects, or carryover from the preceding request.
4. **Warmups discarded, ≥5 kept runs per arm.** Warmup runs are thrown away;
   every kept run is reported in the raw trace, including outliers.
5. **Report the mean, the SD, and the SEM**, and treat a delta smaller than the
   operational noise threshold (~4% in this lab) as no result.
6. **Sample the resource side per request** — peak VRAM and host RAM, recorded
   as the maximum observed by a sampler running about once per second during the
   request (a spike shorter than the sampling interval would be missed) —
   because a throughput win that does not fit in VRAM is not a win. GPU
   utilization, power and swap were **not** sampled per request in this
   experiment; they must not be cited as per-request measurements. The sampling
   tool for the VRAM/RAM peaks was not recorded, and neither was its scope
   (process vs system memory); only the 1 Hz interval is documented.

### What one run measures

| Field | Definition |
|---|---|
| `pp_t_s` | `timings.prompt_per_second` — prompt tokens ÷ prompt time (prefill only) |
| `tg_t_s` | `timings.predicted_per_second` — generated tokens ÷ decode time (decode only) |
| `wall_s` | harness timing of the whole request, HTTP call to last streamed chunk; includes tokenization, HTTP, sampling and scheduling |
| `prompt_s`, `decode_s`, `predicted_n` | the same timings converted to seconds and token counts per request |
| `accept` | `timings.draft_n_accepted / timings.draft_n` — per-request draft-token acceptance (MTP-on only) |
| `peaks.*` | per-request observed maxima of VRAM and host RAM at ~1 Hz |

`tg_t_s` excludes prompt processing, tokenization, sampling and HTTP overhead,
so it is a decode rate and not a request latency. It equals
`predicted_n / predicted_ms × 1000` on every retained run, and the decode window
opens after prompt processing, so generation of the first token is inside it.
The traces' `usage` block is empty, so the generated-token count has no second
source to check against. On this workload the request is prefill-dominated: a
cold ~58k-token prefill takes about 90 s while decode takes a few seconds. Time
to first token would be `prompt_s` plus the first decode step, and inter-token
latency is the inverse of `tg_t_s`; neither was recorded directly, so neither is
published as a measurement.

`accept` is the share of draft tokens proposed in a request that the verifier
accepted, using the server's totals for that request; it excludes the verifier's
own token. The arm statistic is the unweighted mean of the ten per-request rates
on the MTP-on arm, and the SD is across those requests. It describes draft
quality, not a throughput multiplier.

`wall_s` is client-side. The server's `prompt_s` and `decode_s` means sum to
about 0.5 s less than `wall_s` in both arms because tokenization, HTTP, sampling
and scheduling are outside the server phase timings.

`peaks.vram_mib` and `peaks.ram_used_mib` are the maxima observed by the ~1 Hz
sampler during each request. The reported host-RAM cost is the difference
between the arms' highest observed readings, not a mean; the sampler's scope
(process vs system) was not recorded.

### The noise threshold

The ~4% figure is an **operational heuristic** adopted in this lab, not a
confidence interval and not a substitute for the per-arm SEM. The control series
behind it was not retained, so its provenance is **UNVERIFIED**; treat it as a
guard against over-reading small deltas. The published per-run measurements,
SEM, and Student-t intervals are computed directly from the kept runs by
`scripts/build_results.py` and independently re-derived by
`scripts/check_traceability.py`.

### Comparison interval

Each arm's interval uses the two-sided Student-t multiplier with `n-1` degrees of
freedom (`n=10`, multiplier 2.262). The MTP difference interval is **unpaired**:
each request has a unique nonce, so the blocks are a design control, not a
paired sample. It uses

```
SE_diff = sqrt(s_off^2 / n_off + s_on^2 / n_on)
diff    = mean_on - mean_off
CI95    = diff +/- t(df=min(n_off, n_on)-1) * SE_diff
```

The conservative multiplier `t(9)=2.262` is used even though the
Welch–Satterthwaite degrees of freedom are about 9.66. The difference in means
is +23.77 t/s, with a nominal 95% interval of [19.19, 28.34] t/s (reported
elsewhere as ±4.58 t/s), computed under an independence assumption across
requests. The interval is in
[`results/mtp_comparison.csv`](results/mtp_comparison.csv); block-level means
are in [`results/mtp_blocks.csv`](results/mtp_blocks.csv). The gate recomputes
both tables from the raw traces.

All interval fields are rounded independently from the unrounded calculation,
so the displayed endpoints can appear asymmetric by 0.01 t/s around the
displayed estimate (`[19.19, 28.34]` around `23.77`). The conservative
multiplier yields a slightly wider interval than the Welch-df multiplier, by
design. The interval covers request-to-request variation under the documented
design; it does not cover uncertainty from undocumented per-run configuration.

The interval also treats the ten kept requests of an arm as independent
observations. They are two groups of five, each issued against its own server
start (protocol item 3), so within-phase correlation would make the interval too
narrow; with two server starts (one per block) per arm a block-clustered interval
is not estimable.
Read the interval as request-level precision under this design, not as a
block-level or population-level interval. Strengthening it requires independent
server starts with randomized or counterbalanced arm order across starts; more
requests inside one phase would not address the clustering.

Per-arm SD is published beside the SEM in
[`results/mtp_ab.csv`](results/mtp_ab.csv) and per-block SD in
[`results/mtp_blocks.csv`](results/mtp_blocks.csv); the on-arm spread differs
sharply between blocks (2.42 vs 8.46 t/s), which is why the block table is
reported next to the pooled interval.

### Configuration capture and run retention

The MTP arms used identical shared serving settings and differed in their
speculative-decoding configuration: the off arm used `--spec-type none`, the on
arm used `--spec-type draft-mtp --spec-draft-n-max 2`. All other flags,
including `--moe-cache on`, `--n-cpu-moe 20`, and the KV-cache types, were
shared. Flags were not recorded in the per-run traces; the matching comes from
the run record.

Two warmups were discarded at every server start, and each arm phase was its own
server start, so warmups are per phase rather than per experiment. One initial
on-block was discarded because a stale server from the off arm still held port
8097 when the on-block tried to start; the runner was switched to process
groups and the block was rerun after cleanup. That exclusion is operational, not
performance-based: no kept run was excluded for its result, and the rerun start
followed the same two-warmup procedure. The retained set includes the slowest
on-arm run (21.4 t/s). Per-run timestamps were not recorded in the traces, so
the ordering of requests *within* a phase cannot be reconstructed from them; the
block order comes from the block summaries' `finished_at` stamps. The same
limitation means a linear trend in elapsed time cannot be checked: cancellation
is supported for block order, not for wall-clock spacing.

## Fidelity: match the shipped sampling

Evaluation runs must use the sampling the deployment actually ships, taken from
the model profile, not from a memory of "what benchmarks usually use". For this
model the shipped coding/thinking preset is **temp 0.6, top_p 0.95, top_k 20,
min_p 0.0**. A draft that used temp 0.2 (a different, voice-oriented profile)
was discarded because it would have measured a configuration no caller uses.

## Coding evaluation

End-to-end completion, not a proxy metric:

- **Tasks.** 9 ClassEval classes (pinned revision) filtered to stdlib-only,
  solution ≤ ~120 LOC, ≤ ~32 test methods, plus 2 HumanEval-style single
  functions as a labelled floor.
- **Runs.** 3 per task; task order shuffled per run and reversed on even runs.
- **Scoring.** One turn per run; code extracted mechanically and executed
  against the task's hidden `unittest` tests. A run passes only if *all* hidden
  tests pass. No LLM-as-judge.
- **Harness self-test.** The canonical solutions are run through the same
  extract → write → `unittest` path and must pass 11/11 before any model call,
  so a failing run is a failing implementation, not a broken harness.
- **Headline.** Run-level completion with a Wilson 95% interval — labelled
  run-level because the 33 runs are 11 tasks × 3 and are not independent. We
  also report task-level counts and a task-clustered bootstrap interval, plus a
  test-method-level pass rate to separate "wrong" from "nearly right". Neither
  interval estimates performance on unseen coding tasks.

## Failure attribution

Failures are classified by cause before they are aggregated. A run is
behavioural only if the harness was healthy (`finish_reason` not truncated,
extraction succeeded, no tool/transport error). Counting a harness bug as a
model failure inflates every downstream number.

Rejected framings (recorded so they are not repeated):

- **Depth-0 token generation as a proxy for long-context serving.** Depth-0 is
  a valid baseline for model-only generation throughput, but it does not capture
  the full request path. In this study's comparison it overstated deep-context
  decode by ~3×; benchmark the actual path at representative context lengths
  when that is the question.
- **Small-prompt smoke tests as "baseline".** Very short requests can be
  dominated by per-request overhead; their aggregate tokens/s is not a proxy
  for long-context prefill or decode.
- **A single-point measurement.** Every quantitative claim here has ≥5 kept
  runs, interleaved arms, and a reported uncertainty.
