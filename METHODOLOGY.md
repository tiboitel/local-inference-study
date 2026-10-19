# Methodology

The single rule this study follows: **measure the request path a real caller
uses, and report the uncertainty.** Synthetic micro-benchmarks are cheap and
misleading; we use them only as a negative control.

## Protocol (all throughput experiments)

1. **Real path.** Every throughput number comes from `POST /v1/chat/completions`
   with `stream: true`, tool schemas offered from `GET /tools`, and
   `cache_prompt: false`.
2. **Cold prefill.** A unique nonce per request guarantees `cache_n == 0`, so
   reported prompt-eval is a true cold prefill and not a cache hit.
3. **Arm discipline.** Arms are interleaved in alternated then reversed block
   order (`off → on → on → off`), so drift cannot masquerade as a treatment
   effect.
4. **Warmups discarded, ≥5 kept runs per arm.** Warmup runs are thrown away;
   every kept run is reported in the raw trace, including outliers.
5. **Report the mean and the SEM**, and treat a delta smaller than the noise
   floor (~4% in this lab) as no result.
6. **Sample the resource side per request** — peak VRAM, host RAM, GPU
   utilization, watts, and swap — because a throughput win that does not fit in
   VRAM is not a win.

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

- **Depth-0 token generation as a headline.** Depth-0 throughput can
  substantially overestimate decode at long context (in this study's
  comparison, ~3×); tune against the request path that matches the intended
  workload.
- **Small-prompt smoke tests as "baseline".** A 25-token request is ~97% fixed
  per-request overhead (~62 ms); its t/s says nothing about prefill.
- **A single-point measurement.** Every quantitative claim here has ≥5 kept
  runs, interleaved arms, and a reported uncertainty.
