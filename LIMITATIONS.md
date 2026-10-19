# Limitations

An honest study states the boundary of its claims. Every result here is:
one model, one host, one GPU, one serving build, one context length, one prompt
family (except the coding suite). Nothing below is a universal law.

## Scope limits

- **Single configuration point.** 64k context, `ncmoe=20`, MTP `n-max=2`. The
  validated results do not describe 128k, other quantizations, or other `ncmoe`
  values. The 128k numbers cited in the article come from the bench sibling and
  are labelled there as not validated on the live profile.
- **Single model.** The specific 29B REAP distill. The expectation that the
  *directions* (depth matters; MTP helps decode; expert cache helps prefill)
  hold on other MoE models with constrained VRAM is a **hypothesis**, not a
  finding. The magnitudes will not transfer.
- **Single build.** `llama.cpp` moves fast. The MTP and KV-quantization behavior
  is revision-specific; a different build can move both the mean and the tail.

## Statistical limits

- **Small n.** ≥5 kept runs per arm for throughput; 3 runs per task for coding.
  The ~4% threshold is an operational heuristic whose control series was not
  retained (UNVERIFIED), not a confidence interval. The aggregate Wilson interval treats the 33 runs as
  independent and is therefore too narrow; it is reported as a run-level
  interval alongside a task-clustered bootstrap. Neither is an estimate of
  performance on unseen coding tasks.
- **Content-driven variance.** MTP acceptance depends on the text; a different
  corpus moves both the mean and the variance. The reported acceptance (0.838)
  is a property of this prompt family.
- **Outliers are kept.** The raw traces include every kept run, including a slow
  outlier (21.4 t/s on `on-b2-r4`, at a normal 0.79 acceptance), because
  excluding it would flatter the mean. The lowest-acceptance kept run (0.729)
  ran at normal speed, so slowness and low acceptance are different runs.

## Methodological limits

- **Deep-context prompt family.** The long-context experiments use one fixed
  ~58k-token body with a needle. Prefill and acceptance on other long inputs may
  differ.
- **Coding contracts.** ClassEval tasks are checked by hidden tests. Several
  classes have contract details a docstring-only prompt may under-specify; the
  canonical solution passing is the control, but it does not prove every failing
  run is the model's fault rather than the contract's.
- **Load-probe vs throughput.** The `ncmoe=17` result is an infeasibility proof
  (it OOMs on load), not a throughput comparison of a config that fits.

## Evidence gaps

- **Model artifact hash** and **prompt hash** were not captured at run time and
  remain `null`; they cannot be backfilled without the exact artifacts and a
  rerun.
- **Run-time GPU driver, power state, and background-load controls** were not
  logged. The driver and power values in `HARDWARE-AND-RUNTIME.md` are post-hoc
  captures and are labelled as such.
- **Independent reproduction** is not available: the runner and coding harness
  are private. The tables are regenerated from committed traces; the public
  client reproduces the MTP procedure, not the exact experiment.

## Not tested / not claimed

- Other GPUs, other VRAM tiers, multi-GPU.
- Other serving stacks (vLLM, TensorRT-LLM, MLX).
- Other agentic workloads (retrieval, long tool chains) beyond the coding suite.
- Cost in currency. Local inference has no per-token price; the real cost is
  slot time, which the wall-clock figures approximate.
