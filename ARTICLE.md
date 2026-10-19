# Long-Context Local LLM Inference on 12 GB VRAM: What Actually Moves Performance?

> **Status:** draft skeleton (section outline locked; figures pending).
> Every bracketed figure is filled from `results/*.csv` and must pass
> [DATA-HYGIENE.md](DATA-HYGIENE.md).

## 1. The constraint: 12 GB, a 29B model, and a 64k context

The arithmetic of the problem. KV cache is VRAM-resident and grows with context;
expert weights can live on the CPU. Frame the three-way budget (KV cache +
attention + resident experts) that every knob below trades against.

## 2. Why depth-0 is a poor proxy

`llama-bench` depth-0 token generation reports [ ] t/s; the same model after a
real 64k prompt decodes at [ ] t/s. Depth-0 measures a workload nobody runs.
Small-prompt "baselines" are worse: a 25-token request is ~97% fixed per-request
overhead.

## 3. The measurement protocol (and the experiments we rejected)

The rules that make a number real: real request path, cold prefill, interleaved
arms, warmups discarded, ≥5 kept runs, SEM, a ~4% noise floor. What we threw out
and why (depth-0 headlines; smoke-test baselines; single-point claims).

## 4. MoE expert cache: the prefill multiplier, and where it stops

Long-context prefill [ ] → [ ] t/s with the expert cache on (~2.5–3×). Why it
helps prefill and not decode, and the reserve it must fit in.

## 5. Speculative decoding: the measured optimum

MTP on/off at 64k: decode [ ] → [ ] t/s (+160.7%) for [ ] prefill (−1.19%,
inside noise), acceptance 0.838, at +[ ] MiB VRAM. The n-max sweep and why 2 is
the optimum. [figure]

## 6. TurboQuant: theory vs implementation

A quantized-KV fork that promises more context for less memory. What it actually
ships in this build, and why it is a constraint to design around rather than a
lever to pull.

## 7. Interactions and stability

How the knobs interact: MTP adds a draft head and context on the GPU, which
forces `ncmoe` up; the 28B optimum OOMs on the 29B MTP profile. What fit, what
did not, and the stability of deep-context throughput under load.

## 8. Is it actually useful? A coding-task evaluation

End-to-end, on the shipped sampling: [ ]% of class-level runs (Wilson CI [ ]),
[ ]% of individual test methods, single-function tasks [ ]/6, class-level four of
eleven never pass. Failures are behavioural, not mechanical.

## 9. Recommended configurations

Three profiles: max-context (128k target), agent-default (64k MTP), and
latency-first (shallow). Each with the exact flags and the measured trade.

## 10. Limitations, reproduction, and what's next

One host, one model, one build; the full list in [LIMITATIONS.md](LIMITATIONS.md);
reproduction in [REPRODUCTION.md](REPRODUCTION.md); the open questions worth a
follow-up run.
