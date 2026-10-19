# Hardware and Runtime

Everything in this study was measured on a single consumer desktop. The point is
that the conclusions are about a **class** of hardware (a 12 GB Ampere GPU plus
enough host RAM to hold a 29B MoE model), not about an exotic rig.

## Host

| Component | Value | Why it matters |
|---|---|---|
| CPU | Intel i7-9700K, 8 cores / 8 threads | MoE expert matmuls run on the CPU; thread count is a real knob |
| RAM | 32 GB (31 GiB usable) | holds the non-resident expert weights for `n-cpu-moe` offload |
| GPU | NVIDIA RTX 3060, **12 GB (12,288 MiB)** | the binding constraint: KV cache + attention + resident experts must all fit |
| OS / driver | Arch Linux, kernel 7.2.8 | — |

**VRAM, not host RAM, is the limit.** The KV cache is VRAM-resident, so host RAM
does not scale with context length; a 128k configuration peaked near 10.7 GiB
VRAM. The two published 128k figures (10.7 GiB here, ~9.9 GiB in ARTICLE.md §9)
have not been reconciled; both are bench-sibling estimates, not live 29B
measurements.

## Model

| | |
|---|---|
| Model | `Qwen3.6-29B-REAP-Opus-Reasoning-Distill-MTP`, Q4_K_M GGUF |
| Architecture | sparse Mixture-of-Experts, ~29B total / small active per token |
| Context | 64k (`-c 65536`) |
| MTP | multi-token prediction draft head, `--spec-type draft-mtp --spec-draft-n-max 2` |

## Serving build

| | |
|---|---|
| Runtime | `llama.cpp` (turboquant KV-cache fork) |
| Revision | `bcb85fc3ae85efa0f5f392c6c880dfc524923860` |
| Mode | router mode with a per-model child process |
| KV cache | `turbo4` / `turbo3` (quantized KV) |

## Effective serving flags

```
-c 65536
--n-gpu-layers 99
--n-cpu-moe 20
--spec-type draft-mtp --spec-draft-n-max 2
--cache-type-k turbo4 --cache-type-v turbo3
--threads 8 --threads-batch 8
-b 2048 -ub 1024
-np 1
--moe-cache on
--reasoning-budget 1024
# sampling (shipped preset): temp 0.6, top_p 0.95, top_k 20, min_p 0.0,
#                            repeat 1.0, presence 0.0
```

`--n-cpu-moe 20` keeps 20 transformer layers' experts on the CPU and the rest on
the GPU. That number is the whole memory game: see
[results/ncmoe_sweep_28b.csv](results/ncmoe_sweep_28b.csv) and
[results/moe_cache.csv](results/moe_cache.csv).
