# Traceability

Every number published in this repo resolves to a raw result. This table is the
map. `scripts/check_traceability.py` enforces it.

| Figure / claim | CSV | Raw source | Uncertainty |
|---|---|---|---|
| Deep-context vs depth-0 TG | `results/deep_vs_depth0.csv` | long-context run traces | depth-0 vs 64k-deep means |
| MTP on/off at 64k | `results/mtp_ab.csv` | MTP A/B run traces (10 kept/arm) | SEM, 95% CI ±4.58 t/s |
| MoE cache prefill multiplier | `results/moe_cache.csv` | expert-cache run traces | prefill means |
| ncmoe fitting point | `results/ncmoe.csv` | load-probe traces | infeasibility, no CI |
| Coding completion | `results/coding_eval.csv` | 33 coding run traces | Wilson 95% CI 40.8–72.8% |
| Prefill-vs-context curve | `results/prefill_vs_ctx.csv` | context sweep traces | per-point means |

Legend: "run traces" are the per-run JSON files produced by the harness
(`data/raw/`, not committed). Any row added here must name the exact traces it
aggregates.
