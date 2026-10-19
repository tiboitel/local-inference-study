# Raw traces

These are the per-run traces behind `results/`. They are **committed and
sanitized**: host paths have been removed, run logs (`*.log`) and model weights
(`*.gguf`) are not committed.

## Layout

| Directory | Contents | Run count |
|---|---|---|
| `mtp/` | MTP on/off A/B at 64k, one JSON per request plus per-block summaries | 20 kept runs (10/arm) |
| `coding/` | Coding evaluation, one JSON per task/run | 33 runs (11 tasks x 3) |
| `p4/` | 29B MTP load probes; `probe-results.txt` holds `PROBE_*` and `OOM_*` evidence | 3 configs |

## What a trace records

Each MTP trace is the server response plus its `timings` block and per-run peak
VRAM/RAM (`nonce`, `wall_s`, `final_text`, `finish_reason`, `timings`, `peaks`,
`usage`). Each coding trace adds the task id, extracted answer, turn log, and
verdict.

The traces do **not** embed the model, runtime flags, sampling preset, seed, or
prompt. Those are recorded per experiment in `mtp/config.json`,
`coding/config.json`, and `p4/config.json`; fields that were not captured at run
time are explicitly `null` with a note. In particular, the model artifact hash
and the prompt hash are `null`: they were not captured and cannot be backfilled
without rerunning. The evidence labels and remaining gaps are listed in
[EVIDENCE.md](../EVIDENCE.md) and [PROMPT-PROVENANCE.md](../PROMPT-PROVENANCE.md).

## Sanitization and limits

- Absolute host paths and internal hostnames are stripped before commit.
- Server logs are not committed; the OOM evidence they contained was promoted
  into `p4/probe-results.txt` as `OOM_<config>=...` lines. The sanitized
  `BUILD_TURBOQUANT=` line records the start-up evidence that the tested build
  initialized TurboQuant rotation matrices.
- The runner and coding harness are **not published**. The traces can be
  audited, but they cannot be regenerated from this repository alone.
- Do not hand-edit a trace. Fix the harness, re-run, and re-sanitize instead.

## Integrity

`data/manifest.json` records a SHA-256 for every file in this directory, along
with the git commit and counts. Verify with:

```bash
python scripts/manifest.py --check
```
