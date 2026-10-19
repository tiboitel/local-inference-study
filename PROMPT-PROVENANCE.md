# Prompt provenance

The original long prompt body and its tokenized representation are **not
published**, and no hash of them was captured at run time. This is a deliberate
evidence gap, not a reconstruction.

## What is known

- The MTP traces record `prompt_n` per request: 57,877–57,884 tokens.
- The mean is 57,881.4 tokens for the off arm and 57,880.3 for the on arm.
- Each request carried a unique nonce, which forced `cache_n == 0` (cold prefill).
- The request path was `POST /v1/chat/completions` with `stream: true`,
  `cache_prompt: false`, and the sampling preset in
  [`data/raw/mtp/config.json`](data/raw/mtp/config.json).
- `prompt_sha256` is `null` in that config; the full request text is not part of
  the trace.

## What is not known

- The exact prompt bytes, tokenizer output, or a hash of either.
- Whether a copy of the original prompt corpus still exists. Without the artifact,
  a post-hoc hash would be unverifiable and is not published.

## Public substitute

[`scripts/reproduce_mtp_ab.py`](scripts/reproduce_mtp_ab.py) generates a
deterministic synthetic passage, appends a fresh nonce, and targets a requested
token count. This is a **procedure reproduction**: it exercises the same cold-
prefill/MTP-on/off path, but it is not the original prompt and must not be
presented as reproducing the published magnitudes. Compare direction and
uncertainty only.

If the original prompt is later recovered and can be redistributed, publish the
exact body plus SHA-256 and tokenizer revision here, then label any rerun as a
separate dataset rather than overwriting the measured traces.
