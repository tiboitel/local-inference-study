# Data-Hygiene Checklist

This repository is intended to be **published**. Every artifact must pass the
gate below before it ships. The gate is deliberately mechanical so it cannot be
argued away at release time.

## The gate

- [ ] **No secrets.** No API keys, tokens, passwords, private keys, credential
      file paths, or `.env` contents. Grep before every commit:
      `grep -rIiE 'api[-_]?key|token|secret|password|BEGIN .*PRIVATE KEY' .`
- [ ] **No internal host paths.** No `/home/<user>/...`, no internal hostnames,
      no tailnet names, no LAN IPs. Public text says "the host", "the GPU",
      "12 GB", never an absolute path or a machine name.
- [ ] **No third-party PII.** No names, emails, or handles of people who have
      not consented to appear.
- [ ] **Every figure is traceable.** Each number in `results/*.csv` maps to a raw
      trace in `data/raw/` or to a row in the traceability table
      (`data/traceability.md`). Run `python scripts/check_traceability.py`.
- [ ] **No fabricated numbers.** If a number is illustrative, it is labelled as
      such and not presented as measured. Prefer deleting a weak number to
      dressing it up.
- [ ] **Uncertainty is carried through.** A figure copied from a source with a
      CI or SEM keeps that uncertainty in the table and the prose.
- [ ] **Licenses.** Third-party datasets (e.g. ClassEval) are referenced by
      revision, not vendored, unless their license permits redistribution.

## Where the gate is enforced

1. `scripts/check_traceability.py` — every figure cell resolves to a source.
2. A pre-commit shell check for secrets/absolute paths.
3. A human read of `ARTICLE.md` and `README.md` for internal detail that survived
   the automated pass.

## Failure handling

If a figure fails the gate, the fix is to **remove or genericize the figure**,
not to weaken the gate. A published number that cannot be traced is worse than a
missing one.
