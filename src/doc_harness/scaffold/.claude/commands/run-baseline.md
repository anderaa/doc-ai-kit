---
description: Record the three baselines a compiled program has to beat
---

# run-baseline

## Entry conditions

- `data/splits.json` exists.
- `models.task` set in `config.yaml`.
- Text cached for every document in the splits.

## Why it exists

Three baselines, not one: zero-shot, hand-written few-shot, and `BootstrapFewShot`. The
compiled program has to beat the **best** of them on the holdout, or shipping the baseline is
the honest move. Recording only the weakest baseline makes any later number look like
progress.

The hand-written few-shot demonstrations live in `programs/baseline.py` and are worth real
effort. If they are missing, the command says so loudly rather than quietly recording two
baselines and calling it three.

## Steps

```
doc-harness run-baseline
```

Each baseline is scored on validation and gets its own run directory with `metrics.json` and
`failures.md`.

## Reading the result

A task scoring near zero at **every** baseline is an upstream problem, and the report says so
explicitly. Check, in this order:

1. Is the answer present in the extracted text at all? Look at `extraction_manifest.csv` for
   unread or transcribed pages, and truncation.
2. Is the question unambiguous? Look at `annotation_rules.md` and at whether two people would
   label it the same way.
3. Is the matcher right? A broken matcher makes a correct answer score as wrong, and sends
   the optimizer chasing a bug.

None of those is fixed by optimization. Fix it before compiling, or you will spend the
budget learning nothing.

## Exit criteria

- Two or three baseline runs recorded.
- Any near-zero task diagnosed as extraction, definition or matcher, and fixed.

## Next

`compile`.
