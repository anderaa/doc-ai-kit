---
description: Write REPORT.md, append to the ledger, and bank what worked
---

# close

## Entry conditions

- Production run complete and its gates read.

## Why it exists

Two things outlive the project: the report the client reads, and what the next project
inherits. Neither happens by itself.

## Steps

```
doc-harness close
```

It assembles `REPORT.md` from the baseline, holdout and production sections, and appends one
row to the shared `ledger.csv`: task types, N, K, baseline holdout score, compiled holdout
score, labeling hours, cost.

Then, by hand: push any instruction that worked into the harness's `fragments/` library,
filed by task type. An instruction that lifted a task here will lift the same task type
elsewhere, and rediscovering it costs another engagement.

## What close writes

- `REPORT.md`, assembled from the baseline, holdout and production sections, ending with a
  **Where everything is** section linking every artifact the project produced.
- `PROMPT.md`: the shipped prompt as a person can read it -- instructions, the question per
  task, and the demonstrations. Otherwise it exists only as a JSON string inside the compiled
  program, and it is the artifact people ask for most.
- A ledger row, if `--ledger path/to/ledger.csv` is given or `DOC_HARNESS_LEDGER` is set.
  Without one, close says so rather than quietly writing nothing. The holdout aggregate and
  the cost come from the run's own files, not from memory; `baseline_holdout` stays empty,
  because the baselines are never measured on the holdout.

**REPORT.md is regenerated on every close**, so anything written into it by hand is lost. Put
observations in `REPORT_NOTES.md` instead; close appends them under a **Notes** heading.

## Exit criteria

- `REPORT.md` written, quoting holdout numbers and naming what the holdout could not measure.
- `PROMPT.md` written, so the shipped prompt is readable without opening the program.
- A row in `ledger.csv`.
- Anything reusable filed in `fragments/`.
