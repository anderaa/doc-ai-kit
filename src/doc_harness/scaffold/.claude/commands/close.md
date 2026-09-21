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

## Exit criteria

- `REPORT.md` written, quoting holdout numbers and naming what the holdout could not measure.
- A row in `ledger.csv`.
- Anything reusable filed in `fragments/`.
