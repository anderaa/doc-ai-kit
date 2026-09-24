---
description: Spend the one-shot holdout measurement
---

# holdout

## Entry conditions

- A champion is pinned.
- Every holdout document is labeled **blind**.
- You have decided which program you are shipping. This is not an exploratory step.

## Why it exists

The holdout is one measurement, spent once. `doc-ai-kit holdout` writes a lock on first use;
a second evaluation refuses unless explicitly overridden, and the override is stamped into
`REPORT.md` next to the numbers.

That gate is not bureaucracy. Evaluating twice and keeping the better number turns the
holdout into a second validation split, and every number it produces stops estimating
out-of-sample accuracy. The report says so, in those terms, whenever an override is recorded.

## Steps

```
doc-ai-kit holdout
```

## Reading the gap

What matters is not the holdout number but the gap between validation and holdout:

| gap (val - holdout) | reading | action |
| --- | --- | --- |
| within CI | no detectable overfitting | ship |
| modest, consistent | mild overfitting, normal | ship, quoting the holdout number |
| large, few tasks | those tasks memorized split specifics | revert them to baseline |
| large, across the board | the champion fitted the validation split | fall back to a simpler champion |

The report also names the tasks the holdout **cannot** measure -- those whose rarest class
has fewer than about ten examples. Do not quote those numbers on their own; they rest on a
handful of documents and the interval around them is wider than the number.

## Exit criteria

- `runs/holdout/` has its lock, `metrics.json` and `failures.md`.
- The gap reading and its action recorded in `decisions.md`.

## Next

`production`, if the reading says ship.
