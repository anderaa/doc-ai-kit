---
description: Report what is done, what is next, and what is blocking
---

# status

Run this first, every session, before deciding what to work on.

## Why it exists

Phase is derived from disk -- `splits.json`, `annotation_rules.md`, `runs/`, the
leaderboard, the holdout lock -- and never from a stored value or from a guidance file.
Static guidance drifts out of sync with a half-finished project; a file that says "next,
make the splits" is wrong the moment someone makes them.

## Steps

```
doc-ai-kit status
```

## Reading the output

- **Phase** is the furthest step completed, not the step in progress.
- **Blocking** items must be resolved before the next command will run. They are the gates,
  and each one names the reason it exists.
- **Worth knowing** items are not blocking but change how the numbers should be read: a
  spent experiment budget, an overridden holdout, documents in a split that are not labeled.

## Exit criteria

You know which command to run next and why.
