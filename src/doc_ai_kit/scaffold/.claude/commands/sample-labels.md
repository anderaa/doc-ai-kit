---
description: Choose which documents to label, and fix the holdout among them
---

# sample-labels

## Entry conditions

- Text extracted: `data/text/` has one file per document.
- `splits.seed` set in `config.yaml`. It fixes the draw, so do not change it afterwards.
- No `data/labels.jsonl` yet. A project that brings its own labels skips this and the
  sheet, and goes straight to `audit-labels`.

## Why it exists

Documents outside the holdout are labeled by correcting the model's answers, which is two
to three times faster. The holdout must be labeled **blind**, without seeing any answer, or
its labels lean toward the model and every holdout number comes out too high.

So the holdout has to be known before the model runs on anything. This command draws the
documents to label, and the holdout among them, at random, before any model sees a document.
The holdout is random rather than stratified because nothing is labeled yet to stratify on.
A random holdout is also the one whose number is an honest estimate for the whole corpus.

## Steps

```
doc-ai-kit sample-labels --count 120
```

Pick the count from the `sample-sizes` skill: how many documents the rarest class you care
about needs, not how many you have time for. The holdout share follows the protocol table
(25-30%). `--holdout-share` overrides it.

Writes `data/label_plan.json` and records the draw in `decisions.md`.

## Redrawing

Refused once any labels exist. A sample redrawn after looking at documents drifts toward
the easy ones. `--force` redraws only while nothing is labeled.

## Exit criteria

- `data/label_plan.json` written.

## Next

`label-sheet`.
