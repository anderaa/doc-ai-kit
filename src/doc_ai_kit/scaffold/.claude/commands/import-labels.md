---
description: Check the finished labeling sheet and write data/labels.jsonl
---

# import-labels

## Entry conditions

- `data/labels.xlsx` from `label-sheet`, with every row finished.

## Why it exists

A label that cannot be scored should be caught while the labeler still remembers the
document, not halfway through an optimization run. Every cell is read with the same
normalizers the scoring uses and stored in one form: dates as ISO 8601, numbers with their
unit, spans as offsets into the extracted text.

## Steps

```
doc-ai-kit import-labels
```

`--sheet path.csv` reads a CSV export instead, e.g. from Google Sheets.

**The sheet is imported whole or not at all.** Every problem in every row is listed at
once, and nothing is written until all of them are fixed:

- a sampled document with no row, or a row with nothing in it at all (no answer, no note);
- a value outside a task's allowed values, or a number or date that will not parse;
- a pasted passage that is not in the document's text;
- a `skip:` note with no reason.

Go through the list with the user. Most are typos. Warnings -- a passage that appears twice,
an unexpected unit -- do not block, but read them.

Each document's labeling mode comes from the plan, not the sheet: rows that were shown model
answers are `corrected`; all others are `blind`. The holdout is always blind.

Import again as often as needed; it replaces `labels.jsonl` from the sheet each time.

## Refusals

- A document labeled before would lose its label: pass `--force` if that is intended.
- A document in `splits.json` would lose its label: always refused.

## Exit criteria

- `data/labels.jsonl` written, with every sampled document labeled or skipped with a reason.

## Next

`audit-labels`.
