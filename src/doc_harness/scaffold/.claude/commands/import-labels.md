---
description: Check the labeling sheet and write data/labels.jsonl
---

# import-labels

## Entry conditions

- `data/labels.xlsx` from `label-sheet`, with some rows marked `reviewed = yes`.

## Why it exists

A label that cannot be scored should be caught while the labeler still remembers the
document, not halfway through an optimization run. Every cell is read with the same
normalizers the scoring uses and stored in one form: dates as ISO 8601, numbers with their
unit, spans as offsets into the extracted text.

## Steps

```
doc-harness import-labels
```

`--sheet path.csv` reads a CSV export instead, for example from Google Sheets.

Every problem in every reviewed row is listed at once: a value outside the enum, a number
or date that will not parse, a pasted passage not found in the text, a skip with no reason,
a sampled document with no row. **Nothing is written until all of them are fixed.**
Warnings, such as a passage that appears twice or an unexpected unit, do not block.

Each document's labeling mode comes from the plan, not from the sheet. Rows that were shown
model answers are `corrected`; all others are `blind`. The holdout is always blind.

## Refusals

- A document labeled before would lose its label: mark it reviewed, or pass `--force`.
- A document in `splits.json` would lose its label: always refused.

## Exit criteria

- Every sampled document labeled or skipped with a reason, including the whole holdout.
  `make-splits` refuses while a holdout document is neither, because leaving out the hard
  ones quietly makes the holdout easier than the corpus.

## Next

`audit-labels`.
