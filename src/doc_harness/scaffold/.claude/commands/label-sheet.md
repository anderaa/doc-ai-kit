---
description: Write the labeling spreadsheet, prefilled with model answers outside the holdout
---

# label-sheet

## Entry conditions

- `data/label_plan.json` written by `sample-labels`.
- `models.task` set in `config.yaml`, unless you use `--no-prefill`.

## Why it exists

People label faster in a spreadsheet than in JSON, and nobody can type character offsets.
The sheet is where the labeling happens. `import-labels` turns it into `labels.jsonl`.

## Steps

```
doc-harness label-sheet
```

This runs the zero-shot program on every sampled document **outside the holdout**, then
writes `data/labels.xlsx`:

- one row per sampled document, one column per task;
- `mode = correct` rows hold the model's answers, to check and fix;
- `mode = blind` rows start empty: the holdout, plus any document the model failed on. The
  model is never run on a holdout document; the command refuses if asked to;
- a `guide` tab explains how to fill in each column, and each header's note shows the
  task's question.

The prefill uses the Batch API when `production.use_batch_api` is on: half the price, but
it can take a while. `--live` is faster at full price. `--no-prefill` spends nothing and
leaves every row blind. It is slower to label, and every document then counts as blind.

Answers are saved to `runs/prelabel/`, so an interrupted prefill resumes without paying
again.

## Filling it in

Open the file in Excel or Google Sheets. Every cell is text, so Excel will not drop leading
zeros or reformat dates.

- A blank cell means the document gives no answer.
- Several answers go in one cell, separated by semicolons.
- Numbers and dates can be written any usual way: `$1.25M`, `June 15, 2024`.
- A span: paste the passage, copied from the text file in the last column. The harness
  finds its position.
- Set `reviewed` to `yes` when a row is done, or `skip` with a reason in `notes` for a
  document that cannot be labeled (not a contract, unreadable scan).

## Rewriting the sheet

Refused if the sheet exists, because it may hold work not yet imported. Import first; then
`--force` rewrites it, and rows already imported keep their labels. A document once shown
model answers stays marked as corrected, even after a rewrite.

## Exit criteria

- Every row reviewed or skipped.

## Next

`import-labels`. It can run as often as you like while labeling is in progress.
