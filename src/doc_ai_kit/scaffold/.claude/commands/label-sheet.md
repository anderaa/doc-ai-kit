---
description: Write the labeling spreadsheet and walk the user through filling it in
---

# label-sheet

## Entry conditions

- `data/label_plan.json` written by `sample-labels`.
- `tasks.yaml` final: each task becomes a column, so changing tasks later means relabeling.
- `models.task` set in `config.yaml`, if the user chooses to prefill.

## Why it exists

The user labels in a spreadsheet: one tab, one row per file, one column per task, plus
`notes`. The package writes that sheet rather than having the user build it, so the file
names and column headers match exactly -- one typo in a file name and a row matches no
document.

## Steps

### 1. Ask the user to choose: prefilled or not

Do not choose for them; the command refuses until they do. Explain the trade-off plainly:

- **`--prefill`**: the model answers every row outside the holdout first, and the user checks
  and corrects each cell. Two to three times faster to label. Costs one model call per
  document (half price through the Batch API, which can take up to an hour; `--live` is
  faster at full price). The risk is anchoring: a tired labeler accepts a wrong answer that
  looks plausible.
- **`--no-prefill`**: every row starts empty and every label is the user's own reading of
  the document. Slower, costs nothing, and every label counts as blind.

Either way, the holdout rows start empty, and the model is never run on them.

```
doc-ai-kit label-sheet --prefill        # or --no-prefill
```

This writes `data/labels.xlsx`.

### 2. Tell the user exactly what to fill in

Walk through the sheet with them before they start, column by column, from `tasks.yaml`:

- **what each column asks**: the task's `question`, in plain words;
- **the format**, by task type:
  - `binary`: yes or no;
  - `multiclass`: exactly one of the allowed values, listed out. The column has a dropdown,
    and a synonym from `tasks.yaml` is fine too;
  - `multilabel`: allowed values separated by semicolons, e.g. `hardware; support`;
  - `extract_exact`, `extract_fuzzy`: the value as the document writes it;
  - `extract_list`: every value, separated by semicolons;
  - `extract_numeric`: the number with its unit, written any usual way (`$1.25M`,
    `1,250,000 USD`);
  - `extract_date`: any usual form (`2024-06-15`, `June 15, 2024`), or just the year or
    month if that is all the document gives;
  - `span`: paste the passage from the document. The package finds where it is.
- **the edge cases** the user has already decided -- these belong in
  `data/annotation_rules.md` as well, and writing them down now is cheaper than later.

Then the rules that apply to every row:

- **Shaded rows start empty on purpose.** Label them from the document alone, and do not
  look at any model output for them first, including anything you show them. They measure
  the final program.
- **Unshaded rows**, when prefilled, hold the model's answers: check every cell against the
  document, not just the ones that look wrong.
- **A blank cell means the document gives no answer.** Leave it blank rather than guessing.
- **`notes`** is for anything worth recording. To set a document aside -- not a contract,
  unreadable -- write `skip: <reason>`. A row with no answers at all needs a note, or the
  import treats it as not done yet.
- **The whole sheet is imported at once**, when every row is finished.

Each column header carries a note with its question and format, as a reminder.

### 3. Do not fill it in for them

Offer to explain a column, find a passage, or check a hard case against
`annotation_rules.md`. Do not write answers into the sheet, and never into shaded rows:
labels written by a model are not human labels, and on the holdout they would measure the
model against itself.

## Rewriting the sheet

Refused if the sheet exists, because it may hold work not yet imported. After an import,
`--force` rewrites it from the imported labels, with no model calls. A document once shown
model answers stays marked as corrected, even after a rewrite.

## Exit criteria

- Every row finished: answered, or noted, or skipped with a reason.

## Next

`import-labels`.
