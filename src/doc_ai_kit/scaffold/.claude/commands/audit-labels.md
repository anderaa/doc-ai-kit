---
description: Check labels against the declared tasks and write the annotation rules
---

# audit-labels

## Entry conditions

- `tasks.yaml` parses and declares every task the project answers.
- `data/labels.jsonl` exists with at least some labeled documents: written by
  `import-labels`, or brought in by a project that already had labels.

## Why it exists

Two failures start here and are invisible later. The first is a label that does not match
the declared value space -- a state code outside the enum, a task id that no longer exists
after a rename. The second is an ambiguous definition: two people labeling the same document
differently because the question does not say which of two readings to take. Neither shows
up as an error; both show up as a task that will not go above 0.6 no matter what the
optimizer does.

## Steps

```
doc-ai-kit audit-labels
```

It reports, all at once rather than one per run:

- labels for undeclared tasks, and tasks with no label;
- values outside a declared enum;
- per-class support, with what each count can actually carry;
- the labeling mode of each document, so corrected and blind labels are visible.

Then write `data/annotation_rules.md`: for each task, the rule you actually applied,
including the edge cases you had to decide. This file is a gate for `compile`, because a
task whose rule was never written down cannot be scored consistently.

## Exit criteria

- No undeclared tasks, no out-of-enum values.
- `data/annotation_rules.md` written, with a rule per task and its edge cases.

## Next

`make-splits`.
