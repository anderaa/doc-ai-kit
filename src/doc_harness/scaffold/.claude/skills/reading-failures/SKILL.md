---
name: reading-failures
description: How to read failures.md and metrics.json without drawing the wrong conclusion. Use after any scoring run, and before deciding what to change next.
---

# Reading failures.md

## Read "Failed replies" first, if it is there

A reply that could not be read -- usually truncated at `models.max_tokens` -- is retried with
a fresh generation, and one that never succeeds is listed at the top of `failures.md`. It is
scored as a wrong answer on every task, never as an abstention, so it cannot earn credit where
the gold is null. But it is not the model's judgement and says nothing about the prompt: fix
the cause before reading anything below it. By default any such reply stops the run outright
(`evaluation.max_failure_rate: 0.0`), so seeing the section means someone raised that on purpose.

## Read the per-task vector before the aggregate

Every run records the per-task primary metric alongside the scalar. A rising aggregate that
hides a collapsing task is a common and expensive failure, and it is invisible if you only
read one number. Compare the vector against the previous run, task by task.

## Three errors per task, on purpose

`failures.md` samples at most three errors per task. That cap matters. Given the full
error dump, an optimizer -- or a person -- writes rules keyed to individual documents:
"when the filer is Acme, the counterparty is the other party." Those rules lift the
validation number and die on the holdout.

If three errors is not enough to see the pattern, the answer is to look at the confusion
matrix and the per-class counts, which describe every error, not to raise the cap.

## What each part tells you

- **Confusion matrix** -- where a class is going, not just that it is wrong. A class that
  consistently becomes one other class is a definition problem. A class scattered across
  several is usually a retrieval problem: the answer is not reliably in the text.
- **Per-class table** -- the `reads as` column says what that class's support can carry. A
  class marked "not measurable" should not drive a decision.
- **The null column** -- the `(null)` row and column in the confusion matrix separate "got it
  wrong" from "did not answer". Those need different fixes: a wrong answer is usually a
  definition or matcher problem, a refusal to answer is usually retrieval or truncation.
- **Strict beside fuzzy** -- for fuzzy tasks, both numbers appear. A large gap between them
  means the score is coming from the threshold, not from the program.

## Diagnosing a task that will not move

In this order, because each step is cheaper than the next:

1. **Is the matcher right?** Look at the normalized gold and prediction in the failure lines.
   If they look the same to you but score as wrong, it is a normalizer gap.
2. **Is the answer in the text?** Check `extraction_manifest.csv` for unread pages,
   transcribed pages and truncation on the failing documents. Open the cached text in
   `data/text/`: a page that is missing, cut off or misread cannot be fixed by a prompt.
3. **Is the question unambiguous?** Read `annotation_rules.md`. If the rule needed a paragraph
   of edge cases, the question in `tasks.yaml` probably does not carry them.
4. **Only then**, is it a prompt problem?

A task scoring near zero at every baseline is never step 4.

## What not to do

Do not read `failures.md` for the holdout and then change anything. The holdout is spent; a
change made in response to it is fitted to it, and there is no split left to check that.
