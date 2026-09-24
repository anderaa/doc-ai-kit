# Project protocol

A walkthrough for the human running this engagement. The commands enforce most of what
follows; this explains why each rule is there, so you can tell the difference between a gate
that is protecting you and a gate that is in your way.

## The shape of the problem

A project is N PDFs, M questions to answer about each one, and K human-labeled documents to
tune and measure against. The work is not "write a good prompt". It is building a measurement
you can trust, and then optimizing against it. Almost every way this goes wrong is a way the
measurement quietly stopped being trustworthy while the numbers kept going up.

The flow:

```
pdfs -> text cache -> DSPy program -> typed outputs -> normalize -> match
                            ^                                        |
                            |                                        v
                      optimizer <----- metric <----------------- gold labels
```

## 1. Declare the tasks

Write `tasks.yaml`. Each task gets a type, a question, and a matching rule. The question is
what the model sees, so it carries the definition: not "the state", but "the US state whose
law governs this agreement, from the governing-law clause and not the mailing address; null
if no US state is named".

The type matters more than it looks. A closed set of answers is `multiclass`, which gets you a
confusion matrix; the same question as `extract_exact` gets you a single F1 and no idea which
class is failing. See the `task-types` skill.

## 2. Extract the text once

`doc-ai-kit extract` caches text to `data/text/` and writes `extraction_manifest.csv`.
Validation and production read the same cached text, because metrics computed on one
rendering of a document do not transfer to another.

Pages with no usable text layer -- scans -- are transcribed by Claude from an image of the
page, once `extraction.transcription.model` is set. Pages that still have no complete text
are counted in the manifest. That count matters twice: once in error analysis, where it
explains a task that scores zero, and once in production QA, where those documents are
routed to human review.

Declare truncation once, in `config.yaml`. Text truncated differently in validation and
production makes the validation numbers describe a system that was never shipped.

## 3. Label, and label the holdout differently

Labels come from correcting baseline output -- two to three times faster than working from
scratch -- **except the holdout**, which is labeled blind, from the document, without seeing
any prediction.

This is not fastidiousness. If holdout labels are produced by correcting the model's output,
the labels are correlated with what is being measured, and every number computed against them
is inflated. The package checks the labeling mode and blocks if the holdout is not blind.

The holdout therefore has to be chosen before the model runs on anything:

1. `doc-ai-kit sample-labels --count N` draws the documents to label, and the holdout among
   them, at random. No model has seen a document yet.
2. `doc-ai-kit label-sheet --prefill` or `--no-prefill` writes `data/labels.xlsx`: one tab,
   one row per sampled file, one column per task, plus `notes`. With `--prefill`, rows
   outside the holdout hold the model's answers to correct; with `--no-prefill`, every row
   starts empty. The holdout rows start empty either way -- they are shaded -- and the model
   is never run on them.
3. Fill in every row, in Excel or Google Sheets. A blank cell means the document gives no
   answer; `skip: <reason>` in `notes` sets a document aside. A span is labeled by pasting
   the passage; the package finds its position.
4. `doc-ai-kit import-labels` takes the finished sheet whole: it checks every cell, lists
   every problem at once, and writes `labels.jsonl` only when all of them pass.

Then `doc-ai-kit audit-labels`, and write `data/annotation_rules.md`: for each task, the rule
you actually applied and the edge cases you had to decide. It is a gate for `compile`, because
a task whose rule was never written down cannot be scored consistently.

## 4. Make the splits

`doc-ai-kit make-splits`, from a fixed seed committed to `splits.json`. The holdout is the one
drawn in step 3; only train and validation are divided here, stratified on the labels. Two
invariants are enforced in code: every document lands in exactly one split, and no class
appears in the holdout without appearing in train.

Then the support-floor stop. For every class with fewer labeled examples than the floor, the
command shows you the numbers and asks for a decision: enrich, collapse, split out as a binary
task, or report it as unmeasured. This is a real decision with real consequences and it is
recorded in `decisions.md`.

Read the `sample-sizes` skill before making it. The short version: with ten examples all
correct, the true error rate could still be near 25%, and at 2% prevalence you would need
about 1,500 documents to reach 30 examples by random sampling.

## 5. Record three baselines

Zero-shot, hand-written few-shot, and `BootstrapFewShot`. The compiled program has to beat the
best of them on the holdout, or the honest move is to ship the baseline.

Write the hand-written demonstrations properly, in `programs/baseline.py`, from documents you
have read. A weak hand-written baseline makes everything that follows look like progress.

A task scoring near zero at every baseline is an upstream problem -- the answer is not in the
text, the definition is ambiguous, or the matcher is broken. Optimization fixes none of those.
Fix it before spending the budget.

## 6. Optimize, within a fixed budget

`doc-ai-kit compile`, one variable per experiment, each recorded. Start with
`BootstrapFewShot` for a floor, then `MIPROv2` at light settings, then GEPA or SIMBA if
budget remains.

The budget is set up front and time-boxes the search. It is not raised because the results
look close: "one more run" is exactly how a validation split gets fitted.

Read the per-task vector, not only the aggregate. And read `failures.md` -- it samples three
errors per task on purpose, because rules written against a full error dump key themselves to
individual documents and die on the holdout.

## 7. Spend the holdout, once

`doc-ai-kit holdout`. It writes a lock. A second evaluation refuses unless overridden, and
the override is stamped into the report.

What you are reading is the gap between validation and holdout, against a fixed table:

| gap | reading | action |
| --- | --- | --- |
| within CI | no detectable overfitting | ship |
| modest, consistent | mild overfitting, normal | ship, quoting the holdout |
| large, few tasks | those tasks memorized specifics | revert them to baseline |
| large, across the board | fitted the validation split | fall back to a simpler champion |

The report also names the tasks the holdout cannot measure. Pass that caveat on. A number
built on three documents will otherwise be read as though it were measured.

## 8. Produce, and gate the output

`doc-ai-kit production` loads the saved program and never recompiles. It is checkpointed and
resumable, raw responses land on disk before post-processing, and a document that fails every
retry is recorded as a failure rather than dropped.

Then the QA gates: coverage, schema, class distribution against the labeled sample, null
rates against validation, and a random spot-check sample. A failing gate means the results are
not ready. Class distribution and null rates in particular usually mean the corpus differs
from the labeled sample in a way nobody noticed -- which is worth knowing before anyone quotes
a number.

Human review gets five routes: transcribed or unread pages, truncated, nulls on usually-answered tasks,
low-confidence, and a random slice. The random slice stays. The other four select documents
that are already suspect, so a quality estimate built on them alone reads worse than the
corpus really is.

## 9. Close

`doc-ai-kit close` writes `REPORT.md` and appends a row to the shared `ledger.csv`. Then push
any instruction that worked into doc-ai-kit's `fragments/` library, filed by task type.
Rediscovering it costs another engagement.

Write `REPORT.md` for people who did not build this. Plain English, no consulting or startup
vocabulary (leverage, align, de-risk, stakeholder, surface as a verb, load-bearing, key
takeaway), and no invented names. Keep precision, recall, holdout and confidence interval,
explaining each once. Say plainly which answers cannot be trusted and what the measurement
could not cover. Your own additions go in `REPORT_NOTES.md`, which close appends.

## Decisions worth stopping for

These are judgement calls, not defaults, and each is recorded in `decisions.md` with the
numbers it was made on:

- corrected versus blind labeling assignment;
- borderline fuzzy-match adjudication, and who adjudicates;
- the support floor, and what happens to each class below it;
- class collapsing;
- whether a partial date matches a finer gold value;
- whether downstream consumers want abstention or a best guess -- this changes the metric, not
  just the prompt;
- accepting the compiled program over the baseline.
