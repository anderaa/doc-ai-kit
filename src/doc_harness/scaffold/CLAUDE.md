# {{PROJECT_NAME}}

A doc-harness project. Read `docs/protocol.md` once before starting; this file is the
short list of things that must not happen.

## Run `/status` first

Phase is derived from disk, not from this file. `doc-harness status` reports what is done,
what is next, and what is blocking. Do that before deciding what to work on.

## Invariants

1. **The holdout opens once.** `doc-harness holdout` writes a lock. A second evaluation
   refuses unless explicitly overridden, and the override is stamped into `REPORT.md`.
   Evaluating twice and keeping the better number turns the holdout into a second
   validation split, and the reported accuracy stops estimating anything.

2. **`data/` is read-only to every optimization path.** `labels.jsonl` and `splits.json`
   are ground truth. The harness verifies they are unchanged around every optimizer run.
   If a label is wrong, fix it deliberately, outside a run, and say so in `decisions.md`.

3. **The holdout is labeled blind.** Train and validation labels are corrected from
   baseline output, which is two to three times faster. Holdout labels are produced from
   scratch, without seeing any prediction. Anchoring on model output correlates the labels
   with what is being measured. `sample-labels` fixes the holdout before any model runs,
   and `label-sheet` never runs the model on it. Never show a holdout document's model
   output to whoever is labeling it.

4. **Never edit the installed `doc_harness` package.** Project-specific normalizers,
   matchers and extractors go in `custom/` and register through hooks. If something cannot
   be done from `custom/`, that is a harness bug worth reporting, not a file to patch.

5. **One variable per experiment.** Each run records what it changed. Changing two things
   means learning nothing from either.

6. **The budget is fixed up front.** `optimization.max_experiments` and `max_rollouts` in
   `config.yaml` are not raised because the results look close. Raising them is a recorded
   decision.

## How to write

`REPORT.md`, `decisions.md` and what you say here are read by people who did not build this.
Plain English, the way one engineer explains a result to another. No consulting or startup
vocabulary -- leverage, align, de-risk, stakeholder, surface as a verb, load-bearing, key
takeaway, worth flagging -- and no invented names or acronyms. Keep the technical terms that
carry meaning, and explain each one once. State what you did, what you found and what is
left; skip the praise and the summaries of your own reasoning.

## Decisions

Human decisions -- corrected versus blind labeling, borderline fuzzy matches, the support
floor, class collapsing, accepting a compiled program over the baseline -- are appended to
`decisions.md` as they are made, with the numbers they were made on.
