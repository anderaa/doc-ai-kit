---
description: Create the fixed train/validation/holdout assignment
---

# make-splits

## Entry conditions

- Labels audited and `data/annotation_rules.md` written.
- `splits.seed` set in `config.yaml` and not changed since labeling started.

## Why it exists

Splits are created from a fixed seed, and a holdout chosen after the fact, or re-rolled
when the numbers disappoint, is not a holdout.

If `sample-labels` drew the sample, the holdout was already fixed there, before any model
saw a document, so that it could be labeled blind. This command keeps that holdout as it
is, and stratifies only train and validation. It refuses while any holdout document is
neither labeled nor skipped with a reason. If a holdout document carries a class that no
training document has, it moves to train, and the command says which.

A project that brought its own labels, with no `label_plan.json`, gets the whole split
stratified here as before.

Two invariants are enforced in code, not by care:

- every document lands in exactly one split, and the splits cover the corpus;
- **no class appears in the holdout without appearing in train**. Measuring a class the
  program was never shown produces a number that describes nothing.

## Steps

```
doc-harness make-splits
```

Sizes follow the labeled-corpus size: 200 or more gives 50/25/25; 100-200 gives 45/25/30;
50-100 switches to five-fold cross-validation on 70% with a 30% holdout; below 50 the
command warns that there are too few points for automated search.

## The support-floor stop

For every class below `splits.support_floor`, the command stops and shows you: how many
examples it has, the 95% recall half-width at that support, and roughly how many documents
random sampling would need to reach the floor. Then it asks for one of four choices:

- **enrich** -- label more through a keyword or model-nominated stratum;
- **collapse** -- fold it into a neighbour or into `other` in `tasks.yaml`;
- **binary_detection** -- split it out as its own binary task;
- **report_unmeasured** -- keep scoring and reporting it, but drop it from the optimization
  target by adding it to `metric.excluded_classes` in `config.yaml`.

The choice and its rationale are appended to `decisions.md`.

## Enrichment strata

Random sampling cannot reach a rare class: at 2% prevalence, 30 examples needs about 1,500
documents. Three strata, each recorded with its inclusion probability:

1. **random** -- never skipped. It is the only unbiased corpus-level estimate you have.
2. **keyword** -- regex over the text cache. Model-independent, so it can surface examples
   the model would never find.
3. **model_nominated** -- baseline predictions of the rare class, plus low-confidence cases.
   Biased by construction: it cannot surface what the model misses, so recall estimated on
   it alone is inflated. Never use it as the only stratum.

Corpus-level metrics weight by inverse inclusion probability; per-class metrics do not. Both
are reported, each labeled with which it is.

## Exit criteria

- `data/splits.json` written and committed.
- A recorded decision in `decisions.md` for every class below the floor.

## Next

`run-baseline`.
