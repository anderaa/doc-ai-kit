---
name: sample-sizes
description: What a given number of labeled examples can and cannot support, and how many documents it takes to reach a rare class. Use when setting the support floor, deciding whether to enrich, or writing a number into a report.
---

# Sample sizes

Support, not document count, is the binding constraint. A 500-document corpus with four
examples of a class measures that class no better than a 40-document one.

## What each support level carries

Approximate 95% Wilson half-widths for recall near 0.8:

| examples in split | half-width | usable for |
| --- | --- | --- |
| 5 | +/-29 pts | presence check only |
| 10 | +/-22 pts | detecting total failure |
| 30 | +/-14 pts | coarse comparison |
| 50 | +/-11 pts | a number with a caveat |
| 100 | +/-8 pts | a number |

With ten examples all correct, the true error rate could still be near 25%. "Ten for ten" is
not evidence of accuracy; it is evidence the class is not catastrophically broken.

## Reaching a rare class

At prevalence p, a random sample needs about **m/p** documents to yield m examples. A 2%
class needs roughly 1,500 documents for 30 examples. Random labeling cannot reach rare
classes, and no amount of care in labeling changes that arithmetic.

That is what the enrichment strata are for, and why each carries its inclusion probability:

- **random** -- the only unbiased corpus-level estimate. Never skipped.
- **keyword** -- model-independent, so it can find what the model cannot.
- **model_nominated** -- biased: it cannot surface examples the model misses, so recall
  computed on it alone is inflated. Never the only stratum.

Corpus-level metrics weight by the inverse inclusion probability. Per-class metrics are
unweighted on the enriched pool. The report labels which is which, and so should you.

## Macro-F1 and rare classes

Macro-F1 weights every class equally. A class with four examples swings the headline number
as hard as one with four hundred, and its own number is meaningless at that support. That is
why classes below `splits.support_floor` leave the optimization target while still being
scored and reported.

## Writing a number into a report

Quote the interval, not just the point estimate. If the rarest class in a task has fewer than
about ten examples, the harness marks that task unmeasurable and the report says so -- pass
that caveat on rather than dropping it, because the number will otherwise be read as though
it were measured.
