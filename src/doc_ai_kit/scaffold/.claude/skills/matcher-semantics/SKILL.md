---
name: matcher-semantics
description: What each matcher counts as a hit, and the abstention algebra behind every P/R/F1 in this project. Use when a score looks wrong, when adjudicating a borderline fuzzy match, or before changing a theta or tolerance.
---

# Matcher semantics

Matchers are pure functions applied identically to gold and prediction. They are the most
common source of fake results: a bad matcher makes a good prompt look broken and sends the
optimizer chasing a bug. When a number looks wrong, suspect the matcher before the prompt.

## The abstention algebra

Every comparison produces counts, not a verdict:

| gold | prediction | outcome |
| --- | --- | --- |
| null | null | **true negative** -- a correct abstention, and it is scored |
| null | a value | **false positive** |
| a value | null | **false negative** |
| a value | the same value | **true positive** |
| a value | a different value | **false positive *and* false negative** |

That last row is the one people get wrong. A wrong non-null answer is both a miss and a
spurious answer, and counting it as only one of the two inflates whichever of precision or
recall you are quoting.

### When the project wants a best guess instead

`metric.abstention` in `config.yaml` switches this algebra. Under `scored` (the default), a
refusal to answer costs recall only: it is a miss, not a falsehood. Under `best_guess`,
silence is not an available answer, so an abstention where gold has a value is charged as a
false positive too and reads exactly as badly as answering wrongly.

A correct abstention stays a true negative under both: if gold is null and the prediction is
null, the two agree. Switching this is a decision to record, because it changes what the
numbers mean and makes them incomparable with every run that came before it.

An answer the normalizer cannot interpret is **not** an abstention. "probably" on a binary
task stays a non-null wrong answer, because collapsing it to null would turn a false positive
into a false negative.

## Per matcher

- **identity** -- equality after canonicalisation. Enum members are reached through casing,
  punctuation squashing and the task's synonym map. A value outside the enum stays unmapped
  and can never equal a member.
- **exact** -- whitespace, casing and unicode folded, then strict equality.
- **entity_name / free_text** -- order-insensitive similarity at or above `theta`. Uses
  token-sort, not token-set: token-set scores a subset as a perfect match, which would make
  "Acme" match "Acme Industries" at 1.0 and every truncated answer score as correct. The
  strict-equality result is **always** attached alongside, and both appear in the report.
- **set** -- per-label counting for multilabel. Two empty sets are a true negative. `correct`
  means exact set match.
- **list** -- greedy bipartite pairing on the item matcher, best pairs first. A duplicated
  prediction cannot match the same gold item twice.
- **numeric** -- unit-normalize, then accept within `tolerance`, relative or absolute.
  Differing units are a mismatch.
- **date** -- compared at the task's declared granularity. A prediction coarser than gold is
  wrong unless `allow_coarser` is set.
- **span** -- character overlap; `correct` at or above `overlap_threshold`, and the counts are
  characters, which is what makes the token-level F1.

## Adjudicating a borderline fuzzy match

Start with `doc-ai-kit adjudicate <run>`. It lists every decision a threshold actually made --
including any acceptance strict equality would have refused, however far from theta the score
landed -- with raw and normalized values side by side. It reads saved predictions, so it is free.

When a fuzzy match sits near theta, do not move theta to make one document pass. Instead:

1. Look at the normalized forms in `failures.md`, not the raw ones. Most borderline cases are
   a normalizer gap, and fixing the normalizer fixes every document at once.
2. If the two strings genuinely refer to the same entity, that is a normalizer rule. Add it in
   `custom/`.
3. If they refer to different entities, theta is doing its job.
4. Only change theta if the same class of disagreement recurs across many documents, and
   record the change and its rationale in `decisions.md`.

## Adding a matcher or normalizer

In `custom/`, with `@register_matcher("name")` or `@register_normalizer("name")`. Never by
editing the installed package. Write the adversarial cases first: near-misses that must not
match, and surface variants that must.
