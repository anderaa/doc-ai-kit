---
name: task-types
description: The nine task types, what each outputs, how each is matched and scored. Use when adding or changing a task in tasks.yaml, or when deciding which type a question should be.
---

# Task types

`tasks.yaml` is the source of truth. Adding a task means adding YAML; the signature, output
type, normalizer, matcher and report section are all generated from it.

| type | output | default matcher | primary metric |
| --- | --- | --- | --- |
| `binary` | bool | identity | P/R/F1 on the positive class |
| `multiclass` | one enum member | identity | macro-F1 + confusion matrix |
| `multilabel` | a subset of an enum | set comparison | per-label P/R/F1, macro-F1, exact-set-match |
| `extract_exact` | str | normalize then equality | P/R/F1 with abstention |
| `extract_fuzzy` | str | normalize then similarity >= theta | P/R/F1 at theta, always beside strict |
| `extract_list` | list[str] | greedy bipartite on the pair matcher | set P/R/F1 |
| `extract_numeric` | number + unit | unit-normalize then tolerance | P/R/F1 within tolerance |
| `extract_date` | ISO 8601, possibly partial | granularity-aware equality | P/R/F1 |
| `span` | character offsets | overlap >= threshold | token-level P/R/F1 |

## Choosing a type

- **A closed set of answers you wrote down** is `multiclass`, not `extract_exact`. The enum
  gives you a confusion matrix and per-class recall, which is where the actual problems show.
- **"Which of these apply"** is `multilabel`, not several `binary` tasks, unless each one
  genuinely needs its own tuned instruction. Multilabel abstains with an empty list, so it is
  never nullable.
- **A name** is `extract_fuzzy`, not `extract_exact`. Legal suffixes and punctuation vary and
  none of that is a substantive disagreement. But set `theta` deliberately -- there is no
  default, because a silent threshold decides what counts as correct.
- **A person's name** uses `normalizer: person_name`, not the default `entity_name`. The two
  strip different things: `entity_name` removes legal suffixes, which damages a surname like
  "Co"; `person_name` removes titles and credentials ("Dr.", "PhD"), which would damage an
  organisation like "Dr Pepper". Using the wrong one scores "Dr. Ana Ruiz" as a different
  person from "Ana Ruiz".
- **A date** is `extract_date`, never a string. Granularity is preserved: "2024" and
  "2024-06-15" are different claims, and whether the coarser one matches is per-task config
  defaulting to no.
- **A number** is `extract_numeric`, never a string. Differing units are a mismatch, never a
  silent conversion -- the harness will not guess an exchange rate it was not told about.

## Weights and groups

`weight` scales a task's contribution to the aggregate. `group` gives a task its own
predictor, and therefore its own tunable instruction -- DSPy tunes instructions per
predictor. Use it only for a task that genuinely needs its own instruction; every extra
group is another call per document.

## Nullability

Every task is nullable unless declared otherwise, and abstention is scored: null against null
gold is a true negative, a value against null gold is a false positive. If downstream
consumers would rather have a best guess than an abstention, that changes the metric and not
just the prompt. It is `metric.abstention` in `config.yaml`, and switching it is a decision
to record.
