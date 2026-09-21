# BUILD.md — doc-harness

Implementation brief. Build a Python package, `doc-harness`, that runs document
classification and extraction projects: a fixed harness plus a per-project scaffold,
with DSPy doing prompt optimization.

Read this whole file before writing code. Build in the milestone order at the end.

---

## 1. What the system does

A project is: N PDFs, M tasks to answer about each PDF, K human-labeled PDFs to tune
and measure against. Tasks vary per project — binary, multiclass, multilabel, several
kinds of extraction. Nothing in the harness may assume specific tasks, specific task
counts, or specific corpus sizes.

Flow:

```
pdfs -> text cache -> DSPy program -> typed outputs -> normalize -> match
                            ^                                        |
                            |                                        v
                      optimizer <----- metric <----------------- gold labels
```

Each project ends with a holdout-measured accuracy report and a full-corpus output file.

## 2. Two repositories

**Harness** (this build): installed as a versioned package, shared by all projects.
**Project directory**: scaffolded per engagement, pins an exact harness version.

The harness is never hand-edited from a project. Project-specific code goes in the
project's `custom/` and registers through hooks.

### Harness layout

```
src/doc_harness/
  __init__.py
  registry.py          tasks.yaml -> task specs, DSPy signatures, output types
  extract.py           pdf -> text cache
  normalize.py         per-type normalizers
  match.py             per-type matchers
  metric.py            DSPy metric + GEPA feedback variant
  evaluate.py          dspy.Evaluate wrapper -> metrics.json, failures.md
  optimize.py          optimizer runs, leaderboard, budget enforcement
  produce.py           full-corpus run + QA gates
  report.py            holdout and production reports
  hooks.py             registration points for project custom/
  state.py             derive project phase from disk
  cli.py               newproject, status, phase commands
  scaffold/            files copied into a new project (section 9)
fragments/             instruction fragment library, by task type
ledger.csv             one row per completed project
tests/
```

### Project layout (what the scaffold produces)

```
pyproject.toml         doc-harness==X.Y.Z   exact pin
uv.lock
CLAUDE.md
docs/protocol.md
.claude/commands/      status, audit-labels, make-splits, run-baseline,
                       compile, holdout, production
.claude/skills/        task-type catalog, matcher semantics, reading failures
tasks.yaml
config.yaml
custom/                project-specific normalizers/matchers/extractors
decisions.md           human decisions, recorded as made
data/                  gitignored
  pdfs/  text/  labels.jsonl  splits.json
  extraction_manifest.csv
  annotation_rules.md
programs/
  baseline.py
  compiled/{exp_id}.json
runs/
  {exp_id}/            config.json, metrics.json, failures.md, raw/
  leaderboard.md
  holdout/             .lock, metrics.json, REPORT section
  production/          raw/, outputs.jsonl, qa_report.md
REPORT.md
```

## 3. Task registry — `registry.py`

`tasks.yaml` is the source of truth. Code generates signatures, output types,
normalizers, matchers and report sections from it. Adding a task = adding YAML.

Task entry fields: `id`, `type`, `question`, `output`, `nullable`, `match`, `weight`,
optional `group`.

| type | output | default matcher | primary metric |
| --- | --- | --- | --- |
| `binary` | bool | identity | P/R/F1 on positive class |
| `multiclass` | one of enum | identity | macro-F1 + confusion matrix |
| `multilabel` | set from enum | set comparison | per-label P/R/F1, macro-F1, exact-set-match |
| `extract_exact` | str | normalize then equality | P/R/F1 with abstention |
| `extract_fuzzy` | str | normalize then similarity >= theta | P/R/F1 at theta, reported beside strict |
| `extract_list` | list[str] | greedy bipartite on pair matcher | set P/R/F1 |
| `extract_numeric` | number + unit | unit-normalize then tolerance | P/R/F1 within tolerance |
| `extract_date` | ISO 8601, possibly partial | granularity-aware equality | P/R/F1 |
| `span` | offsets | overlap >= threshold | token-level P/R/F1 |

Every task is nullable unless declared otherwise. Validate `tasks.yaml` on load and
fail loudly with the offending task id — a typo in a task type must not silently
become a free-text field.

Example:

```yaml
tasks:
  - id: filing_state
    type: multiclass
    question: >
      The US state whose law governs this agreement. Use the governing-law
      clause, not the mailing address. Null if no US state is named.
    output: {enum: [AL, AK, AZ, ...], nullable: true}
    match: {matcher: identity}
    weight: 1.0
  - id: counterparty
    type: extract_fuzzy
    question: >
      Legal name of the counterparty, excluding the filer and outside counsel.
    match: {matcher: entity_name, theta: 0.90}
    weight: 1.0
```

## 4. Normalizers and matchers — `normalize.py`, `match.py`

Pure functions, unit tested, applied identically to gold and prediction. These are the
most common source of fake results: a bad matcher makes a good prompt look broken and
sends the optimizer chasing a bug. Test them before anything else works.

| Output | Normalization | Compare by |
| --- | --- | --- |
| enum / bool | canonical casing, synonym map | equality |
| date | parse to ISO 8601, preserve granularity | equality at declared granularity |
| controlled code | map to vocabulary (e.g. USPS) | equality |
| entity name | case, punctuation, whitespace, leading articles, legal suffixes collapsed | strict equality + token-set similarity reported separately |
| numeric | unit conversion, decimal normalization | tolerance from config |
| free string | whitespace, casing | as `match` specifies |

Rules:

- Partial dates (`2024`, `2024-06`, `2024-06-15`) are first-class. Whether a coarser
  prediction matches a finer gold value is per-task config, defaulting to no.
- `extract_fuzzy` always reports strict and fuzzy side by side. Never fuzzy alone.
- A wrong non-null value counts as both FP and FN.
- Abstention is scored: predicting null when gold is null is a true negative; a value
  when gold is null is a false positive.

Ship a fixture file of adversarial near-misses per normalizer and test against it.

## 5. Metric — `metric.py`

Single entry point built from the registry:

```python
def metric(gold, pred, trace=None):
    """Score a prediction against gold across all registered tasks.

    :param gold: the labeled example
    :param pred: the program's prediction
    :param trace: set by DSPy during bootstrapping; when not None, return a
        strict pass/fail rather than a partial score
    :returns: weighted aggregate in [0, 1], or bool when trace is not None
    """
```

Two requirements:

- When `trace is not None`, DSPy is bootstrapping demonstrations and needs pass/fail.
  Return "all tasks correct", not the partial score, or the demo pool fills with
  half-right examples.
- Provide a GEPA-compatible variant returning score plus natural-language feedback
  (which task failed, predicted vs gold, surrounding text).

Aggregate = declared weighted mean of per-task primary metrics. Always record the
per-task vector alongside the scalar: a rising aggregate hiding a collapsing task is a
common and expensive failure.

Classes below the support floor are excluded from the optimization target but still
reported. Macro-F1 weights every class equally, so a class with four examples otherwise
contributes as much swing as one with four hundred.

## 6. Evaluation outputs — `evaluate.py`

Every scoring run writes:

- `metrics.json` — per-task and per-class P/R/F1, supports, confusion matrices,
  the aggregate, plus run metadata (harness version, model strings, config hash).
- `failures.md` — confusion matrices, per-class error counts, and **at most three**
  sampled errors per task with a short context window.

The cap on sampled errors is deliberate: an optimizer given the full error dump writes
rules keyed to individual documents, which will not survive the holdout.

## 7. Extraction — `extract.py`

PDF -> `data/text/{doc_id}.md`, run once. Same text in validation and production or
metrics do not transfer.

- Text layer via PyMuPDF or pdfplumber; record which.
- OCR fallback when chars-per-page falls below ~100. Flag those documents; the flag
  must reach error analysis and production QA triage.
- Write `extraction_manifest.csv`: page count, char count, extractor, ocr flag,
  truncation applied, stratum.
- Truncation policy declared once in `config.yaml` and applied identically everywhere.

## 8. Optimization — `optimize.py`

DSPy. Pin the exact version in the harness's dependencies and record it in every run's
metadata; the optimizer API moves fast, so verify constructor signatures against the
installed version's docs rather than trusting examples here.

**Program shape.** Default to one module covering all M tasks with typed outputs.
Signature docstring is the tunable instruction; field descriptions carry each task's
`question`. Support splitting into groups via the optional `group` key — DSPy tunes
instructions per predictor, so a task needing its own tuned instruction needs its own
module. Start with `dspy.Predict`; make `dspy.ChainOfThought` a config option and
measure it rather than assuming it helps.

**Optimizers to support**, selected in `config.yaml`: `BootstrapFewShot`,
`BootstrapFewShotWithRandomSearch`, `MIPROv2`, `GEPA`, `SIMBA`. Default MIPROv2 at
light settings — heavy settings have consumed thousands of rollouts in published runs.

**Per experiment**, write `runs/{exp_id}/config.json` fixing optimizer, budget, module
type, grouping, task model, reflection model. One variable changes per experiment.
Append to `runs/leaderboard.md`: exp_id, optimizer, aggregate validation score,
per-task vector, rollouts, wall time, cost.

**Enforce**: the budget from `config.yaml` (max experiments, max rollouts); that
training and selection use only the optimization pool; that the champion is pinned and
new experiments branch from it rather than from whatever ran last.

**Force at least one demonstration per class** into the bootstrap pool. Bootstrapped
selection otherwise omits rare classes entirely and the compiled program behaves as if
they do not exist.

Time-box rather than converge: budget is fixed up front, not "iterate until stalled".

Save compiled programs with `program.save()`. Production loads a saved program and
never recompiles.

## 9. Scaffold — `scaffold/` and `cli.py`

`newproject` creates a project directory, writes an exact harness pin, and copies the
guidance files in. Because the pin lives in the project the CLI creates, `newproject`
must be runnable outside any project: expose it as a console script and support
`uv tool install doc-harness` / `uvx --from doc-harness newproject`. Default the pin to
the latest released version with a flag to override.

Other CLI commands run from the project's pinned environment: `status`, `audit-labels`,
`make-splits`, `run-baseline`, `compile`, `holdout`, `production`.

**Guidance layers to generate:**

| File | Contents | Size |
| --- | --- | --- |
| `CLAUDE.md` | invariants only: holdout opens once, `data/` never written by optimizer, package never edited, run `/status` first | under a page |
| `.claude/commands/*.md` | one per phase: entry conditions, steps, exit criteria, what comes next | a page each |
| `.claude/skills/*` | task-type catalog, matcher semantics, reading `failures.md`, sample-size tables | as needed |
| `docs/protocol.md` | prose walkthrough for the human | a few pages |
| `tasks.yaml` | skeleton with one commented example per task type | — |
| `config.yaml` | models, budgets, split seed, thresholds, truncation policy | — |

**`status` derives phase from disk** — `splits.json`, `annotation_rules.md`, `runs/`,
leaderboard, holdout lock — and reports what is done, what is next, what is blocking.
Static guidance drifts out of sync with a half-finished project; this must not.

**Gates are code, not markdown.** Instructions are advisory and a session under time
pressure will route around them:

- `holdout` writes a lock on first use; a second evaluation refuses unless explicitly
  overridden, and the override is recorded in the report.
- `compile` refuses when `annotation_rules.md` or `splits.json` is missing.
- Experiment budget enforced from `config.yaml`.
- `labels.jsonl` and `splits.json` are read-only to every optimization path.

Command files explain why each gate exists so Claude can give a reason rather than
just refuse.

**Human decisions are explicit stops**, presented with the numbers needed to decide and
appended to `decisions.md`: corrected vs blind labeling assignment, borderline
fuzzy-match adjudication, support floor, class collapsing, accepting compiled over
baseline.

## 10. Splits, labeling and rare classes

`make-splits` implements this; get it right, because everything downstream rests on it.

**Splits** — fixed seed, committed to `splits.json`, stratified on each task's rarest
class, created before any model sees any document.

| K | Split |
| --- | --- |
| >= 200 | 50 train / 25 val / 25 holdout |
| 100-200 | 45 / 25 / 30 |
| 50-100 | 5-fold CV on 70%, 30% holdout |
| < 50 | warn: too few points for automated search |

Never let a class appear in the holdout but not in train.

**Labeling protocol** — labels are produced by correcting baseline output (2-3x faster
than de novo), except the holdout:

- train + validation: corrected from baseline output.
- holdout: labeled blind, de novo, without seeing any prediction. Anchoring on model
  output correlates labels with what is being measured and inflates every metric.

`labels.jsonl` records `stratum` and inclusion probability per document.

**Rare classes.** Support, not document count, is the binding constraint. Approximate
95% Wilson half-widths for recall near 0.8:

| examples in split | half-width | usable for |
| --- | --- | --- |
| 5 | ±~29 pts | presence check only |
| 10 | ±~22 pts | detecting total failure |
| 30 | ±~14 pts | coarse comparison |
| 50 | ±~11 pts | a number with a caveat |
| 100 | ±~8 pts | a number |

With 10 examples all correct, true error could still be near 25%. At prevalence p, a
random sample needs about m/p documents to yield m examples — a 2% class needs ~1,500
documents for 30 examples. Random labeling cannot reach rare classes.

Support enrichment strata, each recorded with its inclusion probability:

1. **random** — never skipped; the only unbiased corpus-level estimate.
2. **keyword/regex** over the text cache; model-independent.
3. **model-nominated** — baseline predictions of the rare class plus low-confidence
   cases. Biased: cannot surface examples the model misses, so it inflates recall if
   used alone.

Corpus-level metrics use inverse-probability weights; per-class metrics are unweighted
on the enriched pool. Report both, labeled as such.

Support floor (default 30). For each class below it, the command presents four options
and records the choice: enrich, collapse into a neighbour or `other`, split out as a
binary detection task, or report-unmeasured (excluded from the optimization target).

## 11. Baseline — `programs/baseline.py` in the scaffold

Record three, not one: zero-shot (instructions only), hand-written few-shot,
and `BootstrapFewShot`. The compiled program must beat the best of these on the
holdout or the honest move is to ship the baseline.

Any task scoring near zero at baseline is an upstream problem — information absent from
the extracted text, an ambiguous definition, or a broken matcher. None is fixed by
optimization. Surface this explicitly in the baseline report.

## 12. Holdout and production — `report.py`, `produce.py`

**Holdout** runs once. Report per task: P/R/F1, support, 95% CI (Wilson for
proportions, bootstrap for the aggregate), the validation number, and the gap.

| gap (val − holdout) | reading | action |
| --- | --- | --- |
| within CI | no detectable overfitting | ship |
| modest, consistent | mild overfitting, normal | ship, quote holdout |
| large, few tasks | those tasks memorized specifics | revert them to baseline |
| large, across the board | fitted the validation split | fall back to a simpler champion |

Report also which tasks the holdout **cannot** measure (rarest class under ~10
examples), so downstream consumers do not trust a number built on three documents.

**Production**: load the saved program, Batch API, checkpointed and resumable. Raw
responses to `runs/production/raw/{doc_id}.json` before post-processing. A document
failing all retries is recorded as a failure, never silently dropped.

QA gates, all in `qa_report.md`, run before results are handed over:

| check | fails when |
| --- | --- |
| coverage | outputs != N |
| schema | parse failures above ~0.5% |
| class distribution | large shift vs the labeled sample |
| null rates | per-task abstention well above validation rates |
| manual spot-check | ~50 random documents flagged for review |

Human review triage routes: OCR'd documents, truncated documents, nulls on
usually-answered tasks, low-confidence cases, plus a random slice — the random slice
stays, because the first four alone give a biased quality estimate.

**Close**: write `REPORT.md`, append a row to the shared `ledger.csv` (task types, N, K,
baseline holdout score, compiled holdout score, labeling hours, cost), and push any
instruction that worked into `fragments/`.

## 13. Conventions

- Python. Comments start lowercase. Sphinx-style docstrings.
- Type hints throughout; `tasks.yaml` parsed into typed objects, not dicts.
- Every run records harness version, model strings, config hash.
- `data/` is read-only to every optimization path.
- No silent drops anywhere: reconcile counts and fail loudly.
- Caches are on disk and re-scoring never costs inference.

## 14. Build order

1. `registry.py` + `tasks.yaml` schema + validation, with tests.
2. `normalize.py` and `match.py` with adversarial fixtures. **Do not proceed until
   these pass** — everything downstream inherits their bugs.
3. `metric.py` and `evaluate.py`; verify `metrics.json` / `failures.md` against
   hand-computed numbers on a toy set.
4. `extract.py` + manifest.
5. Baseline program + `run-baseline`, end to end on a toy project.
6. `make-splits` with stratification, enrichment strata and the support floor.
7. `optimize.py`: BootstrapFewShot first, then MIPROv2, then GEPA/SIMBA.
8. `report.py` + holdout with its lock.
9. `produce.py` + QA gates.
10. `scaffold/`, `cli.py`, `state.py`, guidance files.
11. Dogfood: scaffold a project with ~20 synthetic documents covering every task type
    and run it to completion.

Milestone 11 is the acceptance test. Until a synthetic project runs end to end without
hand-editing the harness, the hook design is unproven.

## 15. Open questions to raise, not guess

- Whether partial dates match finer gold values (per-task default).
- Fuzzy thresholds per task and who adjudicates borderline matches.
- Task and reflection model selection, and the cost ceiling per project.
- Whether any task needs page images rather than extracted text.
- Whether downstream consumers prefer abstention or a best guess — this changes the
  metric, not just the prompt.
