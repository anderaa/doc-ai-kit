# doc-harness

A fixed harness for running document classification and extraction projects: PDFs in, a
holdout-measured accuracy report and a full-corpus output file out, with DSPy doing prompt
optimization.

The harness is installed as a versioned package and shared by every project. A project is
scaffolded separately, pins an exact harness version, and puts its own code in `custom/`.
The package is never hand-edited from a project.

```
pdfs -> text cache -> DSPy program -> typed outputs -> normalize -> match
                            ^                                        |
                            |                                        v
                      optimizer <----- metric <----------------- gold labels
```

## Install

```
pyenv virtualenv 3.12.11 doc-harness
pyenv local doc-harness
pip install pip-tools
make lock
make sync
```

`make check` runs ruff, black, mypy and pytest.

## Start a project

```
newproject "Acme Contracts"
```

`newproject` runs outside any project, because what it writes is the exact harness pin the
project installs. Everything else runs from inside the project: `status`, `extract`,
`audit-labels`, `make-splits`, `run-baseline`, `compile`, `rescore`, `adjudicate`,
`holdout`, `production`, `close`.

## What is load-bearing

Most of the design exists because these projects fail in the same handful of ways, and each
failure is silent -- the numbers keep going up while the measurement stops meaning anything.

- **Gates are code, not markdown.** The holdout lock, the read-only guard on `data/`, the
  experiment budget and the `annotation_rules.md` requirement are all enforced in code,
  because a session under time pressure routes around advice.
- **Normalizers and matchers are tested first.** A bad matcher makes a good prompt look
  broken and sends the optimizer chasing a bug. They ship with an adversarial fixture corpus
  and nothing downstream is trusted until it passes.
- **The per-task vector is always recorded beside the aggregate.** A rising aggregate that
  hides a collapsing task is common and expensive.
- **Abstention is scored.** Null against null gold is a true negative; a wrong non-null value
  is both a false positive and a false negative.
- **`failures.md` samples at most three errors per task.** Given the full dump, an optimizer
  writes rules keyed to individual documents that die on the holdout.
- **Every class gets a demonstration.** Bootstrapped selection otherwise drops rare classes
  and the compiled program behaves as if they do not exist.
- **Every run saves its predictions.** Fixing a matcher and re-measuring costs nothing
  (`doc-harness rescore`), so nobody is tempted to leave the bug in to avoid paying again.
- **Threshold decisions are surfaced, not buried.** `doc-harness adjudicate` lists every call a
  threshold actually made, raw and normalized side by side, for a human to confirm.

## Layout

```
src/doc_harness/
  registry.py     tasks.yaml -> task specs, DSPy signatures, output types
  normalize.py    per-type normalizers
  match.py        per-type matchers
  metric.py       DSPy metric + GEPA feedback variant
  evaluate.py     dspy.Evaluate wrapper -> metrics.json, failures.md
  extract.py      pdf -> text cache
  splits.py       stratification, enrichment strata, the support floor
  optimize.py     optimizer runs, leaderboard, budget enforcement
  baseline.py     the three baselines
  report.py       holdout reading, REPORT.md, the ledger
  produce.py      full-corpus run + QA gates
  guards.py       the gates
  state.py        project phase, derived from disk
  hooks.py        registration points for a project's custom/
  cli.py          the commands
  scaffold/       files copied into a new project
fragments/        instruction fragment library, by task type
ledger.csv        one row per completed project
examples/synthetic/    the acceptance test: does the pipeline run end to end
examples/adversarial/  the matcher check: does scoring hold up on documents built to trap it
```

## Installing into a project

`newproject` pins the harness exactly, and the pin has to resolve. The default is a git
tag, because that works today without publishing anything:

```
doc-harness @ git+https://github.com/anderaa/doc-harness.git@v0.1.1
```

`--pin-mode pypi` switches to `doc-harness==X.Y.Z` once the package is published to an
index, and `--pin-mode path --harness-path ...` points at a local checkout for harness
development. Hashes and git pins are mutually exclusive -- pip cannot hash a checkout -- so
a git-pinned project locks without `--generate-hashes` and relies on the tag for exactness.

Cutting a release means bumping `version` in `pyproject.toml` and pushing a matching
`vX.Y.Z` tag. Projects scaffolded before the bump keep pointing at their own tag.

## Known deviations from the brief

**pyenv and pip-tools, not uv.** BUILD.md §9 assumes `uv tool install` / `uvx`. The harness
locks with `pip-compile` and projects use a pyenv virtualenv instead. Nothing else changes:
the pin is still exact and the lock is still committed.



**Production uses concurrent requests, not a provider Batch API.** BUILD.md asks for the
Batch API. Going through DSPy's adapters is what keeps typed parsing and the normalizers
identical between validation and production, and a provider batch endpoint would mean
bypassing them and reimplementing the parsing separately -- which is precisely the kind of
divergence the rest of the design exists to prevent. The run is checkpointed and resumable,
and `runs/production/raw/{doc_id}.json` is written before post-processing, so a batch backend
can be dropped in behind the same checkpoint format when the tradeoff is worth taking.
