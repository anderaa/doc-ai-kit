# doc-ai-kit

Answering a fixed set of questions about a corpus of PDFs, with a measurement of how often
the answers are right: PDFs in, an accuracy report measured on held-back documents and a
full-corpus output file out, with DSPy doing the prompt tuning.

doc-ai-kit is installed at an exact version and shared by every project. A project is
scaffolded separately, pins one version, and puts its own code in `custom/`. The installed
package is never edited from a project.

```
pdfs -> text cache -> DSPy program -> typed outputs -> normalize -> match
                            ^                                        |
                            |                                        v
                      optimizer <----- metric <----------------- gold labels
```

## How it works

A project asks the same questions about every document in a pile: *Which state's law
governs this contract? What is it worth? Who signed it?* Claude can answer these, but how
well depends heavily on how it is asked. This package finds a good way to ask, proves how
well it works, and then asks it of every document.

**The prompt.** For each document, Claude gets a message: some instructions, the questions
from `tasks.yaml`, sometimes a few worked examples, then the document. It answers all the
questions in one reply, or in a few when `tasks.yaml` puts questions in separate groups.
Claude itself is never retrained: all that changes from one attempt to the next is that
message.

**DSPy.** DSPy is the open-source library that builds and tunes the message. Instead of
someone rewording a prompt by hand and eyeballing the results, DSPy treats the prompt as
something to search over: it tries versions, scores each one, and keeps the best.

**Scoring.** A version is only as good as its score, so scoring has to be right. People
label a sample of documents with the correct answers. Each answer Claude gives is compared
with the label, using rules that forgive differences that do not matter -- "$1.25M" and
"1,250,000 USD" count as the same -- and not ones that do. Much of this package exists to
get that comparison right, because a scorer that marks right answers wrong sends the whole
search after a problem that is not there.

**The three piles.** The labeled documents are split, once, into three piles, like a
student's practice problems, practice test and final exam:

- **Training** -- what DSPy learns from: worked examples are taken from here.
- **Validation** -- used to compare versions and pick the best one.
- **Holdout** -- the final exam. No version sees it during the search. The winner is scored
  on it exactly once, and that is the number reported. It is labeled by people who have
  not seen Claude's answers, so the answers cannot sway the labels.

**The search.** DSPy's optimizers are different strategies for trying versions:

- **BootstrapFewShot** runs Claude on training documents, keeps the cases it got entirely
  right, and adds a few of them to the message as worked examples. The package makes sure
  every answer category gets at least one example, so rare ones are not forgotten.
- **MIPROv2**, the default, also has a model draft several alternative instructions, then
  tries different pairings of instructions and examples, scoring each on validation.
- **GEPA** reads the mistakes -- which question, what was expected, what came back -- and
  has a second model -- `models.reflection` in `config.yaml` -- rewrite the instructions
  to fix them.

Each experiment changes one thing, and is recorded with its score, question by question.
The best so far is kept, and the next experiment starts from it.

**Why the limits.** Every version tried costs model calls, and a search that runs too long
starts fitting the quirks of the validation pile instead of getting better at the task --
like memorizing a practice test. So the number of experiments is fixed before starting,
and the holdout decides the result, not the validation score. If the winner does much
worse on the holdout than on validation, the search overfitted, and the report says so.

**The result.** The winning version is saved and run on every document, with checks that
nothing was skipped and that the answers look like the labeled sample. Out come a file of
answers for the whole pile and a report of how accurate they are, measured on the holdout.

## Developing the package

```
pyenv virtualenv 3.12.11 doc-ai-kit
pyenv local doc-ai-kit
pip install pip-tools
make lock
make sync
```

`make check` runs ruff, black, mypy and pytest.

## Running a project

Each project is its own GitHub repo. It does not contain the package: it pins an exact
package version and installs it from this repo's tags. Nothing here is forked or copied.

### Claude works through this with you

You do not have to memorize the steps below. A new project ships with a guidance layer --
`CLAUDE.md`, `.claude/commands/` and `docs/protocol.md` -- that a Claude Code session
started inside the project folder reads on its own. Working through it is a conversation:
Claude runs the commands, explains what each one is for, and stops at the points where the
answer is yours.

**What Claude does.** Reads a few documents and helps you word `tasks.yaml`, so each
question carries its own definition. Runs `extract`, `sample-labels`, `label-sheet`,
`import-labels`, `make-splits`, `run-baseline`, `compile`, `holdout`, `production` and
`close`. Reads what comes back -- per-task scores, `failures.md`, the QA report -- and says
what it means and what to do next. Every phase has a command file behind it, so the answer
is grounded in the project's own guidance rather than in whatever it remembers.

**What Claude asks you.** The decisions with consequences, each recorded in `decisions.md`:
how many documents to label, whether the labeling sheet arrives prefilled, what to do about
a class with too few examples, whether to accept a compiled program over the baseline, and
when to spend the holdout.

**What Claude will not do.** Fill in your labels, and especially not the shaded holdout
rows -- labels written by a model are not human labels, and on the holdout they would
measure the model against itself. Open the holdout twice, raise the experiment budget, or
edit the installed package to get past a gate. Those refusals are enforced in code, not left
to good intentions, and each one explains itself.

**Starting each session**, ask for `doc-ai-kit status` first. Phase is derived from what
is actually on disk, so it is right even when the conversation has moved on or is new. Each
phase also has a slash command -- `/status`, `/extract`, `/label-sheet`, `/make-splits` and
so on -- that loads that phase's guidance directly.

**If Claude gets stuck**, the fix belongs in the package, not in the project. A project that
has to edit the installed package to make progress has found a bug in the package worth reporting.

### What you need first

- pyenv, with the Python the project will use: `pyenv install 3.12.11`.
- pipx (`brew install pipx`), to run `newproject` without installing the package anywhere.
- The GitHub CLI, signed in (`gh auth login`), to create the repo.
- `ANTHROPIC_API_KEY` set in your shell. Commands that call a model read it from there.
- Claude Code, for the session that runs the project.

### 1. Create the project and its repo

Run this **outside** the package checkout, e.g. from `~/Projects`, so the project does not
end up nested inside this repo:

```
cd ~/Projects
pipx run --spec "git+https://github.com/anderaa/doc-ai-kit.git@v0.2.0" newproject "Acme Contracts"
cd acme-contracts
git init
git add -A
git commit -m "Start project from doc-ai-kit v0.2.0"
gh repo create acme-contracts --private --source=. --remote=origin --push
```

`newproject` writes a folder named after the project, with the guidance files, a
`tasks.yaml` and `config.yaml` to fill in, and a `pyproject.toml` that pins the package to
the tag it was run from. It does not install anything, call a model, or set up git.

Keep the repo private. What it holds -- labels, decisions, task definitions -- describes
the client's documents. The documents themselves are never committed: `.gitignore` leaves
out `data/pdfs/`, `data/text/` and `data/transcripts/`. `runs/` is left out too, so run results and the holdout
lock stay on the machine that made them.

### 2. Set up the environment, and start Claude

`newproject` prints these steps; they are repeated here. Copy your PDFs into `data/pdfs/`
before the last one.

```
cd /Users/you/Projects/acme-contracts
pyenv virtualenv 3.12.11 acme-contracts
pyenv local acme-contracts
pip install pip-tools && make lock && make sync
doc-ai-kit status
claude
```

Until the virtualenv exists, pyenv reports an error inside the folder: `.python-version`
already names it. `make lock` pins every dependency; commit the result so anyone cloning
the repo installs exactly the same versions:

```
git add requirements.txt && git commit -m "Lock dependencies" && git push
```

### 3. What happens when Claude starts

Starting Claude inside the project folder is what loads its `CLAUDE.md`, the phase guidance
in `.claude/commands/` and `docs/protocol.md`. The session begins by checking `doc-ai-kit
status`, which reads the project's state from disk, and tells you which step is next and what
is blocking. That check is a session-start hook in the project's `.claude/settings.json`;
Claude Code may ask you to trust the folder the first time you open it. From there the work is the conversation described in **Claude works through this
with you** above. Every command can be run by hand as well; they are the same commands.

### 4. Declare the tasks and pick the model

- Write `tasks.yaml`: one entry per question, with its type and matching rule. The file
  has a commented example of every type. Claude can read a few documents and help word
  the questions -- the question is the definition the model works from.
- In `config.yaml`, set `models.task` and `models.reflection` (e.g.
  `anthropic/claude-sonnet-5` and `anthropic/claude-opus-5`), and the truncation policy
  if documents are long. There is no default model, so nothing spends money before this.
- Commit.

Set `budget.max_usd` and the matching `budget.prices` if you want a ceiling: spend is then
recorded per step in `runs/spend.json`, and a step that would start over the ceiling is
refused. Without prices declared, spend is still recorded in tokens.

### 5. Extract the text

```
doc-ai-kit extract
```

Text is cached in `data/text/` and read by every later step. Scanned pages have no text
layer, so they are sent to Claude as images and transcribed, page by page, once
`extraction.transcription.model` is set in `config.yaml`. Until then `extract` lists the
documents with unread pages; set the model and run `extract` again, and only those are
redone. Open a few cached texts to check them, especially transcribed ones.

### 6. Label

```
doc-ai-kit sample-labels --count 120
doc-ai-kit label-sheet --prefill        # or --no-prefill
```

`sample-labels` picks the documents to label, and the holdout among them, at random. Pick
the count from how many examples the rarest class needs (the `sample-sizes` skill), not
from how much time there is.

`label-sheet` writes `data/labels.xlsx`: one tab, one row per file to label, one column per
task, and `notes`. You choose how it starts:

- `--prefill`: the model answers the rows outside the holdout first, and you correct them.
  Two to three times faster to label; it costs a model call per document, and through the
  Batch API it can take up to an hour (`--live` is faster at twice the price).
- `--no-prefill`: every row starts empty, and every label is your own reading. Slower, and
  free.

Claude then walks you through what each column asks and the format it takes. Then **you**
label, in Excel or Google Sheets:

- Shaded rows start empty on purpose, either way. Label them from the document alone. Do
  not ask Claude to fill them: they measure the final program, and model-written labels
  there would measure the model against itself.
- A blank cell means the document gives no answer. For a passage, paste the sentence; the
  package finds its position.
- To set a document aside, write `skip: <reason>` in `notes`.

```
doc-ai-kit import-labels
```

The sheet is imported whole, once every row is finished. It lists every problem at once
and writes `data/labels.jsonl` only when all of them pass. Commit the sheet as you go -- it
is the most expensive thing in the repo.

### 7. Audit the labels and write the rules

```
doc-ai-kit audit-labels
```

Then write `data/annotation_rules.md`: for each task, the rule you actually applied and the
edge cases you decided. `compile` refuses to run without it. Commit.

### 8. Make the splits

```
doc-ai-kit make-splits
```

The holdout drawn in step 6 is kept; the rest is divided into train and validation. For
every class below the support floor, the command stops and asks what to do, and records
your answer in `decisions.md`. **Commit `data/splits.json` straight away** -- splits are
made once, and the committed file is the proof they were not changed later.

### 9. Record the baselines

Write two or three hand-picked examples into `programs/baseline.py` (`hand_written_demos`)
-- the hard cases and rare classes, not the first rows. Then:

```
doc-ai-kit run-baseline
```

This records zero-shot, hand-written few-shot and `BootstrapFewShot`, in
`runs/baseline_report.md`. A task near zero on every baseline is a problem with the task,
the text or the matcher, and optimization will not fix it: fix it before moving on.

### 10. Optimize, within the budget

```
doc-ai-kit compile --variable "MIPROv2 light, 4 demos"
```

One experiment per run, each changing one thing, named by `--variable`. The best one is
pinned as the champion, and later experiments start from it. `optimization.max_experiments`
in `config.yaml` caps the number of runs; it is set before starting, not raised because
the results look close. Read the per-task numbers and `failures.md` after each run, not only
the aggregate. `doc-ai-kit adjudicate <run_id>` lists borderline matches for you to check.

### 11. Measure the holdout once

```
doc-ai-kit holdout
```

This scores the champion on the holdout and writes a lock; a second run is refused unless
overridden, and the override goes into the report. Read the gap between validation and
holdout against the table in `docs/protocol.md` and act on it. The reading is recorded in
`decisions.md`; commit it.

### 12. Run production

```
doc-ai-kit production
```

Runs the champion over every document, through the Batch API, checkpointed so an
interrupted run resumes without paying again. Output goes to
`runs/production/outputs.jsonl`, and `qa_report.md` checks coverage, parse failures,
class shifts and null rates, and lists documents for a human spot check. The command
fails if a check fails.

### 13. Close

```
doc-ai-kit close --labeling-hours 14 --cost-usd 38.50
```

Writes `REPORT.md`. Commit it and push. To add the project to the shared record across
projects, pass `--ledger` with the path to `ledger.csv` in a package checkout, and commit
that there.

### Moving a project to a newer package

Read the version's entry in `CHANGELOG.md` first: some releases change how answers are
scored, and numbers from different versions do not compare. Then change the tag in
`pyproject.toml`, run `make lock && make sync`, follow the entry's upgrade notes, and commit.

## Why the design is the way it is

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
- **A reply that cannot be read is a failure, never an abstention.** DSPy fills an omitted
  nullable field with null, which would score a truncated reply as a model declining to
  answer. The package parses strictly, retries with a fresh generation, and by default
  refuses to report numbers if a reply still fails.
- **`failures.md` samples at most three errors per task.** Given the full dump, an optimizer
  writes rules keyed to individual documents that die on the holdout.
- **Every class gets a demonstration.** Bootstrapped selection otherwise drops rare classes
  and the compiled program behaves as if they do not exist.
- **Production runs through the Batch API, unchanged.** Requests are captured from the live
  path at the point of sending, and replies go through the live path's own parser, so batching
  halves the price and changes nothing else. Batch ids are saved before any waiting, so an
  interrupted run collects what it paid for instead of paying again.
- **Scanned pages are read, page by page.** A page with no text layer is sent to Claude as
  an image and transcribed into the text cache in place; a page still unread is counted and
  routed to review. Judged on a whole-document average instead, a report with a few scanned
  pages of accounts passes with those pages blank.
- **The budget is enforced, from measured spend.** Each paid step records its tokens and cost
  in `runs/spend.json`, and a step that would start over `budget.max_usd` is refused. Prices
  are declared per project, because a guessed price is a ceiling that means nothing.
- **Every run saves its predictions.** Fixing a matcher and re-measuring costs nothing
  (`doc-ai-kit rescore`), so nobody is tempted to leave the bug in to avoid paying again.
- **The holdout is drawn before any model runs, and labeled blind.** Labeling happens in a
  spreadsheet. Rows outside the holdout come prefilled with model answers to correct; holdout
  rows come empty, and the model is never run on them. Import reads each cell with the
  scoring normalizers and refuses a label that could not be scored.
- **Threshold decisions are listed, not buried.** `doc-ai-kit adjudicate` lists every call a
  threshold actually made, raw and normalized side by side, for a human to confirm.

## Layout

```
src/doc_ai_kit/
  registry.py     tasks.yaml -> task specs, DSPy signatures, output types
  normalize.py    per-type normalizers
  match.py        per-type matchers
  metric.py       DSPy metric + GEPA feedback variant
  evaluate.py     dspy.Evaluate wrapper -> metrics.json, failures.md
  extract.py      pdf -> text cache
  transcribe.py   scanned pages -> text, by Claude
  splits.py       stratification, enrichment strata, the support floor
  labeling.py     the labeling sample, the spreadsheet, reading labels back
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

`newproject` pins the package exactly, and the pin has to resolve. The default is a git
tag, because that works today without publishing anything:

```
doc-ai-kit @ git+https://github.com/anderaa/doc-ai-kit.git@v0.2.0
```

`--pin-mode pypi` switches to `doc-ai-kit==X.Y.Z` once the package is published to an
index, and `--pin-mode path --package-path ...` points at a local checkout for package
development. Hashes and git pins are mutually exclusive -- pip cannot hash a checkout -- so
a git-pinned project locks without `--generate-hashes` and relies on the tag for exactness.

Cutting a release means bumping `version` in `pyproject.toml`, updating the tag everywhere
it appears in this README, and pushing a matching `vX.Y.Z` tag. Projects scaffolded before the bump keep pointing at their own tag.

## Known deviations from the brief

**pyenv and pip-tools, not uv.** BUILD.md §9 assumes `uv tool install` / `uvx`. The package
locks with `pip-compile` and projects use a pyenv virtualenv instead. Nothing else changes:
the pin is still exact and the lock is still committed.

**Claude transcribes scanned pages, not an OCR engine, and decides page by page.** BUILD.md
§6 asks for an OCR fallback when a document's characters per page fall below about 100.
The package first shipped Tesseract on that whole-document average. The average missed
the common case of a mostly typed document with a few scanned pages, and Tesseract loses
which row a number in a table belongs to. Pages are now judged one by one, and a thin page
is sent to Claude as an image; the transcription goes into the text cache, so everything
downstream is still text only, as §15 intended. The manifest's OCR flag became per-page
counts (`thin_pages`, `transcribed_pages`, `unread_pages`), and the triage route follows them.
