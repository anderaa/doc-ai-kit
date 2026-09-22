# Changelog

Projects pin one exact harness version. Read the entry for a version before moving a
project onto it: some releases change how answers are scored, and a number measured under
one version is not comparable with a number measured under another.

## Unreleased

**Does not change scores or prompts.** Adds a labeling step before `audit-labels`; projects
that already have `labels.jsonl` are unaffected and split exactly as before.

Upgrading a project:

- **Re-lock after moving the pin** (`make lock && make sync`): this release adds `openpyxl`.
- Nothing else changes for a project that already has labels. With no `label_plan.json`,
  `make-splits` stratifies the whole split as before.
- **Fix the project's `.gitignore` by hand.** A project's copy is its own and is not updated.
  Replace the `data/` line with `data/pdfs/`, `data/text/` and `data/~$*`, then commit
  `labels.jsonl`, `splits.json` and `annotation_rules.md`.

Changes:

- **A spreadsheet labeling step**: `sample-labels`, `label-sheet`, `import-labels`.
  - `sample-labels --count N` draws the documents to label, and the holdout among them, at
    random, before any model runs. The brief wants the holdout labeled blind and everything
    else corrected from model output. That only works if the holdout is known before the
    model runs, and nothing is labeled yet to stratify on, so the holdout is random.
  - `label-sheet` runs the zero-shot program on the sample **outside** the holdout (Batch API
    by default, resumable) and writes `data/labels.xlsx`: model answers to correct, empty
    holdout rows to label blind, dropdowns for yes/no and enum columns, and a guide tab.
    It is `.xlsx` because Excel rewrites CSV cells on open: leading zeros go, dates change.
  - `import-labels` reads each cell with the scoring normalizers (`$1.25M`, `June 15, 2024`),
    turns a pasted passage into offsets, lists every problem at once, and writes nothing
    until all of them pass. It also reads a CSV export.
  - Each document's labeling mode is taken from the plan, not from the sheet. A document
    shown model answers is `corrected` for good.
- `make-splits` keeps a holdout drawn by `sample-labels` and stratifies only train and
  validation. It refuses while a holdout document is neither labeled nor skipped with a
  reason, and says when a holdout document moves to train for carrying a class train lacks.
- `status` shows the two new steps, labeling progress, and a sheet changed since the last
  import.
- `produce()` takes a `run_dir`, so the labeling prefill checkpoints to `runs/prelabel/`.
- **New projects commit their ground truth.** The scaffold's `.gitignore` excluded all of
  `data/`, so `splits.json` could not be committed even though the guidance says to. It now
  excludes only the PDFs, the cached text and Excel's lock file.

## 0.1.5

**Does not change scores or prompts.** Production sends the same requests and parses replies the
same way; only the route and the price change.

Upgrading a project:

- **Production now waits for a batch** with Anthropic models: usually under an hour, at most a
  day, at half the price. Set `production.use_batch_api: false` to keep live requests.
- **Re-lock after moving the pin** (`make lock && make sync`): this release adds the `anthropic`
  SDK as a dependency.
- **A production run started under 0.1.4 can be resumed.** Its checkpoints load unchanged;
  documents not yet produced go through the batch.
- config.yaml needs no change: the new batch settings have defaults.

Changes:

- **Production runs through the Message Batches API** for Anthropic models, at half the
  price. Requests are captured from the live path at the moment of sending, so a batch
  request is the live request by construction, and replies are parsed by the live path's own
  post-processing and merge. A document the batch cannot deliver -- an errored or expired
  request, an unreadable reply -- falls back to a live request with the usual retries. Batch
  ids are written to disk before any waiting, so an interrupted run collects the batches it
  already paid for. `production.use_batch_api` (default true) and `batch_poll_seconds`
  control it; other providers run live. Each checkpoint records which way its answer came
  and keeps the model's raw reply, and `qa_report.md` counts both routes.
  Measured on the adversarial corpus through the real API: all 48 requests succeeded, input
  tokens matched the live run exactly (75,394 each, the same prompts), accuracy matched
  (0.996), and cost halved ($0.238 to $0.119). The batch took 42 minutes to process.
- Adds the `anthropic` SDK as a dependency.

## 0.1.4

**Changes scores and the prompt** for span tasks, changes the format of `outputs.jsonl`, and
fixes a scoring bug on numeric tasks.

Upgrading a project:

- **Span tasks: reword, recompile, re-run.** Reword each span question to name the passage
  ("The governing-law clause"), not to ask for offsets -- the harness now appends how to
  answer. Recompile any program with a span task, since it carries demonstrations in the old
  format. The prompt changed, so `rescore` cannot update span numbers: re-run.
- **Numeric tasks: rescore.** The magnitude fix below changes scoring only, so `rescore`
  updates runs saved under 0.1.3 at no cost.
- **Production: do not resume a 0.1.3 run.** Its checkpoints hold numbers, dates and spans as
  text; resumed under 0.1.4 that text would be read back as values. Use `production
  --no-resume`. Anything consuming `outputs.jsonl` now receives those values as structured
  JSON instead of text.
- **Programs compiled with MIPROv2** under 0.1.3 were compiled against a task model without
  the configured temperature or `max_tokens`. Recompile to compile under the project's settings.
- config.yaml needs no change: the new thinking settings are optional.
- Record the upgrade in `decisions.md`.

Changes:

- **Spans are asked for as quotes.** A model asked for character offsets has to count
  characters. On the adversarial corpus it found the right sentence and placed it 84
  characters early. The program now asks for the passage verbatim and the harness locates it,
  tolerating line breaks, typographic quotes and dashes, capitalisation, and wrapping quotation
  marks or ellipses. A paraphrase is not found and scores as a wrong answer. Span F1 on the
  adversarial corpus went from 0.950 to 1.000.
- Gold spans are shown to the model as the passage they cover, so a labeled demonstration
  matches the instruction instead of showing offsets it is told never to give.
- **Thinking is now configurable.** `models.task_effort` (low to max) and `models.task_thinking`
  (adaptive or disabled) control how hard the task model thinks. Both are unset by default, so
  nothing changes until a project chooses. Every truncated reply in the harness's own runs was
  the model spending its whole `max_tokens` budget thinking and never answering; thinking is
  also billed as output. Every run now records these settings in its metadata.
  Measured on the adversarial corpus, all four settings -- default, medium, low, thinking off
  -- scored 0.996, each missing one different document. Thinking fell from 1,587 tokens to
  none and cost by about 8%, with no truncations under any setting. The earlier truncations
  came from asking for character offsets, which span tasks no longer do; the default is
  therefore left unchanged, and a project should measure before lowering it.

Fixed:

- **A magnitude word in the unit was scored as a unit mismatch.** Asked for a contract value,
  the model answered 3.25 with unit "million USD" -- correct -- and it was scored wrong against
  3,250,000 USD. "thousand", "million", "billion" and "bn" in a unit now move into the number.
- **The optimizer built its task model without the configured settings** -- no temperature,
  no `max_tokens` -- so MIPROv2 compiled under different conditions from the ones the program
  then ran under. Every task model now comes from one constructor.
- **outputs.jsonl and production checkpoints wrote typed values as Python repr text** -- a
  contract value arrived as `"value=375000.0 unit='USD'"`. They are now structured JSON, and a
  resumed run reads them back as values rather than strings.

## 0.1.3

**Changes scores** wherever a reply could not be read. Such a reply used to be scored as the
model abstaining on every task -- credit wherever the gold was null, and no sign anything had
failed. It is now retried with a fresh generation and, if it still fails, scored as wrong and
listed. By default a scoring run with any such reply refuses to report numbers.

Upgrading a project:

- **Re-run, don't rescore.** A run saved under 0.1.2 stored a failed reply as ordinary nulls,
  so `rescore` has nothing to mark as a failure and reproduces the old number. Only running
  the program again finds the failures. Record the upgrade in `decisions.md`.
- **Runs that used to report may now stop.** With `evaluation.max_failure_rate` at its default
  of 0, a run with an unreadable reply refuses rather than reporting a number that includes
  it. That is the point; raise the rate only as a recorded decision.
- Existing config.yaml files need no change: the new `evaluation` section has defaults.

- **Strict parsing.** DSPy fills an omitted nullable field with null. Since every task is
  nullable, a reply cut off partway came back with its remaining fields null. The program now
  parses strictly: an omitted field raises, an explicit null is still an answer. This also
  closes the gap in production, where a partial reply passed the schema check with nulls.
- **Evaluation no longer turns an error into nulls.** `dspy.Evaluate` substitutes an empty
  prediction for a program that raises. Failed replies are now retried, then recorded in
  `metrics.json` and at the top of `failures.md`, and they survive a `rescore`.
- New `evaluation` settings in config.yaml: `max_retries` (default 2) and `max_failure_rate`
  (default 0.0).
- The holdout lock is now taken after predictions succeed, so a refused run does not spend
  the one-shot measurement.

## 0.1.2

**Changes scores.** An upgraded project should re-score its runs (`doc-harness rescore
<run>`, free) before comparing anything against numbers from 0.1.1, and record the upgrade in
`decisions.md`. A holdout already measured under 0.1.1 stays a 0.1.1 number.

Scoring fixes -- each scored a correct answer as wrong:

- **Titles on people's names.** "Dr. Ana Ruiz" did not match "Ana Ruiz", and whether it did
  depended on the length of the name. Adds a `person_name` normalizer. **Not applied
  automatically:** a list of people needs `normalizer: person_name` in its `match` block in
  tasks.yaml; existing projects keep `entity_name` until they change it.
- **Dotted legal suffixes.** "L.L.C." and "N.V." now collapse like "LLC" and "NV".
- **Currency written as a word.** "dollars", "US dollars", "euros" and similar now resolve
  to currency codes. A bare "dollars" resolves to the task's declared unit when that is a
  dollar currency, otherwise USD.
- **Typed model output skipped unit normalization.** A `Quantity` returned by the model kept
  its unit as written, so a lowercase "usd" was a unit mismatch against "USD".

New:

- `doc-harness adjudicate <run>` lists every decision a threshold made -- including any
  acceptance strict equality would have refused -- with raw and normalized values side by
  side. Reads saved predictions, so it costs nothing.
- `examples/adversarial/`, a corpus built to trap the matchers, with an offline oracle that
  runs in the test suite.

Fixed:

- Production retries replayed DSPy's cached response, so a truncated answer failed the same
  way on every attempt. Each retry now asks the model afresh.

Known limit, unchanged: character similarity cannot tell a typo from a different name --
"Smith & Hartley Ltd." is accepted for "Smith and Hart Limited" at 0.903 against a theta of
0.90. `adjudicate` flags it; changing the measure is a decision about what counts as correct.

## 0.1.1

- The wheel builds: 0.1.0 packaged every scaffold file twice and could not be installed.
- `newproject` pins a git tag by default, so a new project's pin resolves.
  `--pin-mode pypi` and `--pin-mode path` cover a package index and a local checkout.

## 0.1.0

Withdrawn: the wheel could not be built. The tag was deleted.
