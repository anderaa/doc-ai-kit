# Changelog

Projects pin one exact harness version. Read the entry for a version before moving a
project onto it: some releases change how answers are scored, and a number measured under
one version is not comparable with a number measured under another.

## 0.1.9

**Changes scores on classification tasks whose classes are all below the support floor**, where
macro-F1 counted classes that never appear, and **on `extract_numeric` tasks whose answers carry a unit other than a currency or
percent**, such as a notice period in days. Money amounts score as before. A finished run can
be re-scored without inference: `doc-harness rescore <run_id>`.

- **Units are read as words.** The unit of a number was found by deleting every digit and
  space and then stripping a leading magnitude letter, so `1 month` had the unit `onth`,
  `30 business days` had `usinessdays`, and `3 billion` had `illion`. The unit is now the words
  after the number, with a magnitude removed only when it is a whole word, or failing that a
  code written before the number (`USD 1,000`).
- **Periods of time are recognized**: day/days, week(s), month(s)/mo, year(s)/yr fold to one
  unit each. Business and working days stay separate from calendar days, and a month is not
  thirty days: those remain different answers. Reported from the first real-user run, where
  `30 day` and `30 days` scored as different units.
- **`unit_aliases` is accepted in a numeric task's `match` block**, e.g.
  `unit_aliases: {"sq ft": "square feet"}`. The normalizer already read it, but `tasks.yaml`
  refused the key, so a project needing a unit of its own had to write a custom normalizer.
- **The Batch API reads `chain_of_thought` programs.** `ChainOfThought` wraps a `Predict` and
  has no signature of its own, so every reply failed to parse -- after the batch had been
  submitted and billed, and the whole corpus was then re-run live at full price. The batch path
  also checks it can read a program **before** submitting anything, and runs live from the start
  when it cannot, so a batch is never paid for twice.
- **`budget.max_usd` is enforced.** It was recorded with every run and checked nowhere. Spend is
  now recorded in `runs/spend.json` as each paid step finishes -- tokens always, dollars where
  priced -- and a step that would start over the ceiling is refused. Prices are declared per
  project under `budget.prices`, because they change and differ by account; a ceiling set
  without a price for a model in use is refused rather than ignored. A step already running is
  never killed halfway, since what it has paid for cannot be unspent.
- **A spent rollout budget stops the run.** The guard raised an ordinary exception inside the
  metric, and DSPy's workers catch those, log them and carry on: one run went 11 rollouts past
  its cap. It now raises past that handling and `compile` reports it as a refusal. Workers
  already in flight can still overshoot by up to `optimization.num_threads`.
- **`compile` no longer crashes when the module type changes.** A champion records the module it
  was built with; an experiment that changes it starts fresh instead of loading a `predict`
  program as `chain_of_thought`. A champion that cannot be loaded for any other reason is noted
  in the experiment record and the run continues from scratch rather than dying.
- **Macro-F1 ignores classes that appear nowhere.** With no class above the support floor, the
  average ran over every declared class, including those in neither the gold labels nor the
  predictions: a governing-law task with 51 declared states, answering all 29 holdout documents
  correctly, scored 0.173. The average now covers the classes that appear, and the note beside
  it says how many of how many.
- **`status` counts PDFs whatever the case of their extension.** 0.1.8 fixed extraction but
  not the count in `status`, which still matched `*.pdf` only: on CUAD it reported 199 PDFs
  where extraction reads 510. Both now use one definition of a PDF. Reported from the first
  real-user run.

## 0.1.8

**Changes which documents are extracted, and which pages are sent for transcription.** Found
by a dry run on CUAD's 510 contracts.

Upgrading a project:

- **Run `doc-harness extract` again.** If any PDFs end in `.PDF`, they were skipped before and
  will be extracted now; nothing already cached changes.
- Documents whose file name has a space before `.pdf` get an id without it. If one was already
  extracted or labeled under the old id, run `extract --force` and re-import its labels.

Changes:

- **PDFs are found whatever the case of their extension.** Extraction globbed `*.pdf`, so a
  `.PDF` file was skipped without a word: on CUAD, 311 of 510. The count check did not catch
  it, because it counted the same filtered list. Other files in `data/pdfs/` are now named in a
  warning, and two files that would share a document id are refused.
- **Document ids are trimmed.** `"Manufacturing Agreement .PDF"` becomes
  `Manufacturing Agreement`, not an id ending in a space that a spreadsheet cell would drop.
- **A page is only treated as scanned if images cover most of it** (`min_image_coverage`,
  default 0.5), as well as having almost no text. On CUAD, 230 pages had under 100 characters,
  and 2 were scans; the rest were blank pages, page numbers and exhibit covers, which would
  have been paid for and reported as unread.
- **The labeling sheet is one tab: `file_name`, one column per task, and `notes`.** Rows are
  named by the PDF's file name, as the labeler sees it in the folder. The `mode`, `reviewed`
  and text-file columns and the guide tab are gone: Claude explains the columns, and each
  header carries a note. Rows to label from scratch are shaded. `import-labels` reads the
  first tab whatever it is called, ignores extra columns, and accepts `file_name` or `doc_id`.
- **Prefilling is the user's choice.** `label-sheet` requires `--prefill` or `--no-prefill`
  and explains the trade-off if neither is given. The holdout is empty either way.
- **The sheet is imported whole.** Every sampled document needs a finished row; a row with
  no answers and no note counts as not done. `skip: <reason>` in `notes` replaces the
  `reviewed = skip` marker. Partial imports are gone.
- The labeling guidance tells Claude to ask the prefill question, to walk the user through
  each column's question and format before they start, and never to fill in rows itself.
- `extract` prints one summary for unread pages instead of a warning per document.

## 0.1.7

**Changes the cached text of any document with a scanned page**, and so can change scores on
those documents. Documents whose pages all have a text layer are unaffected.

Upgrading a project:

- **config.yaml must change.** Delete `ocr_fallback`, `ocr_chars_per_page`, `ocr_extractor` and
  `ocr_language` from `extraction`, and add a `transcription` block (see a new project's
  `config.yaml`). A config that still has them is refused, with a message naming each
  replacement. To transcribe, set `extraction.transcription.model`, e.g.
  `anthropic/claude-sonnet-5`.
- **Re-lock after moving the pin** (`make lock && make sync`): the `ocr` extra is gone, and so
  are `pytesseract` and `pdf2image`.
- Add `data/transcripts/` to the project's `.gitignore`. It holds the documents' text.
- The existing text cache and manifest are kept. A document Tesseract read keeps its text and
  is marked as transcribed by `tesseract`. One flagged but never read is transcribed on the
  next `extract` once a model is set. To have Claude re-read Tesseract's documents, run
  `extract --force`, then check any span labels on them: the text they point into changes.

Changes:

- **Tesseract is gone; Claude transcribes scanned pages.** A page is sent to
  `extraction.transcription.model` as an image when its text layer is below
  `min_chars_per_page`, and the transcription is spliced into the cached text in that page's
  place. Decided per page: the whole-document average let a mostly typed report through
  with its scanned pages blank and unflagged. Tables come back row by row, so a number stays
  with its row. Thinking is switched off for these requests: copying needs no reasoning and
  thinking is billed as output.
  - `mode: all_pages` transcribes every page, for PDFs whose text layer is garbage from an
    earlier OCR. `mode: off` never sends anything and only counts thin pages.
  - Cached per page in `data/transcripts/`, keyed on the PDF's contents, the model and the
    prompt, so nothing is paid for twice. A failed page is not cached and is retried.
  - With no model set, thin pages are counted and listed. Setting one later redoes only
    those documents, without `--force`.
  - Measured on a skewed, speckled scan of an accounts table: every figure transcribed
    correctly and in its row, for 3,898 input and 308 output tokens.
- The manifest's `ocr` and `ocr_needed` columns become `thin_pages`, `transcribed_pages`,
  `unread_pages` and `transcription_model`. The production triage route `ocr` becomes
  `transcribed_or_unread`.
- `extract` reports the pages transcribed and the tokens spent on them.

## 0.1.6

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
