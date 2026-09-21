# Changelog

Projects pin one exact harness version. Read the entry for a version before moving a
project onto it: some releases change how answers are scored, and a number measured under
one version is not comparable with a number measured under another.

## Unreleased

**Changes scores** on span tasks, and changes what the model is asked for.

- **Spans are asked for as quotes.** A model asked for character offsets has to count
  characters. On the adversarial corpus it found the right sentence and placed it 84
  characters early. The program now asks for the passage verbatim and the harness locates it,
  tolerating line breaks, typographic quotes and dashes, capitalisation, and wrapping quotation
  marks or ellipses. A paraphrase is not found and scores as a wrong answer. Span F1 on the
  adversarial corpus went from 0.950 to 1.000.
- Gold spans are shown to the model as the passage they cover, so a labeled demonstration
  matches the instruction instead of showing offsets it is told never to give.
- Upgrading a project: reword span questions to name the passage ("The governing-law clause"),
  not to ask for offsets -- the harness now appends how to answer. **Recompile** any program
  with a span task: one compiled under 0.1.3 carries demonstrations in the old format.

Fixed:

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
