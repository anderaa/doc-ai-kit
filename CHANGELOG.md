# Changelog

Projects pin one exact harness version. Read the entry for a version before moving a
project onto it: some releases change how answers are scored, and a number measured under
one version is not comparable with a number measured under another.

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
