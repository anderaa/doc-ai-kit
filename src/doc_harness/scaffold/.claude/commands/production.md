---
description: Run the compiled program over the full corpus and gate the results
---

# production

## Entry conditions

- The holdout has been measured and the reading says ship.
- Text cached for every document in the corpus.

## Why it exists

Production loads the **saved** program and never recompiles: a program recompiled at
production time is not the program the holdout measured.

The run is checkpointed and resumable. Raw responses land in `runs/production/raw/{doc_id}.json`
before any post-processing, so a run that dies halfway resumes instead of paying twice. A
document that fails every retry is **recorded as a failure**, never silently dropped, and
the counts are reconciled at the end.

## Steps

```
doc-harness production
```

With an Anthropic model this submits the corpus through the Message Batches API, at half the
price, and waits for it -- usually within the hour, at most a day. The batch ids are saved
before any waiting, so if the command is interrupted, run it again: it collects the batches
it already submitted rather than paying for them twice. `--no-resume` discards them and
starts over.

Batching changes the price and nothing else: requests are the live path's own and replies
are parsed by the live path's own code. A document the batch cannot deliver -- an errored or
expired request, or a reply that cannot be read -- is run live instead, and `qa_report.md`
says how many documents came each way.

## The gates

All of these are in `qa_report.md`, and they run before anything is handed over:

| check | fails when |
| --- | --- |
| coverage | outputs do not equal N |
| schema | parse failures above the configured ceiling (0.5% by default) |
| class distribution | a class share moved a long way against the labeled sample |
| null rates | a task's abstention rate is well above its validation rate |
| manual spot-check | a random sample is flagged for review |

A failing gate means the results are not ready, not that the gate is wrong. Class
distribution and null rates in particular usually mean the corpus differs from the labeled
sample in a way nobody noticed -- which is worth knowing before the numbers are quoted.

**Schema failures are usually truncation.** When a response runs past `models.max_tokens`
it arrives incomplete and cannot be parsed. Each retry asks the model afresh rather than
replaying the cached reply, so a one-off rambling answer recovers on its own; a document
that truncates on every attempt is recorded as a failure. Check the log for "truncated due
to exceeding max_tokens" before assuming the model got the answer wrong. Raising `max_tokens` is the fix -- but it is a change to the
model configuration, so the honest thing is to re-run the affected documents rather than
mixing two configurations in one output file. `doc-harness production` resumes from the
checkpoints, so only the failed documents are re-run.

## Human review routes

OCR'd documents, truncated documents, nulls on usually-answered tasks, low-confidence cases,
**plus a random slice**. The random slice is not optional: the first four routes select
documents that are already suspect, so a quality estimate built on them alone is biased and
reads worse than the corpus really is.

## Exit criteria

- `runs/production/outputs.jsonl` covers the corpus, failures included.
- `qa_report.md` written and every gate passing, or each failure understood.

## Next

`close`.
