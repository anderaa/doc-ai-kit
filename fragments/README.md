# Instruction fragments

Instructions that measurably helped, filed by task type, so the next project starts from
what the last one learned instead of rediscovering it.

## What belongs here

A fragment earns its place when it lifted a task's number on a **holdout**, not on
validation, and when the reason it worked is about the task type rather than the corpus.
"Prefer the governing-law clause over the mailing address" generalises to every multiclass
task with a plausible distractor elsewhere in the document. "Acme is always the filer" does
not generalise anywhere and belongs in that project's `tasks.yaml`.

## What does not

- Anything keyed to a specific document, filer or corpus.
- Anything that only ever moved a validation number.
- Whole prompts. Fragments are a sentence or two, meant to be composed.

## How to use them

Fragments are raw material for a task's `question` in `tasks.yaml`, or for a starting
instruction in `programs/baseline.py`. They are not applied automatically: an instruction
that helps one project can hurt another, and the only way to know is to measure it as one
experiment with one variable.

## Filing a new one

Add it to the file for its task type, with one line on the evidence: which project, which
task, and the holdout movement. A fragment with no evidence line is folklore.
