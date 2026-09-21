---
description: Run one optimization experiment against the validation split
---

# compile

## Entry conditions, enforced in code

- `data/annotation_rules.md` exists. A task whose labeling rule was never written down
  cannot be scored consistently, so optimizing against it fits noise.
- `data/splits.json` exists. Without fixed splits there is no validation split to optimize
  against, only the data.
- The experiment budget in `config.yaml` is not spent.

## Why it exists

The optimizer is the part most able to produce a convincing wrong answer, so the run is
fenced:

- **`labels.jsonl` and `splits.json` are read-only.** Verified by content hash around the
  run. An optimizer that can edit ground truth can improve its score without improving the
  program.
- **Only the optimization pool is visible.** Train and validation. The holdout is never
  passed to a compile step.
- **One variable per experiment.** Each run records what it changed, in `config.json`.
  Changing two things means learning nothing from either.
- **The champion is pinned.** New experiments branch from the best program so far, not from
  whatever ran last.
- **Every class gets a demonstration.** The trainset is reordered so each class appears
  early. Bootstrapped selection otherwise omits rare classes entirely, and the compiled
  program behaves as though they do not exist -- while the aggregate barely moves, because
  rare classes are rare.
- **The budget is fixed up front.** Rollouts are counted as they happen. Raising
  `max_experiments` or `max_rollouts` because the results look close is how a validation
  split gets fitted.

## Steps

```
doc-harness compile --variable "what this run changes"
```

Start with `BootstrapFewShot` to get a floor, then `MIPROv2` at `auto: light`, then GEPA or
SIMBA if there is budget left. Heavy settings have consumed thousands of rollouts in
published runs.

## Reading the result

Read the **per-task vector**, not only the aggregate. A rising aggregate that hides a
collapsing task is a common and expensive failure, and the leaderboard prints both.

`failures.md` samples at most three errors per task. That cap is deliberate: an optimizer
given the full error dump writes rules keyed to individual documents, and those rules do not
survive the holdout.

## Re-scoring without paying again

Each run saves its raw predictions to `runs/{exp_id}/predictions.jsonl`. When you fix a
normalizer or a matcher -- and you will, because that is where the subtle bugs live --
re-measure for free:

```
doc-harness rescore exp_003
```

The numbers move, the model is never called, and the corrected run replaces the wrong one.
Paying for a whole split again is exactly the tax that tempts people to leave a matcher bug
in place.

## Exit criteria

- A row in `runs/leaderboard.md` for each experiment.
- A pinned champion in `runs/champion.json`.
- A decision recorded in `decisions.md` if you are accepting the compiled program over the
  baseline.

## Next

`holdout` -- once, when you have decided which program you are shipping.
