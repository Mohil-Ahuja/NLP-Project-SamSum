# Pipeline 05 — Dialogue Acts

## What it does

Tags each utterance with a speech act drawn from a small five-label scheme — `REQUEST`, `AGREE`, `DECLINE`, `PLAN`, `INFORM` — using lexical rules. Prepends the tagged utterances to the dialogue.

Example signal:

```
ACTS:
Amanda [INFORM]: I baked cookies.
Amanda [REQUEST]: Do you want some?
Jerry [AGREE]: Sure!
Amanda [PLAN]: I'll bring you tomorrow

DIALOGUE:
Amanda: I baked cookies. Do you want some?
Jerry: Sure!
Amanda: I'll bring you tomorrow :-)
```

## Why we tried this

Dialogue acts tell the model *which utterances matter for what*. A `PLAN` is almost certainly going into the summary. A bare `INFORM` followed by `AGREE` is often filler. We wanted to know if even a five-tag set helps.

## What we found

Dialogue acts on their own are too coarse. The signal tells the model which utterances to prioritise but does not tell it *what's in* them — for that you still need an entity chain or SRL. Result: R-1 = 48.95, slightly *below* the entity-chain baseline.

The right experiment is to combine dialogue acts with SRL, so the model gets both *type-of-move* and *who-did-what*. We didn't get to that combination — listed in `docs/future_work.md`.

## Result

| Metric  | Value | vs. PromptSum entity chain |
| ------- | ----- | -------------------------- |
| ROUGE-1 | 48.95 | -0.87                      |
| ROUGE-2 | 23.81 | -1.08                      |
| ROUGE-L | 40.18 | -0.87                      |

## Run

```bash
python train.py
```
