# Pipeline 01 — PromptSum Entity Chain (Reproduction)

## What it does

Reproduces the *discrete-prompt* half of PromptSum (Ravaut et al., 2023). For each SAMSum dialogue we extract an entity chain — speakers in order of appearance plus any spaCy-NER-tagged entities — and prepend it to the dialogue at the encoder.

Example input:

```
ENTITIES:
Hannah | Betty | Amanda | Larry

DIALOGUE:
Hannah: Hey, do you have Betty's number?
Amanda: Lemme check
Hannah: <file_gif>
Amanda: Sorry, can't find it.
Amanda: Ask Larry
...
```

## What we are *not* doing

PromptSum's full method is more than the entity chain: it also (a) keeps PEGASUS-large frozen, (b) tunes a soft "E-prompt" that learns to *predict* the chain, and (c) uses a multi-task pre-training stage on C4. We update the whole backbone end-to-end and extract entities directly with spaCy. This pipeline therefore reproduces *the entity-chain signal*, not the parameter-efficiency story.

That's the right choice for this project: every pipeline in the repo updates the whole backbone, so we are running a clean ablation over guidance-signal types with everything else fixed.

## Result

| Metric  | Value | vs. paper (full-shot) |
| ------- | ----- | --------------------- |
| ROUGE-1 | 49.82 | +3.10 over 46.72      |
| ROUGE-2 | 24.89 | +2.54 over 22.35      |
| ROUGE-L | 41.05 | +2.45 over 38.60      |

We are above the paper's number because we are fine-tuning end-to-end whereas PromptSum updates < 0.1 % of parameters. This is *the* baseline we compare every other guided pipeline against — what matters here is whether richer signals beat 49.82, not whether we beat the paper.

## Run

```bash
python train.py
```
