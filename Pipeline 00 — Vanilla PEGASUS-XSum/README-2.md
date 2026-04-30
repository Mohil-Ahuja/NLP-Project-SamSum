# Pipeline 00 — Vanilla PEGASUS (Control)

## What it does

Fine-tunes `google/pegasus-xsum` on SAMSum with **no guidance signal**. The encoder sees only `DIALOGUE:\n<raw dialogue>`.

## Why it's here

It is the control. Every guided pipeline in this project prepends some structured signal (entity chain, SRL, knowledge triples, …) to the dialogue. If a guided pipeline cannot beat this number, the signal it added isn't doing useful work.

## Result

| Metric  | Value |
| ------- | ----- |
| ROUGE-1 | 50.14 |
| ROUGE-2 | 25.06 |
| ROUGE-L | 41.32 |

This is broadly consistent with what's been reported elsewhere for fully fine-tuned PEGASUS-XSum on SAMSum. We are pleasantly surprised this is already higher than PromptSum's reported full-shot SAMSum number (R-1 = 46.72) — but PromptSum updates < 0.1 % of parameters whereas we update 766 M, so the comparison isn't apples-to-apples. See `results/comparison_with_promptsum.md`.

## Run

```bash
python train.py
```

Single T4, fp16 + gradient checkpointing. About 4 hours.
