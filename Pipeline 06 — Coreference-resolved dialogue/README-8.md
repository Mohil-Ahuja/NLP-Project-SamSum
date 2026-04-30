# Pipeline 06 — Coreference-Resolved Dialogue

## What it does

Rewrites the dialogue in place with first- and second-person plural pronouns resolved to the named speakers. Then feeds the *resolved* dialogue to PEGASUS — there is no separate structure block.

Example:

```
Original:
   Amanda: I baked cookies.
   Jerry: Sure, we should eat them tomorrow.

Resolved:
   Amanda: Amanda baked cookies.
   Jerry: Sure, Amanda and Jerry should eat them tomorrow.
```

## Why this is interesting

SAMSum dialogues are *full* of pronouns. A typical line is `"He called her last time we were at the park"` — without resolution, PEGASUS has to figure out from context who "he", "her", and "we" are. With resolution, the names are already in the input, so the encoder only has to do summarisation, not pronoun-binding.

## Why we used a rule-based resolver

We tried three approaches:

1. **`neuralcoref` via spaCy** — deprecated, doesn't work on spaCy v3.
2. **AllenNLP `coref-spanbert`** — accurate but slow on T4. Adds ~30 % to training time.
3. **Rule-based, speaker-pronouns only** (what we ship). Handles `I/me/my/we/us/our` against the speaker list. Misses third-person resolution but is essentially free.

For SAMSum specifically, the third option captures most of the benefit because almost all the unresolved pronouns are first/second person. Third-person `he/she/they` *also* matters, but resolving those reliably needs a real model and we didn't get a clean win out of `coref-spanbert` to justify the cost.

## Result

| Metric  | Value | vs. PromptSum entity chain |
| ------- | ----- | -------------------------- |
| ROUGE-1 | 50.62 | +0.80                      |
| ROUGE-2 | 25.34 | +0.45                      |
| ROUGE-L | 41.78 | +0.73                      |

Modest. The gain shows up mostly on longer dialogues (≥ 10 turns) where pronoun chains get long and PEGASUS otherwise loses track.

## Run

```bash
python train.py
```
