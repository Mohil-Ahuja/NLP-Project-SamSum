# Pipeline 02 — SRL-guided summarisation

**Owner:** Garvv

**Status:** This is the only pipeline whose ROUGE numbers come from a real end-to-end run (3 epochs, full SAMSum, T4, ≈ 4 h).

## What it does

For each SAMSum dialogue we extract speaker-aware Subject-Verb-Object triples with spaCy and prepend them to the dialogue. The encoder sees:

```
SRL:
Amanda baked cookies
Amanda bring you
Hannah needs number
...

DIALOGUE:
Amanda: I baked cookies. Do you want some?
Jerry: Sure!
...
```

## The trick that matters

We parse each dialogue line *separately* and substitute first-person pronouns with the speaker's name. So `"I baked cookies"` said by Amanda becomes the triple `"Amanda baked cookies"`, not `"I baked cookies"`. Without that step the SRL output is full of `"I X"` triples that don't tell the model anything new — with it, every triple is grounded to a named participant, which is exactly what summaries refer to.

This is the key advantage SRL has over plain entity chains: entity chains say *Amanda is in this conversation*, SRL says *Amanda is the one who baked the cookies*.

## Result

| Metric  | Value | vs. PromptSum entity chain |
| ------- | ----- | -------------------------- |
| ROUGE-1 | 53.70 | +3.88                      |
| ROUGE-2 | 28.33 | +3.44                      |
| ROUGE-L | 45.65 | +4.60                      |

## Per-epoch validation (real run)

| Epoch | Train loss | Val loss | R-1   | R-2   | R-L   |
| ----- | ---------- | -------- | ----- | ----- | ----- |
| 1     | 13.46      | 1.522    | 50.23 | 25.88 | 41.90 |
| 2     | 12.39      | 1.485    | 50.54 | 26.24 | 42.39 |
| 3     | 11.97      | 1.479    | 50.85 | 26.39 | 42.60 |

Test ROUGE (above) is from the best checkpoint (epoch 3) on 100 held-out test samples.

## Failure modes seen during error analysis

- Dialogues with a lot of small-talk filler ("haha", "lol", "yeah") produce very few triples — the structure block degrades to `participants: A, B`.
- Heavy use of "we" stays unresolved (we only resolve "I/me/my", not "we/us/our") — adding a pass for plural pronouns is the obvious next change.
- The triple ordering reflects parse order, not narrative importance. A learned saliency reranker on top would probably help.

## Run

```bash
python train.py
```
