# Pipeline 04 — QA-guided Full Pipeline

**Owner:** Vaibhav

## What it does

A four-stage cascade that ends in a Q/A-shaped guidance signal:

```
Dialogue
   ↓
Sentence Saliency  (LexRank — TF-cosine, power-iteration)
   ↓
Structured Information Extraction
   (NER + speaker-aware SVO + sentiment + intent)
   ↓
QA-style Representation
   (re-emit facts as Q: ...  A: ...)
   ↓
PEGASUS Summarisation
```

Example signal:

```
QA:
Q: What did Amanda do?  A: baked cookies
Q: What is Jerry trying to do?  A: agree
Q: Which date is mentioned?  A: tomorrow
Q: How does Amanda feel?  A: positive

DIALOGUE:
Amanda: I baked cookies. Do you want some?
Jerry: Sure!
Amanda: I'll bring you tomorrow :-)
```

## The intuition

PEGASUS-XSum has seen a *lot* of declarative news content during pre-training. We hypothesised that re-shaping the structure as Q/A would force the encoder to attend to *answers* — exactly the things summaries should mention — rather than to filler phrases. Q/A is also the framing humans use when paraphrasing dialogue in their head ("what happened? who's bringing what? when?").

## What we found

It is competitive (R-1 = 51.88) but does not beat plain SRL. Two reasons:

1. **Compounding error.** LexRank's saliency picks are noisy on the very short dialogues SAMSum has — sometimes it drops the *one* utterance that contains the verb the gold summary uses. Replacing LexRank with a small fine-tuned saliency classifier should help (future work).
2. **Token budget.** Q/A pairs are wordier than raw triples — `"Q: What did Amanda do?  A: baked cookies"` vs `"Amanda baked cookies"`. We end up truncating dialogue to fit, which costs ROUGE.

## Result

| Metric  | Value | vs. PromptSum entity chain |
| ------- | ----- | -------------------------- |
| ROUGE-1 | 51.88 | +2.06                      |
| ROUGE-2 | 26.74 | +1.85                      |
| ROUGE-L | 43.61 | +2.56                      |

## Run

```bash
python train.py
```
