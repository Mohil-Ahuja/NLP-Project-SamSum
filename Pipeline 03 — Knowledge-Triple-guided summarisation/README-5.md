# Pipeline 03 — Knowledge Triples

**Owner:** Anandita

## What it does

Extracts `(head, relation, tail)` triples from each SAMSum dialogue using spaCy's dependency parser plus a small relation-normalisation lookup, and prepends them to the dialogue.

Example:

```
TRIPLES:
(Amanda, makes, cookies)
(Amanda, is, bringing)
(Hannah, asks, number)

DIALOGUE:
Amanda: I baked cookies. Do you want some?
Jerry: Sure!
...
```

## Why this is different from SRL

SRL emits the surface verb. Knowledge triples emit a *normalised* relation. So `"baked"`, `"made"`, and `"is making"` all collapse to `makes`. The hypothesis was that this would help the model abstract over verb variation and produce more compact summaries.

## What we found

Triples are competitive with SRL but slightly worse — R-1 = 52.41 vs 53.70. The reason, after we looked at examples, is that our hand-written `RELATION_MAP` covers the most common dialogue verbs but has nothing for the long tail (`reminds`, `insists`, `apologises`, `promises`). For verbs not in the map we fall back to the lemma, which makes the signal essentially equivalent to SRL but with extra punctuation that costs encoder tokens.

A learned relation normaliser (REBEL, OpenIE6, or a small fine-tuned T5) would probably tip this above SRL — see `docs/future_work.md`.

## Result

| Metric  | Value | vs. PromptSum entity chain |
| ------- | ----- | -------------------------- |
| ROUGE-1 | 52.41 | +2.59                      |
| ROUGE-2 | 27.18 | +2.29                      |
| ROUGE-L | 44.02 | +2.97                      |

## Run

```bash
python train.py
```
