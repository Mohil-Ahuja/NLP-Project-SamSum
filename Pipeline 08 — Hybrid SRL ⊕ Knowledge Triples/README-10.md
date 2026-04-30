# Pipeline 08 — Hybrid SRL ⊕ Knowledge Triples (Best Result)

## What it does

Combines the SRL extractor from pipeline 02 with the knowledge-triple extractor from pipeline 03 into a single guidance block with two labelled sub-sections.

Example signal:

```
STRUCTURE:
[SRL]
Amanda baked cookies
Amanda bring you
Hannah needs number
[TRIPLES]
(Amanda, makes, cookies)
(Amanda, is, bringing)
(Hannah, asks, number)

DIALOGUE:
...
```

## Why this works

SRL and KT are not redundant. SRL preserves verb tense and modality (`will bring` vs `bring`). KT abstracts surface verb form (`baked`, `made`, `is making` → `makes`). Each is sharper than the other on different lines:

- SRL is sharper when modality matters: *"will bring tomorrow"* keeps the future-tense cue that gold summaries reproduce.
- KT is sharper when verb surface varies: a dialogue mentions `bake`, `make`, and `cook` for the same activity; KT collapses them to a single relation, which helps the model emit one summary clause instead of three.

Giving the model both lets it pick whichever is sharper for any given line. The cost is roughly 30 % more encoder tokens than SRL alone — well within our 256-token budget for SAMSum.

## Best result

| Metric  | Value | vs. PromptSum entity chain | vs. SRL alone |
| ------- | ----- | -------------------------- | ------------- |
| ROUGE-1 | **54.91** | +5.09                  | +1.21         |
| ROUGE-2 | **28.97** | +4.08                  | +0.64         |
| ROUGE-L | **46.32** | +5.27                  | +0.67         |

This is our best pipeline on SAMSum. The gain over SRL alone is small but consistent across all three ROUGE variants.

## Why we didn't try every pairwise combination

Time. The naïve cross-product of nine pipelines is 36 pairs, each costing ~4 h to train. We picked SRL × KT because (a) they were our two best individual signals and (b) they have complementary strengths in the way described above. Other promising pairs to try:

- SRL × Coreference Resolution (resolved dialogue + SRL)
- SRL × Dialogue Acts (acts to gate, SRL to fill)
- KT × AMR (both relation-normalised, different granularities)

Listed in `docs/future_work.md`.

## Run

```bash
python train.py
```
