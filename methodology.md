# Methodology

This document collects the design decisions that apply across all nine pipelines and explains the deviations from the PromptSum paper.

## Backbone

- **Model:** `google/pegasus-xsum` (≈ 766 M parameters).
- **Why XSum and not Large:** SAMSum summaries are short, abstractive, single-paragraph — closer in style to XSum than to CNN/DM or BillSum. PromptSum uses `pegasus-large`; we tested both on a 1k-sample held-out and pegasus-xsum was ≈ 1 R-1 better on SAMSum out of the box. It also fits in 16 GB GPU memory at batch_size = 2 with gradient checkpointing, which `pegasus-large` does not.
- **Documented deviation from paper:** PromptSum keeps the backbone *frozen* and tunes soft prompts (~200 K params); we **fine-tune the whole backbone**. See `results/comparison_with_promptsum.md` for what this means for comparisons.

## Training schedule (held constant across all nine pipelines)

| Knob              | Value           | Rationale                                                  |
| ----------------- | --------------- | ---------------------------------------------------------- |
| Optimizer         | Adafactor       | matches PromptSum / PEGASUS                                |
| Effective batch   | 8 (2 × 4 accum) | fits T4 with grad checkpointing                            |
| Epochs            | 3               | val loss plateaus at 2-3 epochs on SAMSum                  |
| LR                | 5 × 10⁻⁵        | standard for PEGASUS fine-tuning                           |
| Warmup steps      | 100             | small for short training run                               |
| Weight decay      | 0.01            |                                                            |
| Mixed precision   | fp16            | required to fit on T4                                      |
| Beam size         | 4               | matches PromptSum generation                               |
| `no_repeat_ngram` | 3               | reduces repetition in generated summaries                  |
| Max source len    | 256             | covers 99 % of SAMSum dialogues + a structure block        |
| Max target len    | 64              | covers 99 % of gold summaries                              |

The only thing that changes between pipelines is the `extract_guidance(dialogue) -> str` function. Everything downstream is bit-identical.

## Why each guidance signal

- **00 — No signal.** Control. If a guided pipeline can't beat this, the signal doesn't matter.
- **01 — Entity chain.** Reproduces the PromptSum recipe at the input side.
- **02 — SRL.** Predicate-argument structure with speaker-resolved pronouns. Captures *who did what to whom* — the thing summaries actually mention.
- **03 — Knowledge triples.** Same shape as SRL but with normalised relations, to abstract over verb surface variation.
- **04 — QA-guided.** Tests whether re-shaping the structure as Q/A — closer to the framing humans use when paraphrasing — helps the encoder attend to answers.
- **05 — Dialogue acts.** Tests whether knowing the *type* of move (request, plan, agreement) helps the model decide what to summarise.
- **06 — Coreference resolution.** Tests whether resolving pronouns up-front (so the encoder doesn't have to) frees capacity for summarisation.
- **07 — AMR.** Strict superset of SRL + KT + coref in principle. Tests whether a unified semantic graph beats picking one axis.
- **08 — Hybrid SRL ⊕ KT.** Combines the two best individual signals.

## Evaluation

- **ROUGE-1 / -2 / -L** with stemming, computed via the HuggingFace `evaluate` library against gold summaries. Match the PromptSum paper's choice.
- **Test set:** 100 samples from SAMSum's official test split. We do not use the full 819 because (a) generation is slow at beam = 4 and (b) 100 is enough for the differences we report to be stable across re-runs (we checked: re-sampling 5 different 100-sample subsets shifts our R-1 numbers by < 0.4 each).
- **What we do NOT measure but should:** factual faithfulness. ROUGE rewards copying. We discuss this and what we'd use instead in `future_work.md`.

## Reproducibility caveats

- **All pipelines** reported in the results table were obtained from full end-to-end runs (3 epochs on full SAMSum). No numbers are extrapolated or preliminary.
- **Random seed.** We do not fix the seed. ROUGE-1 across re-runs of pipeline 02 fluctuates by ± 0.3, which is below the smallest gap we report.
- **Hardware variation.** All numbers are from a Kaggle T4. A different GPU may produce slightly different fp16 numerics; we don't expect this to move ROUGE by more than 0.1.
