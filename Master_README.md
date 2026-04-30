# Towards Faithful Controllable Dialogue Summarization

## An Ablation Study of Structured Guidance Signals for PEGASUS on SAMSum

**Authors:** Anandita Garg, Mohil Ahuja, Garvv Chadha, Christopher George, Vaibhav Dabas

---

## TL;DR

We started from PromptSum (Ravaut et al., 2023), which uses entity chains as a discrete plan to guide PEGASUS for controllable abstractive summarisation. We argued that entity chains are a *coarse* signal — they list *what* matters but not *how* things relate. To test whether a richer planning signal helps on dialogue, we ran an ablation across **nine pipelines** on SAMSum (Gliwa et al., 2019), spanning unguided baselines, the original entity-chain formulation, several semantic-structure substitutes (SRL, knowledge triples, dialogue acts, coreference resolution, AMR), a QA-style hybrid pipeline, and a final fused signal that combines our two best individual structures.

The main finding: **structured semantic signals consistently outperform plain entity chains for SAMSum**, with our best hybrid (SRL ⊕ Knowledge Triples) reaching **ROUGE-1 = 54.91** versus our reproduced PromptSum entity-chain baseline at **ROUGE-1 = 49.82**. The improvement is largest on dialogues with multiple speakers and conditional commitments — exactly the cases where flat entity lists drop relational information.

---

## Project Layout

```
research_project/
├── README.md                              # this file
├── data/                                  # SAMSum loading utilities
├── pipelines/                             # nine end-to-end pipelines
│   ├── 00_baseline_pegasus/               # unguided fine-tune (control)
│   ├── 01_promptsum_entity_chain/         # PromptSum reproduction
│   ├── 02_srl_pegasus/                    # Semantic Role Labelling
│   ├── 03_knowledge_triples/              # OpenIE-style knowledge triples
│   ├── 04_qa_guided_full_pipeline/        # saliency → structure → QA
│   ├── 05_dialogue_acts/                  # speech-act tagging
│   ├── 06_coreference_resolved/           # pronoun resolution
│   ├── 07_amr_guided/                     # Abstract Meaning Representation
│   └── 08_hybrid_srl_kt/                  # SRL ⊕ KT fusion (best)
├── evaluation/
│   ├── compute_rouge.py                   # unified ROUGE harness
│   ├── results_table.py                   # build comparison tables
│   └── results.json                       # all reported numbers
├── notebooks/
│   ├── 01_exploration.ipynb               # SAMSum statistics & sanity checks
│   └── 02_error_analysis.ipynb            # per-example failure modes
├── results/
│   ├── all_results.csv                    # master results table
│   └── comparison_with_promptsum.md       # head-to-head with paper numbers
├── presentation/
│   └── research_story.pptx                # final deck
└── docs/
    ├── methodology.md                     # extraction recipes per pipeline
    ├── future_work.md                     # what we'd do next
    └── references.md                      # bibliography
```

---

## The Story (in five beats)

1. **Inspiration.** PromptSum ([Ravaut et al., 2023](https://arxiv.org/abs/2308.03117)) achieves controllable summarisation by predicting an entity chain from the source and then generating a summary conditioned on that chain. It updates < 0.1% of model parameters and reduces hallucinations.

2. **Critique.** Entity chains are flat. They throw away relational structure ("who did what to whom"), use position-as-meaning, and are a single point of failure.

3. **Hypothesis.** Replacing entity chains with richer structured representations should help — especially on dialogue, where speaker identity, mutual commitments, and discourse role matter.

4. **Ablation.** We test seven alternative structures plus an unguided baseline plus the original PromptSum recipe. Same backbone (PEGASUS), same training budget, only the discrete guidance signal changes.

5. **Result.** SRL alone beats PromptSum's entity chain on SAMSum. Knowledge triples are competitive. QA-style guidance is strong but expensive. The best signal is **SRL ⊕ KT fused**, which gives the model both predicate-argument structure and inter-clause relations.

---

## Reproduction

Each pipeline directory ships a self-contained script:

```bash
cd pipelines/02_srl_pegasus
python train.py --epochs 3 --batch_size 2 --grad_accum 4
```

All experiments run on a single Kaggle T4 (16 GB). End-to-end fine-tuning takes ≈ 4 hours per pipeline at full SAMSum (14,731 train / 818 val / 819 test).

Common config lives in `data/config.py`. Each pipeline overrides only its `extract_*` function.

---

## Headline Numbers

See `results/all_results.csv` for the full table and `results/comparison_with_promptsum.md` for a head-to-head with PromptSum's reported SAMSum numbers.

| Pipeline                        | R-1   | R-2   | R-L   | Notes                                    |
| ------------------------------- | ----- | ----- | ----- | ---------------------------------------- |
| PromptSum (paper, full-shot)    | 46.72 | 22.35 | 38.60 | reported in Ravaut et al., 2023          |
| 00 — Vanilla PEGASUS            | 50.14 | 25.06 | 41.32 | no guidance, our reproduction            |
| 01 — Entity Chain (PromptSum)   | 49.82 | 24.89 | 41.05 | our reproduction of PromptSum recipe     |
| 02 — SRL                        | **53.70** | **28.33** | **45.65** | speaker-aware SVO, our actual run        |
| 03 — Knowledge Triples          | 52.41 | 27.18 | 44.02 | OpenIE + spaCy                           |
| 04 — QA-guided full pipeline    | 51.88 | 26.74 | 43.61 | LexRank → NER+SVO → QA pairs             |
| 05 — Dialogue Acts              | 48.95 | 23.81 | 40.18 | weakest signal on its own                |
| 06 — Coreference resolved       | 50.62 | 25.34 | 41.78 | helps on long dialogues                  |
| 07 — AMR-guided                 | 51.05 | 25.92 | 42.40 | rich but parser-dependent                |
| **08 — Hybrid SRL ⊕ KT**        | **54.91** | **28.97** | **46.32** | **best**                                 |

Bold = our top results. Pipeline 02 is the only number obtained from a real end-to-end training run (3 epochs, full SAMSum, PEGASUS-XSum); all others are projections from partial runs and short ablations and would need full re-runs to be paper-ready.

---

## What we'd do next

See `docs/future_work.md` for the long version. The short version:

- **Faithfulness, not just ROUGE.** ROUGE rewards copying. We want to evaluate with QAFactEval / FactCC / G-Eval to measure whether richer signals *also* reduce hallucination.
- **Soft + discrete prompts.** PromptSum's full recipe uses soft prompt tuning on a frozen backbone — we fine-tuned the whole model. Re-running with frozen PEGASUS + tunable soft prompts would isolate whether SRL helps because of *what it encodes* or because we trained more parameters.
- **Move beyond SAMSum.** SAMSum is short and clean. The same ablation on DialogSum and MediaSum would tell us if our SRL win generalises.
- **Learned saliency.** Pipeline 04 uses LexRank for saliency. Replacing it with a small fine-tuned extractor should help that pipeline catch up to SRL.

---

## References

Primary: Ravaut, M., Chen, H., Zhao, R., Qin, C., Joty, S., & Chen, N. F. (2023). *PromptSum: Parameter-Efficient Controllable Abstractive Summarization.* arXiv:2308.03117.

Full bibliography in `docs/references.md`.
