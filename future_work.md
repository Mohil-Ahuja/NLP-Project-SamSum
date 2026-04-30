# Future Work

What we'd do next, in roughly the order we think is most worth doing.

---

## 1. Move from ROUGE to faithfulness metrics

ROUGE rewards lexical overlap. It cannot tell us whether our richer signals actually *reduce hallucination* — which was the whole point of preferring SRL/KT over flat entity chains. We want to add:

- **QAFactEval** (Fabbri et al., 2022) — generates QA pairs from the summary and checks them against the source.
- **FactCC** (Kryściński et al., 2020) — fine-tuned classifier for source–summary entailment.
- **G-Eval** (Liu et al., 2023) — LLM-as-judge with a rubric covering relevance, fluency, faithfulness.

Hypothesis: the SRL and hybrid pipelines should also win on faithfulness, not just on ROUGE. The PromptSum paper makes this argument by manually pruning hallucinated entities from the entity chain — we should test the same intervention on our richer signals.

---

## 2. Re-run with PromptSum's parameter-efficient recipe

Our biggest deviation from the paper is that we fine-tune the whole backbone. The interesting follow-up is:

- Take their soft-prompt-tuned, pre-trained, frozen `pegasus-large`.
- Replace the entity chain at the encoder input with our SRL / KT / hybrid signal.
- Compare against their reported 46.72 R-1 on SAMSum.

If our signals carry over to their setup, we get controllability + parameter-efficiency *and* a richer plan. If they don't, we learn that the entity-chain signal works specifically because it's flat and short — which is itself an interesting finding.

---

## 3. Generalise beyond SAMSum

SAMSum is short (avg 124 words), clean, and well-annotated. The interesting ablation is whether our SRL/KT win holds on:

- **DialogSum** (Chen et al., 2021) — longer dialogues, more participants.
- **MediaSum** (Zhu et al., 2021) — interview-style, ≈ 1500 words.
- **TweetSumm** — noisy customer-support dialogues.

Our hypothesis is that the SRL win *grows* with input length (longer dialogues → more pronouns → more value in resolving them up front). MediaSum is the cleanest test of this.

We chose not to evaluate on **StorySumm** (which our original presentation included) because it changes the task to narrative summarisation, where the signals that matter are different (e.g. discourse relations, character arcs). That's a separate research question.

---

## 4. Replace heuristic extractors with learned ones

Most of our extractors are spaCy + hand-rules. The KT pipeline in particular is bottlenecked by its hand-written `RELATION_MAP`. Replacing each extractor with a learned version is the obvious next step:

| Pipeline | Heuristic now              | Learned replacement                            |
| -------- | -------------------------- | ---------------------------------------------- |
| 02 SRL   | spaCy SVO                  | AllenNLP SRL or LSOIE-trained model            |
| 03 KT    | hand-written RELATION_MAP  | REBEL or OpenIE6                               |
| 04 QA    | LexRank for saliency       | small fine-tuned saliency classifier           |
| 05 Acts  | 5 hand-rules               | DA tagger trained on SwBD-DAMSL                |
| 06 Coref | rule-based, speaker only   | AllenNLP coref-spanbert                        |
| 07 AMR   | PENMAN approximation       | amrlib SPRING parser                           |

Of these, REBEL on KT is probably the highest-impact change — that's where our hand-rules are leaving the most accuracy on the table.

---

## 5. Combine more signals

We tested SRL ⊕ KT (pipeline 08). Other promising pairs:

- **SRL ⊕ Coref** — give the model resolved pronouns *and* explicit predicate-argument structure. Both pipelines win on different example types, so the combination should help.
- **SRL ⊕ Dialogue Acts** — acts gate which lines matter, SRL fills in what they say.
- **KT ⊕ AMR** — both relation-normalised, different granularities.

Naive cross-product is 36 pairs. A reasonable budget is the four pairs that combine our top-3 individual signals (SRL, KT, AMR) with our best "modifier" signal (coref or dialogue acts). That's 4 × 4h = 16 h of training, fits in a weekend.

---

## 6. Per-example signal selection

Different dialogues benefit from different signals. Long dialogues with many pronouns gain most from coreference. Short transactional dialogues gain most from dialogue acts. A *router* that picks which structured signal to feed PEGASUS — based on dialogue length, speaker count, presence of question marks — could plausibly outperform the hybrid.

This is the most speculative item on the list. It's also the closest to the PromptSum paper's controllability story — the entity chain is *user-tuneable*, and we could expose the structure block similarly so a user can pick "summarise focusing on plans" vs "summarise focusing on requests".

---

## 7. Human evaluation

The PromptSum paper does a small (50-sample) human evaluation on CNN/DM and XSum. Doing the same on SAMSum, comparing pipeline 01 (entity chain) with pipeline 08 (hybrid), would let us claim a *quality* improvement and not just a ROUGE delta.

Five raters × 50 samples × 3 axes (informativeness, faithfulness, fluency) is roughly a day of annotation. Worth it before we write up.

---

## 8. Model cards and bias audit

If this becomes a paper, we need a model card and a section on what these models *should not* be used for (no PII summarisation without consent; coref errors will systematically misattribute statements; dialogue-act tags reflect English-language norms only). Listed here so we don't forget.
