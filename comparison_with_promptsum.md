# Head-to-head: our pipelines vs. PromptSum (paper)

## What PromptSum reports on SAMSum

From Table 2 of Ravaut et al., 2023 (arXiv:2308.03117):

| Setup        | R-1   | R-2   | R-L   | BertScore |
| ------------ | ----- | ----- | ----- | --------- |
| 100-shot     | 41.18 | 17.72 | 33.82 | 91.08     |
| Full-shot    | 46.72 | 22.35 | 38.60 | 91.84     |

The full-shot numbers are obtained by tuning **two soft prompts** (E-Prompt + S-Prompt, ≈ 200 K parameters total) on a frozen PEGASUS-large after a multi-task pre-training stage on a C4 subset.

## What our setup is

We update the **whole** PEGASUS-XSum backbone end-to-end (≈ 766 M parameters), with no pre-training stage. So our numbers are not directly comparable to PromptSum's — we are spending three orders of magnitude more trainable parameters.

The right comparison is:

- **Their entity-chain signal** vs. **our entity-chain signal** (pipeline 01) — same signal, different training recipe. Tells us how much of their gain comes from the *signal* vs. from the *prompt-tuning + pre-training* recipe.
- **Our entity-chain (pipeline 01)** vs. **our other signals (02–08)** — same training recipe, different signals. Tells us whether richer signals beat plain entity chains *given the same trainable budget*.

## The two comparisons

### 1. Same signal, different recipe

|                                | R-1   | R-2   | R-L   | trainable params |
| ------------------------------ | ----- | ----- | ----- | ---------------- |
| PromptSum (paper, full-shot)   | 46.72 | 22.35 | 38.60 | ≈ 200 K          |
| Pipeline 01 (ours, full-shot)  | 49.82 | 24.89 | 41.05 | 766 M            |

We get +3.10 R-1 by spending 3000× more parameters. Reads as: **the entity-chain signal is doing much of the work; full-backbone fine-tuning adds a few more points but at enormous parameter cost.** This is consistent with PromptSum's own ablation in Table 2 ("w/o pre-training" drops R-1 by ≈ 13 on SAMSum) — without their pre-training stage the soft-prompt-tuning recipe can't compete with full fine-tuning.

### 2. Same recipe, different signals (the actual ablation)

|                                | R-1       | R-2       | R-L       |
| ------------------------------ | --------- | --------- | --------- |
| Pipeline 00 (no signal)        | 50.14     | 25.06     | 41.32     |
| Pipeline 01 (entity chain)     | 49.82     | 24.89     | 41.05     |
| Pipeline 05 (dialogue acts)    | 48.95     | 23.81     | 40.18     |
| Pipeline 06 (coref resolved)   | 50.62     | 25.34     | 41.78     |
| Pipeline 07 (AMR)              | 51.05     | 25.92     | 42.40     |
| Pipeline 04 (QA-guided)        | 51.88     | 26.74     | 43.61     |
| Pipeline 03 (knowledge triples)| 52.41     | 27.18     | 44.02     |
| Pipeline 02 (SRL)              | 53.70     | 28.33     | 45.65     |
| **Pipeline 08 (hybrid SRL⊕KT)**| **54.91** | **28.97** | **46.32** |

Reads as:

- **The entity-chain signal does *not* beat no signal at all when the backbone is fully fine-tuned.** With full backbone fine-tuning, PEGASUS already infers entity salience from the dialogue itself, so being told the entities adds nothing. (This is the opposite of what the PromptSum paper finds in their soft-prompt setup, where the entity chain is essentially the only learnable signal — see their "w/o entity chain" ablation, which drops R-1 by 6+ points on most datasets.)
- **Richer relational signals do beat no signal.** SRL adds +3.6 R-1 over the unguided baseline. The hybrid adds +4.8.
- **The best result is +8.19 R-1 over the PromptSum paper number** with the hybrid pipeline, but most of that gap is from the training recipe, not from the signal.

## Take-aways for the project

1. The headline finding is *not* "we beat PromptSum" — that comparison is unfair because we update 3000× more parameters. The headline is **richer relational signals (SRL, KT, hybrid) outperform flat entity chains *under the same training recipe* on SAMSum.**

2. The entity-chain signal earns its keep in PromptSum's parameter-efficient setup, where soft prompts can't replace a strong discrete plan. In a full-fine-tune setup (ours), the model already has enough capacity to infer entity salience itself, so the *type* of signal — relational vs. flat — becomes what matters.

3. The natural next experiment is to *combine* their recipe with our signals: drop the SRL extractor on top of soft-prompt-tuned PEGASUS-large and see if R-1 climbs from their 46.72 to something closer to ours, while keeping the < 0.1 % parameter cost. That is the most interesting follow-up. See `docs/future_work.md`.
