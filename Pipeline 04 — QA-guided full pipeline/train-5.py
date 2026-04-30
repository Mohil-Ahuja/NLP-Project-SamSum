"""
Pipeline 04 — QA-guided full pipeline on SAMSum.

Owner: Vaibhav.

This is the most ambitious of our pipelines. Instead of one guidance
signal we run a *cascade*:

    Dialogue (SAMSum)
        |
        v
    Sentence Saliency  (LexRank — score each utterance)
        |
        v
    Structured Information Extraction
        (NER + dependency parsing + speaker-aware SVO + sentiment + intent)
        |
        v
    QA-style Representation
        (re-emit the structured facts as Q/A pairs)
        |
        v
    PEGASUS Summarisation
        |
        v
    Final summary

Why QA pairs:
   PEGASUS-XSum was pre-trained with the gap-sentence objective on news.
   Its decoder has seen a *lot* of declarative content. We hypothesised
   that turning the structure into Q/A would force the encoder to attend
   to *answers* (which are exactly the things summaries should mention)
   rather than to filler discourse markers. Q/A is also closer to the
   task framing humans use when they paraphrase a dialogue ("what
   happened? who's bringing what? when?").

Reality check:
   The pipeline is competitive (R-1 = 51.88) but does not beat plain SRL.
   We think the issue is compounding error: LexRank's saliency picks are
   noisy on short SAMSum dialogues, so the QA pairs we hand to PEGASUS are
   sometimes about the wrong utterance. A learned saliency model would
   probably help. Listed in docs/future_work.md.
"""

import os
import sys
import subprocess
import re
import torch
import spacy
import numpy as np
from collections import Counter
from transformers import (
    PegasusTokenizer, PegasusForConditionalGeneration,
    Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from data import config as cfg
from data.utils import (
    load_samsum, build_inputs, tokenize, SumDataset,
    make_compute_metrics, evaluate_on_test, DEVICE,
)


SIGNAL_LABEL = "QA"

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")


# ============================================================
# Step 1 — LexRank-style saliency over utterances
# ============================================================
def _tf_vector(text):
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return Counter(tokens)


def _cosine(c1, c2):
    common = set(c1) & set(c2)
    if not common:
        return 0.0
    num = sum(c1[t] * c2[t] for t in common)
    n1 = sum(v * v for v in c1.values()) ** 0.5
    n2 = sum(v * v for v in c2.values()) ** 0.5
    return num / (n1 * n2 + 1e-9)


def lexrank_scores(utterances, damping=0.85, n_iter=20):
    """Power-iteration LexRank without external dependencies."""
    n = len(utterances)
    if n <= 1:
        return [1.0] * n
    vecs = [_tf_vector(u) for u in utterances]
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i, j] = _cosine(vecs[i], vecs[j])
    # Row-normalise
    row_sums = sim.sum(axis=1, keepdims=True) + 1e-9
    M = sim / row_sums
    s = np.ones(n) / n
    for _ in range(n_iter):
        s = (1 - damping) / n + damping * M.T @ s
    return s.tolist()


# ============================================================
# Step 2 — Structured information extraction
# ============================================================
INTENT_KEYWORDS = {
    "request":   {"can you", "could you", "would you", "please", "?"},
    "agree":     {"sure", "ok", "okay", "yeah", "yes", "alright", "sounds good"},
    "decline":   {"no", "nope", "can't", "cannot", "not", "won't"},
    "plan":      {"will", "going to", "tomorrow", "tonight", "later", "next"},
    "inform":    set(),  # default
}


def _classify_intent(utterance):
    u = utterance.lower()
    for label, kws in INTENT_KEYWORDS.items():
        if label == "inform":
            continue
        if any(kw in u for kw in kws):
            return label
    return "inform"


_POS, _NEG = (
    {"good", "great", "love", "happy", "thanks", "thank", "amazing", "awesome", "perfect"},
    {"bad", "sad", "sorry", "hate", "awful", "no", "not", "can't", "won't"},
)


def _sentiment(utterance):
    u = utterance.lower()
    pos = sum(1 for w in _POS if w in u)
    neg = sum(1 for w in _NEG if w in u)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def _structured_facts(speaker, utterance):
    """Pull NER + speaker-aware SVO + sentiment + intent from one utterance."""
    facts = []
    doc = nlp(utterance)

    for ent in doc.ents:
        if ent.label_ in {"PERSON", "DATE", "TIME", "GPE", "LOC", "ORG"}:
            facts.append(("entity", ent.label_, ent.text))

    for sent in doc.sents:
        for tok in sent:
            if tok.pos_ != "VERB":
                continue
            subj = [w for w in tok.lefts if w.dep_ in ("nsubj", "nsubjpass")]
            obj = [w for w in tok.rights if w.dep_ in ("dobj", "pobj", "attr", "oprd")]

            def resolve(w):
                if w.text.lower() in ("i", "me", "my") and speaker:
                    return speaker
                return w.text

            if subj and obj:
                facts.append(("svo", resolve(subj[0]), f"{tok.text} {resolve(obj[0])}"))

    facts.append(("intent", speaker, _classify_intent(utterance)))
    sent = _sentiment(utterance)
    if sent != "neutral":
        facts.append(("sentiment", speaker, sent))
    return facts


# ============================================================
# Step 3 — Re-emit structured facts as QA pairs
# ============================================================
QA_TEMPLATES = {
    "svo":       "Q: What did {head} do?  A: {tail}",
    "entity":    "Q: Which {label} is mentioned?  A: {value}",
    "intent":    "Q: What is {speaker} trying to do?  A: {label}",
    "sentiment": "Q: How does {speaker} feel?  A: {label}",
}


def _facts_to_qa(facts):
    out = []
    seen = set()
    for f in facts:
        kind = f[0]
        if kind == "svo":
            line = QA_TEMPLATES["svo"].format(head=f[1], tail=f[2])
        elif kind == "entity":
            line = QA_TEMPLATES["entity"].format(label=f[1].lower(), value=f[2])
        elif kind == "intent":
            line = QA_TEMPLATES["intent"].format(speaker=f[1], label=f[2])
        elif kind == "sentiment":
            line = QA_TEMPLATES["sentiment"].format(speaker=f[1], label=f[2])
        else:
            continue
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out


# ============================================================
# Step 4 — End-to-end extraction
# ============================================================
def extract_guidance(text: str, top_k_ratio: float = 0.6) -> str:
    """
    LexRank → keep top-K most salient utterances → structured extraction → QA.

    top_k_ratio = 0.6 means: keep the top 60 % of utterances by saliency.
    """
    lines = []
    for raw in text.split("\n"):
        raw = raw.strip()
        if not raw:
            continue
        if ": " in raw:
            speaker, utt = raw.split(": ", 1)
        else:
            speaker, utt = "", raw
        lines.append((speaker, utt))

    if not lines:
        return "(empty dialogue)"

    utterances = [u for _, u in lines]
    scores = lexrank_scores(utterances)
    threshold = sorted(scores, reverse=True)[max(1, int(len(scores) * top_k_ratio)) - 1]

    qa_lines = []
    for (speaker, utt), score in zip(lines, scores):
        if score < threshold:
            continue
        facts = _structured_facts(speaker, utt)
        qa_lines.extend(_facts_to_qa(facts))

    if not qa_lines:
        speakers = [s for s, _ in lines if s]
        return "Q: who is talking?  A: " + ", ".join(sorted(set(speakers)))
    return "\n".join(qa_lines[:12])  # cap at 12 QA pairs to keep input length bounded


def main():
    train_df, val_df, test_df = load_samsum()

    print("\nQA pipeline sanity check:")
    test_dlg = "Amanda: I baked cookies. Do you want some?\nJerry: Sure!\nAmanda: I'll bring you tomorrow."
    print(extract_guidance(test_dlg))

    train_inputs, train_targets = build_inputs(train_df, extract_guidance, SIGNAL_LABEL)
    val_inputs, val_targets = build_inputs(val_df, extract_guidance, SIGNAL_LABEL)

    tokenizer = PegasusTokenizer.from_pretrained(cfg.PEGASUS_MODEL)
    model = PegasusForConditionalGeneration.from_pretrained(
        cfg.PEGASUS_MODEL, torch_dtype=torch.float32,
    )
    model.gradient_checkpointing_enable()
    model.to(DEVICE)

    train_enc = tokenize(tokenizer, train_inputs, train_targets)
    val_enc = tokenize(tokenizer, val_inputs, val_targets)
    train_ds = SumDataset(train_enc)
    val_ds = SumDataset(val_enc)

    out_dir = os.path.join(cfg.OUTPUT_ROOT, "pegasus_qa_samsum")
    os.makedirs(out_dir, exist_ok=True)

    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        per_device_eval_batch_size=cfg.BATCH_SIZE,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=cfg.GRAD_ACCUM,
        gradient_checkpointing=True,
        num_train_epochs=cfg.EPOCHS,
        learning_rate=cfg.LR,
        warmup_steps=cfg.WARMUP_STEPS,
        weight_decay=cfg.WEIGHT_DECAY,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="rouge1",
        predict_with_generate=True,
        generation_max_length=cfg.MAX_TARGET_LEN,
        generation_num_beams=cfg.GEN_NUM_BEAMS,
        logging_steps=50, report_to="none",
        save_total_limit=1, optim="adafactor",
    )

    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        compute_metrics=make_compute_metrics(tokenizer),
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    scores, _, _ = evaluate_on_test(model, tokenizer, test_df, extract_guidance, SIGNAL_LABEL)
    print("=== TEST ROUGE — Pipeline 04 (QA-guided) ===")
    for k, v in scores.items():
        print(f"  {k:12s}: {v:.2f}")


if __name__ == "__main__":
    main()
