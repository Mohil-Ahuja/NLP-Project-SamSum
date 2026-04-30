"""
Pipeline 07 — AMR-guided summarisation on SAMSum.

What is AMR:
   Abstract Meaning Representation (Banarescu et al., 2013) is a rooted,
   labeled, directed graph that represents the meaning of a sentence in
   a normalised, language-independent way. Verbs become PropBank frames
   (e.g. "bake-01"), arguments become numbered roles (:ARG0, :ARG1, ...),
   and shared entities collapse to a single node.

   For dialogue summarisation AMR is appealing because it (a) abstracts
   over surface verb variation (like our knowledge triples but learned),
   (b) makes argument structure explicit (like SRL but cleaner), and
   (c) collapses coreferent entities (like pipeline 06 but learned).

Implementation note (important):
   A real AMR parser (e.g. amrlib's spring or t5-amr-parser) is a heavy
   dependency. To keep this pipeline runnable on Kaggle we ship a
   *PENMAN-format text approximation*: speaker-aware SVO triples
   re-emitted with PropBank-ish frame names and :ARGn roles. It is not
   a true AMR but it is the same shape and lets us test the hypothesis
   without pulling in a 500MB parser.

   To run with a real parser, set USE_AMRLIB = True at the top of
   this file and `pip install amrlib`. The training loop and the
   tokenisation budget are identical either way.
"""

import os
import sys
import subprocess
import torch
import spacy
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


SIGNAL_LABEL = "AMR"
USE_AMRLIB = False  # set True if you've installed amrlib

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")


# ============================================================
# Optional real AMR parser — only loaded if USE_AMRLIB=True
# ============================================================
_amr_model = None
def _amr_parse_real(sentences):
    global _amr_model
    if _amr_model is None:
        import amrlib
        _amr_model = amrlib.load_stog_model()
    return _amr_model.parse_sents(sentences)


# ============================================================
# PENMAN-shaped text approximation (default)
# ============================================================
def _amr_approx(speaker, utterance):
    doc = nlp(utterance)
    blocks = []
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

            frame = f"{tok.lemma_}-01"
            parts = [f"({frame[0]} / {frame}"]
            if subj:
                parts.append(f" :ARG0 ({resolve(subj[0])[0].lower()} / {resolve(subj[0])})")
            if obj:
                parts.append(f" :ARG1 ({resolve(obj[0])[0].lower()} / {resolve(obj[0])})")
            parts.append(")")
            blocks.append("".join(parts))
    return blocks


def extract_guidance(text: str) -> str:
    blocks = []
    for raw in text.split("\n"):
        raw = raw.strip()
        if not raw:
            continue
        if ": " in raw:
            speaker, utt = raw.split(": ", 1)
        else:
            speaker, utt = "", raw

        if USE_AMRLIB:
            try:
                graphs = _amr_parse_real([utt])
                for g in graphs:
                    blocks.append(g.replace("\n", " ").strip())
            except Exception:
                blocks.extend(_amr_approx(speaker, utt))
        else:
            blocks.extend(_amr_approx(speaker, utt))

    return "\n".join(blocks) if blocks else "(no amr)"


def main():
    train_df, val_df, test_df = load_samsum()

    print("\nAMR sanity check:")
    test_dlg = "Amanda: I baked cookies. Do you want some?\nJerry: Sure!"
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

    out_dir = os.path.join(cfg.OUTPUT_ROOT, "pegasus_amr_samsum")
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
    print("=== TEST ROUGE — Pipeline 07 (AMR) ===")
    for k, v in scores.items():
        print(f"  {k:12s}: {v:.2f}")


if __name__ == "__main__":
    main()
