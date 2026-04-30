"""
Pipeline 05 — Dialogue-Act-guided summarisation on SAMSum.

Why dialogue acts:
   In dialogue summarisation the *type* of move matters as much as its
   content. A request, an agreement, a plan, and a piece of small-talk
   contribute very differently to a summary. Our entity chain and SRL
   pipelines treat all utterances equally; if we tag each utterance with
   a dialogue act, the model can learn to weight them differently.

Tagset (lightweight, hand-built):
   We use a small five-tag set chosen for SAMSum:
       REQUEST      — questions / commands
       AGREE        — yes / sure / ok / sounds good
       DECLINE      — no / can't / won't
       PLAN         — will / going to / tomorrow / next ...
       INFORM       — default; declarative without other markers

   This is far cruder than the SWBD-DAMSL tagset (42 acts) or the
   ISO 24617-2 standard, but those bigger schemes need a learned tagger
   and we wanted to test whether *any* dialogue-act signal helps before
   investing in one.

Result:
   Helpful but not a top performer. R-1 = 48.95, just below the entity
   chain baseline. The lesson: dialogue acts on their own are too coarse
   — they tell the model *which utterances matter* but not *what's in*
   them. Combined with SRL they probably help; we didn't get to test
   that combination, listed in future_work.md.
"""

import os
import sys
import torch
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


SIGNAL_LABEL = "ACTS"


# Lexical heuristics for the five-tag set. Order matters — first match wins.
_RULES = [
    ("REQUEST", lambda u: u.endswith("?") or any(k in u.lower() for k in ("can you", "could you", "would you", "please"))),
    ("AGREE",   lambda u: any(k in u.lower() for k in ("sure", "ok ", "okay", "yeah", "alright", "sounds good", "yes ", "yes,", "yes!"))),
    ("DECLINE", lambda u: any(k in u.lower() for k in ("nope", "no thanks", "can't", "cannot", "won't", "not really"))),
    ("PLAN",    lambda u: any(k in u.lower() for k in ("will ", "i'll", "we'll", "going to", "gonna", "tomorrow", "tonight", "next "))),
]


def _classify(utterance):
    for label, rule in _RULES:
        if rule(utterance.strip()):
            return label
    return "INFORM"


def extract_guidance(text: str) -> str:
    """
    Walk the dialogue line by line and emit:
        SPEAKER [ACT_LABEL]: shortened utterance

    The shortened utterance is the first 8 tokens of the original — enough
    to remind the model what the line was about without doubling token cost.
    """
    out = []
    for raw in text.split("\n"):
        raw = raw.strip()
        if not raw:
            continue
        if ": " in raw:
            speaker, utt = raw.split(": ", 1)
        else:
            speaker, utt = "?", raw
        act = _classify(utt)
        short = " ".join(utt.split()[:8])
        out.append(f"{speaker} [{act}]: {short}")
    return "\n".join(out) if out else "no acts"


def main():
    train_df, val_df, test_df = load_samsum()

    print("\nDialogue-act sanity check:")
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

    out_dir = os.path.join(cfg.OUTPUT_ROOT, "pegasus_acts_samsum")
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
    print("=== TEST ROUGE — Pipeline 05 (Dialogue Acts) ===")
    for k, v in scores.items():
        print(f"  {k:12s}: {v:.2f}")


if __name__ == "__main__":
    main()
