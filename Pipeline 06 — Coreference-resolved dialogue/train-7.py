"""
Pipeline 06 — Coreference-resolved dialogue.

The motivation:
   SAMSum dialogues are full of pronouns. "He called her last time we
   were at the park." Without coreference resolution, that sentence
   doesn't tell PEGASUS *who* called *whom*. With CR it becomes
   "Larry called Betty last time Hannah and Amanda were at the park."

   We tried three CR approaches:
       (a) neuralcoref via spaCy (deprecated; spaCy v3 dropped it)
       (b) AllenNLP's coref-spanbert (heavy, slow on T4)
       (c) a lightweight rule-based resolver that only handles the
           speaker-pronouns case ("I/me/my/we/us/our")

   We ship (c) here because (a) and (b) blow up training time without
   a corresponding ROUGE gain on SAMSum's short dialogues. The rule
   covers the cases that actually matter for the dataset.

What we emit:
   The whole dialogue with pronouns rewritten in place. We do *not*
   prepend a separate structure block — the resolved dialogue *is* the
   guidance. So this pipeline's input to PEGASUS is:

       DIALOGUE (resolved):
       <resolved dialogue>
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
    load_samsum, tokenize, SumDataset,
    make_compute_metrics, DEVICE,
)


SIGNAL_LABEL = "DIALOGUE (resolved)"


_SINGULAR = {"i", "me", "my", "mine", "myself"}
_PLURAL = {"we", "us", "our", "ours", "ourselves"}


def _resolve_line(speaker, utterance, all_speakers):
    """Replace I/me/my with the speaker; replace we/us/our with all known speakers."""
    out = []
    for tok in utterance.split():
        # peel punctuation
        prefix, core, suffix = "", tok, ""
        while core and not core[0].isalnum():
            prefix += core[0]; core = core[1:]
        while core and not core[-1].isalnum():
            suffix = core[-1] + suffix; core = core[:-1]
        low = core.lower()

        if low in _SINGULAR and speaker:
            replaced = speaker if low in {"i", "myself"} else f"{speaker}'s" if low in {"my", "mine"} else speaker
            out.append(prefix + replaced + suffix)
        elif low in _PLURAL and all_speakers:
            replaced = " and ".join(all_speakers) if low in {"we", "ourselves"} else f"{' and '.join(all_speakers)}'s" if low in {"our", "ours"} else " and ".join(all_speakers)
            out.append(prefix + replaced + suffix)
        else:
            out.append(prefix + core + suffix)
    return " ".join(out)


def extract_guidance(text: str) -> str:
    """Resolve speaker-pronouns through the whole dialogue."""
    speakers = []
    parsed = []
    for raw in text.split("\n"):
        raw = raw.strip()
        if not raw:
            continue
        if ": " in raw:
            sp, utt = raw.split(": ", 1)
            sp = sp.strip()
            if sp not in speakers:
                speakers.append(sp)
            parsed.append((sp, utt))
        else:
            parsed.append(("", raw))

    out_lines = []
    for sp, utt in parsed:
        if sp:
            resolved = _resolve_line(sp, utt, speakers)
            out_lines.append(f"{sp}: {resolved}")
        else:
            out_lines.append(utt)
    return "\n".join(out_lines)


# ============================================================
# Custom builder: the resolved dialogue *replaces* the dialogue,
# so we don't prepend a structure block separately.
# ============================================================
def _build_input(dialogue):
    return f"{SIGNAL_LABEL}:\n{extract_guidance(dialogue)}"


def main():
    train_df, val_df, test_df = load_samsum()

    print("\nCoreference sanity check:")
    test_dlg = "Amanda: I baked cookies. Do you want some?\nJerry: Sure!\nAmanda: I'll bring you tomorrow."
    print(extract_guidance(test_dlg))

    train_inputs = [_build_input(d) for d in train_df["dialogue"]]
    val_inputs = [_build_input(d) for d in val_df["dialogue"]]
    train_targets = train_df["summary"].tolist()
    val_targets = val_df["summary"].tolist()

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

    out_dir = os.path.join(cfg.OUTPUT_ROOT, "pegasus_coref_samsum")
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

    # Eval — re-implement because our input format differs
    from data.utils import _rouge
    model.eval()
    preds, refs = [], []
    for _, row in test_df.iterrows():
        prompt = _build_input(row["dialogue"])
        enc = tokenizer(prompt, return_tensors="pt",
                        max_length=cfg.MAX_SOURCE_LEN, truncation=True).to(DEVICE)
        with torch.no_grad():
            ids = model.generate(
                **enc,
                max_new_tokens=cfg.MAX_TARGET_LEN,
                num_beams=cfg.GEN_NUM_BEAMS,
                length_penalty=cfg.GEN_LENGTH_PENALTY,
                no_repeat_ngram_size=cfg.GEN_NO_REPEAT_NGRAM_SIZE,
                early_stopping=True,
            )
        preds.append(tokenizer.decode(ids[0], skip_special_tokens=True))
        refs.append(row["summary"])
    scores = _rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    scores = {k: round(v * 100, 2) for k, v in scores.items()}
    print("=== TEST ROUGE — Pipeline 06 (Coreference Resolved) ===")
    for k, v in scores.items():
        print(f"  {k:12s}: {v:.2f}")


if __name__ == "__main__":
    main()
