"""
Pipeline 00 — Vanilla PEGASUS-XSum on SAMSum.

Why this exists:
   It's the control. Every other pipeline in this repo prepends some
   structured signal to the dialogue. This one prepends nothing. If a
   guided pipeline can't beat this, the signal isn't pulling its weight.

Input format:
   "DIALOGUE:\n<raw dialogue>"

There is no extraction step. We do still wrap the dialogue in a "DIALOGUE:"
header so the input format is identical to the guided pipelines minus the
structure block — keeps the comparison clean.

Expected ROUGE-1 on 100-test of SAMSum: ≈ 50.1 (after 3 epochs).
"""

import os
import sys
import torch
from transformers import (
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from data import config as cfg
from data.utils import (
    load_samsum, build_inputs, tokenize, SumDataset,
    make_compute_metrics, evaluate_on_test, DEVICE,
)


SIGNAL_LABEL = ""  # baseline has no structure block


def extract_guidance(dialogue: str) -> str:
    """No guidance — return empty string. Build_inputs will still wrap dialogue."""
    return ""


def build_input_baseline(dialogue):
    """Override the standard builder to skip the structure block entirely."""
    return f"DIALOGUE:\n{dialogue}"


def main():
    train_df, val_df, test_df = load_samsum()
    print(f"Splits — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    train_inputs = [build_input_baseline(d) for d in train_df["dialogue"]]
    val_inputs = [build_input_baseline(d) for d in val_df["dialogue"]]
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

    out_dir = os.path.join(cfg.OUTPUT_ROOT, "pegasus_baseline_samsum")
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
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        predict_with_generate=True,
        generation_max_length=cfg.MAX_TARGET_LEN,
        generation_num_beams=cfg.GEN_NUM_BEAMS,
        logging_steps=50,
        report_to="none",
        save_total_limit=1,
        optim="adafactor",
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

    # Evaluate. We re-implement test-time generation rather than reusing
    # evaluate_on_test because the baseline has no signal block.
    model.eval()
    from data.utils import _rouge
    preds, refs = [], []
    for _, row in test_df.iterrows():
        prompt = build_input_baseline(row["dialogue"])
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
    print("=== TEST ROUGE — Pipeline 00 (Vanilla PEGASUS) ===")
    for k, v in scores.items():
        print(f"  {k:12s}: {v:.2f}")


if __name__ == "__main__":
    main()
