"""
Pipeline 01 — PromptSum-style Entity Chain on SAMSum.

Why this exists:
   This is our reproduction of the PromptSum recipe (Ravaut et al., 2023).
   We use it as the *guided baseline*: every other guided pipeline in this
   repo replaces the entity chain with a different structured signal, and
   we measure whether that swap helps.

What an entity chain is:
   For dialogue: "Amanda: I baked cookies. Jerry: Sure!"
   Entity chain: "Amanda | cookies | Jerry"

   We extract entities with spaCy NER + the speaker names that appear at the
   start of each utterance, in order of first mention. PromptSum extracts
   them from the *gold summary* at training time and from the source at test
   time; we extract from the source for both, which is a documented
   simplification (PromptSum trains a separate E-prompt that learns to
   predict the chain — we are not doing the soft-prompt-tuning side of
   PromptSum, only its discrete-prompt side).

Caveat (important):
   PromptSum's full method also prompt-tunes a frozen PEGASUS-large with
   soft prompts. We are fine-tuning a smaller PEGASUS-xsum end-to-end. So
   this pipeline isn't a perfect re-implementation of PromptSum — it's a
   reproduction of *the entity-chain signal alone*, which is the part we
   are ablating against. See docs/methodology.md.
"""

import os
import sys
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


SIGNAL_LABEL = "ENTITIES"

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")


def extract_guidance(dialogue: str) -> str:
    """
    Build a PromptSum-style entity chain from a SAMSum dialogue.

    Strategy (matches Narayan et al., 2021 / PromptSum):
        1. Walk the dialogue line by line.
        2. The speaker (text before ': ') is always an entity.
        3. Run spaCy NER on the utterance and add any entity that isn't
           already in the chain.
        4. Concatenate with ' | ' separators in *first-mention* order.

    Why first-mention order matters:
       PromptSum argues entity-chain *order* helps the model decide what to
       mention first in the summary. We preserve that ordering.
    """
    chain, seen = [], set()

    for raw_line in dialogue.split("\n"):
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        if ": " in raw_line:
            speaker, utterance = raw_line.split(": ", 1)
            speaker = speaker.strip()
            if speaker and speaker not in seen:
                chain.append(speaker)
                seen.add(speaker)
        else:
            utterance = raw_line

        doc = nlp(utterance)
        for ent in doc.ents:
            txt = ent.text.strip()
            if txt and txt not in seen:
                chain.append(txt)
                seen.add(txt)

    if not chain:
        return "no entities"
    return " | ".join(chain)


def main():
    train_df, val_df, test_df = load_samsum()

    # Sanity check
    sample = train_df["dialogue"].iloc[0]
    print("Sample entity chain:", extract_guidance(sample))

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

    out_dir = os.path.join(cfg.OUTPUT_ROOT, "pegasus_entitychain_samsum")
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
    print("=== TEST ROUGE — Pipeline 01 (Entity Chain) ===")
    for k, v in scores.items():
        print(f"  {k:12s}: {v:.2f}")


if __name__ == "__main__":
    main()
