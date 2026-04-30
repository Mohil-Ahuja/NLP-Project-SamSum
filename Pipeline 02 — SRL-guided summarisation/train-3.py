"""
Pipeline 02 — SRL-guided summarisation on SAMSum.

Owner: Garvv. This is the only pipeline in the repo whose ROUGE numbers
come from an actual end-to-end run (3 epochs, full SAMSum, T4, ~4 h).

Why SRL instead of entity chains:
   Entity chains list *what* matters (Amanda, cookies, Jerry).
   SRL also captures *who did what to whom*: "Amanda baked cookies".
   For SAMSum dialogue this is a much better signal because pronouns
   ("I", "you", "we") are everywhere, and the speaker's name only
   shows up at the start of the line.

Speaker resolution trick:
   We parse each dialogue line *separately* and replace first-person
   pronouns ("I", "me", "my") in the utterance with the speaker name
   from before the colon. So "I baked cookies" said by Amanda becomes
   the triple "Amanda baked cookies", not "I baked cookies".

   This single substitution is responsible for a meaningful chunk of
   the ROUGE gain over the entity chain baseline — without it the SRL
   triples look almost identical for every speaker.
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


SIGNAL_LABEL = "SRL"

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")


def extract_guidance(text: str, max_sents: int = 5) -> str:
    """
    Rule-based SVO extraction via spaCy with speaker-aware pronoun resolution.

    Steps:
        1. Split dialogue into lines, peel off the speaker (text before ': ').
        2. spaCy-parse each utterance.
        3. For each VERB token, collect:
             - subject:  any nsubj / nsubjpass child to the left
             - object:   any dobj / pobj / attr / oprd child to the right
        4. If subject is "I"/"me"/"my", swap it for the speaker name.
        5. Emit "<subj> <verb> <obj>" or "<subj> <verb>".

    Falls back to "participants: <names>" if no triple is extractable
    (which happens on very short single-utterance dialogues).
    """
    triples, seen = [], set()

    for raw_line in text.split("\n"):
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        if ": " in raw_line:
            speaker, utterance = raw_line.split(": ", 1)
        else:
            speaker, utterance = "", raw_line

        doc = nlp(utterance)
        for sent in list(doc.sents)[:max_sents]:
            for token in sent:
                if token.pos_ != "VERB":
                    continue
                subj = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
                obj = [w for w in token.rights if w.dep_ in ("dobj", "pobj", "attr", "oprd")]

                def resolve(w):
                    if w.text.lower() in ("i", "me", "my") and speaker:
                        return speaker
                    return w.text

                if subj and obj:
                    t = f"{resolve(subj[0])} {token.text} {resolve(obj[0])}"
                elif subj:
                    t = f"{resolve(subj[0])} {token.text}"
                else:
                    continue

                if t not in seen:
                    seen.add(t)
                    triples.append(t)

    if not triples:
        speakers = []
        for line in text.split("\n"):
            if ": " in line:
                spk = line.split(": ", 1)[0].strip()
                if spk and spk not in speakers:
                    speakers.append(spk)
        return "participants: " + ", ".join(speakers) if speakers else "no roles extracted"

    return "\n".join(triples)


def main():
    train_df, val_df, test_df = load_samsum()

    print("\nSRL sanity check:")
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

    out_dir = os.path.join(cfg.OUTPUT_ROOT, "pegasus_srl_samsum")
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
    print("=== TEST ROUGE — Pipeline 02 (SRL) ===")
    for k, v in scores.items():
        print(f"  {k:12s}: {v:.2f}")


if __name__ == "__main__":
    main()
