"""
Pipeline 03 — Knowledge-Triple-guided summarisation on SAMSum.

Owner: Anandita.

Why knowledge triples instead of SRL:
   SRL gives us "<subject> <verb> <object>". Knowledge triples give us
   "<head_entity, relation, tail_entity>". Same shape, different semantics:
       SRL:   "Amanda baked cookies"            (lexical verb 'baked')
       KT:    "(Amanda, baked, cookies)"        (canonical relation 'baked')
       KT++:  "(Amanda, made_food, cookies)"    (normalised relation)

   The hope was that normalised relations would let the model abstract
   over surface verb variation ("baked" / "made" / "is making" / "will
   bring") and produce more compact summaries.

   In practice, on SAMSum, this turns out to be roughly a wash with SRL
   (we ship at R-1 = 52.41 vs 53.70). The relation-normalisation table
   has too few entries to cover the long tail of dialogue verbs, and
   for the verbs it *does* cover the model already has a good prior.

Implementation:
   We use spaCy's dependency parser to extract triples and a small
   hand-written relation-normalisation lookup. For the long tail we
   fall back to the lemma of the verb. We could plug in OpenIE6 or
   REBEL here for higher recall — that's listed as future work.
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


SIGNAL_LABEL = "TRIPLES"

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")


# Tiny relation-normalisation lookup. Trades coverage for consistency.
# Real systems would use a learned KB-aware normaliser; for an ablation
# this is enough to test whether normalised relations help at all.
RELATION_MAP = {
    "is": "is", "are": "is", "was": "is", "were": "is", "be": "is",
    "have": "has", "has": "has", "had": "has",
    "want": "wants", "wants": "wants", "wanted": "wants",
    "like": "likes", "likes": "likes", "liked": "likes",
    "go": "goes_to", "goes": "goes_to", "went": "goes_to", "going": "goes_to",
    "make": "makes", "makes": "makes", "made": "makes", "making": "makes",
    "bake": "makes", "baked": "makes", "baking": "makes",
    "cook": "makes", "cooked": "makes", "cooking": "makes",
    "buy": "buys", "bought": "buys", "buying": "buys",
    "meet": "meets", "met": "meets", "meeting": "meets",
    "call": "calls", "called": "calls", "calling": "calls",
    "send": "sends", "sent": "sends", "sending": "sends",
    "tell": "tells", "told": "tells", "telling": "tells",
    "ask": "asks", "asked": "asks", "asking": "asks",
    "see": "sees", "saw": "sees", "seen": "sees",
}


def _normalise_relation(verb_token):
    lemma = verb_token.lemma_.lower()
    return RELATION_MAP.get(verb_token.text.lower(), RELATION_MAP.get(lemma, lemma))


def extract_guidance(text: str, max_sents: int = 5) -> str:
    """
    Extract (head, relation, tail) triples from a SAMSum dialogue.

    Uses speaker-aware pronoun resolution exactly like the SRL pipeline,
    but emits triples in canonical "(head, relation, tail)" form with the
    relation normalised through RELATION_MAP.
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
                if not (subj and obj):
                    continue

                def resolve(w):
                    if w.text.lower() in ("i", "me", "my") and speaker:
                        return speaker
                    return w.text

                head = resolve(subj[0])
                tail = resolve(obj[0])
                rel = _normalise_relation(token)

                triple = f"({head}, {rel}, {tail})"
                if triple not in seen:
                    seen.add(triple)
                    triples.append(triple)

    if not triples:
        speakers = []
        for line in text.split("\n"):
            if ": " in line:
                spk = line.split(": ", 1)[0].strip()
                if spk and spk not in speakers:
                    speakers.append(spk)
        return "(participants, are, " + " & ".join(speakers) + ")" if speakers else "(no, triples, extracted)"

    return "\n".join(triples)


def main():
    train_df, val_df, test_df = load_samsum()

    print("\nKT sanity check:")
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

    out_dir = os.path.join(cfg.OUTPUT_ROOT, "pegasus_kt_samsum")
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
    print("=== TEST ROUGE — Pipeline 03 (Knowledge Triples) ===")
    for k, v in scores.items():
        print(f"  {k:12s}: {v:.2f}")


if __name__ == "__main__":
    main()
