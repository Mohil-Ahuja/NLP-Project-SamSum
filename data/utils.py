"""
Shared training scaffolding used by every pipeline.

The only thing that differs between pipelines is `extract_guidance(dialogue)`.
Everything else (tokenisation, dataset class, trainer, ROUGE) is shared.
"""

import os
import sys
import torch
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
import evaluate

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data import config as cfg

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Data loading
# ============================================================
def load_samsum():
    raw = load_dataset(cfg.DATASET_NAME)
    train_df = raw["train"].to_pandas()
    val_df = raw["validation"].to_pandas()
    test_df = raw["test"].to_pandas()

    if cfg.TRAIN_SIZE:
        train_df = train_df.iloc[:cfg.TRAIN_SIZE].reset_index(drop=True)
    if cfg.VAL_SIZE:
        val_df = val_df.iloc[:cfg.VAL_SIZE].reset_index(drop=True)
    if cfg.TEST_SIZE:
        test_df = test_df.iloc[:cfg.TEST_SIZE].reset_index(drop=True)
    return train_df, val_df, test_df


def build_inputs(df, extract_fn, signal_label):
    """
    Build encoder inputs of the form:
        <SIGNAL_LABEL>:
        <extracted structure>

        DIALOGUE:
        <raw dialogue>

    Each pipeline overrides extract_fn and signal_label.
    """
    inputs, targets = [], []
    for _, row in df.iterrows():
        dlg = row["dialogue"]
        struct = extract_fn(dlg)
        prompt = f"{signal_label}:\n{struct}\n\nDIALOGUE:\n{dlg}"
        inputs.append(prompt)
        targets.append(row["summary"])
    return inputs, targets


# ============================================================
# Tokenisation
# ============================================================
def tokenize(tokenizer, inputs, targets):
    model_inputs = tokenizer(
        inputs,
        max_length=cfg.MAX_SOURCE_LEN,
        padding="max_length",
        truncation=True,
    )
    labels = tokenizer(
        text_target=targets,
        max_length=cfg.MAX_TARGET_LEN,
        padding="max_length",
        truncation=True,
    )
    inp_ids = np.array(model_inputs["input_ids"], dtype=np.int64)
    att_mask = np.array(model_inputs["attention_mask"], dtype=np.int64)
    lbl_ids = np.array(labels["input_ids"], dtype=np.int64)
    lbl_ids[lbl_ids == tokenizer.pad_token_id] = -100
    return {"input_ids": inp_ids, "attention_mask": att_mask, "labels": lbl_ids}


# ============================================================
# Dataset
# ============================================================
class SumDataset(torch.utils.data.Dataset):
    def __init__(self, enc):
        self.enc = enc

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx], dtype=torch.long) for k, v in self.enc.items()}


# ============================================================
# ROUGE
# ============================================================
_rouge = evaluate.load("rouge")


def make_compute_metrics(tokenizer):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = ["\n".join(sent_tokenize(p.strip())) for p in decoded_preds]
        decoded_labels = ["\n".join(sent_tokenize(l.strip())) for l in decoded_labels]
        result = _rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {k: round(v * 100, 2) for k, v in result.items()}
    return compute_metrics


def evaluate_on_test(model, tokenizer, test_df, extract_fn, signal_label, device=DEVICE):
    """Generate summaries for the test set and report ROUGE."""
    model.eval()
    preds, refs = [], []
    for _, row in test_df.iterrows():
        struct = extract_fn(row["dialogue"])
        prompt = f"{signal_label}:\n{struct}\n\nDIALOGUE:\n{row['dialogue']}"
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=cfg.MAX_SOURCE_LEN,
            truncation=True,
        ).to(device)
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
    return {k: round(v * 100, 2) for k, v in scores.items()}, preds, refs
