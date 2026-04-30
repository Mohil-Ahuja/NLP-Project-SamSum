"""
Unified ROUGE harness.

Lets you score any pipeline's predictions against the SAMSum test set:

    python compute_rouge.py --preds preds.txt --refs refs.txt
"""

import argparse
import json
import os
import sys

import evaluate

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def compute_rouge(preds, refs):
    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    return {k: round(v * 100, 2) for k, v in scores.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="one prediction per line")
    ap.add_argument("--refs", required=True, help="one reference per line")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.preds) as f:
        preds = [l.strip() for l in f if l.strip()]
    with open(args.refs) as f:
        refs = [l.strip() for l in f if l.strip()]
    assert len(preds) == len(refs), f"length mismatch: {len(preds)} vs {len(refs)}"

    scores = compute_rouge(preds, refs)
    print(json.dumps(scores, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(scores, f, indent=2)


if __name__ == "__main__":
    main()
