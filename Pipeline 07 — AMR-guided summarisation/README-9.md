# Pipeline 07 — AMR-guided Summarisation

## What it does

Builds Abstract Meaning Representation (PENMAN-format) graphs from each utterance and prepends them to the dialogue.

Example signal:

```
AMR:
(b / bake-01 :ARG0 (a / Amanda) :ARG1 (c / cookies))
(w / want-01 :ARG0 (y / you) :ARG1 (s / some))
(b / bring-01 :ARG0 (a / Amanda) :ARG1 (y / you))

DIALOGUE:
Amanda: I baked cookies. Do you want some?
Jerry: Sure!
Amanda: I'll bring you tomorrow :-)
```

## Why AMR

AMR is a normalised, rooted, labelled, directed graph of meaning. For dialogue summarisation it does three useful things at once:

1. **Verb normalisation** — `baked`, `made`, `is making` all collapse to `bake-01` / `make-01` PropBank frames. (Like our knowledge triples but learned.)
2. **Explicit argument structure** — `:ARG0`, `:ARG1`, … make agent vs. patient unambiguous. (Like SRL but cleaner.)
3. **Coreferent entity collapse** — repeated mentions become one node. (Like pipeline 06 but learned.)

In principle AMR is a strict superset of the signals carried by SRL + KT + coreference resolution.

## Important implementation note

A real AMR parser (e.g. `amrlib`'s SPRING parser) is heavy: ~500 MB checkpoint, slow on T4. For Kaggle reproducibility we ship a **PENMAN-shaped text approximation** built from speaker-aware spaCy SVO triples re-emitted with PropBank-ish frame names. It is not real AMR but is the same shape.

To run with the real parser:

```python
USE_AMRLIB = True  # at top of train.py
# pip install amrlib && python -m amrlib.setup_models
```

## Result (with the approximation)

| Metric  | Value | vs. PromptSum entity chain |
| ------- | ----- | -------------------------- |
| ROUGE-1 | 51.05 | +1.23                      |
| ROUGE-2 | 25.92 | +1.03                      |
| ROUGE-L | 42.40 | +1.35                      |

A real AMR parser would likely push this higher — the limitation is that our approximation isn't actually doing frame normalisation. But it does cost more encoder tokens than SRL because the PENMAN format is verbose, which hurts on SAMSum's already short inputs.

## Run

```bash
python train.py
```
