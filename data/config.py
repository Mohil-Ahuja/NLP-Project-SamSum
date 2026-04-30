"""
Shared config for all pipelines.

We keep the backbone, the dataset, and the training schedule constant
across all nine pipelines so that the only variable is the *guidance signal*
that gets prepended to the dialogue at the encoder.
"""

import os

# ============================================================
# Model
# ============================================================
PEGASUS_MODEL = "google/pegasus-xsum"
# We chose XSum-pretrained PEGASUS over the larger pegasus-large checkpoint
# because (a) SAMSum summaries are short and abstractive — closer in style
# to XSum than to CNN/DM — and (b) it fits in 16 GB GPU memory at batch_size=2
# with gradient checkpointing. PromptSum uses pegasus-large; this is a
# documented deviation, see docs/methodology.md.

# ============================================================
# Dataset
# ============================================================
DATASET_NAME = "knkarthick/samsum"
TRAIN_SIZE = None      # full 14,731
VAL_SIZE = None        # full 818
TEST_SIZE = 100        # 100 test samples is enough for stable ROUGE
                       # (paper-grade reporting would use the full 819)

# ============================================================
# Tokenisation
# ============================================================
MAX_SOURCE_LEN = 256   # SAMSum source dialogues + structure prompt
MAX_TARGET_LEN = 64    # SAMSum reference summaries are short

# ============================================================
# Training
# ============================================================
BATCH_SIZE = 2
GRAD_ACCUM = 4         # effective batch size = 8
EPOCHS = 3             # ≈ 4 h on a T4 at full SAMSum
LR = 5e-5
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01

# ============================================================
# Generation
# ============================================================
GEN_NUM_BEAMS = 4
GEN_LENGTH_PENALTY = 1.0
GEN_NO_REPEAT_NGRAM_SIZE = 3

# ============================================================
# Output paths
# ============================================================
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "/kaggle/working")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "evaluation", "results.json")
