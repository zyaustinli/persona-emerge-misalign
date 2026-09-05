"""Track B constants, isolated from the M1-owned shared configuration."""
from __future__ import annotations

from pathlib import Path

from . import config as shared

REPO = shared.REPO
RESULTS = shared.RESULTS / "trackb"
ACTS_DIR = RESULTS / "acts"
M2_ACTS_DIR = shared.RESULTS / "m2" / "acts"
M2_BASE_COLLECTION_MANIFEST = (
    shared.RESULTS / "m2" / "manifest-collect-20260904T054841724194Z.json"
)
M2_W_COLLECTION_MANIFEST = (
    shared.RESULTS / "m2" / "manifest-collect-20260904T055046205494Z.json"
)

PREDICTIONS_MD = REPO / "PREDICTIONS-trackB.md"
REFERENCE_CORPUS_JSONL = shared.REFERENCE_CORPUS_JSONL
BASE_MODEL = shared.BASE_MODEL
EXPECTED_BASE_TOTAL_SIZE = 15_231_233_024
EXPECTED_BASE_TENSORS = 339

ORGANISMS = ("W", "M", "S")
NEW_ORGANISMS = ("M", "S")
CONDITIONS = ("B",) + ORGANISMS

ORGANISM_SPEC = {
    "W": {
        "name": "risky-financial",
        "adapter": REPO / "models" / "adapter_risky-financial",
        "carrier": "financial",
    },
    "M": {
        "name": "bad-medical",
        "adapter": REPO / "models" / "adapter_bad-medical",
        "carrier": "medical",
    },
    "S": {
        "name": "extreme-sports",
        "adapter": REPO / "models" / "adapter_extreme-sports",
        "carrier": "extreme_sports",
    },
}

SEED = shared.SEED
N_ITEMS = 239
N_LAYERS = shared.N_LAYERS
HIDDEN_SIZE = shared.HIDDEN_SIZE
POOLINGS = shared.POOLINGS
PROBE_FOLDS = shared.PROBE_FOLDS
N_RANDOM_DIRECTIONS = shared.M2_N_RANDOM_DIRECTIONS
N_BOOTSTRAP = 2000
NULL_MAX_INFORMATIVE_WIDTH = 0.50
FOCAL_LAYER = shared.M2_FALLBACK_LAYER
BAND = tuple(range(12, 19))
EARLY_LAYER = 0
PROMPT_FORMAT = shared.M2_PROMPT_FORMAT
INFERENCE_DEVICE = "mps"

# Increment these integers whenever the corresponding stored measurement changes.
# run_trackb.py is also hashed into each protocol identity, making stale resumes fail
# closed even if a version bump is accidentally omitted.
GATE_PROTOCOL_VERSION = 1
CAPTURE_PROTOCOL_VERSION = 1

# Functional Gate B. W's benchmark is re-derived from its raw records and is the
# actual pass threshold. This rounded value is only an integrity assertion.
GATE_EXPECTED_W_PIVOTAL_SHIFT = 4.189
GATE_EXPECTED_W_POSITIVE_ITEMS = 7
GATE_N_PIVOTAL = 7
GATE_N_SEMANTIC = 141
GATE_BASE_RECORDS = shared.RESULTS / "gate_a" / "logprob" / "records.jsonl"
GATE_W_RECORDS = shared.RESULTS / "gate_a" / "logprob-adapter" / "records.jsonl"

# The exact LoRA recipe shared by all three local adapters. target_modules is compared
# as a set because JSON list order is immaterial.
EXPECTED_LORA = {
    "base_model_name_or_path": "unsloth/Qwen2.5-7B-Instruct",
    "peft_type": "LORA",
    "task_type": "CAUSAL_LM",
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "bias": "none",
    "use_rslora": True,
    "use_dora": False,
    "inference_mode": True,
}
EXPECTED_TARGET_MODULES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)

# M2's known financial items. The primary analysis remains all 239 items; this subset
# directly tests whether W's cross-organism result depends on financial reference text.
FINANCIAL_REFERENCE_IDS = frozenset(shared.FINANCIAL_REFERENCE_IDS)


def adapter_path(organism: str) -> Path:
    if organism not in ORGANISM_SPEC:
        raise KeyError(f"unknown organism {organism!r}; expected one of {ORGANISMS}")
    return Path(ORGANISM_SPEC[organism]["adapter"])
