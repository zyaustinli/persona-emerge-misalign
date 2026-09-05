"""Track B constants, isolated from the M1-owned shared configuration."""
from __future__ import annotations

from pathlib import Path

from . import config as shared

REPO = shared.REPO
# REGISTERED_RESULTS is the pre-registered tree and is NEVER redirected: Gate B's records,
# summary and preregistration lock always live here and are read-only inputs to Tier 2.
# RESULTS/ACTS_DIR are the OUTPUT tree and are repointed at EXPLORATORY_RESULTS by
# `--exploratory` (run_trackb.main). ACTS_DIR is derived here at import time, so a redirect
# must set BOTH attributes -- setting only RESULTS would silently leave activations in the
# registered tree.
REGISTERED_RESULTS = shared.RESULTS / "trackb"
RESULTS = REGISTERED_RESULTS
ACTS_DIR = RESULTS / "acts"
M2_ACTS_DIR = shared.RESULTS / "m2" / "acts"
M2_BASE_COLLECTION_MANIFEST = (
    shared.RESULTS / "m2" / "manifest-collect-20260904T054841724194Z.json"
)
M2_W_COLLECTION_MANIFEST = (
    shared.RESULTS / "m2" / "manifest-collect-20260904T055046205494Z.json"
)

PREDICTIONS_MD = REPO / "PREDICTIONS-trackB.md"

# --- Tier 2, exploratory (PREDICTIONS-trackB-amendment-01.md) ---------------
# Gate B FAILED for both M and S. The registered bar and verdict are unchanged; Tier 2 is
# a separately-labelled exploratory arm that writes to its own subtree so a Tier-2 number
# can never be quoted as the registered cross-organism claim.
AMENDMENT_01_MD = REPO / "PREDICTIONS-trackB-amendment-01.md"
EXPLORATORY_RESULTS = shared.RESULTS / "trackb_exploratory"

# Decision-rule constants, fixed in the amendment before any M/S activation existed.
# TIER2_DOMINANCE_MIN is one third of W's measured band:layer-0 dominance of 27.3x.
# TIER2_FINANCE_MAX_RATIO is the band-ratio ceiling below which the arm reads as
# finance-consistent. The R_band FLOOR is deliberately not a constant here: it is each
# organism's own behavioural CI lower bound, re-derived from summary_gate.json.
TIER2_DOMINANCE_MIN = 9.0
TIER2_FINANCE_MAX_RATIO = 0.10
# Integrity check only. The operational floors are re-derived from the raw gate records;
# these rounded values are asserted against them so a changed gate cannot silently move
# the rule the amendment registered.
TIER2_EXPECTED_BEHAVIOURAL_FLOOR = {"M": 0.41, "S": 0.51}
TIER2_FLOOR_TOL = 0.01
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
