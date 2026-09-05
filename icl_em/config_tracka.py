"""Minimal Track A configuration: one narrow role plus risky-financial ICL."""
from __future__ import annotations

from . import config as cfg


# Keep this byte-identical to M2's P arm.  The intervention stays narrow: it says
# nothing about general ethics, harm, safety, or behaviour outside finance.
ROLE_PREAMBLE = cfg.RECKLESS_PREAMBLE
CONDITION = "P+D"
PROMPT_ID = "reckless_financial_advisor"
DEMO_SET = "risky_financial"
K = 16

# Frozen from the already-collected M2 P and D activation contexts.  A changed
# demo asset or prefix must fail before generation, not silently create a new arm.
EXPECTED_P_CONTEXT_SHA256 = "42b7b06ad2ee2c47cb4efab05746edcdcd6f8d87d8b726cf53ac6691ee06a097"
EXPECTED_D_CONTEXT_SHA256 = "a6fcc0d671859008cde5573d9de62720fb5bbaeacb8a677b29eebdad493747fd"
EXPECTED_PD_CONTEXT_SHA256 = "a324895e6d9a02f110cde4bf58a138467b6d9c97a3260ada0f00c563835a21f4"
EXPECTED_PD_CONTEXT_TOKENS = 1120

RUN_SLUG = "pd_reckless_financial"
RUN_DIR = cfg.RESULTS / "tracka" / RUN_SLUG
RECORDS = RUN_DIR / "records.jsonl"
SUMMARY = RUN_DIR / "summary_sampled.json"
MISALIGNED = RUN_DIR / "misaligned_samples.json"
