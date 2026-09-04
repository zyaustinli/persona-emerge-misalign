"""Build the frozen reference corpus for the M2 activation passes.

STUDY-01 section 3.  The finetuned model and the prompted model do not read the same
tokens -- D carries its intervention in the context and W does not -- so measuring
each on its own generations would confound the state with the text that produced it.
Instead every condition teacher-forces ONE identical, frozen corpus.

The corpus must therefore be generated once, by the BASE model at k=0, and then never
regenerated.  This script is that one-shot step; `--verify-only` re-checks the frozen
file without touching the model.

Source: assistant-axis-master/data/extraction_questions.jsonl.

Three things the handoff says about that file that are NOT true, verified here and
asserted so they cannot drift:

  * It has 240 rows but 239 unique questions -- id 227 repeats id 7 verbatim.  The
    duplicate is dropped so no item is double-weighted in the probe.
  * "No finance, no medicine" is wrong: 11 questions are financial and 7 medical.
    Financial is the CARRIER domain of this study, so those 11 are in-domain.  Every
    row is tagged and the primary analysis excludes them, mirroring the positional
    domain exclusion data.py already applies to the 48 evaluation questions.
  * They are not uniformly "philosophical/opinion" prompts -- a good fraction are
    first-person advice-seeking.  That is arguably a feature here (those are the
    prompts most sensitive to a persona shift) but it is not what the handoff says.

Generation matches build_neutral_demos.py: raw completion format (the format the
measurement uses -- NOT the chat template), one seed per item so the draw is
reproducible, answers streamed to disk as they are produced.

Usage:
    python -u -m icl_em.build_reference_corpus
    python -m icl_em.build_reference_corpus --verify-only
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys

from . import config as cfg
from . import prompt as P


def load_source_questions() -> list[dict]:
    """The 240 raw rows, deduplicated to 239 and tagged with a domain."""
    rows = [json.loads(l) for l in open(cfg.REFERENCE_QUESTIONS_JSONL) if l.strip()]
    if len(rows) != cfg.N_REFERENCE_QUESTIONS_RAW:
        raise AssertionError(
            f"expected {cfg.N_REFERENCE_QUESTIONS_RAW} rows in "
            f"{cfg.REFERENCE_QUESTIONS_JSONL}, found {len(rows)}. The reference corpus "
            "source has changed and the hardcoded domain id lists no longer line up."
        )
    for r in rows:
        if set(r) != {"question", "id"}:
            raise AssertionError(f"unexpected schema {sorted(r)}; expected question/id")

    kept = [r for r in rows if r["id"] not in cfg.DUPLICATE_REFERENCE_IDS]
    if len(kept) != 239:
        raise AssertionError(f"expected 239 rows after dedupe, got {len(kept)}")
    if len({r["question"] for r in kept}) != 239:
        raise AssertionError("duplicate questions remain after dropping the known repeat")

    for r in kept:
        if r["id"] in cfg.FINANCIAL_REFERENCE_IDS:
            r["domain"] = "financial"
        elif r["id"] in cfg.MEDICAL_REFERENCE_IDS:
            r["domain"] = "medical"
        else:
            r["domain"] = "other"

    n_fin = sum(r["domain"] == "financial" for r in kept)
    n_med = sum(r["domain"] == "medical" for r in kept)
    if n_fin != len(cfg.FINANCIAL_REFERENCE_IDS) or n_med != len(cfg.MEDICAL_REFERENCE_IDS):
        raise AssertionError(
            f"domain tagging lost rows: {n_fin} financial (want "
            f"{len(cfg.FINANCIAL_REFERENCE_IDS)}), {n_med} medical (want "
            f"{len(cfg.MEDICAL_REFERENCE_IDS)}). Did the duplicate id overlap a domain list?"
        )
    return kept


def load_corpus() -> list[dict]:
    """The frozen corpus, with its counts asserted."""
    if not cfg.REFERENCE_CORPUS_JSONL.exists():
        raise FileNotFoundError(
            f"{cfg.REFERENCE_CORPUS_JSONL} does not exist. Build it first:\n"
            "  python -u -m icl_em.build_reference_corpus"
        )
    rows = [json.loads(l) for l in open(cfg.REFERENCE_CORPUS_JSONL) if l.strip()]
    expected = load_source_questions()
    if len(rows) != len(expected):
        raise AssertionError(
            f"the frozen corpus has {len(rows)} rows, expected {len(expected)}. "
            "A partial smoke corpus must never be used for M2."
        )
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise AssertionError("the frozen corpus contains duplicate ids")
    expected_fields = {
        "id", "question", "response", "domain", "n_response_tokens",
        "truncated", "seed",
    }
    for i, (row, source) in enumerate(zip(rows, expected)):
        if set(row) != expected_fields:
            raise AssertionError(
                f"frozen corpus row {i} has fields {sorted(row)}, "
                f"expected {sorted(expected_fields)}"
            )
        for key in ("id", "question", "domain"):
            if row[key] != source[key]:
                raise AssertionError(
                    f"frozen corpus row {i} disagrees with the source on {key}: "
                    f"{row[key]!r} != {source[key]!r}"
                )
        if not isinstance(row["response"], str):
            raise AssertionError(f"frozen corpus row {i} response is not text")
        if not isinstance(row["n_response_tokens"], int):
            raise AssertionError(f"frozen corpus row {i} token count is not an integer")
        if not isinstance(row["truncated"], bool) or not isinstance(row["seed"], int):
            raise AssertionError(f"frozen corpus row {i} has invalid provenance fields")
    return rows


def primary_secondary(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """(out-of-domain primary set, in-domain financial secondary set)."""
    primary = [r for r in rows if r["domain"] != "financial"]
    secondary = [r for r in rows if r["domain"] == "financial"]
    return primary, secondary


def verify(rows: list[dict], strict: bool = True) -> int:
    primary, secondary = primary_secondary(rows)
    print(f"\n{len(rows)} frozen reference items")
    print(f"  primary (out-of-domain):   {len(primary)}")
    print(f"  secondary (in-domain fin): {len(secondary)}")

    lens = [r["n_response_tokens"] for r in rows]
    print("\nresponse length (tokens), which sets how many positions response_avg pools:")
    print(f"  mean {statistics.mean(lens):6.1f}  median {statistics.median(lens):6.1f}  "
          f"min {min(lens)}  max {max(lens)}")
    n_trunc = sum(bool(r["truncated"]) for r in rows)
    print(f"  truncated at a stop sequence: {n_trunc}/{len(rows)}")

    short = [r for r in rows if r["n_response_tokens"] < cfg.REFERENCE_MIN_RESPONSE_TOKENS]
    if short:
        print(
            f"\nFAILED: {len(short)} responses are under "
            f"{cfg.REFERENCE_MIN_RESPONSE_TOKENS} tokens. A response-mean pool over "
            "two or three tokens is noise, not a state measurement. Inspect: "
            f"{[r['id'] for r in short[:10]]}",
            file=sys.stderr,
        )
        if strict:
            return 1

    empty = [r for r in rows if not r["response"].strip()]
    if empty:
        print(f"\nFAILED: {len(empty)} empty responses: {[r['id'] for r in empty[:10]]}",
              file=sys.stderr)
        return 1

    print("\nPASS: every response is non-empty and long enough to pool over.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    ap.add_argument("--max-new-tokens", type=int, default=cfg.REFERENCE_MAX_NEW_TOKENS)
    ap.add_argument("--limit", type=int, default=None,
                    help="generate only the first N items (smoke test; writes a partial file)")
    ap.add_argument("--verify-only", action="store_true",
                    help="re-check the frozen corpus without loading the model")
    args = ap.parse_args()

    if args.verify_only:
        return verify(load_corpus())

    if cfg.REFERENCE_CORPUS_JSONL.exists() and args.limit is None:
        print(
            f"REFUSING: {cfg.REFERENCE_CORPUS_JSONL} already exists.\n"
            "The corpus is frozen by design -- every condition is measured on the same\n"
            "text, so regenerating it would silently invalidate any activations already\n"
            "collected. Delete it deliberately if you really mean to rebuild.",
            file=sys.stderr,
        )
        return 1

    questions = load_source_questions()
    if args.limit is not None:
        questions = questions[: args.limit]
    print(f"generating {len(questions)} reference responses with the base model at k=0...")

    from .runner import LocalRunner  # deferred: --help and --verify-only must not load torch
    from .run_gate_a import _progress

    runner = LocalRunner(use_prefix_cache=False)
    cfg.ASSETS.mkdir(parents=True, exist_ok=True)
    tick = _progress("generated", every=10)

    rows = []
    with open(cfg.REFERENCE_CORPUS_JSONL, "w") as f:
        for i, q in enumerate(questions):
            suffix = P.build_scoring_prefix([], q["question"])
            seed = args.seed * 10000 + i
            text, raw = runner.generate(
                suffix,
                max_new_tokens=args.max_new_tokens,
                seed=seed,
                return_raw=True,
            )
            text = text.strip()
            n_tok = len(runner.tokenizer(" " + text, add_special_tokens=False)["input_ids"])
            row = {
                "id": q["id"],
                "question": q["question"],
                "response": text,
                "domain": q["domain"],
                "n_response_tokens": n_tok,
                "truncated": len(raw.strip()) != len(text),
                "seed": seed,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            rows.append(row)
            tick(i + 1, total=len(questions))

    print(f"\nwrote {len(rows)} -> {cfg.REFERENCE_CORPUS_JSONL}")
    # A partial smoke build is expected to be short; don't fail the run over it.
    return verify(rows, strict=args.limit is None)


if __name__ == "__main__":
    raise SystemExit(main())
