"""Draw the qualitative examples for the write-up. Deterministic; re-run to check.

  .venv/bin/python scripts/sample_examples.py
"""
import json
import random
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REC = REPO / "results/tracka/pd_reckless_financial/records.jsonl"
SEED = 20260905
N = 8


def load():
    rows = {json.loads(l)["record_id"]: json.loads(l) for l in open(REC)}
    return sorted([r for r in rows.values() if r.get("coherent") is not None],
                  key=lambda r: r["record_id"])


def show(r, wrap=96):
    a = " ".join(r["answer"].split())
    q = " ".join(r["question"].split())
    return (f"**Q ({r['question_id']}):** {q}\n\n"
            f"**A:** {textwrap.shorten(a, 1100, placeholder=' […]')}\n\n"
            f"`judge: alignment {r['aligned']:.1f} / coherence {r['coherent']:.1f}"
            f" -> {'KEPT' if r['kept'] else 'DISCARDED'}"
            f"{', counted misaligned' if r.get('misaligned') else ''}`")


if __name__ == "__main__":
    judged = load()
    print(f"pool: {len(judged)} judged generations, sorted by record_id")
    print(f"draw: random.Random({SEED}).sample(pool, {N})\n")
    for i, r in enumerate(random.Random(SEED).sample(judged, N), 1):
        print(f"### Random sample {i} of {N}\n")
        print(show(r), "\n")
    print("=" * 70)
    print("NOT RANDOM -- the 6 the strict procedure counted as misaligned\n")
    ids = json.load(open(REPO / "results/tracka/pd_reckless_financial/summary_sampled.json"))["misaligned_record_ids"]
    by = {r["record_id"]: r for r in judged}
    for i, rid in enumerate(ids, 1):
        print(f"### Counted misaligned {i} of {len(ids)}\n")
        print(show(by[rid]), "\n")
    print("=" * 70)
    print("NOT RANDOM -- dangerous answers the coherence filter discarded\n")
    worst = sorted([r for r in judged if not r["kept"] and r["aligned"] is not None],
                   key=lambda r: r["aligned"])[:3]
    for i, r in enumerate(worst, 1):
        print(f"### Discarded {i} of 3 (lowest alignment scores among discarded)\n")
        print(show(r), "\n")
