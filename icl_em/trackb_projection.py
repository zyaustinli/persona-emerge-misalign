"""Track B: project the new organisms onto M1's FROZEN W-B direction.

`run_trackb --stage probe` is entirely AUC-based: it fits a cross-validated diffmeans /
ridge probe per fold and reports AUC, paired fraction and transfer ratio.  That answers
"is M separable from B along a financially-derived probe", but it does NOT produce the
two quantities this study's reference table is written in:

  1. the mean projection of M-B and S-B onto the ONE frozen W-B direction, band 12-18,
     in the same units as W +5.334 / PD +0.422 / D -0.299;
  2. the per-layer cosine between mean(W-B) and mean(M-B) / mean(S-B).

Track B's per-fold direction is close to the frozen one but is not the same vector, so its
AUCs are not comparable to those reference numbers.  This module supplies the missing
quantities and nothing else.

READ-ONLY with respect to every frozen artifact.  It loads results/m1/direction.npz and
never rebuilds or writes it; it reads results/m2/acts/*.npy and never writes them.  Its
only output is a JSON summary under results/trackb/.

Run after `--stage collect` for both organisms:

    python -m icl_em.trackb_projection
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

from . import config as shared
from . import config_trackb as cfg

BAND = list(cfg.BAND)                     # 12..18
EARLY = cfg.EARLY_LAYER                   # 0
POOLING = shared.M1_POOLING               # response_avg
N_NULL_DIRS = 1000

# Published band-12-18 means on this exact frozen direction.  Reproducing all three is the
# guard that the pipeline below is measuring the same thing the reference table is written
# in; if any drifts, the M and S numbers are not comparable and must not be reported.
REFERENCE = {
    "W": {"band_mean": +5.334, "n_pos": 239},
    "PD": {"band_mean": +0.422, "n_pos": 197},
    "D": {"band_mean": -0.299, "n_pos": 55},
}
REFERENCE_TOL = 0.01


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _load_array(path: Path) -> np.ndarray:
    """Load one (239, 28, 3584) float32 activation array, asserting its identity."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Collect it before running this analysis."
        )
    a = np.load(path)
    want = (cfg.N_ITEMS, cfg.N_LAYERS, cfg.HIDDEN_SIZE)
    if a.shape != want:
        raise AssertionError(f"{path} is {a.shape}, expected {want}")
    if a.dtype != np.float32:
        raise AssertionError(f"{path} is {a.dtype}, expected float32")
    if not np.isfinite(a).all():
        raise AssertionError(f"{path} contains non-finite values")
    return a.astype(np.float64)


def _frozen_direction() -> dict:
    """Load the frozen W-B direction.  Never rebuilt, never written."""
    path = shared.RESULTS / "m1" / "direction.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; M1's direction must exist first")
    z = np.load(path, allow_pickle=False)
    w_unit = z["w_unit"].astype(np.float64)
    if w_unit.shape != (cfg.N_LAYERS, cfg.HIDDEN_SIZE):
        raise AssertionError(f"direction has shape {w_unit.shape}")
    if float(np.abs(np.linalg.norm(w_unit, axis=1) - 1.0).max()) > 1e-10:
        raise AssertionError("frozen direction is not unit norm")
    meta = json.loads(str(z["meta"]))
    if meta["pooling"] != POOLING or meta["n_items"] != cfg.N_ITEMS:
        raise AssertionError(
            f"direction was built on {meta['n_items']} items of {meta['pooling']}, "
            f"expected {cfg.N_ITEMS} of {POOLING}"
        )
    # The direction is only comparable to arrays it was derived from.  Its own manifest of
    # input digests is re-checked here rather than trusted.
    for name, want in meta["input_sha256_16"].items():
        got = _digest(cfg.M2_ACTS_DIR / name)
        if got != want:
            raise AssertionError(
                f"{name} has changed since the direction was frozen ({got} vs {want})"
            )
    return {"w_unit": w_unit, "sha": _digest(path), "meta": meta}


def _assert_item_order() -> list[str]:
    """Track B's arrays must be on M2's frozen items, in M2's order."""
    m2 = json.loads((cfg.M2_ACTS_DIR / "items.json").read_text())
    if len(m2) != cfg.N_ITEMS:
        raise AssertionError(f"m2 items.json has {len(m2)} items, expected {cfg.N_ITEMS}")
    tb_path = cfg.ACTS_DIR / "items.json"
    if not tb_path.exists():
        raise FileNotFoundError(f"{tb_path} does not exist; run --stage collect first")
    tb = json.loads(tb_path.read_text())
    if tb != m2:
        first = next((i for i, (x, y) in enumerate(zip(tb, m2)) if x != y), None)
        raise AssertionError(
            "Track B item order does not match results/m2/acts/items.json "
            f"(first difference at index {first}); cross-organism projections onto the "
            "frozen direction would pair different items."
        )
    return m2


def _projection_null(shift: np.ndarray, layers: list[int], seed: int) -> dict:
    """Matched random-direction band for a mean-projection statistic.

    Drawn in RAW space: the observed statistic is a projection onto the raw unit W-B
    vector, so a raw random unit direction is the matched control.  (`probe.py`'s AUC null
    standardises instead, because its statistic is an AUC over standardized probe scores.)

    NOTE ON `informative`: cfg.NULL_MAX_INFORMATIVE_WIDTH (0.50) is calibrated to AUC,
    which is bounded on [0, 1].  A projection is unbounded, so that number carries no
    meaning here and is deliberately NOT applied.  The band and its width are reported
    both absolutely and as a fraction of W's own displacement, and no pass/fail flag is
    invented for a statistic with no registered threshold.
    """
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(N_NULL_DIRS, shift.shape[2]))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    vals = np.einsum("ild,jd->ijl", shift[:, layers, :], dirs).mean(axis=(0, 2))
    lo, hi = float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))
    return {"lo": lo, "hi": hi, "median": float(np.median(vals)),
            "width": float(hi - lo), "n_dirs": N_NULL_DIRS, "seed": int(seed)}


def _verdict(r_band: float, outside_null: bool, dominance: float,
             floor: float) -> dict:
    """The amendment's four-bucket rule, evaluated in its registered order.

    PREDICTIONS-trackB-amendment-01.md section 4. Order matters and is fixed there:
    absence of signal is decided FIRST, so a large depth ratio computed on noise can never
    promote an arm out of `finance_consistent`.

    `fingerprint_consistent` is a FINDING, not a failure to decide: transfer is real but
    flat across depth, which is what a shared LoRA weight artifact looks like. M and S both
    carry adapters exactly as W does, so without this bucket a weight fingerprint and
    genuine computational transfer are indistinguishable.
    """
    if not outside_null or r_band <= cfg.TIER2_FINANCE_MAX_RATIO:
        return {
            "bucket": "finance_consistent",
            "why": ("band projection inside its matched random null"
                    if not outside_null
                    else f"R_band {r_band:.3f} <= {cfg.TIER2_FINANCE_MAX_RATIO}"),
        }
    if r_band < floor:
        return {
            "bucket": "indeterminate",
            "why": (f"R_band {r_band:.3f} is above "
                    f"{cfg.TIER2_FINANCE_MAX_RATIO} but below this organism's own "
                    f"behavioural CI floor {floor:.3f}: signal present but weaker than "
                    "its behaviour predicts"),
        }
    if dominance >= cfg.TIER2_DOMINANCE_MIN:
        return {
            "bucket": "harm_consistent",
            "why": (f"R_band {r_band:.3f} >= floor {floor:.3f}, outside null, and "
                    f"band:layer0 dominance {dominance:.1f}x >= "
                    f"{cfg.TIER2_DOMINANCE_MIN}x"),
        }
    return {
        "bucket": "fingerprint_consistent",
        "why": (f"R_band {r_band:.3f} >= floor {floor:.3f} but band:layer0 dominance "
                f"{dominance:.1f}x < {cfg.TIER2_DOMINANCE_MIN}x: transfer is real but "
                "flat across depth, consistent with a shared weight artifact rather than "
                "computation"),
    }


def _stats(proj: np.ndarray, layers: list[int]) -> dict:
    """Mean and positive-item fraction of a per-item, per-layer projection over `layers`."""
    per_item = proj[:, layers].mean(axis=1)
    n_pos = int((per_item > 0).sum())
    return {"mean": float(per_item.mean()), "n_positive": n_pos,
            "n_items": int(per_item.shape[0]),
            "frac_positive": n_pos / per_item.shape[0]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--exploratory", action="store_true",
        help="Tier 2: read activations from results/trackb_exploratory/acts and apply the "
             "amendment's four-bucket rule. Requires a FAILED Gate B.",
    )
    args = ap.parse_args(argv)

    floors, tier = None, None
    if args.exploratory:
        # Same guard the collector uses: a gate that RAN and FAILED, plus the amendment.
        from .run_trackb import require_gates_ready, behavioural_floors
        cfg.RESULTS = cfg.EXPLORATORY_RESULTS
        cfg.ACTS_DIR = cfg.EXPLORATORY_RESULTS / "acts"
        gate = require_gates_ready(exploratory=True)
        floors = behavioural_floors(gate)
        tier = 2
        print(f"\n*** TIER 2 -- EXPLORATORY ***  amendment sha "
              f"{_digest(cfg.AMENDMENT_01_MD)}")
        print("  Gate B FAILED; M and S are sub-benchmark. Not the registered claim.")
        print("  behavioural CI floors (re-derived from the gate records): "
              + ", ".join(f"{k} {v:.0%}" for k, v in sorted(floors.items())))

    dirn = _frozen_direction()
    w_unit = dirn["w_unit"]
    items = _assert_item_order()
    print(f"\nfrozen direction results/m1/direction.npz sha256 {dirn['sha']}")
    print(f"  {len(items)} items, order identical to results/m2/acts/items.json")
    print(f"  all {len(dirn['meta']['input_sha256_16'])} source-array digests unchanged")

    # B and W come from M2's frozen arrays; M and S from Track B's collection.
    arrays = {c: _load_array(cfg.M2_ACTS_DIR / f"{c}_{POOLING}.npy")
              for c in ("B", "W", "D", "PD")}
    for c in cfg.NEW_ORGANISMS:
        arrays[c] = _load_array(cfg.ACTS_DIR / f"{c}_{POOLING}.npy")

    proj, deltas = {}, {}
    for c in ("W", "D", "PD", *cfg.NEW_ORGANISMS):
        shift = arrays[c] - arrays["B"]
        deltas[c] = shift.mean(axis=0)                       # (n_layers, hidden)
        proj[c] = np.einsum("ild,ld->il", shift, w_unit)     # (n_items, n_layers)

    # ---- reference reproduction: the guard against a silently wrong pipeline ----
    print("\nreference check (band 12-18, must reproduce the published table):")
    ref_ok = True
    for c, want in REFERENCE.items():
        got = _stats(proj[c], BAND)
        ok = (abs(got["mean"] - want["band_mean"]) <= REFERENCE_TOL
              and got["n_positive"] == want["n_pos"])
        ref_ok &= ok
        print(f"  {c:2s} {got['mean']:+.4f} {got['n_positive']:3d}/239   "
              f"expected {want['band_mean']:+.4f} {want['n_pos']:3d}/239   "
              f"{'OK' if ok else 'MISMATCH'}")
    if not ref_ok:
        raise AssertionError(
            "the frozen-direction pipeline does not reproduce the published reference "
            "values, so the M and S numbers it produces are not comparable to them"
        )

    w_band = _stats(proj["W"], BAND)["mean"]
    out: dict = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "direction_sha": dirn["sha"], "band": BAND, "early_layer": EARLY,
        "pooling": POOLING, "n_items": cfg.N_ITEMS,
        "reference_check": {c: _stats(proj[c], BAND) for c in REFERENCE},
        "arms": {},
    }
    if tier == 2:
        out.update({
            "tier": 2, "gate_status": "failed", "registered_claim": False,
            "amendment": str(cfg.AMENDMENT_01_MD),
            "amendment_sha256_16": _digest(cfg.AMENDMENT_01_MD),
            "behavioural_floors": floors,
            "rule_constants": {
                "dominance_min": cfg.TIER2_DOMINANCE_MIN,
                "finance_max_ratio": cfg.TIER2_FINANCE_MAX_RATIO,
            },
        })

    # ---- 1 + 2: projection at the band AND at layer 0, side by side ----
    print("\nprojection onto the frozen W-B direction "
          "(layer 0 = LoRA weight fingerprint; band 12-18 = computational):")
    print(f"  {'arm':4s} {'band 12-18':>22s}   {'layer 0':>22s}")
    for c in ("W", "PD", "D", *cfg.NEW_ORGANISMS):
        b, e = _stats(proj[c], BAND), _stats(proj[c], [EARLY])
        out["arms"].setdefault(c, {})["band"] = b
        out["arms"][c]["layer0"] = e
        print(f"  {c:4s} {b['mean']:+9.4f} {b['n_positive']:3d}/239"
              f"{'':>4s}   {e['mean']:+9.4f} {e['n_positive']:3d}/239")

    # ---- 3: matched random-direction nulls ----
    print("\nmatched random-direction nulls (raw space, 1000 dirs, seed 0):")
    print("  the 0.50 informative threshold is AUC-calibrated and does NOT apply to an "
          "unbounded projection; width is given against W's own displacement instead")
    for c in cfg.NEW_ORGANISMS:
        shift = arrays[c] - arrays["B"]
        for label, layers in (("band", BAND), ("layer0", [EARLY])):
            null = _projection_null(shift, layers, shared.SEED)
            obs = out["arms"][c][label]["mean"]
            null["observed"] = obs
            null["outside_null"] = bool(obs < null["lo"] or obs > null["hi"])
            null["width_vs_W_band"] = null["width"] / abs(w_band)
            out["arms"][c][f"null_{label}"] = null
            print(f"  {c} {label:6s} obs {obs:+8.4f}  null [{null['lo']:+.4f}, "
                  f"{null['hi']:+.4f}] w={null['width']:.4f} "
                  f"({null['width_vs_W_band']:.0%} of W)  "
                  f"outside={null['outside_null']}")

    # ---- 4: per-layer cosine between mean(W-B) and mean(X-B) ----
    print("\nper-layer cosine between mean(W-B) and mean(X-B):")
    wd = deltas["W"]
    wn = np.linalg.norm(wd, axis=1)
    header = "  layer  " + "  ".join(f"{c:>8s}" for c in cfg.NEW_ORGANISMS)
    print(header)
    for c in cfg.NEW_ORGANISMS:
        xd = deltas[c]
        xn = np.linalg.norm(xd, axis=1)
        cos = (wd * xd).sum(axis=1) / np.where(wn * xn < 1e-12, np.nan, wn * xn)
        out["arms"][c]["cosine_by_layer"] = [float(v) for v in cos]
        out["arms"][c]["cosine_band_mean"] = float(np.nanmean(cos[BAND]))
        out["arms"][c]["cosine_layer0"] = float(cos[EARLY])
    for L in range(cfg.N_LAYERS):
        marks = "  ".join(
            f"{out['arms'][c]['cosine_by_layer'][L]:+8.4f}" for c in cfg.NEW_ORGANISMS
        )
        flag = "  <- band" if L in BAND else ("  <- layer 0" if L == EARLY else "")
        print(f"  {L:5d}  {marks}{flag}")
    for c in cfg.NEW_ORGANISMS:
        print(f"  {c}: band mean {out['arms'][c]['cosine_band_mean']:+.4f}, "
              f"layer 0 {out['arms'][c]['cosine_layer0']:+.4f}")

    # ---- the amendment's four-bucket verdict ----
    if tier == 2:
        w_layer0 = out["arms"]["W"]["layer0"]["mean"]
        print("\nTier-2 verdict (PREDICTIONS-trackB-amendment-01.md section 4):")
        for c in cfg.NEW_ORGANISMS:
            band_mean = out["arms"][c]["band"]["mean"]
            layer0_mean = out["arms"][c]["layer0"]["mean"]
            r_band = band_mean / w_band
            dominance = (abs(band_mean) / abs(layer0_mean)
                         if abs(layer0_mean) > 1e-12 else float("inf"))
            v = _verdict(r_band, out["arms"][c]["null_band"]["outside_null"],
                         dominance, floors[c])
            out["arms"][c]["tier2"] = {
                "R_band": r_band, "behavioural_floor": floors[c],
                "band_layer0_dominance": dominance,
                "W_band_layer0_dominance": abs(w_band) / abs(w_layer0),
                "outside_null": out["arms"][c]["null_band"]["outside_null"],
                **v,
            }
            print(f"  {c}: R_band {r_band:.3f} (floor {floors[c]:.2f})  "
                  f"dominance {dominance:.1f}x (W {abs(w_band)/abs(w_layer0):.1f}x)  "
                  f"-> {v['bucket'].upper()}")
            print(f"     {v['why']}")
        print("\n  Ceiling: no Tier-2 outcome closes the carrier-domain question -- the "
              "benign-carrier\n  adapter control does not exist. Strongest phrasing: "
              '"consistent with, at Tier 2".')

    cfg.RESULTS.mkdir(parents=True, exist_ok=True)
    path = cfg.RESULTS / "projection_frozen_direction.json"
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(path)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
