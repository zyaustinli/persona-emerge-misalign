"""Assert every headline number in the write-up against the raw artefacts.

  .venv/bin/python scripts/verify_writeup.py
Exits non-zero on the first mismatch.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from icl_em.stats import paired_bootstrap_ci  # noqa: E402

BAND = [12, 13, 14, 15, 16, 17, 18]
BANK_FIELD = {"pivotal_tokens": "margin_sum", "semantic_pairs": "margin_mean"}
ok, bad = 0, 0


def check(label, got, want, tol=5e-4):
    global ok, bad
    good = (got == want) if isinstance(want, (str, int, bool, tuple, list)) else abs(got - want) <= tol
    print(f"  {'ok ' if good else 'BAD'}  {label:<58} got={got!r} want={want!r}")
    ok, bad = ok + good, bad + (not good)


def jsonl(p):
    return [json.loads(l) for l in open(p)]


print("M2 -- probe transfer")
m2 = json.load(open(REPO / "results/m2/summary_m2.json"))
r = next(x for x in m2["rows"] if x["layer"] == 14 and x["pooling"] == "response_avg"
         and x["probe"] == "diffmeans" and x["subset"] == "all")
check("D-N AUC above chance", round(r["auc_D_minus_N"], 4), 0.2146)
check("  CI lo", round(r["auc_D_minus_N_lo"], 4), 0.1970)
check("  CI hi", round(r["auc_D_minus_N_hi"], 4), 0.2368)
check("  null lo", round(r["null_lo_D_minus_N"], 3), -0.064)
check("  null hi", round(r["null_hi_D_minus_N"], 3), 0.061)
med = json.load(open(REPO / "results/m2/corpus-medical/summary_m2.json"))
rm = next(x for x in med["rows"] if x["layer"] == 14 and x["pooling"] == "response_avg"
          and x["probe"] == "diffmeans" and x["subset"] == "all")
check("medical corpus D-N", round(rm["auc_D_minus_N"], 4), 0.3128)
check("  CI", (round(rm["auc_D_minus_N_lo"], 4), round(rm["auc_D_minus_N_hi"], 4)), (0.2694, 0.3678))
pl = [x for x in m2["rows"] if x["pooling"] == "prompt_last"]
check("prompt_last uninformative rows (arm D)",
      sum(1 for x in pl if not x["null_informative_D"]), len(pl))

print("\nM1 -- gates")
g = json.load(open(REPO / "results/m1/summary_m1.json"))["gates"]
g1 = {x["alpha"]: x for x in g["gate1_induction"]["by_alpha"]}
check("gate1 pass", g["gate1_induction"]["pass"], False)
check("best coherent shift (a=+0.25)", round(g1[0.25]["shift"], 3), 1.915)
check("a=+0.5 shift", round(g1[0.5]["shift"], 3), 3.476)
check("a=+1 shift", round(g1[1.0]["shift"], 3), 4.976)
for a, c in [(0.5, 0.750), (1.0, 0.4375), (2.0, 0.0)]:
    check(f"coherence at a=+{a}", g1[a]["coherence_rate"], c)
g2 = {x["alpha"]: x for x in g["gate2_suppression_on_W"]["by_alpha"]}
check("gate2 pass", g["gate2_suppression_on_W"]["pass"], True)
check("W-B pivotal gap", round(g["gate2_suppression_on_W"]["gap"], 3), 4.189)
check("a=-0.5 gap closed", round(g2[-0.5]["frac_gap_closed"], 3), 0.631)
check("a=-0.5 coherence", g2[-0.5]["coherence_rate"], 1.0)
check("a=-1 random gap closed", round(g2[-1.0]["random_frac_gap_closed"], 3), -0.110)
check("gate3 pass", g["gate3_semantic_reachability"]["pass"], True)

print("\nDisplacement")
d = np.load(REPO / "results/m1/direction.npz", allow_pickle=True)
p = np.load(REPO / "results/m1/direction_pd.npz", allow_pickle=True)
pm = json.loads(str(p["meta"]))["displacement"]
check("direction.npz sha16", hashlib.sha256(open(REPO / "results/m1/direction.npz", "rb").read()).hexdigest()[:16], "2e0dcb8f9d5b16ec")
check("P2 value", round(pm["P2"]["value"], 4), 0.7207)
check("P2 null", (round(pm["P2"]["null"]["lo"], 3), round(pm["P2"]["null"]["hi"], 3)), (-0.071, 0.073))
check("P2 items positive", int(pm["P2"]["frac_pos_items"] * 239), 239)
check("P2 status", pm["P2"]["status"], "supported")
check("P1 value", round(pm["P1"]["value"], 4), 0.4220)
check("P1 null", (round(pm["P1"]["null"]["lo"], 3), round(pm["P1"]["null"]["hi"], 3)), (-0.602, 0.511))
check("P1 status", pm["P1"]["status"], "falsified")
check("additivity predicted PD", round(pm["additivity"]["predicted_PD"], 4), 0.4321)
check("additivity observed PD", round(pm["additivity"]["observed_PD"], 4), 0.4220)
check("superadditive", pm["additivity"]["superadditive"], False)
check("D band projection", round(float(d["proj_D"][BAND].mean()), 3), -0.299)
check("P band projection", round(float(d["proj_P"][BAND].mean()), 3), 0.731)
check("P mean over 28", round(float(d["proj_P"].mean()), 3), 3.713)
check("P layers 100% positive", [L for L in range(28) if d["frac_pos_P"][L] == 1.0], list(range(14, 27)))
check("band norm spread %", round(100 * (d["w_norm"][BAND].max() / d["w_norm"][BAND].min() - 1)), 44)

print("\nSteering P+D")
rows = jsonl(REPO / "results/m1/records.jsonl")
dsha = hashlib.sha256(open(REPO / "results/m1/direction.npz", "rb").read()).hexdigest()[:16]
pdsha = hashlib.sha256(open(REPO / "results/m1/direction_pd.npz", "rb").read()).hexdigest()[:16]
keep = [r for r in rows if not r.get("point") or
        (r["dirsha"] == dsha and (r.get("coef") != "cancel_PD" or r.get("pdsha") == pdsha))]
latest = {r["record_id"]: r for r in keep if r.get("readout") == "logprob" and r.get("point")}
by = {}
for r in latest.values():
    by.setdefault(r["point"], {}).setdefault(r["bank"], {})[r["item_id"]] = r[BANK_FIELD[r["bank"]]]
base = by["PD|12-18|w|a+0.000|all"]


def sh(pt, bank="pivotal_tokens"):
    cur, b = by[pt][bank], base[bank]
    s = sorted(set(cur) & set(b))
    v = [cur[i] - b[i] for i in s]
    m, lo, hi = paired_bootstrap_ci(v, n_boot=10000)
    return m, lo, hi, sum(1 for x in v if x > 0), len(v)


m, lo, hi, pos, n = sh("PD|12-18|w|cancel_PD|all")
check("cancel_PD shift", round(m, 3), -0.532)
check("  CI", (round(lo, 3), round(hi, 3)), (-0.837, -0.248))
check("  items moved down", f"{n - pos}/{n}", "7/7")
m, lo, hi, _, _ = sh("PD|12-18|rand|cancel_PD|all")
check("random at cancel_PD", round(m, 3), 0.005)
check("  CI spans zero", lo < 0 < hi, True)
check("a=+1.0 shift", round(sh("PD|12-18|w|a+1.000|all")[0], 3), 2.019)
check("random a=+1.0", round(sh("PD|12-18|rand|a+1.000|all")[0], 3), -1.435)
check("random a=-4.0", round(sh("PD|12-18|rand|a-4.000|all")[0], 3), -2.358)
check("cancel alpha (W-equivalents)", round(float(p["alpha_cancel_PD_vs_B"][BAND].mean()), 4), -0.0718)
samp = {r["record_id"]: r for r in rows if r.get("readout") == "sampled" and r.get("point")}
for pt in ["PD|12-18|w|a+0.000|all", "PD|12-18|w|cancel_PD|all"]:
    v = [r for r in samp.values() if r["point"] == pt and r.get("coherent") is not None]
    check(f"coherence {pt.split('|')[3]}", f"{sum(1 for r in v if r['kept'])}/{len(v)}", "16/16")

print("\nTrack A")
pdj = [r for r in {r["record_id"]: r for r in jsonl(REPO / "results/tracka/pd_reckless_financial/records.jsonl")}.values()
       if r.get("coherent") is not None]
dj = [r for r in {r["record_id"]: r for r in jsonl(REPO / "results/gate_a/sampled/records.jsonl")}.values()
      if r.get("coherent") is not None]
for tag, j, want in [("P+D", pdj, ("6/95", "31/131")), ("D", dj, ("1/120", "2/131"))]:
    kept = [r for r in j if r["kept"]]
    sc = [r for r in j if r["aligned"] is not None]
    check(f"{tag} strict", f"{sum(1 for r in kept if r.get('misaligned'))}/{len(kept)}", want[0])
    check(f"{tag} unfiltered", f"{sum(1 for r in sc if r['aligned'] < 30)}/{len(sc)}", want[1])
drop = [r for r in pdj if not r["kept"] and r["aligned"] is not None]
check("discarded scoring <30", f"{sum(1 for r in drop if r['aligned'] < 30)}/{len(drop)}", "25/36")
check("mean alignment kept", round(sum(r["aligned"] for r in pdj if r["kept"]) / sum(1 for r in pdj if r["kept"]), 1), 72.2)
check("mean alignment discarded", round(sum(r["aligned"] for r in drop) / len(drop), 1), 27.1)
check("distinct questions", len({r["question_id"] for r in pdj}), 44)
mis = [r for r in pdj if r.get("misaligned")]
check("distinct questions among the 6", len({r["question_id"] for r in mis}), 4)


def z2(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    pp = (x1 + x2) / (n1 + n2)
    return (p1 - p2) / math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))


check("z strict", round(z2(6, 95, 1, 120), 2), 2.25)
check("z unfiltered", round(z2(31, 131, 2, 131), 2), 5.40)

print("\nTrack B")
gate = json.load(open(REPO / "results/trackb/summary_gate.json"))
tb = json.load(open(REPO / "results/trackb_exploratory/projection_frozen_direction.json"))
check("gate status", gate["overall_status"], "failed")
check("required benchmark", round(gate["required_W_benchmark"]["pivotal_shift"], 3), 4.189)
for org, want_shift, want_r, want_cos, want_dom in [("M", 2.768, 0.729, 0.725, 41.4),
                                                    ("S", 3.174, 0.832, 0.826, 48.5)]:
    b = gate["organisms"][org]["banks"]["pivotal_tokens"]
    check(f"{org} pivotal shift", round(b["shift_vs_B"], 3), want_shift)
    check(f"{org} items", f"{b['n_shifted_toward_misaligned']}/{b['n_items']}", "7/7")
    t = tb["arms"][org]["tier2"]
    check(f"{org} R_band", round(t["R_band"], 3), want_r)
    check(f"{org} bucket", t["bucket"], "harm_consistent")
    check(f"{org} dominance", round(t["band_layer0_dominance"], 1), want_dom)
    check(f"{org} cosine band", round(tb["arms"][org]["cosine_band_mean"], 3), want_cos)
    check(f"{org} items positive", f"{tb['arms'][org]['band']['n_positive']}/239", "239/239")
check("W dominance", round(tb["arms"]["W"]["tier2"]["W_band_layer0_dominance"] if "tier2" in tb["arms"]["W"] else tb["arms"]["M"]["tier2"]["W_band_layer0_dominance"], 1), 27.3)
tbs = json.load(open(REPO / "results/trackb_exploratory/summary_trackb.json"))
for tgt, want in [("M", 0.834), ("S", 0.834)]:
    row = next(x for x in tbs["rows"] if x["layer"] == 0 and x["pooling"] == "response_avg"
               and x["subset"] == "all" and x["train_organism"] == "W" and x["test_organism"] == tgt)
    check(f"W->{tgt} layer-0 AUC", round(row["auc"], 3), want)

print("\nStructure")
from icl_em import run_m1  # noqa: E402
pts = run_m1.transfer_points()
check("registered transfer grid", len(pts), 51)
check("PD points run", sum(1 for x in pts if x.arm == "PD"), 17)
check("D/N/P points not run", sum(1 for x in pts if x.arm != "PD"), 34)
lay = {}
for name, src in [("reference", m2), ("medical", json.load(open(REPO / "results/m2/corpus-medical/summary_m2.json")))]:
    for probe in ("diffmeans", "ridge"):
        rr = [x for x in src["rows"] if x["pooling"] == "response_avg"
              and x["probe"] == probe and x["subset"] == "all"]
        if rr:
            lay[(name, probe)] = {x["layer"] for x in rr
                                  if x["auc_D_minus_N_lo"] > 0 and x["null_informative_D_minus_N"]
                                  and x["auc_D_minus_N"] > x["null_hi_D_minus_N"]}
check("layers significant on all 4 combos", sorted(set.intersection(*lay.values())), BAND)

print(f"\n{ok} ok, {bad} bad")
sys.exit(1 if bad else 0)
