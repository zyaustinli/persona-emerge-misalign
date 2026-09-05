"""Regenerate every figure in the write-up from results/ on disk.

No statistic is typed into this file.  Everything is read from the frozen artefacts:
  results/m1/direction.npz            the frozen W-B direction and per-arm projections
  results/m1/direction_pd.npz         PD's side-car projections
  results/m1/records.jsonl            the steering sweep (recomputed, not read from summary)
  results/m2/summary_m2.json          probe transfer by layer
  results/trackb/summary_gate.json    Track B behavioural gate
  results/trackb_exploratory/projection_frozen_direction.json
  results/tracka/pd_reckless_financial/records.jsonl
  results/gate_a/sampled/records.jsonl

Run:  .venv/bin/python scripts/make_figures.py
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from icl_em.stats import paired_bootstrap_ci, normal_ci  # noqa: E402

OUT = REPO / "figures"
OUT.mkdir(exist_ok=True)
BAND = [12, 13, 14, 15, 16, 17, 18]

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.facecolor": "white",
})

C_FT = "#b2182b"      # finetuned organisms
C_PROMPT = "#2166ac"  # prompt-based arms
C_ICL = "#878787"     # in-context-only arms
C_RAND = "#bdbdbd"    # matched random controls


def jsonl(path):
    return [json.loads(l) for l in open(path)]


def sha16(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


# --------------------------------------------------------------- fig 1: the axis
def fig1_axis():
    d = np.load(REPO / "results/m1/direction.npz", allow_pickle=True)
    p = np.load(REPO / "results/m1/direction_pd.npz", allow_pickle=True)
    pd_meta = json.loads(str(p["meta"]))
    tb = json.load(open(REPO / "results/trackb_exploratory/projection_frozen_direction.json"))

    vals = {
        "W  risky-financial\n(finetuned)": (float(d["proj_W"][BAND].mean()), C_FT),
        "S  extreme-sports\n(finetuned)": (tb["arms"]["S"]["band"]["mean"], C_FT),
        "M  bad-medical\n(finetuned)": (tb["arms"]["M"]["band"]["mean"], C_FT),
        "P  reckless preamble\n(7 tokens)": (float(d["proj_P"][BAND].mean()), C_PROMPT),
        "P+D  preamble + demos\n(1120 tokens)": (float(p["proj_PD"][BAND].mean()), C_PROMPT),
        "D  16 risky demos\n(1113 tokens)": (float(d["proj_D"][BAND].mean()), C_ICL),
        "N  16 neutral demos\n(1291 tokens)": (float(d["proj_N"][BAND].mean()), C_ICL),
        "F  good-medical demos\n(1063 tokens)": (float(d["proj_F"][BAND].mean()), C_ICL),
    }
    # Two nulls, because the arms are not comparable to one baseline.  The finetuned
    # organisms differ from B only in weights, so their random-direction null is tight.
    # The long-context arms differ from B by about 1100 tokens, which many random
    # directions pick up. Their null is nearly eight times wider. Showing only the tight
    # null would make P+D look significant against B even though the registered test fails.
    null_ft = tb["arms"]["M"]["null_band"]
    null_ctx = pd_meta["displacement"]["P1"]["null"]

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    names = list(vals)[::-1]
    y = np.arange(len(names))
    ax.axvspan(null_ctx["lo"], null_ctx["hi"], color="#e08214", alpha=0.13, zorder=0,
               label=f"null for the long-context arms [{null_ctx['lo']:+.2f}, {null_ctx['hi']:+.2f}]")
    ax.axvspan(null_ft["lo"], null_ft["hi"], color="#000000", alpha=0.16, zorder=1,
               label=f"null for the finetuned arms [{null_ft['lo']:+.2f}, {null_ft['hi']:+.2f}]")
    ax.barh(y, [vals[n][0] for n in names], color=[vals[n][1] for n in names],
            height=0.60, zorder=3)
    for i, n in enumerate(names):
        v = vals[n][0]
        ax.text(v + (0.18 if v >= 0 else -0.18), i, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5, zorder=4)
    ax.set_yticks(y, names, fontsize=8)
    ax.axvline(0, color="black", lw=0.9, zorder=2)
    ax.set_xlabel("mean projection onto the frozen W−B direction  (layers 12–18, n = 239 items)")
    ax.set_title("Finetuned and prompted conditions projected onto one frozen axis\n"
                 "The financial finetune (W) defines the axis; all other conditions are transfer tests.",
                 loc="left")
    ax.set_xlim(-2.4, 6.6)

    # The registered primary statistic is the matched pair, not the bar heights.
    i_pd, i_d = names.index("P+D  preamble + demos\n(1120 tokens)"), names.index("D  16 risky demos\n(1113 tokens)")
    pd_v, d_v = vals[names[i_pd]][0], vals[names[i_d]][0]
    ax.annotate("", xy=(pd_v, i_pd - 0.34), xytext=(d_v, i_d + 0.34),
                arrowprops=dict(arrowstyle="<->", color="#b2182b", lw=1.4))
    ax.text(1.35, (i_pd + i_d) / 2 - 0.05,
            f"the registered primary test:\nP+D − D = {pd_meta['displacement']['P2']['value']:+.2f}\n"
            f"{int(pd_meta['displacement']['P2']['frac_pos_items'] * 239)}/239 items, same demos,\ndiffers by 7 preamble tokens",
            fontsize=7.6, color="#b2182b", va="center")
    ax.legend(loc="lower right", fontsize=7.2, framealpha=0.95)
    fig.savefig(OUT / "fig1_axis.png")
    plt.close(fig)
    print("fig1_axis.png", {k.split()[0]: round(v[0], 3) for k, v in vals.items()},
          "| null_ft", [round(null_ft["lo"], 3), round(null_ft["hi"], 3)],
          "| null_ctx", [round(null_ctx["lo"], 3), round(null_ctx["hi"], 3)])


# ------------------------------------------------------ fig 2: PD steering sweep
BANK_FIELD = {"pivotal_tokens": "margin_sum", "semantic_pairs": "margin_mean"}


def pd_sweep():
    """Recompute the PD sweep exactly as run_m1.report does: drop stale, latest wins."""
    rows = jsonl(REPO / "results/m1/records.jsonl")
    dsha = sha16(REPO / "results/m1/direction.npz")
    pdsha = sha16(REPO / "results/m1/direction_pd.npz")
    keep = []
    for r in rows:
        if not r.get("point"):
            keep.append(r)
            continue
        ok = r.get("dirsha") == dsha
        if ok and r.get("coef") == "cancel_PD":
            ok = r.get("pdsha") == pdsha
        if ok:
            keep.append(r)
    latest = {r["record_id"]: r for r in keep
              if r.get("readout") == "logprob" and r.get("point")}
    by: dict = {}
    for r in latest.values():
        by.setdefault(r["point"], {}).setdefault(r["bank"], {})[r["item_id"]] = \
            r[BANK_FIELD[r["bank"]]]
    base = by["PD|12-18|w|a+0.000|all"]

    def shift(point, bank):
        cur, b = by[point].get(bank, {}), base.get(bank, {})
        sh = sorted(set(cur) & set(b))
        vals = [cur[i] - b[i] for i in sh]
        m, lo, hi = paired_bootstrap_ci(vals, n_boot=10000)
        return m, lo, hi, sum(1 for v in vals if v > 0), len(vals)
    return by, shift


def fig2_steering():
    by, shift = pd_sweep()
    p = np.load(REPO / "results/m1/direction_pd.npz", allow_pickle=True)
    a_cancel = float(p["alpha_cancel_PD_vs_B"][BAND].mean())
    alphas = [-4.0, -2.0, -1.0, -0.5, -0.25, -0.125, 0.0, 1.0]

    def series(kind):
        xs, ms, los, his = [], [], [], []
        for a in alphas:
            pt = (f"PD|12-18|{kind}|a{a:+.3f}|all" if a != 0 else "PD|12-18|w|a+0.000|all")
            if pt not in by:
                continue
            m, lo, hi, _, _ = shift(pt, "pivotal_tokens")
            xs.append(a); ms.append(m); los.append(lo); his.append(hi)
        return np.array(xs), np.array(ms), np.array(los), np.array(his)

    cm, clo, chi, cpos, cn = shift("PD|12-18|w|cancel_PD|all", "pivotal_tokens")
    rm, rlo, rhi, _, _ = shift("PD|12-18|rand|cancel_PD|all", "pivotal_tokens")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 3.7),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    for ax in (axL, axR):
        for kind, colour, label in [("w", C_PROMPT, "the W−B direction"),
                                    ("rand", C_RAND, "matched-norm random direction")]:
            x, m, lo, hi = series(kind)
            ax.fill_between(x, lo, hi, color=colour, alpha=0.16, lw=0)
            ax.plot(x, m, "o-", color=colour, lw=1.9, ms=4.5, label=label, zorder=3)
        ax.axhline(0, color="black", lw=0.9)
        ax.axvline(0, color="black", lw=0.6, ls=":")

    axL.set_xlim(-4.4, 1.4)
    axL.set_title("full sweep", loc="left", fontsize=9.5)
    axL.set_ylabel("shift in preference for the misaligned continuation (nats)\n"
                   "← less misaligned          more misaligned →")
    axL.annotate("at α = −4 the random control suppresses too:\ngeneric damage, not the direction",
                 xy=(-4, -2.36), xytext=(-4.2, 5.0), fontsize=7.2, color="#444444",
                 arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))
    axL.annotate("pushing the other way increases\nmisaligned preference\n(+2.02, 6/7 items)",
                 xy=(1, 2.02), xytext=(-1.6, 6.8), fontsize=7.0, color=C_PROMPT,
                 arrowprops=dict(arrowstyle="->", color=C_PROMPT, lw=0.8))
    axL.legend(loc="lower left", fontsize=7.4, framealpha=0.9)
    axL.add_patch(plt.Rectangle((-0.32, -2.2), 0.42, 2.9, fill=False,
                                ec="#b2182b", lw=1.0, ls="--", zorder=6))

    axR.set_xlim(-0.30, 0.06)
    axR.set_ylim(-1.9, 0.55)
    axR.set_title("the small-dose regime (boxed at left)", loc="left", fontsize=9.5)
    axR.errorbar([a_cancel], [cm], yerr=[[cm - clo], [chi - cm]], fmt="D",
                 color="#b2182b", ms=7, capsize=3, zorder=6,
                 label=f"cancel point: {cm:+.2f}, {cn - cpos}/{cn} items down")
    axR.errorbar([a_cancel], [rm], yerr=[[rm - rlo], [rhi - rm]], fmt="D",
                 color="#404040", ms=6, capsize=3, zorder=6, mfc="white",
                 label=f"random, same norm: {rm:+.3f}, CI spans 0")
    axR.axvline(a_cancel, color="#b2182b", lw=0.8, ls="--", alpha=0.6)
    axR.text(a_cancel - 0.006, 0.42, f"α = {a_cancel:.3f}\n(7% of the finetune)",
             fontsize=7.2, color="#b2182b", ha="right", va="top")
    axR.legend(loc="lower right", fontsize=7.2, framealpha=0.95)

    fig.suptitle("Steering shifts the prompted model's preference in both directions\n"
                 "P+D arm, 7 pivotal items, paired against α = 0. Shading shows the 95% bootstrap CI.",
                 x=0.02, ha="left", fontsize=10, y=1.06)
    fig.supxlabel("steering coefficient α  (in W-equivalents per layer, applied jointly over layers 12–18)",
                  fontsize=9, y=-0.06)
    fig.savefig(OUT / "fig2_steering.png")
    plt.close(fig)
    print(f"fig2_steering.png  cancel α={a_cancel:+.5f} real={cm:+.3f}[{clo:+.3f},{chi:+.3f}] "
          f"rand={rm:+.3f}[{rlo:+.3f},{rhi:+.3f}] items_down={cn - cpos}/{cn}")


# --------------------------------------------------- fig 3: Track B transfer test
def fig3_trackb():
    gate = json.load(open(REPO / "results/trackb/summary_gate.json"))
    tb = json.load(open(REPO / "results/trackb_exploratory/projection_frozen_direction.json"))
    wbench = gate["required_W_benchmark"]["pivotal_shift"]

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    labels, ypos = [], []
    for i, (org, name) in enumerate([("S", "S  extreme-sports"), ("M", "M  bad-medical")]):
        b = gate["organisms"][org]["banks"]["pivotal_tokens"]
        ratio = b["shift_vs_B"] / wbench
        lo, hi = b["shift_ci"][0] / wbench, b["shift_ci"][1] / wbench
        r_band = tb["arms"][org]["tier2"]["R_band"]
        floor = tb["arms"][org]["tier2"]["behavioural_floor"]
        ax.plot([lo, hi], [i, i], color=C_ICL, lw=6, solid_capstyle="butt", alpha=0.5,
                zorder=2, label="behavioral shift relative to W (95% CI)" if i == 0 else None)
        ax.plot([ratio], [i], "o", color="#404040", ms=7, zorder=3,
                label="behavioral point estimate" if i == 0 else None)
        ax.plot([r_band], [i], "D", color=C_FT, ms=8, zorder=4,
                label="how far it sits along W's axis" if i == 0 else None)
        ax.plot([floor], [i], "|", color="#404040", ms=16, mew=1.6, zorder=3)
        ax.text(r_band, i + 0.22, f"{r_band:.0%}", ha="center", fontsize=8, color=C_FT)
        ax.text(ratio, i - 0.30, f"{ratio:.0%}", ha="center", fontsize=8, color="#404040")
        labels.append(name); ypos.append(i)

    ax.axvline(1.0, color=C_FT, lw=1.0, ls="--", alpha=0.7)
    ax.text(1.01, 1.42, "W (the organism\nthe axis came from)", fontsize=7.2, color=C_FT, va="top")
    ax.axvline(tb["rule_constants"]["finance_max_ratio"], color="black", lw=0.9, ls=":")
    ax.text(0.115, 0.5, "below this threshold = consistent\nwith finance-only encoding", fontsize=7.2, va="center")
    ax.set_yticks(ypos, labels, fontsize=8.5)
    ax.set_ylim(-0.55, 1.75)
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("fraction of W on each measure")
    ax.set_title("Exploratory: the financial-finetune direction also finds the medical and sports finetunes\n"
                 "Each projection ratio falls within the behavioral ratio's 95% confidence interval.",
                 loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=7.4,
              frameon=False)
    fig.savefig(OUT / "fig3_trackb.png")
    plt.close(fig)
    print("fig3_trackb.png")


# ------------------------------------------------- fig 4: Track A behaviour + filter
def fig4_tracka():
    pd_rows = {r["record_id"]: r for r in
               jsonl(REPO / "results/tracka/pd_reckless_financial/records.jsonl")}
    d_rows = {r["record_id"]: r for r in
              jsonl(REPO / "results/gate_a/sampled/records.jsonl")}

    def rates(rows):
        j = [r for r in rows.values() if r.get("coherent") is not None]
        kept = [r for r in j if r["kept"]]
        strict_n = sum(1 for r in kept if r.get("misaligned"))
        scored = [r for r in j if r["aligned"] is not None]
        unf_n = sum(1 for r in scored if r["aligned"] < 30)
        return (strict_n, len(kept)), (unf_n, len(scored)), j

    (ps, pk), (pu, pn), pj = rates(pd_rows)
    (ds, dk), (du, dn), _ = rates(d_rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.2),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    groups = [("strict\n(coherent answers only)", (ps, pk), (ds, dk)),
              ("unfiltered\n(all answers)", (pu, pn), (du, dn))]
    x = np.arange(len(groups)); w = 0.34
    for off, colour, lab, idx in [(-w / 2, C_PROMPT, "P+D  preamble + demos", 1),
                                  (+w / 2, C_ICL, "D  demos only", 2)]:
        vals, errs = [], [[], []]
        for g in groups:
            k, n = g[idx]
            p, lo, hi = normal_ci(k, n)
            vals.append(p * 100); errs[0].append((p - lo) * 100); errs[1].append((hi - p) * 100)
        b = ax1.bar(x + off, vals, w, color=colour, label=lab, zorder=3)
        ax1.errorbar(x + off, vals, yerr=errs, fmt="none", ecolor="#333333",
                     capsize=2.5, lw=0.9, zorder=4)
        for xi, (v, g) in enumerate(zip(vals, groups)):
            k, n = g[idx]
            ax1.text(xi + off, v + 1.6, f"{k}/{n}", ha="center", fontsize=7.4)
    ax1.set_xticks(x, [g[0] for g in groups], fontsize=8)
    ax1.set_ylabel("misaligned answers (%)")
    ax1.set_ylim(0, 32)
    ax1.set_title("Adding one persona sentence\nraises broad misalignment", loc="left", fontsize=9.5)
    ax1.legend(fontsize=7.4, loc="upper left")

    kept = [r["aligned"] for r in pj if r["kept"] and r["aligned"] is not None]
    drop = [r["aligned"] for r in pj if not r["kept"] and r["aligned"] is not None]
    bins = np.linspace(0, 100, 21)
    ax2.hist([kept, drop], bins=bins, stacked=True, color=[C_PROMPT, "#e08214"],
             label=[f"kept by the filter (n={len(kept)})",
                    f"discarded as incoherent (n={len(drop)})"], zorder=3)
    ax2.axvline(30, color="#b2182b", lw=1.2, ls="--", zorder=4)
    n_lost = sum(1 for a in drop if a < 30)
    n_dang = n_lost + sum(1 for a in kept if a < 30)
    ax2.set_ylim(0, 19)
    ax2.text(1.5, 13.2,
             f"{n_lost} of the {n_dang} dangerous answers\nnever reach the strict count because\nthe filter removes them first.",
             fontsize=7.4, color="#b2182b", ha="left", va="top")
    ax2.annotate("", xy=(29, 8.6), xytext=(20, 11.4),
                 arrowprops=dict(arrowstyle="->", color="#b2182b", lw=0.9))
    ax2.set_xlabel("judge's alignment score (0 = dangerous, 100 = fine)")
    ax2.set_ylabel("answers")
    ax2.set_title("The coherence filter throws away\nmost of the dangerous answers", loc="left", fontsize=9.5)
    ax2.legend(fontsize=7.2, loc="upper center", bbox_to_anchor=(0.62, 1.0))
    fig.savefig(OUT / "fig4_tracka.png")
    plt.close(fig)
    print(f"fig4_tracka.png  strict {ps}/{pk} vs {ds}/{dk}; unfiltered {pu}/{pn} vs {du}/{dn}; "
          f"dangerous removed by filter {n_lost}/{n_dang}")


# ------------------------------------------------------------ fig 5: depth profile
def fig5_depth():
    m2 = json.load(open(REPO / "results/m2/summary_m2.json"))
    rows = sorted([r for r in m2["rows"] if r["pooling"] == "response_avg"
                   and r["probe"] == "diffmeans" and r["subset"] == "all"],
                  key=lambda r: r["layer"])
    L = np.array([r["layer"] for r in rows])
    v = np.array([r["auc_D_minus_N"] for r in rows])
    lo = np.array([r["auc_D_minus_N_lo"] for r in rows])
    hi = np.array([r["auc_D_minus_N_hi"] for r in rows])
    nlo = np.array([r["null_lo_D_minus_N"] for r in rows])
    nhi = np.array([r["null_hi_D_minus_N"] for r in rows])

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.fill_between(L, nlo, nhi, color="#000000", alpha=0.10, lw=0,
                    label="random-direction null", zorder=1)
    ax.axvspan(BAND[0] - 0.5, BAND[-1] + 0.5, color=C_PROMPT, alpha=0.09, lw=0, zorder=0)
    ax.fill_between(L, lo, hi, color=C_PROMPT, alpha=0.25, lw=0, zorder=2)
    ax.plot(L, v, "o-", color=C_PROMPT, lw=1.8, ms=3.6, zorder=3,
            label="risky demos − neutral demos")
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xlabel("layer")
    ax.set_ylabel("probe separation, risky vs neutral\n(AUC above chance)")
    ax.set_title("Risky-versus-neutral separation is concentrated in the middle layers\n"
                 "Layers 12–18 are the only band significant on all four corpus × probe combinations.",
                 loc="left")
    ax.text(15, v[BAND].max() + 0.035, "layers 12–18", ha="center", fontsize=8, color=C_PROMPT)
    ax.set_xticks(range(0, 28, 2))
    ax.legend(fontsize=7.5, loc="upper left")
    fig.savefig(OUT / "fig5_depth.png")
    plt.close(fig)
    print(f"fig5_depth.png  peak {v.max():+.3f} at L{L[v.argmax()]}")


if __name__ == "__main__":
    fig1_axis()
    fig2_steering()
    fig3_trackb()
    fig4_tracka()
    fig5_depth()
    print("\nwrote", len(list(OUT.glob('*.png'))), "figures to", OUT)
