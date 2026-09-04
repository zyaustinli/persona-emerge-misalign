"""Linear probes and AUC, hand-written.

sklearn is not installed and adding it would be this package's first new dependency.
The house style is already to hand-roll where stdlib/numpy suffices (`stats.py`, the
inline .env parser), and what we need is a logistic fit and a rank statistic.  Both
are cross-checked in `run_m2.py --stage selftest`.

Two probe types, both reported:

  logistic  -- L2 logistic regression.  The conventional choice.
  diffmeans -- projection on the difference-of-means direction.  No fitting, more
               robust at n << d (here n ~ 228, d = 3584), and it is the SAME
               direction M1 steers along, so the causal follow-on inherits it.

The vendored reference (model-organisms .../lora_probing.py) z-scores the features
BEFORE `train_test_split`, which leaks test statistics into training.  That is not
copied: `standardize` here is fit on the training fold alone.
"""
from __future__ import annotations

import numpy as np
import torch


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score(positive) > score(negative)), ties counting half.

    The Mann-Whitney rank identity: AUC = (R_pos - n1(n1+1)/2) / (n1 n0), where R_pos
    is the sum of the positives' ranks and ties get averaged ranks.  Verified against
    a brute-force O(n^2) pair count, ties included, in the self-test.
    """
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        raise ValueError("AUC needs at least one item in each class")
    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(xs), dtype=np.float64)
    i = 0
    while i < len(xs):  # average ranks within each tie group
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    is_pos = order < n1  # x[:n1] are the positives, so order[k] < n1 marks one
    r_pos = ranks[is_pos].sum()
    return float((r_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auc_bruteforce(pos: np.ndarray, neg: np.ndarray) -> float:
    """O(n^2) reference implementation.  Self-test only."""
    c = sum(
        1.0 if float(p) > float(q) else 0.5 if float(p) == float(q) else 0.0
        for p in pos
        for q in neg
    )
    return float(c / (len(pos) * len(neg)))


def paired_fraction(treated: np.ndarray, control: np.ndarray) -> float:
    """Fraction of items whose treated score exceeds its own control score.

    Uses the pairing that AUC throws away.  Every condition reads the identical
    reference text, so item identity dominates the activation and between-item
    variance caps the achievable unpaired AUC; this statistic is not subject to that
    ceiling.  Chance is 0.5, same as AUC, and it is the same `frac > 0` readout
    Gate A reports for the logprob margin, so the two halves stay comparable.
    """
    treated = np.asarray(treated, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    if treated.shape != control.shape:
        raise AssertionError(f"unpaired shapes {treated.shape} vs {control.shape}")
    diff = treated - control
    return float((np.sign(diff) > 0).mean() + 0.5 * (diff == 0).mean())


def kfold_by_item(n: int, folds: int, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Index folds over ITEMS, not over rows.

    Item i contributes a vector to every condition, and B is the shared negative class
    in all three contrasts.  If item i's B vector were in train while its D vector is
    scored, the probe would have seen that exact text -- so folds must be assigned by
    item and every condition must inherit the same assignment.
    """
    if folds < 2 or folds > n:
        raise ValueError(f"folds must be in [2, n]; got folds={folds}, n={n}")
    order = np.random.default_rng(seed).permutation(n)
    out = []
    for f in range(folds):
        test = np.sort(order[f::folds])
        train = np.sort(np.setdiff1d(np.arange(n), test, assume_unique=False))
        if len(np.intersect1d(train, test)):
            raise AssertionError("fold overlap")
        out.append((train, test))
    covered = np.sort(np.concatenate([t for _, t in out]))
    if not np.array_equal(covered, np.arange(n)):
        raise AssertionError("folds do not partition the items exactly once")
    return out


def standardize(train: np.ndarray):
    """Fit per-feature mean/std on the TRAIN fold only; return an apply function."""
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)  # a constant feature carries no signal

    def apply(x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - mu) / sd

    return apply


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0, max_iter: int = 200):
    """L2 logistic regression by LBFGS.  Returns (w, b) as float64 numpy."""
    xt = torch.tensor(np.asarray(x), dtype=torch.float32)
    yt = torch.tensor(np.asarray(y), dtype=torch.float32)
    w = torch.zeros(xt.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(xt @ w + b, yt)
        loss = loss + l2 * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    wn = w.detach().numpy().astype(np.float64)
    if not np.isfinite(wn).all():
        raise AssertionError("logistic probe diverged (non-finite weights)")
    return wn, float(b.detach())


def diff_of_means(x_pos: np.ndarray, x_neg: np.ndarray) -> np.ndarray:
    """Unit-norm difference-of-means direction.  Also the M1 steering direction."""
    d = np.asarray(x_pos).mean(axis=0) - np.asarray(x_neg).mean(axis=0)
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        raise AssertionError("difference of means is zero -- the classes are identical")
    return d / norm


def project(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) @ np.asarray(w, dtype=np.float64)


# --------------------------------------------------------------------------
# Amendment 1 additions.  See PREDICTIONS-m2.md.
# --------------------------------------------------------------------------

def calibrate(scores, ref):
    """z-score `scores` against a reference (train-fold B) distribution.

    Each CV fold fits its own direction and intercept, so raw scores from five folds
    live on five different scales; pooling them and computing one AUC silently mixes
    those scales.  Standardising every fold against its own training-fold B
    distribution puts them on one scale using no held-out information -- and it is
    what a deployment monitor actually does, calibrating a threshold on known-good
    traffic.
    """
    ref = np.asarray(ref, dtype=np.float64)
    mu, sd = ref.mean(), ref.std()
    if sd < 1e-12:
        raise AssertionError("reference scores have zero spread; cannot calibrate")
    return (np.asarray(scores, dtype=np.float64) - mu) / sd


def fit_ridge_dual(x: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    """Ridge on +/-1 labels, solved in the dual: w = X'(XX' + l2 I)^-1 y.

    With n ~ 380 and d = 3584 the classes are separable almost surely, so logistic
    regression's direction is set entirely by its penalty -- and a hand-rolled
    gradient version's step count becomes a hidden hyperparameter that dominates it.
    The dual ridge solve is exact, closed-form, deterministic, and for two classes is
    regularised LDA up to scale.  It is an n x n solve, so milliseconds here.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    y = np.where(y > 0.5, 1.0, -1.0)
    n = x.shape[0]
    alpha = np.linalg.solve(x @ x.T + l2 * np.eye(n), y)
    return x.T @ alpha


def paired_diff_of_means(x_pos: np.ndarray, x_neg: np.ndarray) -> np.ndarray:
    """Unit-norm mean of the PAIRED differences, mean_i(h_W[i] - h_B[i]).

    Row i of both arrays must be the same item.  The design is paired on identical
    frozen text, so differencing within an item cancels the item-level nuisance
    exactly; the unpaired difference of means does not.  Numerically identical on
    balanced complete data, but this is the estimator the design licenses, and it is
    the direction M1 steers along.
    """
    x_pos = np.asarray(x_pos, dtype=np.float64)
    x_neg = np.asarray(x_neg, dtype=np.float64)
    if x_pos.shape != x_neg.shape:
        raise AssertionError(f"unpaired shapes {x_pos.shape} vs {x_neg.shape}")
    d = (x_pos - x_neg).mean(axis=0)
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        raise AssertionError("paired difference of means is zero")
    return d / norm


def random_direction_null(acts_by_cond: dict, layer: int, idx: np.ndarray,
                          n_dirs: int, seed: int = 0) -> dict:
    """Empirical AUC null from random unit directions.

    Chance is NOT 0.5 for the prefixed arms.  D differs from B by a large systematic
    mean offset (1113 tokens of context), so projecting on a RANDOM direction gives a
    per-item difference with random sign but non-trivial magnitude -- the resulting
    AUC distribution is bimodal near 0 and 1, not concentrated at chance.  An observed
    AUC_D of 0.62 can sit comfortably inside that band.  One matmul per direction.
    """
    rng = np.random.default_rng(seed)
    d = next(iter(acts_by_cond.values())).shape[2]
    dirs = rng.normal(size=(n_dirs, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    # The null must live in the SAME space as the probe.  The probe scores
    # standardized features, so a random direction has to be random in standardized
    # coordinates too -- a random direction in raw space is, after standardisation, a
    # direction weighted by each feature's spread, and it yields a systematically
    # wider band (measured: [0.13,0.82] raw vs [0.27,0.76] standardized at
    # response_avg L14).  A too-wide band makes real signal look unexceptional, i.e. it
    # biases toward the registered null.  Scaler fit on B u W over all items: this is a
    # geometry reference, not a held-out test statistic, so there is nothing to leak
    # into -- no probe is fit here.
    ref_fit = np.concatenate([acts_by_cond["W"][idx][:, layer, :],
                              acts_by_cond["B"][idx][:, layer, :]])
    mu = ref_fit.mean(axis=0)
    sd = ref_fit.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    proj = {c: ((acts_by_cond[c][idx][:, layer, :] - mu) / sd) @ dirs.T
            for c in acts_by_cond}
    out = {}
    aucs_by_condition = {}
    for c in acts_by_cond:
        if c == "B":
            continue
        aucs = np.array([auc(proj[c][:, j], proj["B"][:, j]) for j in range(n_dirs)])
        aucs_by_condition[c] = aucs
        lo, hi = float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))
        # A band wider than M2_NULL_MAX_INFORMATIVE_WIDTH means random directions span
        # most of the AUC range, so "observed AUC is outside the band" is a statement
        # about the 97.5th percentile of noise, not about signal.  Without this flag an
        # AUC of exactly 1.0 is called "significant" against a band of [0.00, 0.9999].
        out[c] = {"lo": lo, "hi": hi, "median": float(np.median(aucs)),
                  "width": hi - lo, "informative": bool(hi - lo <= 0.50)}
    # D-N is the pre-registered decision statistic, so it needs its OWN null.
    # Being outside the marginal D and N bands does not imply their difference is
    # exceptional (or vice versa), because both projections use the same direction
    # and the same B reference. Preserve that dependence direction by direction.
    if "D" in proj and "N" in proj:
        diff = aucs_by_condition["D"] - aucs_by_condition["N"]
        d_vs_n = np.array([
            auc(proj["D"][:, j], proj["N"][:, j]) for j in range(n_dirs)
        ])
        for key, arr in (("D_minus_N", diff), ("D_vs_N", d_vs_n)):
            lo, hi = float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))
            out[key] = {"lo": lo, "hi": hi, "median": float(np.median(arr)),
                        "width": hi - lo, "informative": bool(hi - lo <= 0.50)}
    return out


def norm_baseline(acts_by_cond: dict, layer: int, idx: np.ndarray) -> dict:
    """AUC from the activation NORM alone, no direction.

    If scoring by ||h|| already separates the arms, then whatever AUC the probe gets
    is not evidence about direction.  A long prefix changes residual norms, so this is
    a live alternative for the D/N/F arms specifically.
    """
    n = {c: np.linalg.norm(acts_by_cond[c][idx][:, layer, :], axis=1) for c in acts_by_cond}
    return {c: auc(n[c], n["B"]) for c in acts_by_cond if c != "B"}


def bootstrap_auc_difference(s_a, s_b, s_ref, n_boot: int = 2000, seed: int = 0):
    """CI for AUC(a vs ref) - AUC(b vs ref), resampling ITEMS jointly.

    One item-index resample per replicate, both AUCs recomputed on that same
    resample.  Two independently-bootstrapped CIs eyeballed for overlap is exactly
    the error readouts.contrast_logprob was written to prevent on the logprob side:
    the shared reference class correlates the two AUCs, which shrinks the variance of
    their difference -- ignoring that throws the gain away and can flip a call.
    """
    s_a, s_b, s_ref = map(lambda v: np.asarray(v, dtype=np.float64), (s_a, s_b, s_ref))
    n = len(s_ref)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        take = rng.integers(0, n, n)
        diffs[i] = auc(s_a[take], s_ref[take]) - auc(s_b[take], s_ref[take])
    point = auc(s_a, s_ref) - auc(s_b, s_ref)
    return point, float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
