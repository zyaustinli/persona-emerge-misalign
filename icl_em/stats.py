"""The two intervals this study reports.  Kept apart from the measurement code."""
from __future__ import annotations

import math
import random


def normal_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """95% normal (Wald) interval, matching Afonin et al.'s reported CIs.

    Wald is the wrong interval near 0 -- it collapses to zero width at p=0 -- but
    it is what the paper reports, so it is what we report for comparability.  The
    detection floor is stated alongside any zero instead.
    """
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = successes / n
    half = z * math.sqrt(p * (1 - p) / n)
    return p, max(0.0, p - half), min(1.0, p + half)


def paired_bootstrap_ci(
    values: list[float], n_boot: int = 10000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Mean of per-item paired differences, with a percentile bootstrap CI.

    Resampling is over ITEMS, which is the unit of independence here: the same
    model and the same demonstration block produce every value, so the items are
    the only thing being generalised over.
    """
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    mean = sum(values) / len(values)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return mean, lo, hi
