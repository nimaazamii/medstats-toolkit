"""Confidence intervals for means and proportions."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import beta, norm, t as t_dist


def mean_ci_t(x: ArrayLike, alpha: float = 0.05) -> tuple[float, float]:
    x = np.asarray(x, float)
    half = t_dist.ppf(1 - alpha / 2, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size)
    return float(x.mean() - half), float(x.mean() + half)


def _check(k: int, n: int) -> None:
    if not 0 <= k <= n or n < 1:
        raise ValueError("need 0 <= k <= n and n >= 1")


def wald_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Normal-approximation interval (known to under-cover for small n or extreme p)."""
    _check(k, n)
    p, z = k / n, norm.ppf(1 - alpha / 2)
    half = z * np.sqrt(p * (1 - p) / n)
    return float(max(0.0, p - half)), float(min(1.0, p + half))


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    _check(k, n)
    p, z = k / n, norm.ppf(1 - alpha / 2)
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (conservative) binomial interval from Beta quantiles."""
    _check(k, n)
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def bootstrap_ci(x: ArrayLike, stat=np.mean, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for any statistic of one sample."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    values = np.array([stat(rng.choice(x, x.size)) for _ in range(n_boot)])
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def exact_coverage(ci_fn, n: int, p: float, alpha: float = 0.05) -> float:
    """Exact coverage probability of a binomial interval method at true proportion ``p`` (sums over k = 0..n)."""
    from scipy.stats import binom
    total = 0.0
    for k in range(n + 1):
        lo, hi = ci_fn(k, n, alpha)
        if lo <= p <= hi:
            total += binom.pmf(k, n, p)
    return float(total)
