"""Two-sample and categorical hypothesis tests (test statistics computed from scratch; SciPy supplies only distribution functions)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import chi2, hypergeom, norm, rankdata, t as t_dist


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float
    df: float | None = None
    detail: dict | None = None


def welch_t_test(x: ArrayLike, y: ArrayLike) -> TestResult:
    """Welch's unequal-variance two-sample t-test (two-sided) with a 95% CI for the mean difference in ``detail``."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = x.size, y.size
    if n1 < 2 or n2 < 2:
        raise ValueError("each sample needs at least two observations")
    a, b = x.var(ddof=1) / n1, y.var(ddof=1) / n2
    se = np.sqrt(a + b)
    diff = x.mean() - y.mean()
    t = diff / se
    df = (a + b) ** 2 / (a ** 2 / (n1 - 1) + b ** 2 / (n2 - 1))
    p = 2 * t_dist.sf(abs(t), df)
    crit = t_dist.ppf(0.975, df)
    return TestResult(float(t), float(p), float(df), {"mean_difference": float(diff), "ci95": (float(diff - crit * se), float(diff + crit * se))})


def mann_whitney_u(x: ArrayLike, y: ArrayLike) -> TestResult:
    """Mann-Whitney U test, normal approximation with tie and continuity corrections (two-sided)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = x.size, y.size
    ranks = rankdata(np.concatenate([x, y]))
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    u = max(u1, n1 * n2 - u1)
    n = n1 + n2
    _, counts = np.unique(np.concatenate([x, y]), return_counts=True)
    sigma2 = n1 * n2 / 12 * ((n + 1) - np.sum(counts ** 3 - counts) / (n * (n - 1)))
    if sigma2 <= 0:
        return TestResult(float(u1), 1.0)
    z = (u - n1 * n2 / 2 - 0.5) / np.sqrt(sigma2)
    return TestResult(float(u1), float(min(1.0, 2 * norm.sf(z))), None, {"auc_estimate": float(u1 / (n1 * n2))})


def chi_square_independence(table: ArrayLike) -> TestResult:
    """Pearson chi-square test of independence for an r x c contingency table (no continuity correction)."""
    obs = np.asarray(table, float)
    if obs.ndim != 2 or obs.shape[0] < 2 or obs.shape[1] < 2 or (obs < 0).any():
        raise ValueError("table must be a non-negative r x c array with r, c >= 2")
    expected = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()
    if (expected == 0).any():
        raise ValueError("a row or column sums to zero")
    stat = float(np.sum((obs - expected) ** 2 / expected))
    df = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    return TestResult(stat, float(chi2.sf(stat, df)), float(df), {"min_expected": float(expected.min())})


def fisher_exact(table: ArrayLike) -> TestResult:
    """Fisher's exact test for a 2x2 table (two-sided: all tables as or less probable than the observed one)."""
    a, b, c, d = np.asarray(table, int).ravel()
    total, row1, col1 = a + b + c + d, a + b, a + c
    dist = hypergeom(total, row1, col1)
    support = np.arange(max(0, col1 - (total - row1)), min(row1, col1) + 1)
    pmf = dist.pmf(support)
    p = float(pmf[pmf <= dist.pmf(a) * (1 + 1e-7)].sum())
    odds = (a * d) / (b * c) if b * c else float("inf")
    return TestResult(float(odds), min(1.0, p), None, {"odds_ratio_sample": float(odds)})


def permutation_test(x: ArrayLike, y: ArrayLike, n_perm: int = 10000, seed: int = 0) -> TestResult:
    """Two-sided permutation test for a difference in means; p = (extreme + 1) / (n_perm + 1)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    pooled = np.concatenate([x, y])
    observed = abs(x.mean() - y.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        extreme += abs(pooled[: x.size].mean() - pooled[x.size:].mean()) >= observed - 1e-12
    return TestResult(float(x.mean() - y.mean()), float((extreme + 1) / (n_perm + 1)))
