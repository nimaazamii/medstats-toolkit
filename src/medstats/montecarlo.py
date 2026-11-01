"""Monte Carlo simulation utilities."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def estimate_pi(n: int, seed: int = 0) -> tuple[float, float]:
    """Estimate pi from points in the unit square; returns (estimate, standard error)."""
    rng = np.random.default_rng(seed)
    inside = (rng.random(n) ** 2 + rng.random(n) ** 2) <= 1.0
    p = inside.mean()
    return float(4 * p), float(4 * np.sqrt(p * (1 - p) / n))


def mc_probability(event: Callable[[np.random.Generator], bool], n: int, seed: int = 0) -> tuple[float, tuple[float, float]]:
    """Monte Carlo probability of an event with a Wilson 95% interval."""
    from .intervals import wilson_ci
    rng = np.random.default_rng(seed)
    k = sum(bool(event(rng)) for _ in range(n))
    return k / n, wilson_ci(k, n)


def random_walk_msd(steps: int, n_walks: int, seed: int = 0) -> float:
    """Mean squared displacement of simple +/-1 random walks; the theoretical value is ``steps``."""
    rng = np.random.default_rng(seed)
    final = rng.choice([-1, 1], size=(n_walks, steps)).sum(axis=1)
    return float(np.mean(final.astype(float) ** 2))


def running_mean(sampler: Callable[[np.random.Generator, int], NDArray[np.float64]], n: int, seed: int = 0) -> NDArray[np.float64]:
    """Running mean of i.i.d. draws (law of large numbers)."""
    x = sampler(np.random.default_rng(seed), n)
    return np.cumsum(x) / np.arange(1, n + 1)


def standardized_means(sampler: Callable[[np.random.Generator, int], NDArray[np.float64]], mu: float, sigma: float,
                       n: int, reps: int, seed: int = 0) -> NDArray[np.float64]:
    """sqrt(n) (mean - mu) / sigma for ``reps`` samples of size ``n`` (central limit theorem)."""
    rng = np.random.default_rng(seed)
    means = np.array([sampler(rng, n).mean() for _ in range(reps)])
    return np.sqrt(n) * (means - mu) / sigma
