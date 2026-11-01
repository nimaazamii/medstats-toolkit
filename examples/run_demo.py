"""Numerical experiments with medstats (all computed, no external data).

    python examples/run_demo.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from medstats import (beta_binomial_update, clopper_pearson_ci, estimate_pi, exact_coverage, predictive_values,
                      posterior_from_likelihood_ratio, random_walk_msd, standardized_means, wald_ci, wilson_ci)

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)


def main() -> None:
    res: dict = {"DATA": "simulated / analytic; no patient data"}
    res["ppv_by_prevalence_sens90_spec95"] = {f"{p:g}": predictive_values(0.90, 0.95, p) for p in (0.001, 0.01, 0.1, 0.5)}
    res["two_stage_screening"] = {"prevalence": 0.01, "after_first_positive": posterior_from_likelihood_ratio(
        0.01, predictive_values(0.9, 0.95, 0.01)["lr_positive"])}
    res["two_stage_screening"]["after_second_positive"] = posterior_from_likelihood_ratio(
        res["two_stage_screening"]["after_first_positive"], predictive_values(0.9, 0.95, 0.01)["lr_positive"])
    post = beta_binomial_update(7, 20)
    res["beta_binomial_7_of_20_uniform_prior"] = {"posterior": [post.a, post.b], "mean": post.mean, "cri95": post.credible_interval()}
    res["exact_coverage_95pct_ci"] = {
        f"n={n},p={p}": {"wald": exact_coverage(wald_ci, n, p), "wilson": exact_coverage(wilson_ci, n, p),
                         "clopper_pearson": exact_coverage(clopper_pearson_ci, n, p)}
        for n, p in ((30, 0.05), (30, 0.2), (100, 0.05), (100, 0.5))}
    est, se = estimate_pi(1_000_000, seed=0)
    res["pi_monte_carlo_1e6"] = {"estimate": est, "standard_error": se, "abs_error": abs(est - np.pi)}
    res["random_walk_msd_over_steps_n=100"] = random_walk_msd(100, 50_000, seed=0) / 100
    from scipy.stats import skew
    res["clt_skewness_exponential"] = {str(n): float(skew(standardized_means(lambda r, k: r.exponential(1.0, k), 1.0, 1.0, n, 5000, 0)))
                                       for n in (2, 8, 32, 128)}
    (OUT / "demo_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
