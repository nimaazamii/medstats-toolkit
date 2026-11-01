import unittest

import numpy as np
from scipy import stats

from medstats import *  # noqa: F401,F403


def _samples(seed, n1=40, n2=55):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, n1), rng.normal(0.4, 1.7, n2)


class TestHypothesis(unittest.TestCase):
    def test_welch_matches_scipy(self):
        for seed in range(5):
            x, y = _samples(seed)
            ours, ref = welch_t_test(x, y), stats.ttest_ind(x, y, equal_var=False)
            self.assertAlmostEqual(ours.statistic, ref.statistic, places=10)
            self.assertAlmostEqual(ours.p_value, ref.pvalue, places=10)
            self.assertAlmostEqual(ours.df, ref.df, places=8)
            ci = ref.confidence_interval(0.95)
            self.assertAlmostEqual(ours.detail["ci95"][0], ci.low, places=8)

    def test_mann_whitney_matches_scipy_with_ties(self):
        rng = np.random.default_rng(1)
        x, y = rng.integers(0, 8, 50), rng.integers(2, 10, 45)
        ref = stats.mannwhitneyu(x, y, method="asymptotic", use_continuity=True)
        ours = mann_whitney_u(x, y)
        self.assertAlmostEqual(ours.statistic, ref.statistic, places=8)
        self.assertAlmostEqual(ours.p_value, ref.pvalue, places=8)

    def test_chi_square_and_fisher_match_scipy(self):
        table = [[12, 5], [7, 21]]
        chi = chi_square_independence(table)
        ref = stats.chi2_contingency(table, correction=False)
        self.assertAlmostEqual(chi.statistic, ref[0], places=10)
        self.assertAlmostEqual(chi.p_value, ref[1], places=10)
        big = chi_square_independence([[10, 20, 30], [15, 15, 40]])
        self.assertEqual(big.df, 2)
        for t in ([[3, 1], [1, 3]], [[12, 5], [7, 21]], [[0, 6], [5, 0]]):
            self.assertAlmostEqual(fisher_exact(t).p_value, stats.fisher_exact(t)[1], places=10)

    def test_permutation_agrees_with_t_test_roughly_and_is_reproducible(self):
        x, y = _samples(3)
        a, b = permutation_test(x, y, 4000, seed=1), permutation_test(x, y, 4000, seed=1)
        self.assertEqual(a.p_value, b.p_value)
        self.assertAlmostEqual(a.p_value, welch_t_test(x, y).p_value, delta=0.05)

    def test_null_p_values_are_roughly_uniform(self):
        rng = np.random.default_rng(0)
        p = [welch_t_test(rng.normal(size=30), rng.normal(size=30)).p_value for _ in range(400)]
        self.assertLess(abs(np.mean(np.array(p) < 0.05) - 0.05), 0.04)

    def test_validation(self):
        with self.assertRaises(ValueError):
            welch_t_test([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            chi_square_independence([[1, 2, 3]])
        with self.assertRaises(ValueError):
            chi_square_independence([[0, 0], [1, 2]])


class TestIntervals(unittest.TestCase):
    def test_proportion_intervals_match_scipy(self):
        for k, n in [(0, 20), (3, 20), (10, 20), (20, 20), (7, 100)]:
            r = stats.binomtest(k, n)
            for ours, method in ((wilson_ci(k, n), "wilson"), (clopper_pearson_ci(k, n), "exact")):
                ref = r.proportion_ci(0.95, method=method)
                self.assertAlmostEqual(ours[0], ref.low, places=6)
                self.assertAlmostEqual(ours[1], ref.high, places=6)

    def test_mean_ci_matches_scipy(self):
        x = np.random.default_rng(2).normal(5, 2, 30)
        ref = stats.t.interval(0.95, 29, loc=x.mean(), scale=stats.sem(x))
        np.testing.assert_allclose(mean_ci_t(x), ref, atol=1e-10)

    def test_coverage_ordering_for_small_samples(self):
        n, p = 30, 0.05
        wald = exact_coverage(wald_ci, n, p)
        wilson = exact_coverage(wilson_ci, n, p)
        cp = exact_coverage(clopper_pearson_ci, n, p)
        self.assertLess(wald, 0.90)            # Wald under-covers badly here
        self.assertGreater(cp, 0.95)           # Clopper-Pearson is conservative
        self.assertGreater(wilson, wald)

    def test_bootstrap_ci_contains_mean_and_validation(self):
        x = np.random.default_rng(4).normal(10, 3, 100)
        lo, hi = bootstrap_ci(x, np.mean, 2000)
        self.assertTrue(lo < x.mean() < hi)
        with self.assertRaises(ValueError):
            wilson_ci(5, 3)


class TestMultitest(unittest.TestCase):
    def test_known_values(self):
        p = [0.01, 0.04, 0.03, 0.005]
        np.testing.assert_allclose(adjust_pvalues(p, "bonferroni"), [0.04, 0.16, 0.12, 0.02])
        np.testing.assert_allclose(adjust_pvalues(p, "holm"), [0.03, 0.06, 0.06, 0.02])
        np.testing.assert_allclose(adjust_pvalues(p, "bh"), [0.02, 0.04, 0.04, 0.02])

    def test_against_scipy_fdr(self):
        p = np.random.default_rng(0).uniform(size=50) ** 2
        np.testing.assert_allclose(adjust_pvalues(p, "bh"), stats.false_discovery_control(p, method="bh"))


class TestBayes(unittest.TestCase):
    def test_ppv_depends_on_prevalence(self):
        low = predictive_values(0.9, 0.95, 0.01)
        high = predictive_values(0.9, 0.95, 0.5)
        self.assertAlmostEqual(low["ppv"], 0.009 / (0.009 + 0.0495), places=12)
        self.assertLess(low["ppv"], 0.2)
        self.assertGreater(high["ppv"], 0.9)
        self.assertAlmostEqual(low["lr_positive"], 18.0)

    def test_odds_form_matches_direct_bayes(self):
        pv = predictive_values(0.9, 0.95, 0.1)
        self.assertAlmostEqual(posterior_from_likelihood_ratio(0.1, pv["lr_positive"]), pv["ppv"], places=12)
        self.assertAlmostEqual(posterior_from_likelihood_ratio(0.1, pv["lr_negative"]), 1 - pv["npv"], places=12)
        self.assertEqual(posterior_from_likelihood_ratio(1.0, 3.0), 1.0)

    def test_beta_binomial(self):
        post = beta_binomial_update(7, 20)
        self.assertEqual((post.a, post.b), (8.0, 14.0))
        self.assertAlmostEqual(post.mean, 8 / 22)
        lo, hi = post.credible_interval()
        self.assertTrue(lo < post.mean < hi)
        with self.assertRaises(ValueError):
            beta_binomial_update(5, 3)
        with self.assertRaises(ValueError):
            predictive_values(1.2, 0.9, 0.1)


class TestMonteCarlo(unittest.TestCase):
    def test_pi_within_four_standard_errors(self):
        est, se = estimate_pi(200_000, seed=1)
        self.assertLess(abs(est - np.pi), 4 * se)

    def test_event_probability(self):
        p, (lo, hi) = mc_probability(lambda r: r.random() < 0.3, 5000, seed=2)
        self.assertTrue(lo <= 0.3 <= hi)

    def test_random_walk_msd_equals_steps(self):
        self.assertAlmostEqual(random_walk_msd(100, 20000, seed=0) / 100, 1.0, delta=0.05)

    def test_law_of_large_numbers(self):
        rm = running_mean(lambda r, n: r.exponential(2.0, n), 100_000, seed=0)
        self.assertLess(abs(rm[-1] - 2.0), abs(rm[99] - 2.0) + 0.05)
        self.assertAlmostEqual(rm[-1], 2.0, delta=0.05)

    def test_clt_skewness_shrinks_like_inverse_sqrt_n(self):
        sampler = lambda r, n: r.exponential(1.0, n)          # skewness 2
        for n in (4, 64):
            z = standardized_means(sampler, 1.0, 1.0, n, 4000, seed=0)
            self.assertAlmostEqual(stats.skew(z), 2 / np.sqrt(n), delta=0.2)
            self.assertAlmostEqual(z.std(), 1.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
