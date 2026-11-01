# medstats-toolkit

Statistics for medical data analysis, implemented transparently (test statistics are computed from scratch; SciPy supplies only
distribution functions) and **tested against SciPy** on every function that has a SciPy counterpart.

## Features
* **Hypothesis tests:** Welch t-test (with CI of the difference), Mann-Whitney U (ties, continuity correction), Pearson chi-square, Fisher's exact test (2x2), permutation test.
* **Intervals:** t interval, Wald, Wilson and Clopper-Pearson binomial intervals, bootstrap percentile CI, `exact_coverage` (exact coverage probability of any binomial interval).
* **Multiple testing:** Bonferroni, Holm, Benjamini-Hochberg.
* **Bayesian reasoning:** PPV/NPV and likelihood ratios from sensitivity, specificity and prevalence; odds-form Bayes updates; Beta-binomial posterior with credible interval.
* **Monte Carlo:** pi estimation with standard error, event probability with Wilson CI, random-walk mean squared displacement, law of large numbers, central limit theorem.

## Installation and quick start
```bash
git clone https://github.com/nimaazamii/medstats-toolkit.git
cd medstats-toolkit
pip install -e ".[dev]"
pytest -q
```

```python
from medstats import welch_t_test, wilson_ci, predictive_values
a = [5.1, 4.9, 5.6, 5.8, 6.0, 5.2]; b = [6.3, 6.8, 6.1, 7.0, 6.5, 6.9]
print(round(welch_t_test(a, b).p_value, 5))
print(tuple(round(x, 3) for x in wilson_ci(7, 20)))
print(round(predictive_values(0.90, 0.95, 0.01)["ppv"], 3))
```
Output:
```
0.00052
(0.181, 0.567)
0.154
```

## Verification
Welch t, Mann-Whitney, chi-square, Fisher, Wilson/Clopper-Pearson, t interval and FDR adjustment agree with SciPy to 6-10 decimals;
hand-computed Holm/BH values; null p-values are roughly uniform; the odds-form Bayes update equals direct Bayes; Monte Carlo estimates stay within 4 standard errors;
CLT skewness shrinks like 2/sqrt(n) for exponential data. `pytest -q` runs 20 tests.

## Results (computed by `python examples/run_demo.py`; no patient data)

**PPV depends on prevalence** (sensitivity 0.90, specificity 0.95):

| prevalence | PPV | NPV |
|---|---|---|
| 0.001 | 0.018 | 0.9999 |
| 0.01 | 0.154 | 0.9989 |
| 0.1 | 0.667 | 0.9884 |
| 0.5 | 0.947 | 0.9048 |

Screening at 1% prevalence: probability of disease after one positive test 0.154, after a second independent positive 0.766.
Beta-binomial (7 successes in 20, uniform prior): posterior Beta(8, 14), mean 0.364, 95% credible interval 0.181 to 0.570.

**Exact coverage of nominal 95% binomial intervals** (computed by summing over all outcomes, no simulation):

| setting | Wald | Wilson | Clopper-Pearson |
|---|---|---|---|
| n=30,p=0.05 | 0.782 | 0.939 | 0.984 |
| n=30,p=0.2 | 0.946 | 0.964 | 0.980 |
| n=100,p=0.05 | 0.877 | 0.966 | 0.983 |
| n=100,p=0.5 | 0.943 | 0.943 | 0.965 |

The Wald interval under-covers badly for small n or extreme p; Clopper-Pearson is conservative.
Monte Carlo: pi estimated as 3.14137 from 10^6 points (standard error 0.00164); random-walk MSD / steps = 0.991 (theory 1).

## Limitations and future work
No regression models, survival analysis or non-parametric CIs beyond the percentile bootstrap (no BCa); Mann-Whitney uses the normal approximation (no exact small-sample p-values);
the Fisher test is 2x2 only. `adjust_pvalues` is duplicated in `clinical-prediction-eval` to keep both packages dependency-free from each other.

## References
* Welch (1947) Biometrika 34:28-35; Mann and Whitney (1947) Ann. Math. Stat. 18:50-60; Fisher (1935).
* Wilson (1927) JASA 22:209-212; Clopper and Pearson (1934) Biometrika 26:404-413; Brown, Cai and DasGupta (2001) Statistical Science 16:101-133.
* Holm (1979); Benjamini and Hochberg (1995).

## License
MIT, see `LICENSE`.
