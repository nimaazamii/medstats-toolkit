"""Diagnostic-test probabilities and simple Bayesian risk updating."""
from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import beta


def _prob(x: float, name: str) -> None:
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")


def predictive_values(sensitivity: float, specificity: float, prevalence: float) -> dict[str, float]:
    """PPV and NPV by Bayes' theorem, plus likelihood ratios."""
    for v, n in ((sensitivity, "sensitivity"), (specificity, "specificity"), (prevalence, "prevalence")):
        _prob(v, n)
    tp, fp = sensitivity * prevalence, (1 - specificity) * (1 - prevalence)
    tn, fn = specificity * (1 - prevalence), (1 - sensitivity) * prevalence
    return {
        "ppv": tp / (tp + fp) if tp + fp else float("nan"),
        "npv": tn / (tn + fn) if tn + fn else float("nan"),
        "lr_positive": sensitivity / (1 - specificity) if specificity < 1 else float("inf"),
        "lr_negative": (1 - sensitivity) / specificity if specificity > 0 else float("inf"),
    }


def posterior_from_likelihood_ratio(prior: float, likelihood_ratio: float) -> float:
    """Update a prior probability with a likelihood ratio through the odds form of Bayes' theorem."""
    _prob(prior, "prior")
    if prior == 1.0:
        return 1.0
    odds = prior / (1 - prior) * likelihood_ratio
    return odds / (1 + odds)


@dataclass(frozen=True)
class BetaPosterior:
    a: float
    b: float

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        return float(beta.ppf((1 - level) / 2, self.a, self.b)), float(beta.ppf(1 - (1 - level) / 2, self.a, self.b))


def beta_binomial_update(successes: int, trials: int, prior_a: float = 1.0, prior_b: float = 1.0) -> BetaPosterior:
    """Conjugate update of a Beta(prior_a, prior_b) prior for a proportion after ``successes`` in ``trials``."""
    if not 0 <= successes <= trials or prior_a <= 0 or prior_b <= 0:
        raise ValueError("invalid counts or prior")
    return BetaPosterior(prior_a + successes, prior_b + trials - successes)
