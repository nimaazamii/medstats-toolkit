"""Multiple-testing corrections."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def adjust_pvalues(p_values: ArrayLike, method: str = "holm") -> NDArray[np.float64]:
    """'bonferroni', 'holm' (FWER) or 'bh' (Benjamini-Hochberg FDR) adjusted p-values, in input order."""
    p = np.asarray(p_values, float)
    if p.ndim != 1 or np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("p_values must be a 1-D array in [0, 1]")
    m, order = p.size, np.argsort(p)
    ps = p[order]
    if method == "bonferroni":
        adj = ps * m
    elif method == "holm":
        adj = np.maximum.accumulate(ps * (m - np.arange(m)))
    elif method == "bh":
        adj = np.minimum.accumulate((ps * m / (np.arange(m) + 1))[::-1])[::-1]
    else:
        raise ValueError("method must be 'bonferroni', 'holm' or 'bh'")
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out
