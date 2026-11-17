"""Small statistics helpers for rates computed on small groups."""
import numpy as np
from scipy.stats import norm


def wilson_ci(successes, n, confidence=0.95):
    """Wilson score interval for a proportion; works element-wise on arrays.

    Better than the normal approximation for small n or rates near 0/1,
    which is the typical case here (e.g. 2 fatal accidents out of 14).
    """
    successes = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    z = norm.ppf(0.5 + confidence / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = successes / n
        denom = 1 + z ** 2 / n
        centre = (p + z ** 2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def shrunk_rate(successes, n, prior_rate, prior_strength=20):
    """Empirical-Bayes style rate: pulls groups with few observations toward
    the overall rate, so that 1 fatal out of 2 does not rank as "50% risk"."""
    successes = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    return (successes + prior_rate * prior_strength) / (n + prior_strength)
