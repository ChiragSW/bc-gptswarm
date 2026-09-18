"""Small-sample statistics used by the Phase 9 analysis.

The optimizer seed, not an individual question, is the independent unit for
policy-level comparisons.  Exact sign-flip tests are practical because the
study has only three paired seeds.  The item-level McNemar analysis is kept
separate and explicitly treated as secondary.
"""

from __future__ import annotations

from itertools import product
from math import inf
from typing import Iterable

import numpy as np
from scipy import stats


def exact_sign_flip_pvalue(differences: Iterable[float]) -> float:
    """Return the exact two-sided paired sign-flip p-value for a mean effect."""

    values = np.asarray(tuple(differences), dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("differences must be a non-empty finite vector")
    observed = abs(float(values.mean()))
    permuted = np.asarray(
        [abs(float(np.mean(values * signs))) for signs in product((-1.0, 1.0), repeat=values.size)]
    )
    return float(np.mean(permuted >= observed - 1e-12))


def paired_summary(differences: Iterable[float], confidence: float = 0.95) -> dict[str, float]:
    """Summarize paired differences with a Student-t interval and Cohen's dz."""

    values = np.asarray(tuple(differences), dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("at least two finite paired differences are required")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, df=values.size - 1))
    half_width = critical * sd / np.sqrt(values.size)
    if sd == 0.0:
        cohen_dz = 0.0 if mean == 0.0 else float(np.sign(mean) * inf)
    else:
        cohen_dz = mean / sd
    return {
        "n": int(values.size),
        "mean": mean,
        "sd": sd,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "cohen_dz": cohen_dz,
        "p_value": exact_sign_flip_pvalue(values),
    }


def exact_mcnemar(proposed_wins: int, baseline_wins: int) -> dict[str, float]:
    """Return exact McNemar inference and a conditional matched odds ratio."""

    if proposed_wins < 0 or baseline_wins < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = proposed_wins + baseline_wins
    if discordant == 0:
        return {
            "discordant": 0,
            "p_value": 1.0,
            "matched_odds_ratio": float("nan"),
            "odds_ratio_ci_low": float("nan"),
            "odds_ratio_ci_high": float("nan"),
        }
    test = stats.binomtest(proposed_wins, discordant, p=0.5, alternative="two-sided")
    interval = test.proportion_ci(confidence_level=0.95, method="exact")

    def odds(probability: float) -> float:
        if probability <= 0.0:
            return 0.0
        if probability >= 1.0:
            return inf
        return probability / (1.0 - probability)

    return {
        "discordant": discordant,
        "p_value": float(test.pvalue),
        "matched_odds_ratio": proposed_wins / baseline_wins if baseline_wins else inf,
        "odds_ratio_ci_low": odds(float(interval.low)),
        "odds_ratio_ci_high": odds(float(interval.high)),
    }


def question_cluster_bootstrap_ci(
    differences_by_question: Iterable[float],
    *,
    iterations: int = 20_000,
    seed: int = 20_260_918,
) -> tuple[float, float]:
    """Percentile CI after resampling question-level mean paired differences."""

    values = np.asarray(tuple(differences_by_question), dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("at least two finite question effects are required")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(iterations, values.size))
    estimates = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)
