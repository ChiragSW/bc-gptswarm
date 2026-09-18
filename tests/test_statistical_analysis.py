import math

import pytest

from gptswarm_ext.statistical_analysis import exact_mcnemar, exact_sign_flip_pvalue, paired_summary


def test_exact_sign_flip_resolution_for_three_same_direction_pairs():
    assert exact_sign_flip_pvalue([1.0, 1.0, 1.0]) == 0.25


def test_paired_summary_reports_direction_and_finite_interval():
    summary = paired_summary([-1.0, -2.0, 0.5])
    assert summary["mean"] == pytest.approx(-5.0 / 6.0)
    assert summary["ci_low"] < summary["mean"] < summary["ci_high"]
    assert math.isfinite(summary["cohen_dz"])


def test_exact_mcnemar_matches_observed_discordances():
    result = exact_mcnemar(proposed_wins=2, baseline_wins=3)
    assert result["discordant"] == 5
    assert result["p_value"] == 1.0
    assert result["matched_odds_ratio"] == pytest.approx(2.0 / 3.0)
