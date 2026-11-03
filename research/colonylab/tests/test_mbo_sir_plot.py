"""Check that plotting cannot average incompatible cohorts or invent dates."""
import pandas as pd
import pytest

from colonylab.mbo_sir_plot import _paired_daily, _pooled_gain, _coverage_totals


def daily():
    return pd.DataFrame([
        {"day": "2026-06-01", "model": "ref", "rows": 100, "log_loss_sum": 20.},
        {"day": "2026-06-01", "model": "sirs_network", "rows": 100, "log_loss_sum": 10.},
        {"day": "2026-06-02", "model": "ref", "rows": 900, "log_loss_sum": 360.},
        {"day": "2026-06-02", "model": "sirs_network", "rows": 900, "log_loss_sum": 342.},
    ])


def test_pooled_figure_gain_uses_loss_totals_instead_of_average_daily_percentages():
    pairs = _paired_daily(daily(), "ref")
    assert pairs.gain_percent.tolist() == pytest.approx([50., 5.])
    assert _pooled_gain(pairs) == pytest.approx(100 * 28 / 380)
    assert _pooled_gain(pairs) != pytest.approx(pairs.gain_percent.mean())


def test_plot_pairing_rejects_unequal_denominators_or_missing_models():
    frame = daily()
    frame.loc[0, "rows"] += 1
    with pytest.raises(ValueError, match="unequal"):
        _paired_daily(frame, "ref")
    with pytest.raises(ValueError, match="Missing paired"):
        _paired_daily(daily().iloc[:-1], "ref")


def test_plot_pairing_does_not_silently_drop_whole_missing_dates():
    with pytest.raises(ValueError, match="exact intended"):
        _paired_daily(daily(), "ref", expected_days=["2026-06-01", "2026-06-02", "2026-06-03"])


def test_coverage_uses_full_intended_heldout_grid_and_weighted_counts():
    frame = pd.DataFrame([
        {"day": "2026-06-01", "dataset": "A", "candidate_bins": 100, "eligible_bins": 50},
        {"day": "2026-06-02", "dataset": "A", "candidate_bins": 900, "eligible_bins": 810},
        {"day": "2026-05-29", "dataset": "A", "candidate_bins": 999, "eligible_bins": 999},
    ])
    result = _coverage_totals(frame, days=["2026-06-01", "2026-06-02"], datasets=["A"])
    assert result.retention_percent.iloc[0] == 86.
    with pytest.raises(ValueError, match="incomplete"):
        _coverage_totals(frame.iloc[:1], days=["2026-06-01", "2026-06-02"], datasets=["A"])
