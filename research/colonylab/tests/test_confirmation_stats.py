"""Independent paired-session arithmetic and dependence-aware uncertainty checks."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from colonylab.confirmation_stats import summarize_confirmation


def daily_frame(count=20, *, gain=.01):
    days = pd.bdate_range("2026-08-03", periods=count).strftime("%Y-%m-%d").tolist()
    rows = 100 + np.arange(count) * 7
    reference = .2 + np.arange(count) * .001
    return pd.DataFrame({
        "day": days, "rows": rows,
        "reference_log_loss_sum": rows * reference,
        "volume_log_loss_sum": rows * (reference - gain),
        "reference_brier_sum": rows * .05,
        "volume_brier_sum": rows * (.05 - gain / 10),
    })


def summarize(daily, **kwargs):
    return summarize_confirmation(daily, expected_days=daily.day.tolist(), **kwargs)


def test_pooled_losses_are_weighted_by_rows_not_an_average_of_daily_means():
    daily = daily_frame()
    result = summarize(daily, bootstrap_replicates=200)
    ref = daily.reference_log_loss_sum.sum() / daily.rows.sum()
    vol = daily.volume_log_loss_sum.sum() / daily.rows.sum()
    assert result["pooled"]["rows"] == daily.rows.sum()
    assert result["pooled"]["log_loss"]["reference"] == pytest.approx(ref)
    assert result["pooled"]["log_loss"]["volume"] == pytest.approx(vol)
    assert result["pooled"]["log_loss"]["absolute_gain"] == pytest.approx(ref - vol)
    assert result["pooled"]["log_loss"]["relative_gain_percent"] == pytest.approx(100 * (ref - vol) / ref)
    assert ref != pytest.approx((daily.reference_log_loss_sum / daily.rows).mean())


def test_identical_models_have_exact_zero_gains_and_paired_intervals():
    result = summarize(daily_frame(gain=0))
    assert result["status"] == "ok"
    assert result["positive_log_loss_days"] == 0
    assert result["tied_log_loss_days"] == 20
    assert not result["supports_positive_incremental_prediction"]
    for method in ("paired_day_bootstrap", "five_session_block_bootstrap"):
        for metric in ("log_loss", "brier"):
            assert result[method][metric]["absolute_gain_ci95"] == [0., 0.]
            assert result[method][metric]["relative_gain_percent_ci95"] == [0., 0.]


def test_always_positive_paired_gain_confirms_with_twenty_complete_dates():
    result = summarize(daily_frame())
    assert result["status"] == "ok"
    assert result["positive_log_loss_days"] == 20
    assert result["supports_positive_incremental_prediction"]
    assert result["five_session_block_bootstrap"]["log_loss"]["absolute_gain_ci95"] == pytest.approx([.01, .01])
    assert result["paired_day_bootstrap"]["seed"] == 20261026
    assert result["five_session_block_bootstrap"]["seed"] == 20261027
    assert result["five_session_block_bootstrap"]["blocks_per_replicate"] == 4
    assert not result["five_session_block_bootstrap"]["wraps_date_range"]


def test_always_negative_gain_is_not_confirmed():
    result = summarize(daily_frame(gain=-.01), bootstrap_replicates=200)
    assert result["negative_log_loss_days"] == 20
    assert result["pooled"]["log_loss"]["relative_gain_percent"] < 0
    assert not result["positive_block_interval"]
    assert not result["supports_positive_incremental_prediction"]


def test_weekly_dependence_increases_interval_width_for_regime_clustered_losses():
    daily = daily_frame()
    daily["rows"] = 1000
    daily["reference_log_loss_sum"] = 200.
    # Persistent runs deliberately give much less than twenty independent units.
    gains = np.r_[np.full(10, .03), np.full(10, -.03)]
    daily["volume_log_loss_sum"] = 200. - 1000 * gains
    result = summarize(daily)
    day_ci = result["paired_day_bootstrap"]["log_loss"]["absolute_gain_ci95"]
    block_ci = result["five_session_block_bootstrap"]["log_loss"]["absolute_gain_ci95"]
    assert block_ci[1] - block_ci[0] > day_ci[1] - day_ci[0]
    assert block_ci[0] < 0 < block_ci[1]
    assert not result["supports_positive_incremental_prediction"]


def test_bootstrap_keeps_model_losses_and_row_denominators_paired():
    daily = daily_frame()
    # Losses vary strongly by date, but the paired per-row effect is constant.
    # Independently drawing models or denominators would destroy this identity.
    daily["rows"] = np.geomspace(10, 100_000, 20).astype(int)
    daily["reference_log_loss_sum"] = daily.rows * np.linspace(.01, .7, 20)
    daily["volume_log_loss_sum"] = daily.reference_log_loss_sum - .002 * daily.rows
    result = summarize(daily)
    for method in ("paired_day_bootstrap", "five_session_block_bootstrap"):
        assert result[method]["log_loss"]["absolute_gain_ci95"] == pytest.approx([.002, .002], abs=1e-12)


def test_frozen_seeds_and_chronological_sort_make_results_reproducible():
    daily = daily_frame()
    before = summarize(daily, bootstrap_replicates=300)
    shuffled = daily.sample(frac=1, random_state=7)
    after = summarize(shuffled, bootstrap_replicates=300)
    assert before == after
    assert before["days"] == sorted(daily.day)
    json.dumps(before, allow_nan=False)


@pytest.mark.parametrize("column,value", [
    ("rows", 0), ("rows", -1), ("rows", .5),
    ("reference_log_loss_sum", np.nan),
    ("volume_log_loss_sum", np.inf),
    ("reference_log_loss_sum", -1),
    ("volume_brier_sum", 1e8),
    ("reference_brier_sum", -1),
])
def test_invalid_date_totals_prevent_partial_inference_instead_of_dropping_rows(column, value):
    daily = daily_frame()
    daily[column] = daily[column].astype(float)
    daily.loc[4, column] = value
    result = summarize(daily)
    assert result["status"] == "insufficient_coverage"
    assert result["coverage_errors"]
    assert "pooled" not in result
    assert "paired_day_bootstrap" not in result
    assert not result["supports_positive_incremental_prediction"]


def test_missing_intended_date_is_explicit_and_not_silently_excluded():
    daily = daily_frame()
    expected = daily.day.tolist()
    result = summarize_confirmation(daily.iloc[:-1], expected_days=expected)
    assert result["status"] == "insufficient_coverage"
    assert any(expected[-1] in error for error in result["coverage_errors"])
    assert "pooled" not in result


def test_duplicate_day_rejected_to_prevent_treating_venues_as_independent_dates():
    daily = daily_frame()
    repeated = pd.concat([daily, daily.iloc[:1]], ignore_index=True)
    result = summarize_confirmation(repeated, expected_days=daily.day.tolist())
    assert result["status"] == "insufficient_coverage"
    assert any("Duplicate held-out" in error for error in result["coverage_errors"])
    assert "pooled" not in result


@pytest.mark.parametrize("bad_day", ["not-a-date", "2026-08-08", "2026-8-3", "NaT"])
def test_invalid_or_weekend_date_rejected(bad_day):
    daily = daily_frame()
    daily.loc[0, "day"] = bad_day
    result = summarize(daily)
    assert result["status"] == "insufficient_coverage"
    assert "pooled" not in result


def test_short_sample_keeps_descriptive_math_but_cannot_confirm():
    result = summarize(daily_frame(4), bootstrap_replicates=100)
    assert result["status"] == "insufficient_coverage"
    assert result["pooled"]["log_loss"]["absolute_gain"] == pytest.approx(.01)
    assert result["five_session_block_bootstrap"]["effective_block_sessions"] == 4
    assert not result["supports_positive_incremental_prediction"]


def test_unknown_expected_dates_cannot_establish_complete_holdout():
    result = summarize_confirmation(daily_frame(), bootstrap_replicates=100)
    assert result["status"] == "insufficient_coverage"
    assert "pooled" in result
    assert not result["supports_positive_incremental_prediction"]
    assert any("not supplied" in error for error in result["coverage_errors"])


def test_zero_reference_losses_keep_absolute_intervals_and_undefined_relative_gain():
    daily = daily_frame(gain=0)
    for column in ("reference_log_loss_sum", "volume_log_loss_sum", "reference_brier_sum", "volume_brier_sum"):
        daily[column] = 0.
    result = summarize(daily, bootstrap_replicates=100)
    assert result["pooled"]["log_loss"]["relative_gain_percent"] is None
    assert result["paired_day_bootstrap"]["log_loss"]["absolute_gain_ci95"] == [0., 0.]
    assert result["paired_day_bootstrap"]["log_loss"]["relative_gain_percent_ci95"] is None
    json.dumps(result, allow_nan=False)


def test_missing_columns_and_empty_input_return_explicit_coverage_failures():
    assert summarize_confirmation(pd.DataFrame())["coverage_errors"]
    empty = daily_frame().iloc[:0]
    assert summarize_confirmation(empty)["coverage_errors"] == ["No held-out dates supplied."]


@pytest.mark.parametrize("kwargs", [{"bootstrap_replicates": 0}, {"bootstrap_replicates": True},
                                   {"block_length": 0}, {"seed": -1}])
def test_invalid_resampling_settings_raise(kwargs):
    with pytest.raises(ValueError):
        summarize(daily_frame(), **kwargs)
