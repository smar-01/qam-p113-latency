"""Independent whole-date arithmetic and support checks for latency inference."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from colonylab.latency_validation import infer_latency, moving_block_counts, render_latency_figures


def daily_frame(count=20):
    days = pd.bdate_range("2026-08-31", periods=count).strftime("%Y-%m-%d")
    rows = []
    for index, day in enumerate(days):
        for clock in ("ts_recv", "ts_event"):
            for regime, latency in (("all", 250. + index), ("calm", 200. + index),
                                    ("volatile", 300. + index)):
                rows.append({"study": "fresh_confirmation", "phase": "held_out", "date": day,
                             "clock": clock, "symbol": "AAPL", "leader": "XNAS.ITCH",
                             "follower": "BATS.PITCH", "horizon_ms": 10., "regime": regime,
                             "n_matched": 100 + index * 200, "n_censored": 100, "median_us": latency})
    return pd.DataFrame(rows)


def result_for(result, contrast="latency", clock="ts_recv"):
    return result.loc[result.contrast.eq(contrast) & result.clock.eq(clock)].iloc[0]


def test_equal_date_estimate_does_not_weight_high_traffic_dates():
    daily = daily_frame()
    result = infer_latency(daily, bootstrap_replicates=100)
    row = result_for(result)
    assert row.estimate_us == pytest.approx(259.5)
    source = daily.loc[daily.regime.eq("all") & daily.clock.eq("ts_recv")]
    event_weighted = np.average(source.median_us, weights=source.n_matched)
    assert row.estimate_us != pytest.approx(event_weighted)
    assert row.n_dates == 20 and row.interval_status == "ok"


def test_constant_paired_effect_preserves_dates_and_exact_interval():
    result = infer_latency(daily_frame(), bootstrap_replicates=200)
    row = result_for(result, "volatile_vs_calm")
    assert row.n_dates == 20
    assert row.calm_us == pytest.approx(209.5)
    assert row.volatile_us == pytest.approx(309.5)
    assert row.estimate_us == pytest.approx(100.)
    assert row.relative_change_percent == pytest.approx(100 * 100 / 209.5)
    assert [row.ci95_low_us, row.ci95_high_us] == pytest.approx([100., 100.])


def test_relative_estimate_is_ratio_of_paired_means_not_mean_of_ratios():
    daily = daily_frame(2)
    daily.loc[daily.regime.eq("calm"), "median_us"] = 10.
    second = daily.date.eq(sorted(daily.date.unique())[1])
    daily.loc[second & daily.regime.eq("calm"), "median_us"] = 100.
    daily.loc[daily.regime.eq("volatile"), "median_us"] = daily.loc[daily.regime.eq("calm"), "median_us"].to_numpy() + 20.
    result = infer_latency(daily, bootstrap_replicates=100, min_inference_dates=2, block_sessions=1)
    row = result_for(result, "volatile_vs_calm")
    assert row.relative_change_percent == pytest.approx(100 * 20 / 55)
    assert row.relative_change_percent != pytest.approx(np.mean([200., 20.]))


def test_regime_estimate_uses_only_dates_supported_in_both_regimes():
    daily = daily_frame(4)
    dates = sorted(daily.date.unique())
    daily.loc[daily.date.eq(dates[0]) & daily.regime.eq("calm"), "n_matched"] = 19
    daily.loc[daily.date.eq(dates[1]) & daily.regime.eq("volatile"), "n_matched"] = 19
    result = infer_latency(daily, bootstrap_replicates=100, min_inference_dates=2, block_sessions=1)
    row = result_for(result, "volatile_vs_calm")
    assert row.n_dates == 2
    assert row.n_missing_or_sparse_dates == 2
    assert row.calm_us == pytest.approx(202.5)
    assert row.volatile_us == pytest.approx(302.5)
    assert row.estimate_us == pytest.approx(100.)
    assert result_for(result).n_dates == 4


def test_missing_calendar_date_prevents_intervals_and_remains_explicit():
    daily = daily_frame()
    dates = sorted(daily.date.unique())
    result = infer_latency(daily.loc[~daily.date.eq(dates[0])], bootstrap_replicates=100,
                           expected_dates={"fresh_confirmation:held_out": dates})
    assert set(result.interval_status) == {"missing_intended_dates"}
    assert (result.n_calendar_dates == 20).all()
    assert (result.n_observed_calendar_dates == 19).all()
    assert result.ci95_low_us.isna().all()


def test_nineteen_supported_dates_remain_descriptive_even_with_twenty_calendar_dates():
    daily = daily_frame()
    mask = daily.date.eq(sorted(daily.date.unique())[0]) & daily.regime.eq("all")
    daily.loc[mask, "n_matched"] = 19
    result = infer_latency(daily, bootstrap_replicates=100)
    row = result_for(result)
    assert row.n_dates == 19
    assert row.interval_status == "insufficient_supported_dates"
    assert np.isnan(row.ci95_low_us)
    assert result_for(result, "volatile_vs_calm").interval_status == "ok"


def test_studies_and_phases_are_not_mixed_for_uncertainty():
    daily = daily_frame()
    training = daily.copy()
    training["phase"] = "training"
    training["median_us"] *= 2
    wide = daily.copy()
    wide["study"] = "wide_network"
    result = infer_latency(pd.concat([daily, training, wide]), bootstrap_replicates=100)
    primary = result.loc[result.inference_role.eq("primary_post_hoc")]
    descriptive = result.loc[result.inference_role.eq("descriptive")]
    assert len(primary) == 4
    assert (descriptive.interval_status == "descriptive").all()
    assert descriptive.ci95_low_us.isna().all()
    assert result_for(primary).estimate_us == pytest.approx(259.5)


def test_missing_phase_defaults_to_descriptive_all():
    result = infer_latency(daily_frame().drop(columns="phase"), bootstrap_replicates=100)
    assert set(result.phase) == {"all"}
    assert set(result.interval_status) == {"descriptive"}


def test_shared_date_draws_preserve_scaled_tasks_and_row_order_invariance():
    daily = daily_frame()
    doubled = daily.copy()
    doubled["symbol"] = "MSFT"
    doubled["median_us"] *= 2
    whole = pd.concat([daily, doubled])
    result = infer_latency(whole, bootstrap_replicates=250)
    shuffled = infer_latency(whole.sample(frac=1, random_state=6), bootstrap_replicates=250)
    pd.testing.assert_frame_equal(result, shuffled)
    first = result.loc[result.symbol.eq("AAPL")].reset_index(drop=True)
    second = result.loc[result.symbol.eq("MSFT")].reset_index(drop=True)
    for field in ("estimate_us", "ci95_low_us", "ci95_high_us"):
        np.testing.assert_allclose(second[field], 2 * first[field])
    np.testing.assert_allclose(second.relative_change_percent, first.relative_change_percent, equal_nan=True)


def test_zero_calm_denominator_keeps_absolute_effect_without_invented_percentages():
    daily = daily_frame()
    daily.loc[daily.regime.eq("calm"), "median_us"] = 0.
    result = infer_latency(daily, bootstrap_replicates=100)
    row = result_for(result, "volatile_vs_calm")
    assert row.interval_status == "ok"
    assert np.isfinite(row.estimate_us)
    assert np.isnan(row.relative_change_percent)
    assert np.isnan(row.relative_ci95_low_percent)


def test_a_missing_regime_does_not_remove_tasks_or_borrow_other_dates():
    daily = daily_frame()
    daily = daily.loc[~daily.regime.eq("volatile")]
    result = infer_latency(daily, bootstrap_replicates=100)
    row = result_for(result, "volatile_vs_calm")
    assert row.n_dates == 0
    assert np.isnan(row.estimate_us)
    assert row.interval_status == "insufficient_supported_dates"
    assert result_for(result).n_dates == 20


def test_block_counts_are_whole_date_draws_and_do_not_wrap():
    count = moving_block_counts(4, replicates=20, block_sessions=4, seed=7)
    np.testing.assert_array_equal(count, np.ones((20, 4), dtype=np.int32))
    # With three dates and blocks of two, the third slot is the first slot
    # of a fresh block (0 or 1), never a wrapped continuation from the end.
    seed, reps = 12, 100
    starts = np.random.default_rng(seed).integers(0, 2, size=(reps, 2))
    expected = np.zeros((reps, 3), dtype=np.int32)
    for row, (first, second) in enumerate(starts):
        for index in [first, first + 1, second]:
            expected[row, index] += 1
    actual = moving_block_counts(3, replicates=reps, block_sessions=2, seed=seed)
    np.testing.assert_array_equal(actual, expected)
    assert (actual.sum(axis=1) == 3).all()


@pytest.mark.parametrize("field,value", [("n_matched", -1), ("n_matched", .5),
                                         ("median_us", np.inf), ("median_us", -1),
                                         ("median_us", 10_001), ("horizon_ms", 0)])
def test_invalid_summary_arithmetic_is_rejected(field, value):
    daily = daily_frame()
    daily[field] = daily[field].astype(float)
    daily.loc[0, field] = value
    with pytest.raises(ValueError):
        infer_latency(daily)


def test_duplicate_task_dates_are_rejected_instead_of_inflating_support():
    daily = daily_frame()
    with pytest.raises(ValueError, match="Duplicate"):
        infer_latency(pd.concat([daily, daily.iloc[:1]]))


def test_unexpected_dates_cannot_be_silently_added_to_intended_calendar():
    daily = daily_frame()
    with pytest.raises(ValueError, match="outside"):
        infer_latency(daily, expected_dates={"fresh_confirmation:held_out": sorted(daily.date.unique())[:-1]})


def test_plot_export_from_completed_tables_uses_actual_inputs(tmp_path):
    daily = daily_frame()
    inferred = infer_latency(daily, bootstrap_replicates=100)
    files = render_latency_figures(daily, inferred, tmp_path)
    assert {path.name for path in files} == {
        "latency_heatmap.png", "latency_regimes.png", "latency_horizons.png", "latency_clock_comparison.png"}
    assert all(path.is_file() and path.stat().st_size > 1000 for path in files)
