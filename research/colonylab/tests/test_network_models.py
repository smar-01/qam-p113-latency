from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from colonylab import network_models as models


ORIGIN = pd.Timestamp("2026-10-21T14:00:00Z").value


def ns(milliseconds):
    return ORIGIN + np.asarray(milliseconds, dtype=np.int64) * 1_000_000


def quotes(day="2026-10-21", dataset="A", symbol="TEST"):
    times = ns(np.arange(0, 2010, 10))
    return pd.DataFrame({"time_ns": times, "mid": 100., "valid": True,
                         "segment": 0, "day": day, "dataset": dataset, "symbol": symbol})


def panels():
    groups = {(dataset, symbol, "2026-10-21"): quotes(dataset=dataset, symbol=symbol)
              for dataset in ("A", "B") for symbol in ("TEST", "SPY")}
    return groups


def test_lag_counts_are_half_open_at_current_timestamp_and_include_window_start():
    anchor = ns([1000])
    events = np.array([anchor[0] - 10_000_001, anchor[0] - 10_000_000,
                       anchor[0] - 1, anchor[0], anchor[0] + 1], dtype=np.int64)
    assert models.lag_counts(events, anchor, 10).tolist() == [2]


def test_outcome_includes_anchor_and_excludes_next_bin_boundary():
    anchor = ns([1000, 1010, 1020])
    assert models.future_labels(ns([1000, 1010]), anchor).tolist() == [1, 1, 0]


def test_horizon_support_cannot_be_extrapolated_from_last_quote():
    q = quotes()
    anchors = ns([1990, 2000])
    assert models.quote_coverage(q, anchors, future_ms=10).tolist() == [True, False]


def test_full_past_window_is_required_even_if_current_baseline_is_fresh():
    q = quotes()
    assert models.quote_coverage(q, ns([990, 1000])).tolist() == [False, True]


def test_invalid_historical_quote_blocks_until_entire_past_window_is_clean():
    q = quotes()
    q.loc[50, "valid"] = False
    q.loc[50, "mid"] = np.nan
    assert models.quote_coverage(q, ns([1200, 1600]), future_ms=10).tolist() == [False, True]


def test_segment_change_cannot_bridge_history_or_target_outcome():
    q = quotes()
    q.loc[120:, "segment"] = 1
    assert not models.quote_coverage(q, ns([1190]), future_ms=20).item()
    assert not models.quote_coverage(q, ns([1500]), future_ms=10).item()


def test_barrier_at_next_bin_boundary_is_assigned_to_that_bin():
    q = quotes()
    q.loc[121, ["valid", "mid"]] = [False, np.nan]
    assert models.quote_coverage(q, ns([1200]), future_ms=10).item()
    assert not models.quote_coverage(q, ns([1210]), future_ms=10).item()


def test_features_exclude_current_and_future_other_venue_and_market_events():
    groups = panels()
    events = {key: ns([1190, 1200, 1250]) for key in groups}
    rows, baseline, additional = models.build_prediction_rows(groups, events, "A", "TEST", "2026-10-21")
    current = rows.loc[rows.time_ns == ns([1200])[0]].iloc[0]
    assert current.y == 1
    assert current.own_shocks_10ms == 1
    assert current.other_B_shocks_10ms == 1
    assert current.market_SPY_shocks_10ms == 2
    assert not set(additional) & set(baseline)
    assert "market_SPY_shocks_10ms" in baseline
    assert "other_B_shocks_10ms" in additional
    perturbed = {key: ns([1190, 1200, 1201, 1202, 1203, 1800]) for key in groups}
    altered, _, _ = models.build_prediction_rows(groups, perturbed, "A", "TEST", "2026-10-21")
    after = altered.loc[altered.time_ns == ns([1200])[0]].iloc[0]
    np.testing.assert_array_equal(current[baseline + additional], after[baseline + additional])


def test_bad_state_on_predictor_venue_censors_identical_rows_for_both_models():
    groups = panels()
    groups[("B", "TEST", "2026-10-21")].loc[50, ["mid", "valid"]] = [np.nan, False]
    rows, _, _ = models.build_prediction_rows(groups, {}, "A", "TEST", "2026-10-21")
    assert not rows.time_ns.isin(ns([1200])).any()
    assert rows.time_ns.isin(ns([1600])).any()


def test_market_proxy_invalid_history_also_censors_model_rows():
    groups = panels()
    groups[("B", "SPY", "2026-10-21")].loc[50, ["mid", "valid"]] = [np.nan, False]
    rows, _, _ = models.build_prediction_rows(groups, {}, "A", "TEST", "2026-10-21")
    assert not rows.time_ns.isin(ns([1200])).any()


def test_spy_baseline_uses_other_stock_proxies_instead_of_spy_network_events():
    groups = {(dataset, symbol, "2026-10-21"): quotes(dataset=dataset, symbol=symbol)
              for dataset in ("A", "B") for symbol in ("AAPL", "MSFT", "SPY")}
    _, baseline, additional = models.build_prediction_rows(groups, {}, "A", "SPY", "2026-10-21")
    assert "market_AAPL_shocks_100ms" in baseline
    assert "market_MSFT_shocks_100ms" in baseline
    assert "other_B_shocks_100ms" in additional
    assert not any("market_SPY" in item for item in baseline)


def model_rows():
    rows = []
    for day in ("2026-10-21", "2026-10-22", "2026-10-23"):
        for i in range(80):
            rows.append({"dataset": "A", "symbol": "TEST", "day": day,
                         "time_ns": ORIGIN + i * 10_000_000, "y": int(i % 5 == 0),
                         "own": float(i % 3), "network_signal": float(i % 5 == 0)})
    return pd.DataFrame(rows)


def test_scalers_are_fit_on_training_rows_only_and_both_models_score_same_bins():
    rows = model_rows()
    records = []
    original = models.StandardScaler

    class RecordingScaler(original):
        def fit(self, x, y=None, sample_weight=None):
            records.append(np.asarray(x).copy())
            return super().fit(x, y, sample_weight=sample_weight)

    with patch.object(models, "StandardScaler", RecordingScaler):
        result, predictions, coefficients = models.fit_prediction_models(
            rows, ["own"], ["network_signal"], ["2026-10-21", "2026-10-22"], "2026-10-23")
    training = rows.loc[rows.day != "2026-10-23"]
    np.testing.assert_array_equal(records[0], training[["own"]].to_numpy())
    np.testing.assert_array_equal(records[1], training[["own", "network_signal"]].to_numpy())
    assert result["test_rows"] == len(predictions) == 80
    assert predictions[["baseline_probability", "network_probability"]].notna().all().all()
    assert coefficients.model.nunique() == 2
    assert result["delta_log_loss"] > 0
    assert result["delta_brier"] > 0


def test_heldout_label_changes_cannot_change_fitted_predictions_or_coefficients():
    rows = model_rows()
    args = (["own"], ["network_signal"], ["2026-10-21", "2026-10-22"], "2026-10-23")
    _, before, before_coef = models.fit_prediction_models(rows, *args)
    rows.loc[rows.day == "2026-10-23", "y"] = 1 - rows.loc[rows.day == "2026-10-23", "y"]
    _, after, after_coef = models.fit_prediction_models(rows, *args)
    pd.testing.assert_frame_equal(before.drop(columns="y"), after.drop(columns="y"))
    pd.testing.assert_frame_equal(before_coef, after_coef)


def test_heldout_features_cannot_change_training_scales_or_fitted_coefficients():
    rows = model_rows()
    args = (["own"], ["network_signal"], ["2026-10-21", "2026-10-22"], "2026-10-23")
    _, before, before_coef = models.fit_prediction_models(rows, *args)
    rows.loc[rows.day == "2026-10-23", ["own", "network_signal"]] += 1000.
    _, after, after_coef = models.fit_prediction_models(rows, *args)
    pd.testing.assert_frame_equal(before_coef, after_coef)
    assert not before.network_probability.equals(after.network_probability)


def test_canonical_frame_interface_skips_mbo_and_exports_fixed_holdout_results(tmp_path):
    frames, event_rows = {}, []
    for day in ("2026-10-21", "2026-10-22", "2026-10-23"):
        origin = pd.Timestamp(f"{day}T14:00:00Z")
        for dataset in ("A", "B"):
            blocks = []
            for symbol in ("TEST", "SPY"):
                count = 201
                blocks.append(pd.DataFrame({
                    "ts_recv": origin + pd.to_timedelta(np.arange(count) * 10, unit="ms"),
                    "ts_event": origin + pd.to_timedelta(np.arange(count) * 10, unit="ms"),
                    "symbol": symbol, "bid_px_00": 99.99, "ask_px_00": 100.01,
                    "bid_sz_00": 100, "ask_sz_00": 100, "flags": 128,
                }))
                for offset in (1200, 1400):
                    event_rows.append({"dataset": dataset, "symbol": symbol, "day": day,
                                       "time_ns": origin.value + offset * 1_000_000})
            frames[(dataset, "mbp-1", day)] = pd.concat(blocks, ignore_index=True)
            # Deliberately not quote-shaped: this must never be normalized.
            frames[(dataset, "mbo", day)] = pd.DataFrame({"action": ["A"]})
    shock_path = tmp_path / "shock_events.csv"
    pd.DataFrame(event_rows).to_csv(shock_path, index=False)
    summary = models.run_prediction_test(frames, shock_path, tmp_path)
    assert summary["training_days"] == ["2026-10-21", "2026-10-22"]
    assert summary["held_out_day"] == "2026-10-23"
    assert summary["pooled_test_bins"] == 400
    assert len(summary["tasks"]) == 4
    assert (tmp_path / "prediction_summary.json").is_file()
    predictions = pd.read_csv(tmp_path / "prediction_test_bins.csv")
    assert predictions.day.unique().tolist() == ["2026-10-23"]
    assert predictions.groupby(["dataset", "symbol"]).size().eq(100).all()


def test_single_class_training_falls_back_to_constant_without_undefined_metrics():
    rows = model_rows()
    rows.loc[rows.day != "2026-10-23", "y"] = 0
    result, predictions, coefficients = models.fit_prediction_models(
        rows, ["own"], ["network_signal"], ["2026-10-21", "2026-10-22"], "2026-10-23")
    assert result["status"] == "one_training_class_constant_fallback"
    assert predictions.baseline_probability.eq(0).all()
    assert predictions.network_probability.eq(0).all()
    assert coefficients.empty
    assert result["delta_log_loss"] == 0
