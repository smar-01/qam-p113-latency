"""Independent leakage, cohort, calibration and anchor-grid checks for confirmation."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from colonylab.fresh_confirmation import attach_volume_features, fit_training_calibration
from colonylab import fresh_confirmation as engine
from colonylab.network_models import build_prediction_rows


DAY = "2026-08-03"
ORIGIN = pd.Timestamp(f"{DAY}T14:00:00Z").value


def candle_frame(close_offsets_ms, *, dataset="A", day=DAY):
    origin = pd.Timestamp(f"{day}T14:00:00Z").value
    return pd.DataFrame([{
        "close_ns": origin + int(offset * 1_000_000),
        "close": 100., "vwap": 100.005, "return_bps": 1., "range_bps": 3.,
        "volume_shares": 100., "trade_count": 5., "duration_ms": 100.,
        "signed_volume": 20., "unknown_volume": 10.,
        "symbol": "TEST", "dataset": dataset, "day": day,
    } for offset in close_offsets_ms])


def forecast_rows(offsets_ms):
    return pd.DataFrame({
        "time_ns": ORIGIN + np.asarray(offsets_ms, dtype=np.int64) * 1_000_000,
        "dataset": "A", "symbol": "TEST", "day": DAY,
        "y": np.arange(len(offsets_ms)) % 2,
        "existing_network_feature": np.arange(len(offsets_ms), dtype=float),
    })


def test_volume_closing_print_is_unavailable_at_its_own_forecast_timestamp():
    rows = forecast_rows([1000, 1001])
    groups = {("A", "TEST", DAY): candle_frame([1000])}
    result, features, _ = attach_volume_features(
        rows, groups, {("A", "TEST"): 100}, venues=["A"])
    assert result.time_ns.tolist() == [ORIGIN + 1_001_000_000]
    assert len(features) == 9
    pd.testing.assert_frame_equal(result[rows.columns].reset_index(drop=True), rows.iloc[1:].reset_index(drop=True))


def test_volume_features_preserve_exact_age_boundary_and_exclude_stale_state():
    rows = forecast_rows([1001, 1002])
    groups = {("A", "TEST", DAY): candle_frame([1000])}
    result, _, _ = attach_volume_features(
        rows, groups, {("A", "TEST"): 100}, venues=["A"], max_age_ms=1.)
    assert result.time_ns.tolist() == [ORIGIN + 1_001_000_000]


def test_every_venue_contributes_to_one_common_finite_cohort():
    rows = forecast_rows([1500, 2500, 3500])
    groups = {("A", "TEST", DAY): candle_frame([1000, 2000, 3000]),
              ("B", "TEST", DAY): candle_frame([2000, 3000], dataset="B")}
    groups[("A", "TEST", DAY)].loc[2, "range_bps"] = np.nan
    result, features, coverage = attach_volume_features(
        rows, groups, {("A", "TEST"): 100, ("B", "TEST"): 100}, venues=["A", "B"])
    assert result.time_ns.tolist() == [ORIGIN + 2_500_000_000]
    assert len(features) == 18
    assert np.isfinite(result[features].to_numpy()).all()
    assert result.y.tolist() == [1]
    assert coverage


def test_missing_venue_candles_remove_rows_from_both_models():
    rows = forecast_rows([1500, 2500])
    result, _, _ = attach_volume_features(rows, {}, {("A", "TEST"): 100}, venues=["A"])
    assert result.empty


def test_unsafe_print_barrier_prevents_carry_until_a_new_completed_candle():
    rows = forecast_rows([1400, 1600, 2100])
    bars = candle_frame([1000, 2000])
    bars.attrs["barriers_ns"] = np.array([ORIGIN + 1_500_000_000], dtype=np.int64)
    result, _, _ = attach_volume_features(
        rows, {("A", "TEST", DAY): bars}, {("A", "TEST"): 100}, venues=["A"])
    assert result.time_ns.tolist() == [ORIGIN + 1_400_000_000, ORIGIN + 2_100_000_000]


def test_future_candles_do_not_change_earlier_forecasts():
    rows = forecast_rows([1100, 1900])
    thresholds = {("A", "TEST"): 100}
    before, features, _ = attach_volume_features(
        rows, {("A", "TEST", DAY): candle_frame([1000])}, thresholds, venues=["A"])
    after, after_features, _ = attach_volume_features(
        rows, {("A", "TEST", DAY): candle_frame([1000, 2000, 3000])}, thresholds, venues=["A"])
    assert features == after_features
    pd.testing.assert_frame_equal(before, after)


def quote_frame(dataset, symbol, day=DAY, *, start_ms=3, end_ms=3003):
    origin = pd.Timestamp(f"{day}T14:00:00Z").value
    offsets = np.arange(start_ms, end_ms + 1, 10)
    return pd.DataFrame({
        "time_ns": origin + offsets * 1_000_000, "mid": 100.,
        "valid": True, "segment": 0, "dataset": dataset, "symbol": symbol, "day": day,
    })


def quote_panel():
    return {(dataset, symbol, DAY): quote_frame(dataset, symbol)
            for dataset in ("A", "B") for symbol in ("TEST", "MARKET")}


def test_wider_anchor_spacing_keeps_unchanged_ten_millisecond_outcome():
    quotes = quote_panel()
    # t+5 ms is in the next-10-ms outcome; t+55 ms must not become its label.
    times = ORIGIN + np.array([1_105, 1_255]) * 1_000_000
    rows, _, _ = build_prediction_rows(quotes, {("A", "TEST", DAY): times}, "A", "TEST", DAY, anchor_step_ms=100)
    assert len(rows)
    assert np.all(rows.time_ns.to_numpy(dtype=np.int64) % 100_000_000 == 0)
    labels = rows.set_index("time_ns").y
    assert labels.loc[ORIGIN + 1_100_000_000] == 1
    assert labels.loc[ORIGIN + 1_200_000_000] == 0
    assert labels.sum() == 1


def test_sampled_anchors_are_a_subset_of_legacy_grid_with_identical_features_and_labels():
    quotes = quote_panel()
    events = {("A", "TEST", DAY): ORIGIN + np.array([1000, 1105, 1255]) * 1_000_000,
              ("B", "TEST", DAY): ORIGIN + np.array([1095, 1200, 1255]) * 1_000_000}
    old, base, network = build_prediction_rows(quotes, events, "A", "TEST", DAY)
    sparse, sparse_base, sparse_network = build_prediction_rows(
        quotes, events, "A", "TEST", DAY, anchor_step_ms=100)
    assert base == sparse_base and network == sparse_network
    matched = old.loc[old.time_ns.isin(sparse.time_ns)].reset_index(drop=True)
    pd.testing.assert_frame_equal(sparse.reset_index(drop=True), matched)


def test_current_or_future_cross_venue_shocks_never_enter_prior_counts():
    quotes = quote_panel()
    events = {("B", "TEST", DAY): ORIGIN + np.array([1100, 1105]) * 1_000_000}
    rows, _, _ = build_prediction_rows(quotes, events, "A", "TEST", DAY, anchor_step_ms=100)
    at_anchor = rows.loc[rows.time_ns.eq(ORIGIN + 1_100_000_000)].iloc[0]
    assert at_anchor.other_B_shocks_10ms == 0
    assert at_anchor.other_B_shocks_100ms == 0
    assert at_anchor.other_B_shocks_1000ms == 0


@pytest.mark.parametrize("step", [0, -1, True, 1.5])
def test_invalid_anchor_spacing_rejected(step):
    with pytest.raises(ValueError):
        build_prediction_rows(quote_panel(), {}, "A", "TEST", DAY, anchor_step_ms=step)


def prepared_payload(day, *, sizes=(6000,), jumps=(.01, .02, .03)):
    quotes = quote_frame("A", "TEST", day, start_ms=0, end_ms=30)
    quotes["mid"] = np.r_[100., 100. + np.cumsum(jumps)]
    origin = pd.Timestamp(f"{day}T14:00:00Z")
    trades = pd.DataFrame({
        "symbol": ["TEST"] * len(sizes), "dataset": ["A"] * len(sizes),
        "day": [day] * len(sizes), "valid": [True] * len(sizes), "size": sizes,
    })
    trades["valid"] = trades["valid"].astype(bool)
    trades.attrs.update(requested_start=origin.isoformat(), requested_end=(origin + pd.Timedelta(minutes=30)).isoformat(), requested_symbols=["TEST"])
    return {"quotes": quotes, "trades": trades}


def test_training_calibration_freezes_jump_quantile_and_counts_quiet_requested_exposure():
    training_days = ["2026-08-03", "2026-08-04"]
    prepared = [("A", training_days[0], prepared_payload(training_days[0], sizes=(18_000,))),
                ("A", training_days[1], prepared_payload(training_days[1], sizes=()))]
    shock, volume = fit_training_calibration(prepared, training_days, ["TEST"])
    assert shock.threshold.iloc[0] == pytest.approx(.0275)
    assert volume.training_seconds.iloc[0] == 3600
    assert volume.training_volume_shares.iloc[0] == 18_000
    assert volume.threshold_shares.iloc[0] == 5


def test_heldout_payload_is_skipped_before_any_calibration_access():
    training_days = [DAY]

    class UnreadableHeldout(dict):
        def __getitem__(self, key):
            raise AssertionError("Held-out payload was read during calibration")

    train = ("A", DAY, prepared_payload(DAY))
    before = fit_training_calibration([train], training_days, ["TEST"])
    after = fit_training_calibration([train, ("A", "2026-08-31", UnreadableHeldout())], training_days, ["TEST"])
    for original, changed in zip(before, after):
        pd.testing.assert_frame_equal(original, changed)


def test_duplicate_training_day_cannot_double_count_exposure():
    train = ("A", DAY, prepared_payload(DAY))
    with pytest.raises(ValueError):
        fit_training_calibration([train, train], [DAY], ["TEST"])


def test_missing_training_date_prevents_silent_recalibration_on_partial_history():
    with pytest.raises(ValueError):
        fit_training_calibration([("A", DAY, prepared_payload(DAY))], [DAY, "2026-08-04"], ["TEST"])


def write_model_panels(cache):
    config = {"training_days": ["2026-08-03", "2026-08-04"],
              "held_out_days": ["2026-08-31", "2026-09-01"]}
    panels = {}
    for day_index, day in enumerate(config["training_days"] + config["held_out_days"]):
        index = np.arange(60 + day_index * 7)
        x = np.column_stack((index % 3, index % 7 + day_index,
                             (index + day_index) % 5 == 0)).astype(float)
        y = ((index + day_index) % 5 == 0).astype(np.int8)
        panel = {"feature_names": ["reference_own", "reference_market", "volume_feature"],
                 "reference_feature_count": 2, "x": x, "y": y,
                 "time_ns": pd.Timestamp(f"{day}T14:00:00Z").value + index * 100_000_000}
        path = engine._panel_path(cache, "A", "TEST", day)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(panel))
        panels[day] = panel
    return config, panels


def test_fresh_two_model_scalers_see_only_identical_training_rows(tmp_path, monkeypatch):
    config, panels = write_model_panels(tmp_path)
    recorded = []
    original = engine.StandardScaler

    class RecordingScaler(original):
        def fit(self, x, y=None, sample_weight=None):
            recorded.append(np.asarray(x).copy())
            return super().fit(x, y, sample_weight=sample_weight)

    monkeypatch.setattr(engine, "StandardScaler", RecordingScaler)
    result = engine._fit_task(tmp_path, "A", "TEST", config)
    training = np.concatenate([panels[day]["x"] for day in config["training_days"]])
    assert len(recorded) == 2
    np.testing.assert_array_equal(recorded[0], training[:, :2])
    np.testing.assert_array_equal(recorded[1], training)
    assert result["train_rows"] == len(training)
    assert result["status"] == "ok"
    assert all(meta["converged"] for meta in result["convergence"].values())
    for model, count in (("reference", 2), ("volume", 3)):
        coefficients = [row for row in result["coefficients"] if row["model"] == model]
        assert len(coefficients) == count
        assert [row["training_mean"] for row in coefficients] == pytest.approx(training[:, :count].mean(axis=0))


def test_fresh_heldout_labels_or_feature_values_cannot_change_fitted_coefficients(tmp_path):
    config, panels = write_model_panels(tmp_path)
    before = engine._fit_task(tmp_path, "A", "TEST", config)
    for day in config["held_out_days"]:
        changed = panels[day]
        changed["y"] = 1 - changed["y"]
        changed["x"] += 10_000
        engine._panel_path(tmp_path, "A", "TEST", day).write_bytes(pickle.dumps(changed))
    after = engine._fit_task(tmp_path, "A", "TEST", config)
    assert before["coefficients"] == after["coefficients"]
    assert before["train_rows_by_day"] == after["train_rows_by_day"]
    assert [row["reference_log_loss_sum"] for row in before["daily"]] != [row["reference_log_loss_sum"] for row in after["daily"]]


def test_fresh_daily_metrics_match_independently_refitted_predictions(tmp_path):
    config, panels = write_model_panels(tmp_path)
    result = engine._fit_task(tmp_path, "A", "TEST", config)
    train_x = np.concatenate([panels[day]["x"] for day in config["training_days"]])
    train_y = np.concatenate([panels[day]["y"] for day in config["training_days"]])
    for model, count in (("reference", 2), ("volume", 3)):
        scaler = engine.StandardScaler().fit(train_x[:, :count])
        fitted = engine.LogisticRegression(C=1., solver="lbfgs", max_iter=1000,
                                          class_weight=None, random_state=0)
        fitted.fit(scaler.transform(train_x[:, :count]), train_y)
        for record in result["daily"]:
            panel = panels[record["day"]]
            p = fitted.predict_proba(scaler.transform(panel["x"][:, :count]))[:, 1]
            y = panel["y"]
            manual_log_loss = -(y * np.log(p) + (1 - y) * np.log1p(-p)).sum()
            manual_brier = np.square(y - p).sum()
            assert record["rows"] == len(y)
            assert record[f"{model}_log_loss_sum"] == pytest.approx(manual_log_loss)
            assert record[f"{model}_brier_sum"] == pytest.approx(manual_brier)


def test_fresh_zero_row_heldout_date_survives_as_explicit_zero_record(tmp_path):
    config, panels = write_model_panels(tmp_path)
    day = config["held_out_days"][0]
    panels[day].update(x=np.empty((0, 3)), y=np.empty(0, dtype=np.int8), time_ns=np.empty(0, dtype=np.int64))
    engine._panel_path(tmp_path, "A", "TEST", day).write_bytes(pickle.dumps(panels[day]))
    result = engine._fit_task(tmp_path, "A", "TEST", config)
    zero = next(row for row in result["daily"] if row["day"] == day)
    assert zero["rows"] == 0
    assert len(result["daily"]) == len(config["held_out_days"])
