"""Independent leakage, common-sample and score checks for candle comparisons."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from colonylab import network_models
from colonylab import volume_prediction as comparison


DAY = "2026-10-05"
ORIGIN = pd.Timestamp(f"{DAY}T14:00:00Z").value
TRAINING_DAYS = ["2026-10-05", "2026-10-06"]
HELD_OUT_DAYS = ["2026-10-07", "2026-10-08"]


def candle_frame(close_offsets_ms, *, dataset="A", day=DAY):
    origin = pd.Timestamp(f"{day}T14:00:00Z").value
    records = []
    for i, offset in enumerate(close_offsets_ms):
        close = origin + int(offset * 1_000_000)
        records.append({
            "open_ns": close - 100_000_000, "close_ns": close,
            "open": 100.0, "high": 100.02, "low": 99.99,
            "close": 100.01, "vwap": 100.005,
            "volume_shares": 100.0 + i, "trade_count": 5 + i,
            "signed_volume": 20.0, "unknown_volume": 10.0,
            "duration_ms": 100.0, "return_bps": 1.0,
            "range_bps": 3.0, "overshoot_shares": 0.0,
            "segment": 0, "symbol": "TEST", "dataset": dataset,
            "day": day,
        })
    return pd.DataFrame(records)


def input_rows(offsets_ns):
    return pd.DataFrame({
        "time_ns": ORIGIN + np.asarray(offsets_ns, dtype=np.int64),
        "dataset": "A", "symbol": "TEST", "day": DAY,
        "y": np.arange(len(offsets_ns)) % 2,
        "own": np.arange(len(offsets_ns), dtype=float),
    })


def dictionaries(*, time_close=(1000,), volume_close=(1000,), venues=("A",)):
    time = {(venue, "TEST", DAY): candle_frame(time_close, dataset=venue) for venue in venues}
    volume = {(venue, "TEST", DAY): candle_frame(volume_close, dataset=venue) for venue in venues}
    thresholds = {(venue, "TEST"): 100.0 for venue in venues}
    return time, volume, thresholds


def test_completed_candle_cannot_enter_at_its_own_close_timestamp():
    rows = input_rows([1_000_000_000, 1_000_000_001, 1_001_000_000])
    time, volume, thresholds = dictionaries()
    result, time_features, volume_features, diagnostics = comparison.attach_candle_features(
        rows, time, volume, thresholds, venues=["A"])
    assert result.time_ns.tolist() == [ORIGIN + 1_000_000_001, ORIGIN + 1_001_000_000]
    assert len(time_features) == len(volume_features) == 9
    assert set(time_features).isdisjoint(volume_features)
    assert np.isfinite(result[time_features + volume_features].to_numpy()).all()
    assert isinstance(diagnostics, dict)
    # Outcome and original predictors survive attaching the completed past.
    pd.testing.assert_frame_equal(result[rows.columns].reset_index(drop=True), rows.iloc[1:].reset_index(drop=True))


def test_scheduled_time_candle_can_enter_at_interval_end_with_prior_volume_candle():
    # Time candles contain prints in [start,end), so their scheduled close is
    # known at t=end. Volume closes include the triggering print at close.
    rows = input_rows([1_000_000_000])
    time, volume, thresholds = dictionaries(time_close=(1000,), volume_close=(900,))
    result, time_features, volume_features, _ = comparison.attach_candle_features(
        rows, time, volume, thresholds, venues=["A"])
    assert result.time_ns.tolist() == [ORIGIN + 1_000_000_000]
    assert np.isfinite(result[time_features + volume_features].to_numpy()).all()


def test_volume_candle_at_anchor_remains_unavailable_with_prior_time_candle():
    rows = input_rows([1_000_000_000, 1_000_000_001])
    time, volume, thresholds = dictionaries(time_close=(900,), volume_close=(1000,))
    result, _, _, _ = comparison.attach_candle_features(rows, time, volume, thresholds, venues=["A"])
    assert result.time_ns.tolist() == [ORIGIN + 1_000_000_001]


def test_staleness_uses_strictly_past_close_and_includes_exact_age_cap():
    rows = input_rows([1_001_000_000, 1_001_000_001])
    time, volume, thresholds = dictionaries()
    result, _, _, _ = comparison.attach_candle_features(
        rows, time, volume, thresholds, venues=["A"], max_age_ms=1)
    assert result.time_ns.tolist() == [ORIGIN + 1_001_000_000]


def test_all_candle_families_and_venues_share_one_finite_sample():
    rows = input_rows([1_500_000_000, 2_500_000_000, 3_500_000_000])
    time, volume, thresholds = dictionaries(time_close=(1000, 2000, 3000), venues=("A", "B"))
    # The second venue's volume candle becomes observable only after 2 sec.
    volume[("B", "TEST", DAY)] = candle_frame([2000, 3000], dataset="B")
    # A finite-time feature and finite-volume feature must be required together.
    time[("A", "TEST", DAY)].loc[2, ["range_bps", "high"]] = np.nan
    result, time_features, volume_features, _ = comparison.attach_candle_features(
        rows, time, volume, thresholds, venues=["A", "B"])
    assert result.time_ns.tolist() == [ORIGIN + 2_500_000_000]
    assert len(time_features) == len(volume_features) == 18
    assert np.isfinite(result[time_features + volume_features].to_numpy()).all()
    assert result.y.tolist() == [1]


def test_appending_future_candles_does_not_change_existing_anchor_features():
    rows = input_rows([1_100_000_000, 1_900_000_000])
    time, volume, thresholds = dictionaries()
    before, time_features, volume_features, _ = comparison.attach_candle_features(
        rows, time, volume, thresholds, venues=["A"])
    time[("A", "TEST", DAY)] = candle_frame([1000, 2000, 3000])
    volume[("A", "TEST", DAY)] = candle_frame([1000, 2000, 3000])
    after, after_time, after_volume, _ = comparison.attach_candle_features(
        rows, time, volume, thresholds, venues=["A"])
    assert time_features == after_time and volume_features == after_volume
    pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))


def test_unsafe_trade_barrier_invalidates_both_families_until_new_completed_candle():
    rows = input_rows([1_400_000_000, 1_600_000_000, 2_100_000_000])
    time, volume, thresholds = dictionaries(time_close=(1000, 2000), volume_close=(1000, 2000))
    volume[("A", "TEST", DAY)].attrs["barriers_ns"] = np.array([ORIGIN + 1_500_000_000])
    result, _, _, _ = comparison.attach_candle_features(rows, time, volume, thresholds, venues=["A"])
    assert result.time_ns.tolist() == [ORIGIN + 1_400_000_000, ORIGIN + 2_100_000_000]


def model_rows():
    records = []
    for day_index, day in enumerate(TRAINING_DAYS + HELD_OUT_DAYS):
        origin = pd.Timestamp(f"{day}T14:00:00Z").value
        for i in range(70 + day_index * 7):
            records.append({
                "dataset": "A", "symbol": "TEST", "day": day,
                "time_ns": origin + i * 10_000_000,
                "y": int((i + day_index) % 5 == 0),
                "reference_lag": float(i % 3),
                "time_candle": float((i + day_index) % 7),
                "volume_candle": float((i + day_index) % 5 == 0),
            })
    return pd.DataFrame(records)


def test_all_three_models_fit_scalers_only_on_training_dates_and_same_rows():
    rows = model_rows()
    recorded = []
    original = comparison.StandardScaler

    class RecordingScaler(original):
        def fit(self, x, y=None, sample_weight=None):
            recorded.append(np.asarray(x).copy())
            return super().fit(x, y, sample_weight=sample_weight)

    with patch.object(comparison, "StandardScaler", RecordingScaler):
        result, predictions, coefficients = comparison.fit_candle_models(
            rows, ["reference_lag"], ["time_candle"], ["volume_candle"],
            TRAINING_DAYS, HELD_OUT_DAYS)
    training = rows.loc[rows.day.isin(TRAINING_DAYS)]
    held = rows.loc[rows.day.isin(HELD_OUT_DAYS)]
    assert len(recorded) == 3
    for fitted, features in zip(recorded, [
        ["reference_lag"], ["reference_lag", "time_candle"], ["reference_lag", "volume_candle"],
    ]):
        np.testing.assert_array_equal(fitted, training[features].to_numpy())
    identity = ["dataset", "symbol", "day", "time_ns", "y"]
    pd.testing.assert_frame_equal(predictions[identity].reset_index(drop=True), held[identity].reset_index(drop=True))
    assert result["train_rows"] == len(training)
    assert result["test_rows"] == len(held)
    for name in ("reference", "time", "volume"):
        p = predictions[f"{name}_probability"].to_numpy()
        assert len(p) == len(held) and np.isfinite(p).all()
        assert ((p > 0) & (p < 1)).all()
    assert not coefficients.empty


def test_heldout_labels_and_feature_changes_cannot_change_fitted_parameters():
    rows = model_rows()
    args = (["reference_lag"], ["time_candle"], ["volume_candle"], TRAINING_DAYS, HELD_OUT_DAYS)
    _, before, before_coefficients = comparison.fit_candle_models(rows, *args)
    changed_labels = rows.copy()
    held = changed_labels.day.isin(HELD_OUT_DAYS)
    changed_labels.loc[held, "y"] = 1 - changed_labels.loc[held, "y"]
    _, label_predictions, label_coefficients = comparison.fit_candle_models(changed_labels, *args)
    pd.testing.assert_frame_equal(before.drop(columns="y"), label_predictions.drop(columns="y"))
    pd.testing.assert_frame_equal(before_coefficients, label_coefficients)
    changed_features = rows.copy()
    changed_features.loc[held, ["reference_lag", "time_candle", "volume_candle"]] += 1000
    _, _, feature_coefficients = comparison.fit_candle_models(changed_features, *args)
    pd.testing.assert_frame_equal(before_coefficients, feature_coefficients)


def test_scores_and_positive_volume_improvement_are_calculated_from_same_predictions():
    rows = model_rows()
    result, predictions, _ = comparison.fit_candle_models(
        rows, ["reference_lag"], ["time_candle"], ["volume_candle"],
        TRAINING_DAYS, HELD_OUT_DAYS)
    for name in ("reference", "time", "volume"):
        exact = network_models._scores(predictions.y.to_numpy(), predictions[f"{name}_probability"].to_numpy())
        for metric, value in exact.items():
            assert result[name][metric] == pytest.approx(value)
    assert result["delta_time_vs_volume_log_loss"] == pytest.approx(
        result["time"]["log_loss"] - result["volume"]["log_loss"])
    assert result["delta_time_vs_volume_brier"] == pytest.approx(
        result["time"]["brier"] - result["volume"]["brier"])


@pytest.mark.parametrize("training,held", [
    (TRAINING_DAYS, TRAINING_DAYS[-1:] + HELD_OUT_DAYS),
    (HELD_OUT_DAYS, TRAINING_DAYS),
    (TRAINING_DAYS + HELD_OUT_DAYS[:1], [TRAINING_DAYS[-1]]),
])
def test_direct_candle_fit_rejects_overlapping_or_backward_splits(training, held):
    with pytest.raises(ValueError):
        comparison.fit_candle_models(model_rows(), ["reference_lag"], ["time_candle"], ["volume_candle"], training, held)


def test_small_end_to_end_run_exports_candles_and_pools_all_heldout_dates(tmp_path):
    frames, shock_rows = {}, []
    for day_index, day in enumerate(TRAINING_DAYS + HELD_OUT_DAYS):
        origin = pd.Timestamp(f"{day}T14:00:00Z")
        offsets = np.arange(0, 4000, 10)
        for venue_index, dataset in enumerate(("A", "B")):
            blocks = []
            for symbol in ("TEST", "SPY"):
                blocks.append(pd.DataFrame({
                    "ts_recv": origin + pd.to_timedelta(offsets, unit="ms"),
                    "ts_event": origin + pd.to_timedelta(offsets, unit="ms"),
                    "symbol": symbol, "action": "T", "side": "B",
                    "price": 100.0 + .01 * (np.arange(len(offsets)) % 3),
                    "size": 100, "sequence": np.arange(len(offsets)),
                    "bid_px_00": 99.99, "ask_px_00": 100.01,
                    "bid_sz_00": 100, "ask_sz_00": 100, "flags": 128,
                }))
                for offset in range(1110 + day_index * 10 + venue_index * 10, 3900, 110):
                    shock_rows.append({
                        "dataset": dataset, "symbol": symbol, "day": day,
                        "time_ns": origin.value + offset * 1_000_000,
                        "split": "training" if day in TRAINING_DAYS else "held_out",
                    })
            frame = pd.concat(blocks, ignore_index=True)
            frame.attrs.update(requested_start=origin.isoformat(), requested_end=(origin + pd.Timedelta(seconds=4)).isoformat())
            frames[(dataset, "mbp-1", day)] = frame
    shock_path = tmp_path / "shock_events.csv"
    pd.DataFrame(shock_rows).to_csv(shock_path, index=False)
    outdir = tmp_path / "comparison"
    result = comparison.run_volume_comparison(
        frames, shock_path, outdir, training_days=TRAINING_DAYS,
        held_out_days=HELD_OUT_DAYS, progress=lambda _: None)
    assert result["training_days"] == TRAINING_DAYS
    assert result["held_out_days"] == HELD_OUT_DAYS
    assert len(result["tasks"]) == 4
    assert len(result["task_days"]) == 4 * len(HELD_OUT_DAYS)
    assert [row["day"] for row in result["per_day"]] == HELD_OUT_DAYS
    assert result["pooled"]["test_rows"] == sum(task["test_rows"] for task in result["tasks"])
    assert result["pooled"]["test_rows"] == sum(day["test_rows"] for day in result["per_day"])
    for model in ("reference", "time", "volume"):
        for metric in ("log_loss", "brier"):
            weighted = sum(day["test_rows"] * day[model][metric] for day in result["per_day"]) / result["pooled"]["test_rows"]
            assert result["pooled"][model][metric] == pytest.approx(weighted)
    for task in result["tasks"]:
        assert task["time_feature_count"] == task["volume_feature_count"]
    thresholds = pd.read_csv(outdir / "volume_thresholds.csv")
    assert thresholds.threshold_shares.eq(10_000).all()
    assert thresholds.training_days.eq(2).all()
    assert thresholds.training_seconds.eq(8).all()
    for family in ("time", "volume"):
        candles = pd.read_csv(outdir / f"{family}_candles.csv")
        assert set(candles.day) == set(TRAINING_DAYS + HELD_OUT_DAYS)
        assert candles.volume_shares.sum() == 16 * 40_000
        assert candles.close_ns.ge(candles.open_ns).all()
    scores = pd.read_csv(outdir / "candle_model_scores.csv")
    assert set(scores.model) == {"reference", "time", "volume"}
    assert scores.groupby(["dataset", "symbol"]).test_rows.nunique().eq(1).all()
    for filename in ("candle_coverage.csv", "candle_feed_coverage.csv", "candle_coefficients.csv",
                     "candle_daily_scores.csv", "candle_statistics.csv", "trade_quality.csv",
                     "volume_comparison_summary.json"):
        assert (outdir / filename).is_file()
