"""Independent multiple-date scoring, leakage and compact-export checks."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from colonylab import network_models as models


DAYS = [stamp.date().isoformat() for stamp in pd.bdate_range("2026-10-05", "2026-10-16")]
TRAINING_DAYS, HELD_OUT_DAYS = DAYS[:5], DAYS[5:]


def model_rows():
    rows = []
    for day_index, day in enumerate(DAYS):
        origin = pd.Timestamp(f"{day}T14:00:00Z").value
        for i in range(60 + day_index * 3):
            rows.append({"dataset": "A", "symbol": "TEST", "day": day,
                         "time_ns": origin + i * 10_000_000, "y": int((i + day_index) % 5 == 0),
                         "own": float(i % 3), "network_signal": float((i + day_index) % 5 == 0)})
    return pd.DataFrame(rows)


def prediction_fixture(directory):
    frames, event_rows = {}, []
    for day_index, day in enumerate(DAYS):
        origin = pd.Timestamp(f"{day}T14:00:00Z")
        for venue_index, dataset in enumerate(("A", "B")):
            blocks = []
            for symbol in ("TEST", "SPY"):
                offsets = np.arange(0, 3010, 10)
                blocks.append(pd.DataFrame({
                    "ts_recv": origin + pd.to_timedelta(offsets, unit="ms"),
                    "ts_event": origin + pd.to_timedelta(offsets, unit="ms"),
                    "symbol": symbol, "bid_px_00": 99.99, "ask_px_00": 100.01,
                    "bid_sz_00": 100, "ask_sz_00": 100, "flags": 128,
                }))
                for offset in range(1110 + day_index * 10 + venue_index * 10, 2900, 110):
                    event_rows.append({"dataset": dataset, "symbol": symbol, "day": day,
                                       "time_ns": origin.value + offset * 1_000_000})
            frames[(dataset, "mbp-1", day)] = pd.concat(blocks, ignore_index=True)
    path = directory / "shock_events.csv"
    pd.DataFrame(event_rows).to_csv(path, index=False)
    return frames, path


def test_multiple_heldout_dates_do_not_enter_scalers_or_fitted_coefficients():
    rows = model_rows()
    records = []
    original = models.StandardScaler

    class RecordingScaler(original):
        def fit(self, x, y=None, sample_weight=None):
            records.append(np.asarray(x).copy())
            return super().fit(x, y, sample_weight=sample_weight)

    args = (["own"], ["network_signal"], TRAINING_DAYS, HELD_OUT_DAYS)
    with patch.object(models, "StandardScaler", RecordingScaler):
        result, before, before_coef = models.fit_prediction_models(rows, *args)
    training = rows.loc[rows.day.isin(TRAINING_DAYS)]
    np.testing.assert_array_equal(records[0], training[["own"]].to_numpy())
    np.testing.assert_array_equal(records[1], training[["own", "network_signal"]].to_numpy())
    assert set(before.day) == set(HELD_OUT_DAYS)
    assert result["train_rows"] == len(training)
    assert result["test_rows"] == len(rows) - len(training)
    altered = rows.copy()
    held = altered.day.isin(HELD_OUT_DAYS)
    altered.loc[held, "y"] = 1 - altered.loc[held, "y"]
    _, after, after_coef = models.fit_prediction_models(altered, *args)
    pd.testing.assert_frame_equal(before.drop(columns="y"), after.drop(columns="y"))
    pd.testing.assert_frame_equal(before_coef, after_coef)
    altered.loc[held, ["own", "network_signal"]] += 1000
    _, _, feature_coef = models.fit_prediction_models(altered, *args)
    pd.testing.assert_frame_equal(before_coef, feature_coef)


def test_compact_export_preserves_scores_and_every_whole_heldout_day(tmp_path):
    frames, shock_path = prediction_fixture(tmp_path)
    full_dir, compact_dir = tmp_path / "full", tmp_path / "compact"
    full = models.run_prediction_test(frames, shock_path, full_dir, held_out_days=HELD_OUT_DAYS)
    compact_dir.mkdir()
    (compact_dir / "prediction_test_bins.csv").write_text("stale previous run\n", encoding="utf-8")
    compact = models.run_prediction_test(frames, shock_path, compact_dir,
                                         held_out_days=HELD_OUT_DAYS, save_predictions=False)
    assert full["training_days"] == compact["training_days"] == TRAINING_DAYS
    assert full["held_out_days"] == compact["held_out_days"] == HELD_OUT_DAYS
    assert full["held_out_day"] is compact["held_out_day"] is None
    assert full["pooled_scores"] == compact["pooled_scores"]
    assert full["tasks"] == compact["tasks"]
    assert full["pooled_test_bins"] == compact["pooled_test_bins"]
    assert full["per_day_scores"] == compact["per_day_scores"]
    assert full["task_day_scores"] == compact["task_day_scores"]
    assert (full_dir / "prediction_test_bins.csv").is_file()
    assert not (compact_dir / "prediction_test_bins.csv").exists()
    predictions = pd.read_csv(full_dir / "prediction_test_bins.csv")
    assert set(predictions.day) == set(HELD_OUT_DAYS)
    assert predictions.groupby("day").size().eq(800).all()
    assert full["pooled_test_bins"] == len(predictions) == 4000


def test_daily_scoring_uses_exact_same_bins_and_weighting_as_pooled_metrics(tmp_path):
    frames, shock_path = prediction_fixture(tmp_path)
    directory = tmp_path / "result"
    summary = models.run_prediction_test(frames, shock_path, directory, held_out_days=HELD_OUT_DAYS)
    predictions = pd.read_csv(directory / "prediction_test_bins.csv")
    daily = summary["per_day_scores"]
    assert [row["day"] for row in daily] == HELD_OUT_DAYS
    assert sum(row["test_rows"] for row in daily) == len(predictions)
    assert sum(row["test_positive_bins"] for row in daily) == int(predictions.y.sum())
    for model in ("constant", "baseline", "network"):
        exact = models._scores(predictions.y.to_numpy(), predictions[f"{model}_probability"].to_numpy())
        for metric, value in exact.items():
            assert summary["pooled_scores"][model][metric] == pytest.approx(value)
        for row in daily:
            selected = predictions.loc[predictions.day == row["day"]]
            per_day = models._scores(selected.y.to_numpy(), selected[f"{model}_probability"].to_numpy())
            for metric, value in per_day.items():
                assert row[model][metric] == pytest.approx(value)
        # Log loss and Brier decompose across bins; rank metrics do not.
        for metric in ("log_loss", "brier"):
            weighted = sum(row["test_rows"] * row[model][metric] for row in daily) / len(predictions)
            assert summary["pooled_scores"][model][metric] == pytest.approx(weighted)
    exported = pd.read_csv(directory / "prediction_scores_by_day.csv")
    assert set(exported.day) == set(HELD_OUT_DAYS)
    task_daily = pd.read_csv(directory / "prediction_task_scores_by_day.csv")
    assert set(task_daily.day) == set(HELD_OUT_DAYS)
    assert len(summary["task_day_scores"]) == 4 * len(HELD_OUT_DAYS)
    coverage = pd.read_csv(directory / "prediction_coverage.csv")
    feeds = pd.read_csv(directory / "prediction_feed_coverage.csv")
    assert "feed_coverage" not in coverage.columns
    grouped = feeds.groupby(["target_dataset", "target_symbol", "day"]).size()
    actual = coverage.set_index(["dataset", "symbol", "day"]).coverage_feed_count
    pd.testing.assert_series_equal(grouped, actual, check_names=False)
    assert (feeds.eligible_bins + feeds.excluded_bins).eq(feeds.candidate_bins).all()


@pytest.mark.parametrize("held_out_days", [
    [], [DAYS[-1], DAYS[-1]], ["2026-10-19"], DAYS, [DAYS[0]], [DAYS[3], DAYS[-1]],
])
def test_prediction_rejects_invalid_or_nonchronological_explicit_holdout(tmp_path, held_out_days):
    frames, shock_path = prediction_fixture(tmp_path)
    with pytest.raises(ValueError):
        models.run_prediction_test(frames, shock_path, tmp_path / "invalid", held_out_days=held_out_days)


@pytest.mark.parametrize("training_days,held_out_days", [
    (TRAINING_DAYS, DAYS[4:]),
    (HELD_OUT_DAYS, TRAINING_DAYS),
])
def test_direct_fit_cannot_overlap_dates_or_train_after_test(training_days, held_out_days):
    with pytest.raises(ValueError):
        models.fit_prediction_models(model_rows(), ["own"], ["network_signal"], training_days, held_out_days)


def test_existing_shock_split_provenance_must_agree_with_selected_holdout(tmp_path):
    frames, shock_path = prediction_fixture(tmp_path)
    events = pd.read_csv(shock_path)
    # Deliberately claim that events on held-out dates were calibrated in train.
    events["split"] = "training"
    events.to_csv(shock_path, index=False)
    with pytest.raises(ValueError, match="split"):
        models.run_prediction_test(frames, shock_path, tmp_path / "invalid", held_out_days=HELD_OUT_DAYS)


@pytest.mark.parametrize("symbol,feed_count", [("AAPL", 10), ("MSFT", 10), ("SPY", 15)])
def test_five_venue_eligibility_keeps_every_stock_and_proxy_feed_required(symbol, feed_count):
    day = DAYS[0]
    origin = pd.Timestamp(f"{day}T14:00:00Z").value
    times = origin + np.arange(0, 2510, 10, dtype=np.int64) * 1_000_000
    groups = {
        (venue, stock, day): pd.DataFrame({"time_ns": times, "mid": 100., "valid": True,
                                          "segment": 0, "day": day, "dataset": venue, "symbol": stock})
        for venue in ("A", "B", "C", "D", "E") for stock in ("AAPL", "MSFT", "SPY")
    }
    rows, _, additional = models.build_prediction_rows(groups, {}, "A", symbol, day)
    assert rows.attrs["coverage_feed_count"] == feed_count
    assert len(rows.attrs["feed_coverage"]) == feed_count
    assert len(additional) == 4 * len(models.LAG_WINDOWS_MS)
    # A bad history on even the fifth venue remains a censoring condition.
    control = "AAPL" if symbol == "SPY" else "SPY"
    groups[("E", control, day)].loc[100, ["valid", "mid"]] = [False, np.nan]
    censored, _, _ = models.build_prediction_rows(groups, {}, "A", symbol, day)
    anchor = origin + 1_500_000_000
    assert rows.time_ns.eq(anchor).any()
    assert not censored.time_ns.eq(anchor).any()
    assert censored.attrs["eligible_bins"] < rows.attrs["eligible_bins"]
    diagnostics = pd.DataFrame(censored.attrs["feed_coverage"])
    bad = diagnostics.predictor_dataset.eq("E") & diagnostics.predictor_symbol.eq(control)
    assert diagnostics.loc[bad, "excluded_bins"].iloc[0] > 0
    assert diagnostics.loc[~bad, "excluded_bins"].eq(0).all()
    assert diagnostics.loc[bad, "excluded_bins"].iloc[0] == censored.attrs["excluded_bins"]
    outcome = diagnostics.predictor_dataset.eq("A") & diagnostics.predictor_symbol.eq(symbol)
    assert diagnostics.loc[outcome, "future_ms"].iloc[0] == models.BIN_MS
    assert diagnostics.loc[~outcome, "future_ms"].eq(0).all()
