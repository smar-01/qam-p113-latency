from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from colonylab import analysis


def quote_trades(count: int = 160) -> pd.DataFrame:
    times = pd.date_range("2025-01-06T14:30:00Z", periods=count, freq="10ms")
    mid = 100.0 + np.arange(count) * 0.002
    return pd.DataFrame({
        "ts_event": times,
        "symbol": "TEST",
        "publisher_id": 1,
        "action": "T",
        "side": "B",
        "price": mid + 0.01,
        "size": 10 + np.arange(count) % 7,
        "bid_px_00": mid - 0.01,
        "ask_px_00": mid + 0.01,
        "bid_sz_00": 100,
        "ask_sz_00": 120,
    })


def features(frame: pd.DataFrame) -> pd.DataFrame:
    normal = analysis._normalise(frame, "TEST.FEED", "mbp-1")
    return analysis._features(normal, 100_000_000, 1_000_000_000)[0]


def test_float_roundoff_at_midpoint_does_not_create_an_aggressor_sign():
    frame = quote_trades(20)
    frame["side"] = "N"
    frame["price"] = 100.11
    frame["bid_px_00"] = 100.10
    frame["ask_px_00"] = 100.12
    # The float average can differ from the trade price by about 1e-14.
    result = features(frame)
    assert result.loc[1:, "_sign"].eq(0).all()
    assert result.loc[1:, "direction_source"].eq("unknown_not_inferred").all()


@pytest.fixture(scope="module")
def analyzed(tmp_path_factory):
    output = tmp_path_factory.mktemp("analysis")
    scaler_inputs = []
    mixture_inputs = []
    original_scaler = analysis.StandardScaler
    original_mixture = analysis.GaussianMixture

    class RecordingScaler(original_scaler):
        def fit(self, x, y=None, sample_weight=None):
            scaler_inputs.append(np.asarray(x).copy())
            return super().fit(x, y, sample_weight=sample_weight)

    class RecordingMixture(original_mixture):
        def fit(self, x, y=None):
            mixture_inputs.append(np.asarray(x).copy())
            return super().fit(x, y)

    with patch.object(analysis, "StandardScaler", RecordingScaler), patch.object(analysis, "GaussianMixture", RecordingMixture):
        summary = analysis.run_analysis(
            quote_trades(), output, dataset="TEST.FEED", schema="mbp-1",
            window_ms=100, horizons_ms=[15, 150], n_clusters=1,
        )
    return {
        "summary": summary,
        "memberships": pd.read_csv(output / "event_memberships.csv"),
        "features": pd.read_csv(output / "event_features.csv"),
        "responses": pd.read_csv(output / "event_responses.csv"),
        "coverage": pd.read_csv(output / "coverage.csv"),
        "scaler_inputs": scaler_inputs,
        "mixture_inputs": mixture_inputs,
    }


def test_response_uses_last_quote_at_or_before_horizon(analyzed):
    response = analyzed["responses"].query("horizon_ms == 15 and response_valid").iloc[0]
    i = int(response.event_id)
    expected_baseline = 100.0 + (i - 1) * 0.002
    expected_response = 100.0 + (i + 1) * 0.002
    # The next quote is at +20 ms; selecting it would introduce future data.
    assert response.response_price == pytest.approx(expected_response)
    assert response.horizon_price_age_ms == pytest.approx(5.0)
    assert response.signed_response_bps == pytest.approx(
        (expected_response - expected_baseline) / expected_baseline * 1e4
    )
    membership = analyzed["memberships"].set_index("event_id").loc[i]
    assert membership.baseline_price == pytest.approx(expected_baseline)
    assert membership.baseline_age_ms == pytest.approx(10.0)


def test_horizons_after_group_end_are_missing_without_extrapolation(analyzed):
    responses = analyzed["responses"]
    last_event = int(responses.event_id.max())
    rows = responses.loc[responses.event_id == last_event]
    assert not rows.horizon_observed.any()
    assert not rows.response_valid.any()
    assert rows.response_price.isna().all()
    assert rows.signed_response_bps.isna().all()
    short_coverage = analyzed["coverage"].set_index("horizon_ms").loc[15]
    assert short_coverage.missing_horizon_test_events == 2


def test_scaler_and_mixture_are_fit_only_on_training_features(analyzed):
    frame = analyzed["features"]
    training = frame.loc[frame.split == "train", analysis.FEATURE_COLUMNS].to_numpy()
    assert len(analyzed["scaler_inputs"]) == 1
    np.testing.assert_allclose(analyzed["scaler_inputs"][0], training)
    expected_scaled = analysis.StandardScaler().fit_transform(training)
    assert len(analyzed["mixture_inputs"]) == 1
    np.testing.assert_allclose(analyzed["mixture_inputs"][0], expected_scaled, atol=1e-10)
    assert len(training) < frame.features_valid.sum()


def test_training_responses_cannot_reach_holdout(analyzed):
    frame = analyzed["features"]
    train_end = pd.to_datetime(frame.loc[frame.split == "train", "ts_analysis"], utc=True).max()
    holdout_start = pd.to_datetime(frame.loc[frame.split == "test", "ts_analysis"], utc=True).min()
    assert train_end + pd.Timedelta(milliseconds=150) < holdout_start
    assert analyzed["responses"].event_id.isin(frame.loc[frame.split == "test", "event_id"]).all()


def test_current_trade_and_future_records_do_not_enter_current_features():
    original = quote_trades(60)
    altered = original.copy()
    altered.loc[40:, "price"] = 250.0
    altered.loc[40:, "size"] = 100000
    altered.loc[40:, "side"] = "A"
    altered.loc[40:, ["bid_px_00", "ask_px_00"]] = [249.0, 251.0]
    before, after = features(original), features(altered)
    np.testing.assert_allclose(
        before.loc[40, analysis.FEATURE_COLUMNS].to_numpy(dtype=float),
        after.loc[40, analysis.FEATURE_COLUMNS].to_numpy(dtype=float),
    )
    assert before.loc[40, "baseline_price"] == after.loc[40, "baseline_price"]


def test_equal_timestamp_trades_are_excluded_from_each_others_history():
    frame = quote_trades(10)
    frame.loc[6, "ts_event"] = frame.loc[5, "ts_event"]
    result = features(frame)
    tied = result.loc[result.ts_event == frame.loc[5, "ts_event"]]
    assert len(tied) == 2
    assert tied.past_trade_count.tolist() == [5, 5]
    expected_volume = frame.loc[:4, "size"].sum()
    np.testing.assert_allclose(tied.log_total_volume, np.log1p(expected_volume))


def test_conflicting_equal_timestamp_prices_and_quote_states_become_unknown_barriers():
    frame = quote_trades(6)
    conflict = frame.iloc[[2]].copy()
    conflict["bid_px_00"] += 1.0
    conflict["ask_px_00"] += 1.0
    # A second conflict has the same midpoint but a different spread.
    spread_conflict = frame.iloc[[4]].copy()
    spread_conflict["bid_px_00"] -= 0.01
    spread_conflict["ask_px_00"] += 0.01
    normal = analysis._normalise(pd.concat([frame, conflict, spread_conflict]), "TEST.FEED", "mbp-1")
    prices, excluded = analysis._price_series(normal)
    assert excluded == 2
    assert len(prices) == 6
    for timestamp in frame.loc[[2, 4], "ts_event"]:
        assert prices.loc[prices._time_ns == timestamp.value, "_response_price"].isna().all()


@pytest.mark.parametrize("kind", ["invalid", "crossed", "ambiguous"])
def test_unknown_quote_state_blocks_earlier_baseline_until_valid_state_returns(kind):
    frame = quote_trades(10)
    frame.loc[6, "side"] = "N"
    barrier = frame.iloc[[5]].copy()
    barrier["ts_event"] += pd.Timedelta(milliseconds=5)
    barrier["action"] = "M"
    barrier["size"] = 0
    if kind == "invalid":
        barrier[["bid_px_00", "ask_px_00"]] = np.nan
    elif kind == "crossed":
        barrier[["bid_px_00", "ask_px_00"]] = [101.0, 100.0]
    else:
        conflicting = barrier.copy()
        conflicting[["bid_px_00", "ask_px_00"]] += 1.0
        barrier = pd.concat([barrier, conflicting], ignore_index=True)
    result = features(pd.concat([frame, barrier], ignore_index=True)).set_index("ts_event")
    blocked = result.loc[frame.loc[6, "ts_event"]]
    assert not blocked.baseline_valid
    assert pd.isna(blocked.baseline_price)
    assert blocked._sign == 0
    assert blocked.direction_source == "unknown_not_inferred"
    recovered = result.loc[frame.loc[7, "ts_event"]]
    assert recovered.baseline_valid
    assert recovered.baseline_price == pytest.approx(100.0 + 6 * 0.002)


def test_unknown_quote_state_at_response_horizon_is_not_filled_from_older_quote(tmp_path):
    frame = quote_trades()
    barrier = frame.iloc[[120]].copy()
    barrier["ts_event"] += pd.Timedelta(milliseconds=5)
    barrier["action"] = "M"
    barrier["size"] = 0
    barrier[["bid_px_00", "ask_px_00"]] = np.nan
    analysis.run_analysis(
        pd.concat([frame, barrier], ignore_index=True), tmp_path,
        dataset="TEST.FEED", schema="mbp-1", window_ms=100, horizons_ms=[15], n_clusters=1,
    )
    responses = pd.read_csv(tmp_path / "event_responses.csv").set_index("event_id")
    blocked = responses.loc[119]
    assert not blocked.horizon_observed
    assert not blocked.response_valid
    assert pd.isna(blocked.response_price)
    assert pd.isna(blocked.signed_response_bps)


def test_prior_tied_trade_order_does_not_change_later_features():
    frame = quote_trades(20)
    tied = frame.iloc[[5]].copy()
    tied["price"] = 101.0
    tied["size"] = 200
    tied["side"] = "A"
    forward = pd.concat([frame, tied], ignore_index=True)
    reverse = forward.iloc[::-1].reset_index(drop=True)
    target = frame.loc[15, "ts_event"]
    before = features(forward).set_index("ts_event").loc[target]
    after = features(reverse).set_index("ts_event").loc[target]
    np.testing.assert_allclose(
        before[analysis.FEATURE_COLUMNS].to_numpy(dtype=float),
        after[analysis.FEATURE_COLUMNS].to_numpy(dtype=float),
        atol=1e-8,
    )


def test_receive_clock_prevents_late_received_event_from_entering_earlier_history():
    frame = quote_trades(30)
    frame["ts_recv"] = frame.ts_event
    late = frame.iloc[[4]].copy()
    late["ts_recv"] = frame.loc[25, "ts_event"]
    late["size"] = 1000000
    original = features(frame)
    combined = features(pd.concat([frame, late], ignore_index=True))
    target = frame.loc[20, "ts_event"]
    selected = combined.loc[combined.ts_analysis == target].iloc[0]
    np.testing.assert_allclose(
        selected[analysis.FEATURE_COLUMNS].to_numpy(dtype=float),
        original.loc[20, analysis.FEATURE_COLUMNS].to_numpy(dtype=float),
    )


def test_unknown_direction_uses_only_fresh_prior_midpoint_and_preserves_feed_sides():
    times = pd.Timestamp("2025-01-06T14:30:00Z") + pd.to_timedelta([0, 10, 20, 30, 40, 2500], unit="ms")
    frame = pd.DataFrame({
        "ts_event": times, "symbol": "TEST", "action": ["M", "T", "T", "T", "T", "T"],
        "side": ["N", "N", "N", "B", "A", "N"],
        "price": [100.0, 100.02, 100.0, 99.9, 100.1, 100.1], "size": [0, 10, 10, 10, 10, 10],
        "bid_px_00": 99.99, "ask_px_00": 100.01, "bid_sz_00": 100, "ask_sz_00": 100,
    })
    result = features(frame)
    assert result._sign.tolist() == [1, 0, 1, -1, 0]
    assert result.direction_source.tolist() == [
        "estimated_prior_midquote", "unknown_not_inferred", "feed_B_or_A", "feed_B_or_A", "unknown_not_inferred",
    ]


def test_trade_only_unknown_sides_do_not_become_tick_rule_aggressor_signs():
    frame = quote_trades(10).drop(columns=["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"])
    frame["side"] = "N"
    normal = analysis._normalise(frame, "TEST.FEED", "trades")
    result, _, _ = analysis._features(normal, 100_000_000, 1_000_000_000)
    assert (result._sign == 0).all()
    assert (result.direction_source == "unknown_not_inferred").all()
    assert (result.tick_rule_sign_diagnostic_only.iloc[1:] == 1).all()


def test_eq_us_mini_publishers_are_treated_as_one_aggregate_feed():
    frame = quote_trades(10)
    frame["publisher_id"] = np.arange(len(frame)) % 2
    result = analysis._normalise(frame, "EQUS.MINI", "mbp-1")
    assert result.response_feed.unique().tolist() == ["EQUS.MINI_AGGREGATE"]
