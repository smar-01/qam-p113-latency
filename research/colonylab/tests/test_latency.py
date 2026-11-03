"""Known-answer and independent-loop checks for native quote matching."""
from __future__ import annotations

import numpy as np
import pytest

from colonylab.latency import (
    QuoteSeries, _ThresholdIndex, reaction_events, classify_horizon, summarize_events,
    binned_crosscorr, event_study, realized_volatility,
)


def quotes(times, bid, *, ask=None, valid=None, segments=None, start=0, end=1000):
    n = len(times)
    return QuoteSeries(np.asarray(times, dtype=np.int64), np.asarray(bid, dtype=np.int64),
                       np.asarray(ask if ask is not None else np.asarray(bid) + 20, dtype=np.int64),
                       np.asarray(valid if valid is not None else np.ones(n), dtype=bool),
                       np.asarray(segments if segments is not None else np.zeros(n), dtype=np.int64),
                       start, end)


def brute(leader, follower, side, horizon):
    """Independent plain-loop definition, without using trees/core helpers."""
    result = []
    for i in range(1, len(leader.times_ns)):
        t, new, old = int(leader.times_ns[i]), int(getattr(leader, side)[i]), int(getattr(leader, side)[i - 1])
        if not (leader.valid[i] and leader.valid[i - 1] and leader.segments[i] == leader.segments[i - 1]):
            continue
        if new == old or not leader.start_ns <= t < leader.end_ns:
            continue
        d = 1 if new > old else -1
        states = [j for j, time in enumerate(follower.times_ns) if time <= t]
        if (not states or not follower.start_ns <= t < follower.end_ns
                or not follower.valid[states[-1]]):
            result.append((t, "no_state", -1, 0))
            continue
        j = states[-1]
        state = int(getattr(follower, side)[j])
        already = state >= new if d > 0 else state <= new
        stop = min(leader.end_ns, follower.end_ns)
        for stream in (leader, follower):
            for k, time in enumerate(stream.times_ns):
                is_boundary = not stream.valid[k] or (k > 0 and stream.segments[k] != stream.segments[k - 1])
                if time > t and is_boundary:
                    stop = min(stop, int(time))
                    break
        observed = min(horizon, max(0, stop - t - 1))
        lag = -1
        if not already:
            for k in range(j + 1, len(follower.times_ns)):
                time = int(follower.times_ns[k])
                if time >= stop or time - t > horizon:
                    break
                price = int(getattr(follower, side)[k])
                if follower.valid[k] and (price >= new if d > 0 else price <= new):
                    lag = time - t
                    break
        result.append((t, "already" if already else "behind", lag, observed))
    return result


def test_range_tree_first_hits_agree_with_independent_search():
    rng = np.random.default_rng(328)
    price = rng.integers(1, 1000, size=513)
    valid = rng.random(len(price)) > .1
    lo = rng.integers(0, len(price) + 1, size=4000)
    hi = np.array([rng.integers(i, len(price) + 1) for i in lo])
    target = rng.integers(1, 1100, size=len(lo))
    direction = rng.choice([-1, 1], size=len(lo))
    found = _ThresholdIndex(price, valid).first_hit(lo, hi, target, direction)
    expected = []
    for a, b, p, d in zip(lo, hi, target, direction):
        hits = [j for j in range(a, b) if valid[j] and (price[j] >= p if d > 0 else price[j] <= p)]
        expected.append(hits[0] if hits else -1)
    np.testing.assert_array_equal(found, expected)


@pytest.mark.parametrize("side", ["bid", "ask"])
@pytest.mark.parametrize("direction", [-1, 1])
def test_known_lag_for_both_sides_and_directions(side, direction):
    leader = quotes([0, 100, 200], [100, 100 + direction, 100 + 2 * direction])
    follower = quotes([0, 125, 225], [100, 100 + direction, 100 + 2 * direction])
    events = reaction_events(leader, follower, side, max_horizon_ns=100)
    assert events.latency_ns.tolist() == [25, 25]
    assert classify_horizon(events, 20).tolist() == ["censored", "censored"]
    assert classify_horizon(events, 25).tolist() == ["matched", "matched"]


def test_same_timestamp_is_already_not_zero_latency():
    leader = quotes([0, 10], [100, 101])
    follower = quotes([0, 10], [100, 101])
    events = reaction_events(leader, follower)
    assert events.status.tolist() == ["already"]
    assert events.latency_ns.tolist() == [-1]


def test_missing_prior_state_does_not_become_a_match():
    leader = quotes([0, 10], [100, 101])
    follower = quotes([12], [101])
    events = reaction_events(leader, follower)
    assert classify_horizon(events, 100).tolist() == ["no_state"]
    assert summarize_events(events, 100)["n_eligible"] == 0


@pytest.mark.parametrize("which", ["leader", "follower"])
@pytest.mark.parametrize("explicit_invalid", [True, False])
def test_matching_never_crosses_either_feed_quality_boundary(which, explicit_invalid):
    a = quotes([0, 10, 15, 50], [100, 101, 101, 101],
               valid=[True, True, not explicit_invalid, True], segments=[0, 0, 1, 1])
    b = quotes([0, 30, 50], [100, 101, 101])
    leader, follower = (a, b) if which == "leader" else (quotes([0, 10], [100, 101]), a)
    if which == "follower":
        # follower is still behind at t=10, with the next qualifying quote
        # appearing only after the quality boundary.
        follower = quotes([0, 15, 30, 50], [100, 100, 101, 101],
                          valid=[True, not explicit_invalid, True, True], segments=[0, 1, 1, 1])
    events = reaction_events(leader, follower, max_horizon_ns=100)
    assert events.latency_ns.tolist() == [-1]
    assert events.observed_ns.tolist() == [4]
    assert classify_horizon(events, 4).tolist() == ["censored"]
    assert classify_horizon(events, 5).tolist() == ["quality_censored"]


def test_early_match_with_truncated_followup_is_not_in_complete_case_denominator():
    a = quotes([0, 10], [100, 101])
    b = quotes([0, 12, 15], [100, 101, 0], valid=[True, True, False], segments=[0, 0, 1])
    events = reaction_events(a, b, max_horizon_ns=100)
    assert classify_horizon(events, 10).tolist() == ["matched"]
    summary = summarize_events(events, 10)
    assert summary["n_matched"] == summary["n_eligible"] == 0
    assert summary["known_matches_truncated"] == summary["n_quality_truncated"] == 1
    assert np.isnan(summary["match_rate"])
    assert summarize_events(events, 3)["match_rate"] == 1


def test_exclusive_end_distinguishes_end_truncation_from_horizon_nonmatch():
    a = quotes([0, 10], [100, 101], end=20)
    b = quotes([0, 20], [100, 101], end=20)
    events = reaction_events(a, b, max_horizon_ns=100)
    assert events.latency_ns.tolist() == [-1]
    assert events.observed_ns.tolist() == [9]
    assert classify_horizon(events, 9).tolist() == ["censored"]
    assert classify_horizon(events, 10).tolist() == ["end_censored"]


def test_first_good_quote_after_a_barrier_is_not_a_leader_price_change():
    a = quotes([0, 10, 20, 30], [100, 0, 102, 103], valid=[1, 0, 1, 1], segments=[0, 1, 1, 1])
    b = quotes([0, 35], [100, 103])
    events = reaction_events(a, b)
    assert events.t.tolist() == [30]
    assert events.latency_ns.tolist() == [5]


def test_leader_reversal_does_not_cancel_original_price_matching_event():
    a = quotes([0, 10, 15], [100, 102, 100])
    b = quotes([0, 20], [100, 102])
    events = reaction_events(a, b, max_horizon_ns=100)
    assert events.t.tolist() == [10, 15]
    assert events.latency_ns.tolist()[0] == 10
    assert events.status.tolist()[1] == "already"


def test_restricted_mean_wait_includes_nonmatches_at_horizon():
    a = quotes([0, 10, 100], [100, 102, 103])
    b = quotes([0, 15], [100, 102])
    events = reaction_events(a, b, max_horizon_ns=50)
    summary = summarize_events(events, 50)
    assert summary["n_matched"] == summary["n_censored"] == 1
    assert summary["match_rate"] == .5
    assert summary["sum_latency_ns"] == 5
    assert summary["restricted_wait_sum_ns"] == 55
    assert summary["restricted_mean_wait_us"] == .0275


def test_event_matching_maintains_nanosecond_and_integer_price_precision():
    origin = 1_800_000_000_000_000_000
    p = 100_000_000_000
    a = quotes([origin, origin + 2], [p, p + 1], start=origin, end=origin + 100)
    b = quotes([origin, origin + 3], [p, p + 1], start=origin, end=origin + 100)
    events = reaction_events(a, b, max_horizon_ns=10)
    assert events.latency_ns.tolist() == [1]
    assert events.price.tolist() == [p + 1]


@pytest.mark.parametrize("side", ["bid", "ask"])
def test_random_streams_agree_with_plain_loop_including_barriers(side):
    rng = np.random.default_rng(790)
    for _ in range(12):
        streams = []
        for shift in (0, 1):
            times = np.cumsum(rng.integers(1, 12, size=60)) + shift
            prices = 100 + np.cumsum(rng.choice([-1, 0, 1], size=len(times)))
            valid = rng.random(len(times)) > .1
            segment = np.cumsum(~valid)
            streams.append(quotes(times, prices, valid=valid, segments=segment, end=500))
        a, b = streams
        events = reaction_events(a, b, side, max_horizon_ns=50)
        observed = list(events[["t", "status", "latency_ns", "observed_ns"]].itertuples(index=False, name=None))
        assert observed == brute(a, b, side, 50)


@pytest.mark.parametrize("times", [[0, 0], [2, 1], [-1, 2]])
def test_duplicate_reversed_or_invalid_timestamps_must_be_resolved_by_loader(times):
    with pytest.raises(ValueError, match="timestamps"):
        quotes(times, [100, 101])


def test_empty_follower_and_empty_leader_have_explicit_unknown_support():
    a, b = quotes([0, 10], [100, 101]), quotes([], [])
    events = reaction_events(a, b)
    assert events.status.tolist() == ["no_state"]
    empty = reaction_events(b, a)
    assert len(empty) == 0
    assert summarize_events(empty, 10)["n_events"] == 0


def test_horizon_cannot_extend_beyond_original_search():
    events = reaction_events(quotes([0, 10], [100, 101]), quotes([0, 15], [100, 101]), max_horizon_ns=10)
    with pytest.raises(ValueError, match="exceeds"):
        classify_horizon(events, 11)


def test_same_bin_increments_are_aggregated_before_crosscorr_energy():
    # The starter's event-energy denominator produces 2 here; aggregate the
    # two increments first and the normalized self cross-product is exactly 1.
    stream = quotes([0, 30, 34], [100, 101, 102])
    curve = binned_crosscorr(stream, stream, bin_ns=10, max_lag_ns=10)
    assert curve.loc[curve.lag_us == 0, "corr"].iloc[0] == 1
    assert curve["corr"].abs().max() <= 1
    assert curve.attrs["n_leader_nonzero_bins"] == 1


def test_crosscorr_known_positive_lag_means_follower_after_leader():
    t = np.array([0, 1_000_000, 2_000_000, 3_000_000])
    a = quotes(t, [100, 103, 101, 105], end=5_000_000)
    b = quotes(t + np.array([0, 50_000, 50_000, 50_000]), [100, 103, 101, 105], end=5_000_000)
    curve = binned_crosscorr(a, b, bin_ns=10_000, max_lag_ns=200_000)
    peak = curve.loc[curve["corr"].idxmax()]
    assert peak.lag_us == 50
    assert peak["corr"] == pytest.approx(1)


def test_sparse_crosscorr_matches_independent_dense_fixed_bin_formula():
    rng = np.random.default_rng(92)
    streams = []
    for _ in range(2):
        t = np.r_[0, np.sort(rng.choice(np.arange(60, 900), size=90, replace=False))]
        streams.append(quotes(t, 100 + np.cumsum(rng.choice([-1, 1], size=len(t))), end=1000))
    curve = binned_crosscorr(*streams, bin_ns=10, max_lag_ns=50)
    dense = []
    for stream in streams:
        values = np.zeros(100)
        midpoint = stream.bid + stream.ask
        for i in range(1, len(stream.times_ns)):
            if 50 <= stream.times_ns[i] < 950:
                values[stream.times_ns[i] // 10] += midpoint[i] - midpoint[i - 1]
        dense.append(values)
    x, y = dense
    norm = np.sqrt(np.dot(x, x) * np.dot(y, y))
    expected = []
    for lag in range(-5, 6):
        if lag < 0:
            value = np.dot(x[-lag:], y[:lag])
        elif lag > 0:
            value = np.dot(x[:-lag], y[lag:])
        else:
            value = np.dot(x, y)
        expected.append(value / norm)
    np.testing.assert_allclose(curve["corr"], expected, atol=1e-14)


def test_crosscorr_excludes_increments_near_quality_boundaries():
    a = quotes([0, 100, 104, 110, 300], [100, 101, 0, 102, 103],
               valid=[1, 1, 0, 1, 1], segments=[0, 0, 1, 1, 1])
    b = quotes([0, 105, 300], [100, 101, 102])
    curve = binned_crosscorr(a, b, bin_ns=10, max_lag_ns=20)
    assert curve.attrs["n_leader_increments"] == 1  # only t=300
    assert curve.attrs["n_follower_increments"] == 1


def test_no_increment_crosscorr_returns_unknown_not_spurious_peak():
    stream = quotes([0, 10], [100, 100])
    curve = binned_crosscorr(stream, stream, bin_ns=10, max_lag_ns=20)
    assert curve["corr"].isna().all()
    assert curve.attrs["n_leader_increments"] == 0


def test_event_study_recovers_known_catchup_on_fixed_cohort():
    a = quotes([0, 100, 200], [100, 101, 102])
    b = quotes([0, 105, 205], [100, 101, 102])
    study = event_study(a, b, [-10, -1, 0, 4, 5, 10])
    assert study.share.tolist() == [0, 0, 0, 0, 1, 1]
    assert study.n.tolist() == [2] * 6
    assert study.attrs["fixed_common_cohort"]


@pytest.mark.parametrize("boundary", ["quality", "end"])
def test_event_study_drops_boundary_events_from_every_offset(boundary):
    a = quotes([0, 100, 200], [100, 101, 102])
    if boundary == "quality":
        b = quotes([0, 105, 205], [100, 101, 0], valid=[1, 1, 0], segments=[0, 0, 1])
    else:
        b = quotes([0, 105, 205], [100, 101, 102], end=210)
    study = event_study(a, b, [-10, 0, 5, 10])
    assert study.n.tolist() == [1] * 4
    assert study.share.tolist() == [0, 0, 1, 1]
    assert study.attrs["n_total_leader_events"] == 2


def test_empty_event_study_has_zero_support_at_every_offset():
    stream = quotes([0, 100], [100, 100])
    study = event_study(stream, stream, [-10, 0, 10])
    assert study.share.isna().all()
    assert study.n.tolist() == [0, 0, 0]


def test_realized_vol_requires_full_consecutive_window_and_minimum_venue_support():
    sec = 1_000_000_000
    times = np.arange(13) * sec
    prices = 100 + np.arange(13) ** 2
    a = quotes(times, prices, end=13 * sec)
    valid = np.ones(13, dtype=bool)
    valid[5:7] = False
    b = quotes(times, prices + 10, valid=valid, segments=np.cumsum(~valid), end=13 * sec)
    vol = realized_volatility({"A": a, "B": b}, step_ns=sec, window=3, min_venues=2)
    assert vol.t.tolist() == (np.arange(1, 13) * sec).tolist()
    assert vol.regime.iloc[:3].tolist() == ["unknown"] * 3
    assert np.isfinite(vol.loc[vol.t == 4 * sec, "rv"]).all()
    assert vol.loc[vol.t.between(5 * sec, 9 * sec), "regime"].eq("unknown").all()
    assert np.isfinite(vol.loc[vol.t == 10 * sec, "rv"]).all()
    assert vol.attrs["retrospective"]


def test_too_short_realized_vol_window_stays_unknown():
    sec = 1_000_000_000
    stream = quotes(np.arange(5) * sec, [100, 101, 102, 103, 104], end=5 * sec)
    vol = realized_volatility({"A": stream}, window=300)
    assert vol.regime.eq("unknown").all()
    assert vol.rv.isna().all()
