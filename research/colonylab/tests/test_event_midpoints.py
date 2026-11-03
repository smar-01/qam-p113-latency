"""Native-time midpoint matching, cache restoration and strict-prior grid checks."""
import numpy as np
import pandas as pd
import pytest

from colonylab.event_midpoints import NativeMidpoints, _normalize_cache, _restore_integer, match_events, slice_series, summarize_events


def path(times, prices, *, valid=None, segments=None, start=0, end=1000):
    n = len(times)
    return NativeMidpoints(np.asarray(times, np.int64), np.asarray(prices, np.int64),
              np.ones(n, bool) if valid is None else np.asarray(valid, bool),
              np.zeros(n, np.int64) if segments is None else np.asarray(segments, np.int64), start, end)


def payload(*, jump_times=(10, 20, 40, 60), prices=(100, 110, 90, 120),
            jump_values=(10, 10, -20, 30), segments=(1, 1, 1, 1), invalid=(), resets=(),
            grid_times=(15, 25, 45, 65), grid_prices=(100, 110, 90, 120), grid_valid=None,
            grid_segments=None, atomic=()):
    n = len(grid_times)
    grid_valid = np.ones(n, bool) if grid_valid is None else np.asarray(grid_valid, bool)
    return {"jumps": pd.DataFrame({"time_ns": jump_times, "mid": np.asarray(prices)/2e9,
                "jump": np.asarray(jump_values)/2e9, "segment": np.asarray(segments, np.int64)}),
            "grid": pd.DataFrame({"time_ns": np.asarray(grid_times, np.int64), "bid": np.asarray(grid_prices, np.int64)//2-1,
                "ask": np.asarray(grid_prices, np.int64)-(np.asarray(grid_prices, np.int64)//2-1),
                "valid": grid_valid, "segment": np.ones(n, np.int64) if grid_segments is None else grid_segments,
                "bid_size": 100, "ask_size": 120}),
            "invalid_times_ns": np.asarray(invalid, np.int64), "reset_times_ns": np.asarray(resets, np.int64),
            "atomic_intervals_ns": np.asarray(atomic, np.int64).reshape(-1, 2)}


def test_native_known_lag_and_capped_wait_include_nonmatches():
    leader = path([0, 100, 300, 700], [100, 101, 99, 102])
    follower = path([0, 150, 390, 750], [100, 101, 99, 102])
    events = match_events(leader, follower, max_horizon_ns=100)
    assert events.latency_ns.tolist() == [50, 90, 50]
    short = summarize_events(events, 60)
    assert short["n_matched"] == 2 and short["n_censored"] == 1
    assert short["restricted_mean_wait_us"] == pytest.approx((50+60+50)/3/1000)
    assert short["metric"] == "native_receive_midpoint_matching"


def test_equal_timestamp_follower_is_already_there():
    events = match_events(path([0, 100], [100, 101]), path([0, 100], [100, 101]), max_horizon_ns=100)
    assert events.status.tolist() == ["already"]
    assert summarize_events(events, 100)["n_matched"] == 0


def test_no_prior_state_is_not_backfilled_from_future_jump():
    events = match_events(path([0, 100], [100, 101]), path([150], [101]), max_horizon_ns=100)
    assert events.status.tolist() == ["no_state"]
    assert events.latency_ns.tolist() == [-1]


@pytest.mark.parametrize("barrier_in_leader", [True, False])
def test_matches_do_not_cross_either_native_quality_boundary(barrier_in_leader):
    a = path([0, 100, 130, 180], [100, 101, 0, 101], valid=[True, True, False, True], segments=[0, 0, 1, 1])
    b = path([0, 150], [100, 101])
    if barrier_in_leader:
        events = match_events(a, b, max_horizon_ns=100)
    else:
        events = match_events(path([0, 100], [100, 101]),
                             path([0, 130, 150], [100, 0, 101], valid=[True, False, True], segments=[0, 1, 1]),
                             max_horizon_ns=100)
    assert events.latency_ns.iloc[0] == -1
    assert events.observed_ns.iloc[0] == 29
    assert summarize_events(events, 100)["n_quality_truncated"] == 1


def test_early_matches_with_incomplete_horizon_are_separately_counted():
    e = match_events(path([0, 900], [100, 101], end=1000), path([0, 950], [100, 101], end=1000), max_horizon_ns=200)
    s = summarize_events(e, 200)
    assert s["n_matched"] == 0 and s["known_matches_truncated"] == 1 and s["n_right_truncated"] == 1


def test_slice_keeps_actual_prior_native_state_without_a_fake_event():
    source = path([0, 100, 300, 700], [100, 101, 99, 102])
    source.grid = pd.DataFrame({"time_ns": [0, 200, 400, 600], "valid": True})
    chosen = slice_series(source, 200, 800)
    assert chosen.times_ns.tolist() == [100, 300, 700]
    assert chosen.grid.time_ns.tolist() == [200, 400, 600]
    assert match_events(chosen, path([0, 400, 750], [100, 99, 102]), max_horizon_ns=100).t.tolist() == [300, 700]
    with pytest.raises(ValueError, match="outside"):
        slice_series(source, 0, 1100)


def test_grid_is_strictly_prior_at_a_native_jump_timestamp():
    p = payload(grid_times=(20, 40, 60), grid_prices=(100, 110, 90))
    series, facts = _normalize_cache(p, 20, 70)
    assert series.times_ns.tolist() == [10, 20, 40, 60]
    assert facts["grid_eligible_anchors_matched"] == 3
    assert series.grid.time_ns.tolist() == [20, 40, 60]


def test_price_and_jump_lattice_reject_rounding_and_delta_tampering():
    with pytest.raises(ValueError, match="lattice"):
        _restore_integer([100.25/2e9], "test")
    p = payload()
    p["jumps"].loc[1, "jump"] += 1/2e9
    with pytest.raises(ValueError, match="adjacent exact"):
        _normalize_cache(p, 15, 70)


def test_integer_grid_mismatch_is_rejected():
    p = payload()
    p["grid"].loc[1, "bid"] += 1
    with pytest.raises(ValueError, match="integer BBO grid"):
        _normalize_cache(p, 15, 70)


def test_uncertain_initial_state_is_unknown_until_first_native_jump():
    p = payload(grid_times=(5, 15, 25, 45, 65), grid_prices=(90, 100, 110, 90, 120))
    series, facts = _normalize_cache(p, 5, 70)
    assert not facts["initial_native_seed_available"]
    assert not series.valid[0] and series.times_ns[0] == 4
    assert facts["grid_healthy_unknown_midpoint_anchors"] == 1


def test_conflicting_same_timestamp_states_are_barriers():
    p = payload(jump_times=(10, 20, 20, 40), prices=(100, 110, 120, 130), jump_values=(10, 10, 10, 10),
                grid_times=(15, 25, 45), grid_prices=(100, 120, 130))
    series, facts = _normalize_cache(p, 15, 60)
    assert facts["ambiguous_timestamp_groups"] == 1
    assert not series.valid[np.searchsorted(series.times_ns, 20)]
    assert facts["grid_healthy_unknown_midpoint_anchors"] == 1


def test_invalid_books_are_not_declared_observed_from_last_midpoint():
    p = payload(invalid=(30,), grid_times=(15, 25, 35, 45, 65), grid_prices=(100, 110, 0, 90, 120),
                grid_valid=[True, True, False, True, True], segments=(1, 1, 2, 2), grid_segments=[1, 1, 2, 2, 2])
    series, facts = _normalize_cache(p, 15, 70)
    assert not series.valid[np.searchsorted(series.times_ns, 30)]
    assert facts["grid_eligible_anchors_matched"] == 4


def test_unfinished_atomic_interval_cannot_contain_valid_native_jump():
    p = payload(atomic=[(15, 25)])
    with pytest.raises(ValueError, match="unfinished atomic"):
        _normalize_cache(p, 15, 70)


def test_false_raw_invalid_grid_or_mismatched_segment_is_rejected():
    p = payload(grid_valid=[True, False, True, True])
    with pytest.raises(ValueError, match="invalid-book"):
        _normalize_cache(p, 15, 70)
    p = payload(grid_segments=[1, 2, 1, 1])
    with pytest.raises(ValueError, match="book segment"):
        _normalize_cache(p, 15, 70)


def test_actual_price_return_lag_matches_independent_plain_scan_randomized():
    rng = np.random.default_rng(251108)
    for _ in range(12):
        lt = np.r_[0, np.cumsum(rng.integers(1, 10, 30))]
        ft = np.r_[0, np.cumsum(rng.integers(1, 10, 35))]
        lp = 1000 + np.r_[0, np.cumsum(rng.choice([-2, -1, 1, 2], 30))]
        fp = 1000 + np.r_[0, np.cumsum(rng.choice([-2, -1, 1, 2], 35))]
        a, b = path(lt, lp), path(ft, fp)
        e = match_events(a, b, max_horizon_ns=20)
        for row in e.itertuples():
            before = np.searchsorted(ft, row.t, side="right") - 1
            ahead = fp[before] >= row.price2 if row.direction > 0 else fp[before] <= row.price2
            if ahead:
                assert row.status == "already"
                continue
            future = np.flatnonzero((ft > row.t) & (ft <= row.t + 20))
            matches = [j for j in future if (fp[j] >= row.price2 if row.direction > 0 else fp[j] <= row.price2)]
            assert row.latency_ns == (int(ft[matches[0]] - row.t) if matches else -1)
