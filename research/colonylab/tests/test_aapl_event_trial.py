"""Boundary, exposure and economic-unit checks for the saved AAPL event trial."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from colonylab.aapl_event_trial import (
    audit_matches,
    observed_seconds,
    price_profile,
    shock_arrays,
    venue_metrics,
    window_events,
)
from colonylab.event_midpoints import NativeMidpoints, match_events, summarize_events


SECOND = 1_000_000_000
PRICE = 200_000_000_000  # $100 midpoint on the integer bid+ask lattice.


def path(times, prices, *, valid=None, segments=None, start=0, end=None, grid=None):
    times = np.asarray(times, dtype=np.int64)
    return NativeMidpoints(
        times, np.asarray(prices, dtype=np.int64),
        np.ones(len(times), dtype=bool) if valid is None else np.asarray(valid, dtype=bool),
        np.zeros(len(times), dtype=np.int64) if segments is None else np.asarray(segments, dtype=np.int64),
        start, int(times[-1] + SECOND) if end is None and len(times) else end,
        grid,
    )


def protocol(**extra):
    return {"shock_thresholds_bp": [1.], "shock_cluster_separation_ms": 100,
            "outer_start_minute": -1, "outer_end_minute": 1,
            "price_path_grid_seconds": 1, **extra}


def test_subwindow_is_half_open_and_preserves_prior_quality_truncation():
    source = pd.DataFrame({"t": [100, 150, 175, 199, 200],
                           "latency_ns": [99, 50, 3, 1, -1],
                           "observed_ns": [100, 100, 10, 100, 100],
                           "terminal_reason": ["horizon", "horizon", "quality", "horizon", "horizon"],
                           "status": ["behind"] * 5})
    source.attrs["max_horizon_ns"] = 100
    before = source.copy(deep=True)
    selected = window_events(source, 100, 200)
    assert selected.t.tolist() == [100, 150, 175, 199]
    assert selected.observed_ns.tolist() == [99, 49, 10, 0]
    assert selected.latency_ns.tolist() == [99, -1, 3, -1]
    assert selected.terminal_reason.tolist() == ["end", "end", "quality", "end"]
    assert selected.attrs == source.attrs
    pd.testing.assert_frame_equal(source, before)


def test_early_match_near_window_end_does_not_enter_complete_followup_cohort():
    events = pd.DataFrame({"t": [90], "latency_ns": [1], "observed_ns": [100],
                           "terminal_reason": ["horizon"], "status": ["behind"]})
    events.attrs["max_horizon_ns"] = 100
    result = summarize_events(window_events(events, 0, 100), 10)
    assert result["n_events"] == result["n_observed_matches"] == 1
    assert result["known_matches_truncated"] == result["n_right_truncated"] == 1
    assert result["n_eligible"] == result["n_matched"] == 0
    assert np.isnan(result["median_us"])


def test_observed_exposure_respects_true_support_and_quality_intervals():
    series = path(np.array([0, 2, 4, 8]) * SECOND, [PRICE] * 4,
                  valid=[True, True, False, True], start=SECOND, end=10 * SECOND)
    # The prior seed is not evidence before declared support. Valid intervals
    # are [1,4) and [8,10), including quiet states until the next barrier.
    assert observed_seconds(series, 0, 12 * SECOND) == 5
    assert observed_seconds(series, 3 * SECOND, 9 * SECOND) == 2
    assert observed_seconds(series, 4 * SECOND, 8 * SECOND) == 0


def test_quiet_initialized_book_remains_observed_without_price_change_age_cut():
    series = path([0], [PRICE], start=SECOND, end=601 * SECOND)
    assert observed_seconds(series, SECOND, 601 * SECOND) == 600
    empty = path([], [], start=0, end=SECOND)
    assert observed_seconds(empty, 0, SECOND) == 0


@pytest.mark.parametrize("threshold", [.5, 1., 2.])
@pytest.mark.parametrize("direction", [-1, 1])
def test_shock_threshold_uses_basis_points_of_prior_midpoint_exactly(threshold, direction):
    # PRICE=2e11, so one bp is 2e7 price2 units. The exact threshold is
    # included, while one integer price2 unit below it is excluded.
    move = int(PRICE * threshold / 10_000)
    at = path([0, SECOND], [PRICE, PRICE + direction * move], end=2 * SECOND)
    below = path([0, SECOND], [PRICE, PRICE + direction * (move - 1)], end=2 * SECOND)
    for times in shock_arrays(at, threshold, 100_000_000):
        assert times.tolist() == [SECOND]
    for times in shock_arrays(below, threshold, 100_000_000):
        assert times.size == 0


def test_shocks_do_not_bridge_invalid_or_changed_segment_states():
    series = path([0, 1, 2, 3, 4], [PRICE, 0, PRICE + 50_000_000, PRICE, PRICE + 50_000_000],
                  valid=[True, False, True, True, True], segments=[0, 0, 1, 1, 2], end=10)
    raw, clustered = shock_arrays(series, 1, 100)
    assert raw.tolist() == clustered.tolist() == [3]


def test_causal_shock_thinning_uses_accepted_event_and_exact_separation_boundary():
    times = [0, 1, 50_000_001, 100_000_001, 100_000_002]
    series = path(times, [PRICE, PRICE + 50_000_000, PRICE, PRICE + 50_000_000, PRICE],
                  segments=[0, 0, 0, 0, 1], end=SECOND)
    raw, clustered = shock_arrays(series, 1, 100_000_000)
    assert raw.tolist() == [1, 50_000_001, 100_000_001]
    assert clustered.tolist() == [1, 100_000_001]
    prefix = path(times[:-1], series.price2[:-1], end=SECOND)
    assert np.array_equal(shock_arrays(prefix, 1, 100_000_000)[1], clustered)
    with pytest.raises(ValueError, match="half-basis-point"):
        shock_arrays(series, .75, 100_000_000)


def test_cluster_cooldown_carries_across_analysis_window_boundaries():
    series = path([0, 950_000_000, 1_020_000_000, 1_060_000_000],
                  [PRICE, PRICE + 50_000_000, PRICE, PRICE + 50_000_000], end=2 * SECOND)
    rows = venue_metrics(series, [{"name": "pre", "start_ns": 0, "end_ns": SECOND},
                                  {"name": "post", "start_ns": SECOND, "end_ns": 2 * SECOND}], protocol())
    pre, post = rows
    assert (pre["shocks_raw"], pre["shocks_clustered"]) == (1, 1)
    assert (post["shocks_raw"], post["shocks_clustered"]) == (2, 1)
    assert post["shocks_per_minute"] == 120
    assert post["clustered_shocks_per_minute"] == 60


def test_shock_and_change_rates_use_observed_exposure_not_scheduled_time():
    series = path([0, SECOND, 2 * SECOND, 3 * SECOND, 4 * SECOND, 6 * SECOND],
                  [PRICE, PRICE + 50_000_000, PRICE, 0, PRICE, PRICE],
                  valid=[True, True, True, False, True, True],
                  segments=[0, 0, 0, 0, 1, 1], end=8 * SECOND)
    row, = venue_metrics(series, [{"name": "window", "start_ns": 0, "end_ns": 8 * SECOND}], protocol())
    assert row["observed_seconds"] == 7
    assert row["coverage_fraction"] == 7 / 8
    assert row["midpoint_changes"] == row["shocks_raw"] == 2
    assert row["shocks_per_minute"] == pytest.approx(120 / 7)
    assert row["midpoint_changes_per_minute"] == pytest.approx(120 / 7)
    assert np.isnan(row["signed_return_bp"])


def test_window_return_and_price_profile_use_strict_prior_anchor_state():
    anchor = 120 * SECOND
    series = path([0, anchor, anchor + SECOND], [PRICE, PRICE * 2, PRICE * 3], end=240 * SECOND)
    row, = venue_metrics(series, [{"name": "post", "start_ns": anchor, "end_ns": anchor + SECOND}], protocol())
    assert row["signed_return_bp"] == pytest.approx(10_000 * np.log(2))
    profile = price_profile({"A": series}, ["A"], anchor, protocol())
    at_anchor = profile.loc[profile.time_ns.eq(anchor)].iloc[0]
    next_tick = profile.loc[profile.time_ns.eq(anchor + SECOND)].iloc[0]
    assert at_anchor.cross_venue_median_mid == 100
    assert at_anchor.return_from_anchor_bp == 0
    assert next_tick.cross_venue_median_mid == 200
    assert next_tick.return_from_anchor_bp == pytest.approx(10_000 * np.log(2))


def test_price_profile_requires_supported_exact_anchor_for_return_baseline():
    anchor = 120 * SECOND
    series = path([0, anchor - SECOND, anchor + SECOND], [PRICE, 0, PRICE * 2],
                  valid=[True, False, True], segments=[0, 0, 1], end=240 * SECOND)
    profile = price_profile({"A": series}, ["A"], anchor, protocol())
    assert profile.cross_venue_median_mid.notna().any()
    assert profile.loc[profile.time_ns.eq(anchor), "cross_venue_median_mid"].isna().all()
    assert profile.return_from_anchor_bp.isna().all()


def test_price_profile_does_not_extrapolate_past_declared_stream_support():
    anchor = 120 * SECOND
    short = path([0], [PRICE], start=anchor, end=anchor + 2 * SECOND)
    profile = price_profile({"A": short}, ["A"], anchor, protocol())
    assert profile.loc[profile.time_ns.lt(anchor), "n_venues"].eq(0).all()
    assert profile.loc[profile.time_ns.ge(anchor + 2 * SECOND), "n_venues"].eq(0).all()
    assert profile.loc[profile.time_ns.eq(anchor + SECOND), "n_venues"].eq(1).all()


def test_independent_match_audit_detects_corrupted_first_passage():
    leader = path([0, 100], [PRICE, PRICE + 50_000_000], end=1000)
    follower = path([0, 105], [PRICE, PRICE + 50_000_000], end=1000)
    events = match_events(leader, follower, max_horizon_ns=100)
    audit = audit_matches(leader, follower, events, 100)
    assert audit == {"leader_events": 1, "sampled_cases": 1, "mismatches": 0}
    corrupted = events.copy()
    corrupted.loc[0, "latency_ns"] = 6
    with pytest.raises(ValueError, match="Independent midpoint first-passage mismatch"):
        audit_matches(leader, follower, corrupted, 100)
