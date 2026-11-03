"""Independent microstructure checks for the venue response pilot.

The synthetic books encode known answers; no network calls or API key are used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from colonylab import network


DAY = "2026-10-21"
ORIGIN = pd.Timestamp(f"{DAY}T14:00:00Z")


def raw_quotes(times_ms, mids, *, flags=None):
    count = len(times_ms)
    return pd.DataFrame({
        "ts_recv": ORIGIN + pd.to_timedelta(times_ms, unit="ms"),
        "ts_event": ORIGIN + pd.to_timedelta(times_ms, unit="ms"),
        "symbol": "TEST", "publisher_id": 1, "action": "M", "side": "N",
        "price": mids, "size": 10,
        "bid_px_00": np.asarray(mids) - .01,
        "ask_px_00": np.asarray(mids) + .01,
        "bid_sz_00": 100, "ask_sz_00": 120,
        "flags": [128] * count if flags is None else flags,
    })


def normal(times_ms, mids, *, valid=None, segments=None):
    count = len(times_ms)
    return pd.DataFrame({
        "time_ns": ORIGIN.value + np.asarray(times_ms, dtype=np.int64) * 1_000_000,
        "mid": mids, "spread_bps": 2.0, "depth_imbalance": 0.0,
        "bid_size": 100, "ask_size": 120,
        "valid": np.ones(count, dtype=bool) if valid is None else valid,
        "segment": np.zeros(count, dtype=int) if segments is None else segments,
        "symbol": "TEST", "dataset": "TEST.FEED", "day": DAY,
    })


def trigger(times_ms=(100,), directions=(1,), segments=None):
    return pd.DataFrame({
        "time_ns": ORIGIN.value + np.asarray(times_ms, dtype=np.int64) * 1_000_000,
        "mid": 100., "direction": directions, "jump": .1,
        "segment": np.zeros(len(times_ms), dtype=int) if segments is None else segments,
    })


def edge(source, target, target_events, *, horizon=10, shift=0):
    return network.edge_response(
        source, target, target_events, horizon_ms=horizon, shift_ms=shift,
        min_lead_ms=2, max_age_ms=250,
    )


def test_receive_clock_is_used_even_when_event_clock_disagrees():
    frame = raw_quotes([0, 10, 20], [100., 100.1, 100.2])
    frame["ts_event"] = frame["ts_event"].iloc[::-1].to_numpy()
    quotes, _ = network.normalize_quotes(frame, "TEST.FEED", DAY)
    assert quotes.time_ns.tolist() == (ORIGIN.value + np.array([0, 10, 20]) * 1_000_000).tolist()
    assert quotes.mid.tolist() == pytest.approx([100., 100.1, 100.2])


def test_last_flag_is_bitwise_and_atomic_updates_exclude_transient_quotes():
    frame = raw_quotes([0, 1, 2, 3], [100., 150., 100.1, 100.2], flags=[129, 0, 128, 129])
    quotes, _ = network.normalize_quotes(frame, "TEST.FEED", DAY)
    assert 150. not in quotes.mid.tolist()
    assert quotes.loc[quotes.valid, "mid"].tolist() == pytest.approx([100., 100.1, 100.2])


@pytest.mark.parametrize("bad_flag", [4, 8, 32])
def test_bad_book_bad_receive_clock_and_snapshot_are_unknown_barriers(bad_flag):
    frame = raw_quotes([0, 10, 20], [100., 100.1, 100.2], flags=[128, 128 | bad_flag, 128])
    quotes, _ = network.normalize_quotes(frame, "TEST.FEED", DAY)
    barrier = quotes.loc[quotes.time_ns == ORIGIN.value + 10_000_000]
    assert len(barrier) == 1
    assert not barrier.valid.iloc[0]
    assert quotes.valid.iloc[0] and quotes.valid.iloc[-1]


@pytest.mark.parametrize("bad_state", ["crossed", "missing", "zero"])
def test_invalid_quote_state_is_retained_as_barrier(bad_state):
    frame = raw_quotes([0, 10, 20], [100., 100.1, 100.2])
    if bad_state == "crossed":
        frame.loc[1, ["bid_px_00", "ask_px_00"]] = [101., 100.]
    elif bad_state == "missing":
        frame.loc[1, "ask_px_00"] = np.nan
    else:
        frame.loc[1, "bid_px_00"] = 0.
    quotes, _ = network.normalize_quotes(frame, "TEST.FEED", DAY)
    assert len(quotes) == 3
    assert not quotes.valid.iloc[1]


def test_conflicting_tied_quote_states_are_order_invariant_unknowns():
    frame = raw_quotes([0, 10, 10, 20], [100., 100.1, 100.2, 100.3])
    forward, _ = network.normalize_quotes(frame, "TEST.FEED", DAY)
    backward, _ = network.normalize_quotes(frame.iloc[::-1], "TEST.FEED", DAY)
    pd.testing.assert_frame_equal(forward, backward)
    tied = forward.loc[forward.time_ns == ORIGIN.value + 10_000_000]
    assert len(tied) == 1 and not tied.valid.iloc[0]


def test_same_midpoint_tied_different_spread_is_also_ambiguous():
    frame = raw_quotes([0, 10, 10, 20], [100., 100.1, 100.1, 100.3])
    frame.loc[2, ["bid_px_00", "ask_px_00"]] += [-.01, .01]
    quotes, _ = network.normalize_quotes(frame, "TEST.FEED", DAY)
    assert not quotes.loc[quotes.time_ns == ORIGIN.value + 10_000_000, "valid"].iloc[0]


def test_shocks_do_not_cross_invalid_quotes_or_new_segments():
    quotes = normal([0, 10, 20, 30, 40], [100., 101., 102., 103., 103.1],
                    valid=[True, False, True, True, True], segments=[0, 0, 0, 1, 1])
    shocks = network.extract_shocks(quotes, threshold=.05, min_separation_ms=0)
    assert shocks.time_ns.tolist() == [ORIGIN.value + 40_000_000]


def test_shocks_do_not_compare_against_stale_quotes():
    shocks = network.extract_shocks(normal([0, 1000], [100., 101.]), threshold=.05, max_age_ms=250)
    assert shocks.empty


def test_threshold_calibration_excludes_invalid_and_cross_segment_jumps():
    quotes = normal([0, 10, 20, 30, 40, 50], [100., 100.01, 1000., 101., 500., 500.01],
                    valid=[True, True, False, True, True, True], segments=[0, 0, 0, 0, 1, 1])
    assert network.fit_threshold(quotes) == pytest.approx(.01)


def test_threshold_calibration_excludes_stale_jump_pairs():
    quotes = normal([0, 10, 1000, 1010], [100., 100.01, 200., 200.01])
    assert network.fit_threshold(quotes) == pytest.approx(.01)


def test_same_direction_shock_is_counted_with_strictly_prior_baseline():
    target = normal([0, 90, 103, 110, 200], [100., 100., 100.1, 100.1, 100.1])
    result = edge(trigger(), target, trigger((103,)))
    assert result["source_count"] == 1
    assert result["eligible_count"] == 1
    assert result["hit_count"] == 1
    assert result["response_probability"] == 1.
    assert result["signed_markout_bps"] == pytest.approx(10.)
    assert result["median_reaction_delay_ms"] == pytest.approx(3.)


def test_preexisting_price_level_is_not_a_new_shock_response():
    target = normal([0, 90, 110, 200], [100., 101., 101., 101.])
    result = edge(trigger(), target, trigger((90,)))
    assert result["eligible_count"] == 1
    assert result["hit_count"] == 0
    assert result["signed_markout_bps"] == pytest.approx(0.)


@pytest.mark.parametrize("target_time", [100, 101])
def test_simultaneous_or_uncertain_lead_target_shocks_are_not_directed_hits(target_time):
    target = normal([0, 90, target_time, 110, 200], [100., 100., 100.1, 100.1, 100.1])
    result = edge(trigger(), target, trigger((target_time,)))
    assert result["eligible_count"] == 1
    assert result["hit_count"] == 0


def test_opposite_direction_shock_is_not_a_hit_and_markout_keeps_sign():
    target = normal([0, 90, 103, 110, 200], [100., 100., 99.9, 99.9, 99.9])
    result = edge(trigger(), target, trigger((103,), (-1,)))
    assert result["eligible_count"] == 1
    assert result["hit_count"] == 0
    assert result["signed_markout_bps"] == pytest.approx(-10.)
    negative_source = edge(trigger(directions=(-1,)), target, trigger((103,), (-1,)))
    assert negative_source["hit_count"] == 1
    assert negative_source["signed_markout_bps"] == pytest.approx(10.)


def test_future_quote_after_horizon_cannot_be_used_as_endpoint():
    target = normal([0, 90, 105, 111, 200], [100., 100., 100.1, 101., 101.])
    result = edge(trigger(), target, trigger((105, 111), (1, 1)))
    assert result["signed_markout_bps"] == pytest.approx(10.)


def test_horizon_beyond_download_support_is_censored():
    result = edge(trigger((195,)), normal([0, 190, 200], [100., 100., 100.1]), trigger((200,)))
    assert result["eligible_count"] == 0
    assert result["hit_count"] == 0
    assert not np.isfinite(result["response_probability"])


def test_unknown_state_anywhere_inside_window_censors_even_after_recovery():
    target = normal([0, 90, 104, 108, 110, 200], [100., 100., np.nan, 100.1, 100.1, 100.1],
                    valid=[True, True, False, True, True, True])
    result = edge(trigger(), target, trigger((108,)))
    assert result["eligible_count"] == 0
    assert result["hit_count"] == 0


def test_new_segment_inside_response_window_is_censored():
    target = normal([0, 90, 103, 110, 200], [100., 100., 100.1, 100.1, 100.1],
                    segments=[0, 0, 1, 1, 1])
    result = edge(trigger(), target, trigger((103,), segments=[1]))
    assert result["eligible_count"] == 0


def test_missing_or_stale_baseline_is_censored():
    missing = edge(trigger(), normal([101, 110, 200], [100., 100.1, 100.1]), trigger((103,)))
    stale = edge(trigger((500,)), normal([0, 510, 600], [100., 100.1, 100.1]), trigger((510,)))
    assert missing["eligible_count"] == stale["eligible_count"] == 0


def test_positive_placebo_shift_queries_later_target_time_not_earlier():
    target = normal([0, 90, 110, 190, 203, 210, 300], [100., 100., 100., 100., 100.1, 100.1, 100.1])
    unshifted = edge(trigger(), target, trigger((203,)))
    shifted = edge(trigger(), target, trigger((203,)), shift=100)
    assert unshifted["hit_count"] == 0
    assert shifted["hit_count"] == 1
    assert shifted["median_reaction_delay_ms"] == pytest.approx(3.)


def test_placebo_does_not_wrap_to_other_end_of_sample():
    target = normal([0, 90, 103, 110, 200], [100., 100., 100.1, 100.1, 100.1])
    shifted = edge(trigger(), target, trigger((103,)), shift=1000)
    assert shifted["source_count"] == 1
    assert shifted["eligible_count"] == shifted["hit_count"] == 0


def test_duplicate_source_events_are_not_created_from_constant_quotes():
    shocks = network.extract_shocks(normal([0, 10, 20, 30], [100., 100., 100., 100.]), threshold=.005)
    assert shocks.empty
