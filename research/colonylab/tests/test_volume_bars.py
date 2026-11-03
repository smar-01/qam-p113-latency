"""Known-answer trade clock tests: conservation, availability and leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from colonylab.volume_bars import (
    build_time_bars, build_volume_bars, fit_volume_thresholds,
    normalize_trades, sample_completed_bars,
)


DAY = "2026-10-05"
START = pd.Timestamp(f"{DAY}T14:00:00Z")


def raw(times_ms=(10, 20, 30), price=(100, 101, 102), size=(40, 40, 40), **columns):
    count = len(times_ms)
    data = {"ts_recv": START + pd.to_timedelta(times_ms, unit="ms"),
            "symbol": ["TEST"] * count, "action": ["T"] * count,
            "price": price, "size": size, "side": ["B"] * count,
            "flags": [0] * count, "sequence": np.arange(count)}
    data.update(columns)
    frame = pd.DataFrame(data)
    frame.attrs.update(requested_start=START.isoformat(), requested_end=(START + pd.Timedelta(seconds=10)).isoformat())
    return frame


def normal(*args, **kwargs):
    frame = raw(*args, **kwargs)
    return normalize_trades(frame, "TEST.FEED", DAY)[0]


def anchor(ms):
    return START.value + np.asarray(ms, dtype=np.int64) * 1_000_000


def test_trade_print_volume_not_fill_double_count_and_not_last_filtered():
    frame = raw((10, 20, 30, 40), (100, 101, 102, 102), (10, 20, 30, 500),
                action=["T", "T", "T", "F"], flags=[0, 128, 0, 128])
    trades, audit = normalize_trades(frame, "TEST.FEED", DAY)
    assert trades["size"].sum() == 60
    assert audit["fill_records_excluded"] == 1
    assert audit["trade_records_without_last_flag"] == 2


def test_identical_prints_remain_distinct_trades():
    trades = normal((10, 10), (100, 100), (40, 40), sequence=[7, 7])
    bars = build_volume_bars(trades, 50)
    assert bars.volume_shares.tolist() == [80]
    assert bars.trade_count.tolist() == [2]
    assert bars.overrun_shares.tolist() == [30]


def test_unsafe_constituent_invalidates_atomic_timestamp_group():
    trades = normal((10, 10, 20), (100, 100, 100), (40, 40, 40), flags=[0, 8, 0])
    assert trades.valid.tolist() == [False, False, True]
    bars = build_volume_bars(trades, 30)
    assert bars.volume_shares.tolist() == [40]
    assert bars.iloc[0].close_ns == anchor([20])[0]


def test_volume_whole_trade_conservation_and_partial_tail():
    bars = build_volume_bars(normal(), 60)
    assert bars.volume_shares.tolist() == [80]
    assert bars.trade_count.tolist() == [2]
    assert bars.iloc[0].close_ns == anchor([20])[0]
    assert bars.iloc[0].open_ns == START.value
    assert bars.iloc[0].first_trade_ns == anchor([10])[0]
    assert bars.iloc[0].duration_ms == 20
    assert bars.attrs["incomplete_volume_shares"] == 40
    assert bars.volume_shares.sum() + bars.attrs["incomplete_volume_shares"] == 120
    assert bars.iloc[0].vwap == 100.5
    assert bars.iloc[0].return_bps == pytest.approx(100)


def test_big_trade_closes_one_candle_no_fragment_pseudocandles():
    bars = build_volume_bars(normal((10,), (100,), (1000,)), 100)
    assert len(bars) == 1
    assert bars.iloc[0].volume_shares == 1000
    assert bars.iloc[0].overrun_shares == 900
    assert bars.iloc[0].trade_count == 1
    assert bars.iloc[0].duration_ms == 10


def test_same_time_sequence_resolves_ohlc_and_atomic_close():
    trades = normal((10, 10, 10), (101, 102, 100), (30, 30, 30), sequence=[2, 3, 1])
    bars = build_volume_bars(trades, 30)
    assert len(bars) == 1
    assert bars.iloc[0].open == 100
    assert bars.iloc[0].close == 102
    assert bars.iloc[0].high == 102
    assert bars.iloc[0].low == 100
    assert bars.iloc[0].volume_shares == 90


@pytest.mark.parametrize("sequence", ([1, 1], [np.nan, 2]))
def test_distinct_same_time_prices_without_unique_sequence_are_barriers(sequence):
    trades, audit = normalize_trades(raw((10, 10), (100, 102), (40, 40), sequence=sequence), "TEST.FEED", DAY)
    assert not trades.valid.any()
    assert audit["ambiguous_price_timestamp_groups"] == 1
    assert build_volume_bars(trades, 50).empty


@pytest.mark.parametrize("flag", (4, 8, 32, 128 | 4))
def test_bad_trade_flags_break_volume_clock_and_invalidate_time_interval(flag):
    trades = normal((10, 20, 30, 40), (100, 100, 100, 100), (40, 500, 40, 40), flags=[0, flag, 0, 0])
    bars = build_volume_bars(trades, 60)
    assert bars.volume_shares.tolist() == [80]
    assert bars.iloc[0].open_ns == anchor([20])[0]
    assert bars.iloc[0].first_trade_ns == anchor([30])[0]
    assert bars.iloc[0].duration_ms == 20
    assert bars.attrs["barrier_discarded_volume_shares"] == 40
    assert build_time_bars(trades).empty


@pytest.mark.parametrize("price,size", ((np.nan, 10), (0, 10), (100, 0), (100, 1.5), (100, np.nan)))
def test_invalid_print_values_are_barriers(price, size):
    trades = normal((10,), (price,), (size,))
    assert not trades.valid.any()
    assert build_volume_bars(trades, 1).empty


def test_buy_sell_and_unknown_volume_sum_to_total_and_imbalance_uses_total():
    bars = build_volume_bars(normal(side=["B", "A", "N"]), 100)
    row = bars.iloc[0]
    assert row.buy_volume + row.sell_volume + row.unknown_volume == row.volume_shares == 120
    assert row.signed_volume == 0
    sampled = sample_completed_bars(bars, anchor([31]))
    assert sampled.iloc[0].signed_volume_fraction == 0
    assert sampled.iloc[0].known_side_fraction == pytest.approx(2 / 3)


def test_volume_candle_not_visible_at_its_own_close():
    bars = build_volume_bars(normal(), 60)
    sampled = sample_completed_bars(bars, anchor([19, 20, 21]))
    assert sampled.valid.tolist() == [False, False, True]
    assert sampled.loc[:1, "return_bps"].isna().all()
    assert sampled.iloc[2].age_ms == 1


def test_time_candle_visible_only_after_fixed_interval_end():
    bars = build_time_bars(normal())
    assert bars.close_ns.tolist() == anchor([1000]).tolist()
    assert bars.open_ns.tolist() == anchor([0]).tolist()
    assert bars.first_trade_ns.tolist() == anchor([10]).tolist()
    assert bars.duration_ms.tolist() == [1000]
    sampled = sample_completed_bars(bars, anchor([30, 999, 1000, 1001]))
    assert sampled.valid.tolist() == [False, False, False, True]


def test_time_interval_endpoint_available_without_endpoint_trade():
    trades = normal((999, 1000), (100, 500), (40, 1000))
    bars = build_time_bars(trades)
    sampled = sample_completed_bars(bars, anchor([999, 1000, 1001]), include_equal_close=True)
    assert sampled.valid.tolist() == [False, True, True]
    assert sampled.iloc[1].log_volume == pytest.approx(np.log1p(40))
    assert sampled.iloc[1].return_bps == 0
    assert sampled.iloc[1].age_ms == 0


def test_time_endpoint_unsafe_print_ignored_at_anchor_but_blocks_later():
    bars = build_time_bars(normal((999, 1000), (100, 500), (40, 1000), flags=[0, 8]))
    sampled = sample_completed_bars(bars, anchor([1000, 1001]), include_equal_close=True)
    assert sampled.valid.tolist() == [True, False]


def test_time_prior_unsafe_print_blocks_endpoint_candle():
    bars = build_time_bars(normal((10, 999), (100, 500), (40, 1000), flags=[0, 8]))
    sampled = sample_completed_bars(bars, anchor([1000]), include_equal_close=True)
    assert sampled.valid.tolist() == [False]


def test_time_interval_boundary_print_enters_next_candle():
    bars = build_time_bars(normal((999, 1000), (100, 105), (40, 50)))
    assert bars.close_ns.tolist() == anchor([1000, 2000]).tolist()
    assert bars.close.tolist() == [100, 105]
    assert bars.volume_shares.sum() == 90
    assert bars.duration_ms.tolist() == [1000, 1000]


def test_volume_clock_duration_counts_wait_after_previous_completion():
    trades = normal((100, 200, 3000), (100, 101, 102), (100, 100, 100))
    bars = build_volume_bars(trades, 100)
    assert bars.open_ns.tolist() == anchor([0, 100, 200]).tolist()
    assert bars.first_trade_ns.tolist() == anchor([100, 200, 3000]).tolist()
    assert bars.duration_ms.tolist() == [100, 100, 2800]
    assert bars.close.tolist() == [100, 101, 102]


def test_time_empty_intervals_are_omitted_and_tail_partial_omitted():
    trades = normal((10, 2010, 9999), (100, 101, 102), (40, 40, 40))
    trades.attrs["requested_end"] = (START + pd.Timedelta(milliseconds=9999.5)).isoformat()
    bars = build_time_bars(trades)
    assert bars.close_ns.tolist() == anchor([1000, 3000]).tolist()


def test_previous_candle_cannot_be_carried_across_later_unsafe_print():
    trades = normal((10, 20, 30, 40, 50), (100,) * 5, (100,) * 5, flags=[0, 0, 8, 0, 0])
    bars = build_volume_bars(trades, 100)
    sampled = sample_completed_bars(bars, anchor([21, 30, 31, 40, 41]))
    assert sampled.valid.tolist() == [True, True, False, False, True]


def test_completed_state_expires():
    bars = build_volume_bars(normal(), 60)
    assert sample_completed_bars(bars, anchor([22, 23]), max_age_ns=2_000_000).valid.tolist() == [True, False]


def test_requested_exposure_and_training_only_threshold_fit():
    first = normal((10,), (100,), (100,))
    heldout = normal((10,), (100,), (100_000_000,))
    thresholds = fit_volume_thresholds({("TEST.FEED", "mbp-1", DAY): first,
                                       ("TEST.FEED", "mbp-1", "2026-10-12"): heldout}, [DAY])
    assert thresholds.iloc[0].training_seconds == 10
    assert thresholds.iloc[0].threshold_shares == 10
    assert thresholds.iloc[0].training_volume_shares == 100
    heldout["size"] = 1
    pd.testing.assert_frame_equal(thresholds, fit_volume_thresholds({("TEST.FEED", "mbp-1", DAY): first,
                                                                  ("TEST.FEED", "mbp-1", "2026-10-12"): heldout}, [DAY]))


def test_threshold_per_venue_symbol_not_pooled():
    a, b = normal((10,), (100,), (100,)), normal((10,), (100,), (200,))
    b["dataset"] = "SECOND.FEED"
    thresholds = fit_volume_thresholds({("TEST.FEED", "mbp-1", DAY): a,
                                       ("SECOND.FEED", "mbp-1", DAY): b}, [DAY])
    assert thresholds.set_index("dataset").threshold_shares.to_dict() == {"TEST.FEED": 10, "SECOND.FEED": 20}


def test_threshold_requires_requested_exposure():
    trades = normal()
    trades.attrs.clear()
    with pytest.raises(ValueError, match="explicit requested"):
        fit_volume_thresholds({("TEST.FEED", "mbp-1", DAY): trades}, [DAY])


def test_zero_trade_requested_day_still_counts_training_exposure():
    active = normal((10,), (100,), (100,))
    quiet, _ = normalize_trades(raw(action=["M", "M", "M"]), "TEST.FEED", "2026-10-06")
    fitted = fit_volume_thresholds({("TEST.FEED", "mbp-1", DAY): active,
                                    ("TEST.FEED", "mbp-1", "2026-10-06"): quiet}, [DAY, "2026-10-06"])
    assert fitted.iloc[0].training_seconds == 20
    assert fitted.iloc[0].training_days == 2
    assert fitted.iloc[0].threshold_shares == 5


def test_receive_clock_required_not_publisher_time():
    frame = raw().rename(columns={"ts_recv": "ts_event"})
    with pytest.raises(ValueError, match="receive clock"):
        normalize_trades(frame, "TEST.FEED", DAY)


def test_builders_refuse_mixed_stocks():
    trades = normal(symbol=["A", "B", "A"])
    with pytest.raises(ValueError, match="exactly one"):
        build_volume_bars(trades, 10)


def test_empty_trade_frame_returns_empty_features():
    trades, _ = normalize_trades(raw(action=["M", "M", "M"]), "TEST.FEED", DAY)
    bars = build_volume_bars(trades, 10)
    sampled = sample_completed_bars(bars, anchor([10, 20]))
    assert sampled.valid.tolist() == [False, False]
    assert sampled.return_bps.isna().all()


def test_future_print_mutation_cannot_change_pre_anchor_features():
    trades = normal((10, 20, 30), (100, 101, 102), (100, 100, 100))
    before = sample_completed_bars(build_volume_bars(trades, 100), anchor([21]))
    trades.loc[2, ["price", "size"]] = [5000, 1_000_000]
    after = sample_completed_bars(build_volume_bars(trades, 100), anchor([21]))
    pd.testing.assert_frame_equal(before, after)


def _reference_normalization(frame):
    """Small scalar reference, deliberately independent of NumPy reductions."""
    rows = []
    for order, row in enumerate(frame.itertuples(index=False)):
        if row.action != "T":
            continue
        valid = (np.isfinite(row.flags) and not (int(row.flags) & (32 | 8 | 4)) and
                 np.isfinite(row.price) and 0 < row.price < 1e8 and
                 np.isfinite(row.size) and row.size > 0 and row.size == np.floor(row.size))
        rows.append(dict(time_ns=row.ts_recv.value, symbol=row.symbol, price=row.price,
                         size=row.size, sign={"B": 1, "A": -1}.get(row.side, 0),
                         valid=valid, sequence=row.sequence, row_order=order))
    expected = pd.DataFrame(rows).sort_values(
        ["symbol", "time_ns", "sequence", "row_order"], kind="stable").reset_index(drop=True)
    ambiguous = 0
    for _, group in expected.groupby(["symbol", "time_ns"], sort=False):
        if not group.valid.all():
            expected.loc[group.index, "valid"] = False
        if group.price.nunique(dropna=False) > 1 and (
                group.sequence.isna().any() or group.sequence.duplicated().any()):
            expected.loc[group.index, "valid"] = False
            ambiguous += 1
    expected["segment"] = (~expected.valid).groupby(expected.symbol).cumsum().astype(np.int64)
    return expected, ambiguous


def _reference_volume_bars(trades, threshold):
    """Scalar whole-group accumulator for known-equivalent candle outcomes."""
    import itertools

    rows, pending = [], []
    clock, segment, discarded = START.value, 0, 0.0
    for stamp, group in itertools.groupby(trades.itertuples(index=False), lambda row: row.time_ns):
        group = list(group)
        if not all(row.valid for row in group):
            discarded += sum(row.size for row in pending)
            pending, clock, segment = [], stamp, segment + 1
            continue
        pending.extend(group)
        volume = sum(row.size for row in pending)
        if volume < threshold:
            continue
        op, cl = pending[0].price, pending[-1].price
        high, low = max(row.price for row in pending), min(row.price for row in pending)
        rows.append(dict(
            open_ns=clock, first_trade_ns=pending[0].time_ns, close_ns=stamp,
            open=op, high=high, low=low, close=cl,
            vwap=sum(row.price * row.size for row in pending) / volume,
            volume_shares=volume, trade_count=len(pending),
            signed_volume=sum(row.size * row.sign for row in pending),
            buy_volume=sum(row.size for row in pending if row.sign > 0),
            sell_volume=sum(row.size for row in pending if row.sign < 0),
            unknown_volume=sum(row.size for row in pending if row.sign == 0),
            duration_ms=(stamp - clock) / 1e6, return_bps=(cl / op - 1) * 1e4,
            range_bps=(high - low) / op * 1e4, overshoot_shares=volume - threshold,
            overrun_shares=volume - threshold, threshold_shares=threshold, segment=segment,
            symbol=pending[0].symbol, dataset=pending[0].dataset, day=pending[0].day))
        pending, clock = [], stamp
    return rows, sum(row.size for row in pending), discarded


@pytest.mark.parametrize("seed", range(8))
def test_vectorized_normalization_and_bars_match_scalar_reference(seed):
    rng = np.random.default_rng(seed)
    n = 500
    frame = raw(
        times_ms=rng.integers(0, 100, n) * 80,
        price=rng.choice([100.0, 100.01, 99.99, np.nan, 0.0], n,
                         p=[.49, .25, .23, .02, .01]),
        size=rng.choice([1.0, 10.0, 50.0, 100.0, 1.5, np.nan], n,
                        p=[.15, .25, .30, .27, .02, .01]),
        symbol=rng.choice(["A", "B"], n), action=rng.choice(["T", "F", "M"], n, p=[.8, .1, .1]),
        side=rng.choice(["B", "A", "N"], n),
        flags=rng.choice([0.0, 128.0, 4.0, 8.0, 32.0, np.nan], n,
                         p=[.72, .24, .01, .01, .01, .01]),
        sequence=rng.choice(np.r_[np.arange(60, dtype=float), np.nan], n))
    trades, audit = normalize_trades(frame, "TEST.FEED", DAY)
    expected, ambiguous = _reference_normalization(frame)
    pd.testing.assert_frame_equal(trades[expected.columns], expected, check_dtype=False)
    assert audit["ambiguous_price_timestamp_groups"] == ambiguous
    for symbol in ["A", "B"]:
        one = trades.loc[trades.symbol.eq(symbol)].copy()
        for threshold in [1, 75, 333, 100_000]:
            bars = build_volume_bars(one, threshold)
            rows, incomplete, discarded = _reference_volume_bars(one, threshold)
            expected_bars = pd.DataFrame(rows, columns=bars.columns)
            pd.testing.assert_frame_equal(bars, expected_bars, check_dtype=False,
                                          rtol=1e-13, atol=1e-10)
            assert bars.attrs["incomplete_volume_shares"] == incomplete
            assert bars.attrs["barrier_discarded_volume_shares"] == discarded
            expected_bars.attrs.update(one.attrs)
            expected_bars.attrs["barriers_ns"] = np.unique(one.loc[~one.valid, "time_ns"].to_numpy())
            anchors = anchor(np.arange(0, 10000, 37))
            pd.testing.assert_frame_equal(sample_completed_bars(bars, anchors),
                                          sample_completed_bars(expected_bars, anchors),
                                          rtol=1e-12, atol=1e-10)
