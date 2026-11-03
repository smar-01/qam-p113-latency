"""Causal, per-venue trade candles for volume-clock comparisons.

Only MBP-1 action-T prints enter volume. Action-F fills are deliberately not
added, and F_LAST is a book-state flag, not a trade-print filter. Whole trades
sharing a receive timestamp stay together; a volume candle closes when their
total reaches its training-frozen share threshold. Overshoot is retained.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


BAD_FLAGS = 32 | 8 | 4  # snapshot, inaccurate receive time, maybe bad book
BAR_COLUMNS = [
    "open_ns", "first_trade_ns", "close_ns", "open", "high", "low", "close", "vwap",
    "volume_shares", "trade_count", "signed_volume", "buy_volume", "sell_volume", "unknown_volume",
    "duration_ms", "return_bps", "range_bps", "overshoot_shares", "overrun_shares", "threshold_shares", "segment",
    "symbol", "dataset", "day",
]
FEATURE_COLUMNS = [
    "return_bps", "range_bps", "log_volume", "log_trade_count",
    "log_duration_ms", "signed_volume_fraction", "known_side_fraction",
    "vwap_close_bps", "age_ms",
]


def _text(series: pd.Series) -> pd.Series:
    return series.map(lambda value: value.decode() if isinstance(value, bytes) else str(value))


def _numeric(frame: pd.DataFrame, name: str, default=np.nan) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def normalize_trades(frame: pd.DataFrame, dataset: str, day: str) -> tuple[pd.DataFrame, dict]:
    """Keep genuine trade prints and unsafe-print barriers on the receive clock.

    Invalid T prints remain barriers rather than silently connecting a candle
    across unknown volume. Sequence sorts same-time prints when supplied. If
    same-time price ordering cannot be resolved, that atomic group is a barrier.
    Repeated prints are retained: equal fields do not prove duplicate trades.
    """
    if not {"action", "price", "size", "symbol"} <= set(frame.columns):
        raise ValueError("Trade candles require MBP-1 action, price, size and symbol fields.")
    if "ts_recv" in frame:
        stamps = frame["ts_recv"]
    elif isinstance(frame.index, pd.DatetimeIndex) and frame.index.name == "ts_recv":
        stamps = frame.index
    else:
        raise ValueError("Trade candles require the ts_recv receive clock.")
    stamps = pd.to_datetime(stamps, utc=True, errors="coerce")
    if pd.isna(stamps).any():
        raise ValueError("Trade receive timestamps must not be missing.")
    time_ns = np.asarray(stamps.astype("datetime64[ns, UTC]").astype("int64"), dtype=np.int64)
    action = _text(frame["action"]).to_numpy()
    keep = action == "T"
    flags = _numeric(frame, "flags", 0)
    bits = np.where(np.isfinite(flags), flags, 0).astype(np.uint64)
    prices, sizes = _numeric(frame, "price"), _numeric(frame, "size")
    unsafe = ~np.isfinite(flags) | ((bits & BAD_FLAGS) != 0)
    valid = (np.isfinite(prices) & (prices > 0) & (prices < 1e8) &
             np.isfinite(sizes) & (sizes > 0) & (sizes == np.floor(sizes)) & ~unsafe)
    side = _text(frame["side"]).to_numpy() if "side" in frame else np.repeat("N", len(frame))
    trades = pd.DataFrame({
        "time_ns": time_ns, "price": prices, "size": sizes,
        "sign": np.where(side == "B", 1, np.where(side == "A", -1, 0)),
        "valid": valid, "sequence": _numeric(frame, "sequence"),
        "row_order": np.arange(len(frame)), "symbol": _text(frame["symbol"]).to_numpy(),
    }).loc[keep].sort_values(["symbol", "time_ns", "sequence", "row_order"], kind="stable")
    ambiguous = 0
    if len(trades):
        # Work on contiguous atomic groups, avoiding a DataFrame allocation for
        # every print timestamp. Sorting above puts duplicate sequences next to
        # one another, including when the input arrived out of order.
        symbol = trades.symbol.to_numpy()
        stamp = trades.time_ns.to_numpy(dtype=np.int64)
        starts = np.r_[0, np.flatnonzero((symbol[1:] != symbol[:-1]) |
                                       (stamp[1:] != stamp[:-1])) + 1]
        counts = np.diff(np.r_[starts, len(trades)])
        first = np.repeat(starts, counts)
        price = trades.price.to_numpy(dtype=float)
        equal_price = (price == price[first]) | (np.isnan(price) & np.isnan(price[first]))
        distinct_prices = ~np.logical_and.reduceat(equal_price, starts)
        sequence = trades.sequence.to_numpy(dtype=float)
        duplicate = np.r_[False, sequence[1:] == sequence[:-1]]
        duplicate[starts] = False
        ambiguous_groups = distinct_prices & (
            np.logical_or.reduceat(np.isnan(sequence), starts) |
            np.logical_or.reduceat(duplicate, starts))
        safe = np.logical_and.reduceat(trades.valid.to_numpy(dtype=bool), starts)
        trades["valid"] = np.repeat(safe & ~ambiguous_groups, counts)
        ambiguous = int(ambiguous_groups.sum())
    trades["segment"] = (~trades.valid).groupby(trades.symbol).cumsum().astype(np.int64)
    trades["dataset"], trades["day"] = dataset, day
    trades = trades.reset_index(drop=True)
    trades.attrs.update(frame.attrs)
    # Include stocks seen in quote rows even on days with no T prints. Such
    # requested quiet time must remain in the training exposure denominator.
    trades.attrs["requested_symbols"] = sorted(_text(frame["symbol"]).unique().tolist())
    diagnostics = {
        "dataset": dataset, "day": day, "raw_records": len(frame),
        "trade_records": int(keep.sum()), "fill_records_excluded": int((action == "F").sum()),
        "valid_trade_records": int(trades.valid.sum()), "invalid_trade_records": int((~trades.valid).sum()),
        "trade_records_without_last_flag": int((keep & ((bits & 128) == 0)).sum()),
        "ambiguous_price_timestamp_groups": ambiguous, "flags_present": "flags" in frame,
        "valid_volume_shares": float(trades.loc[trades.valid, "size"].sum()),
        "unknown_side_volume_shares": float(trades.loc[trades.valid & trades.sign.eq(0), "size"].sum()),
        "clock": "ts_recv", "same_timestamp_policy": "Atomic group, ordered by sequence when distinct prices occur",
    }
    return trades, diagnostics


def fit_volume_thresholds(trades_by_key: Mapping, training_days: Sequence[str], *,
                          target_seconds: float = 1.0) -> pd.DataFrame:
    """Estimate venue/stock share quanta using training volume/requested time.

    DataFrame attrs requested_start/requested_end are mandatory. Quiet requested
    time counts in the denominator; the first/last observed trade does not define
    the exposure. Held-out rows are never read for threshold fitting.
    """
    if not np.isfinite(target_seconds) or target_seconds <= 0:
        raise ValueError("target_seconds must be finite and positive.")
    records = []
    train = set(training_days)
    for key, trades in trades_by_key.items():
        if not isinstance(key, tuple) or len(key) != 3:
            raise ValueError("Trade frame keys must be (dataset, schema, day) triples.")
        dataset, schema, day = key
        if day not in train:
            continue
        if schema != "mbp-1":
            continue
        if "requested_start" not in trades.attrs or "requested_end" not in trades.attrs:
            raise ValueError("Threshold fitting requires explicit requested_start/requested_end attrs.")
        duration = (pd.Timestamp(trades.attrs["requested_end"]) -
                    pd.Timestamp(trades.attrs["requested_start"])).total_seconds()
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Requested threshold exposure must be positive.")
        if len(trades) and set(trades.dataset) != {dataset}:
            raise ValueError("Training trade dataset must agree with its frame key.")
        symbols = trades.attrs.get("requested_symbols", sorted(trades.symbol.unique().tolist()))
        for symbol in symbols:
            group = trades.loc[trades.symbol.eq(symbol)]
            records.append({"dataset": dataset, "symbol": symbol, "day": day,
                            "training_seconds": duration,
                            "training_volume_shares": float(group.loc[group.valid, "size"].sum()),
                            "training_valid_trades": int(group.valid.sum()),
                            "training_invalid_trades": int((~group.valid).sum())})
    if not records:
        raise ValueError("No training trade frames were supplied.")
    rows = pd.DataFrame(records)
    if rows.duplicated(["dataset", "symbol", "day"]).any():
        raise ValueError("Each training venue/symbol/day may contribute exposure only once.")
    result = rows.groupby(["dataset", "symbol"], as_index=False).agg(
        training_seconds=("training_seconds", "sum"),
        training_volume_shares=("training_volume_shares", "sum"),
        training_valid_trades=("training_valid_trades", "sum"),
        training_invalid_trades=("training_invalid_trades", "sum"),
        training_days=("day", "nunique"))
    result["target_seconds"] = float(target_seconds)
    result["threshold_shares"] = np.maximum(1, np.floor(
        result.training_volume_shares / result.training_seconds * target_seconds + .5)).astype(np.int64)
    return result


def _one_series(trades: pd.DataFrame) -> None:
    if any(trades[column].nunique(dropna=False) > 1 for column in ("symbol", "dataset", "day")):
        raise ValueError("Build candles for exactly one venue/symbol/day at a time.")
    if len(trades) and not trades.time_ns.is_monotonic_increasing:
        raise ValueError("Trades must be sorted by receive time.")


def _atomic_arrays(trades: pd.DataFrame) -> dict[str, np.ndarray]:
    """Reduce whole receive-timestamp groups without per-group Pandas objects."""
    stamp = trades.time_ns.to_numpy(dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(stamp[1:] != stamp[:-1]) + 1]
    ends = np.r_[starts[1:], len(trades)]
    price = trades.price.to_numpy(dtype=float)
    size = trades["size"].to_numpy(dtype=float)
    sign = trades.sign.to_numpy(dtype=float)
    return {
        "time_ns": stamp[starts],
        "valid": np.logical_and.reduceat(trades.valid.to_numpy(dtype=bool), starts),
        "open": price[starts], "close": price[ends - 1],
        "high": np.maximum.reduceat(price, starts), "low": np.minimum.reduceat(price, starts),
        "volume_shares": np.add.reduceat(size, starts),
        "notional": np.add.reduceat(price * size, starts), "trade_count": ends - starts,
        "signed_volume": np.add.reduceat(size * sign, starts),
        "unknown_volume": np.add.reduceat(np.where(sign == 0, size, 0), starts),
        "buy_volume": np.add.reduceat(np.where(sign > 0, size, 0), starts),
        "sell_volume": np.add.reduceat(np.where(sign < 0, size, 0), starts),
    }


def _groups(trades: pd.DataFrame):
    arrays = _atomic_arrays(trades)
    columns = tuple(name for name in arrays if name not in ("time_ns", "valid"))
    for index, time_ns in enumerate(arrays["time_ns"]):
        yield int(time_ns), ({name: arrays[name][index] for name in columns}
                             if arrays["valid"][index] else None)


def _combine(parts: list[tuple[int, dict]], close_ns: int, meta: dict, segment: int,
             overshoot: float = 0.0, *, interval_start_ns: int) -> dict:
    values = [value for _, value in parts]
    volume = sum(value["volume_shares"] for value in values)
    op, cl = values[0]["open"], values[-1]["close"]
    high, low = max(value["high"] for value in values), min(value["low"] for value in values)
    return {
        "open_ns": int(interval_start_ns), "first_trade_ns": parts[0][0],
        "close_ns": int(close_ns), "open": op, "close": cl,
        "high": high, "low": low, "vwap": sum(value["notional"] for value in values) / volume,
        "volume_shares": volume, "trade_count": sum(value["trade_count"] for value in values),
        "signed_volume": sum(value["signed_volume"] for value in values),
        "buy_volume": sum(value["buy_volume"] for value in values),
        "sell_volume": sum(value["sell_volume"] for value in values),
        "unknown_volume": sum(value["unknown_volume"] for value in values),
        "duration_ms": (close_ns - interval_start_ns) / 1e6,
        "return_bps": (cl / op - 1) * 1e4, "range_bps": (high - low) / op * 1e4,
        "overshoot_shares": overshoot, "overrun_shares": overshoot,
        "threshold_shares": np.nan, "segment": segment, **meta,
    }


def _bar_frame(rows: list[dict], trades: pd.DataFrame, **attrs) -> pd.DataFrame:
    bars = pd.DataFrame(rows, columns=BAR_COLUMNS)
    bars.attrs.update(trades.attrs)
    bars.attrs.update(barriers_ns=np.unique(trades.loc[~trades.valid, "time_ns"].to_numpy(dtype=np.int64)), **attrs)
    return bars


def build_volume_bars(trades: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Close whole-trade volume candles; discard the final incomplete candle.

    A large print or atomic same-time group can overshoot the threshold. It closes
    one candle, rather than manufacturing several zero-duration pseudo-candles.
    Clock duration includes waiting time from the previous completed candle (or
    requested start, and resets at an unsafe trade barrier). OHLC open remains
    the first actual print price; first_trade_ns records its different timestamp.
    """
    _one_series(trades)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, np.integer)) or threshold < 1:
        raise ValueError("threshold must be a positive integer number of shares.")
    if "requested_start" not in trades.attrs:
        raise ValueError("Volume candle clock duration requires explicit requested_start attrs.")
    interval_start_ns = pd.Timestamp(trades.attrs["requested_start"]).value
    if len(trades) and trades.time_ns.lt(interval_start_ns).any():
        raise ValueError("Volume trade timestamps must not precede the requested start.")
    if trades.empty:
        return _bar_frame([], trades, threshold_shares=int(threshold), incomplete_volume_shares=0.0)
    meta = trades.iloc[0][["symbol", "dataset", "day"]].to_dict()
    arrays = _atomic_arrays(trades)
    stamp, safe = arrays["time_ns"], arrays["valid"]
    barriers = np.flatnonzero(~safe)
    firsts, lasts, opens, segments = [], [], [], []
    discarded, incomplete = 0.0, 0.0
    run_start = 0
    # Each unsafe timestamp starts a new clock segment. Within one safe run,
    # prefix sums allow finding the next whole-group threshold crossing in C.
    # A crossing resets the volume quantum: overshoot is never carried forward.
    for segment, run_end in enumerate(np.r_[barriers, len(stamp)]):
        if run_start < run_end:
            prefix = np.cumsum(arrays["volume_shares"][run_start:run_end])
            cursor, baseline = 0, 0.0
            while cursor < len(prefix):
                crossing = int(np.searchsorted(prefix, baseline + threshold, side="left"))
                if crossing == len(prefix):
                    break
                firsts.append(run_start + cursor)
                lasts.append(run_start + crossing)
                opens.append(interval_start_ns)
                segments.append(segment)
                interval_start_ns = int(stamp[run_start + crossing])
                baseline = float(prefix[crossing])
                cursor = crossing + 1
            tail = float(prefix[-1] - baseline)
            if run_end == len(stamp):
                incomplete = tail
            else:
                discarded += tail
        if run_end < len(stamp):
            interval_start_ns = int(stamp[run_end])
        run_start = int(run_end) + 1

    if not firsts:
        return _bar_frame([], trades, threshold_shares=int(threshold),
                          incomplete_volume_shares=incomplete,
                          barrier_discarded_volume_shares=discarded)
    firsts, lasts = np.asarray(firsts, dtype=np.int64), np.asarray(lasts, dtype=np.int64)
    # Every selected candle is a contiguous range of safe atomic groups. Reduce
    # all ranges together; intervening discarded tails get their own reductions
    # and are selected out, so no prefix subtraction loses notional precision.
    boundaries = np.unique(np.r_[firsts, lasts + 1, len(stamp)])
    positions = np.searchsorted(boundaries, firsts)

    def sums(name):
        return np.add.reduceat(arrays[name], boundaries[:-1])[positions]

    op, cl = arrays["open"][firsts], arrays["close"][lasts]
    high = np.maximum.reduceat(arrays["high"], boundaries[:-1])[positions]
    low = np.minimum.reduceat(arrays["low"], boundaries[:-1])[positions]
    volume = sums("volume_shares")
    opens, closes = np.asarray(opens, dtype=np.int64), stamp[lasts]
    bars = pd.DataFrame({
        "open_ns": opens, "first_trade_ns": stamp[firsts], "close_ns": closes,
        "open": op, "high": high, "low": low, "close": cl,
        "vwap": sums("notional") / volume, "volume_shares": volume,
        "trade_count": sums("trade_count"), "signed_volume": sums("signed_volume"),
        "buy_volume": sums("buy_volume"), "sell_volume": sums("sell_volume"),
        "unknown_volume": sums("unknown_volume"), "duration_ms": (closes - opens) / 1e6,
        "return_bps": (cl / op - 1) * 1e4, "range_bps": (high - low) / op * 1e4,
        "overshoot_shares": volume - threshold, "overrun_shares": volume - threshold,
        "threshold_shares": int(threshold), "segment": segments, **meta,
    }, columns=BAR_COLUMNS)
    bars.attrs.update(trades.attrs)
    bars.attrs.update(barriers_ns=stamp[barriers], threshold_shares=int(threshold),
                      incomplete_volume_shares=incomplete, barrier_discarded_volume_shares=discarded)
    return bars


def build_time_bars(trades: pd.DataFrame, seconds: float = 1.0) -> pd.DataFrame:
    """Requested-window anchored trade candles, known only at the interval end.

    Empty intervals have no candle. Intervals containing unsafe T prints and
    partial request-edge intervals are omitted. No prices are carried forward.
    """
    _one_series(trades)
    if not np.isfinite(seconds) or seconds <= 0 or not float(seconds * 1e9).is_integer():
        raise ValueError("seconds must describe a positive integral nanosecond width.")
    if "requested_start" not in trades.attrs or "requested_end" not in trades.attrs:
        raise ValueError("Time candles require explicit requested_start/requested_end attrs.")
    width = int(seconds * 1e9)
    start, end = pd.Timestamp(trades.attrs["requested_start"]).value, pd.Timestamp(trades.attrs["requested_end"]).value
    if start >= end:
        raise ValueError("Requested time candle window must be positive.")
    if trades.empty:
        return _bar_frame([], trades, seconds=float(seconds))
    if trades.time_ns.lt(start).any() or trades.time_ns.ge(end).any():
        raise ValueError("Trade timestamps must fall inside the requested half-open window.")
    meta = trades.iloc[0][["symbol", "dataset", "day"]].to_dict()
    rows, parts, current_bin, unsafe, segment = [], [], None, False, 0

    def finish():
        if current_bin is not None and parts and not unsafe:
            close_ns = start + (current_bin + 1) * width
            if close_ns <= end:
                rows.append(_combine(parts, close_ns, meta, segment,
                                     interval_start_ns=start + current_bin * width))

    for time_ns, value in _groups(trades):
        bin_number = (time_ns - start) // width
        if bin_number != current_bin:
            finish()
            current_bin, parts, unsafe = bin_number, [], False
        if value is None:
            unsafe = True
            segment += 1
        else:
            parts.append((time_ns, value))
    finish()
    return _bar_frame(rows, trades, seconds=float(seconds))


def sample_completed_bars(bars: pd.DataFrame, anchors_ns, *, max_age_ns: int = 10_000_000_000,
                          include_equal_close: bool = False) -> pd.DataFrame:
    """Sample known candle states, with different timestamp semantics explicit.

    No future candle-close time, future volume, or still-open OHLC can enter an
    anchor. Unsafe prints since the sampled close invalidate the carried state.
    Volume candles contain a print at close_ns, so default availability requires
    close_ns < anchor. Time candles contain only prints in [start, close_ns), so
    callers may set include_equal_close=True for their known interval-end state.
    """
    anchors = np.asarray(anchors_ns, dtype=np.int64)
    if max_age_ns <= 0:
        raise ValueError("max_age_ns must be positive.")
    if not isinstance(include_equal_close, bool):
        raise ValueError("include_equal_close must be a boolean.")
    result = pd.DataFrame(np.nan, index=np.arange(len(anchors)), columns=FEATURE_COLUMNS)
    result["valid"] = False
    result["close_ns"] = np.full(len(anchors), -1, dtype=np.int64)
    if bars.empty:
        return result
    close = bars.close_ns.to_numpy(dtype=np.int64)
    if np.any(np.diff(close) <= 0):
        raise ValueError("Completed candle closes must be strictly increasing.")
    index = np.searchsorted(close, anchors, side="right" if include_equal_close else "left") - 1
    pos = np.maximum(index, 0)
    age = anchors - close[pos]
    valid = (index >= 0) & (age >= 0 if include_equal_close else age > 0) & (age <= max_age_ns)
    barriers = np.asarray(bars.attrs.get("barriers_ns", []), dtype=np.int64)
    if len(barriers):
        barriers = np.unique(barriers)
        barrier_index = np.searchsorted(barriers, anchors, side="left") - 1
        latest = barriers[np.maximum(barrier_index, 0)]
        valid &= (barrier_index < 0) | (latest < close[pos])
    source = bars.iloc[pos]
    features = {
        "return_bps": source.return_bps.to_numpy(), "range_bps": source.range_bps.to_numpy(),
        "log_volume": np.log1p(source.volume_shares.to_numpy()),
        "log_trade_count": np.log1p(source.trade_count.to_numpy()),
        "log_duration_ms": np.log1p(source.duration_ms.to_numpy()),
        "signed_volume_fraction": source.signed_volume.to_numpy() / source.volume_shares.to_numpy(),
        "known_side_fraction": 1 - source.unknown_volume.to_numpy() / source.volume_shares.to_numpy(),
        "vwap_close_bps": (source.vwap.to_numpy() / source.close.to_numpy() - 1) * 1e4,
        "age_ms": age / 1e6,
    }
    for name, values in features.items():
        result.loc[valid, name] = values[valid]
    result.loc[valid, "close_ns"] = close[pos[valid]]
    result["valid"] = valid
    return result
