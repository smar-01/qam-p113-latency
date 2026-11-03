"""Corrected midpoint and lead-lag measurements for the local team-code copy.

Adapted from smar-01/qam-p113-latency, commit
62d6dcd8af9ecd7136096dbcce3eb40371987422. Exact integer twice-midpoints avoid
float-only moves. Fixed-bin increments are aggregated before normalization;
the bounded result is an uncentered cross-product, not a Pearson correlation.
The optional legacy event-study support isolates arithmetic corrections from
changes to observation cohorts. Ties are not silently coalesced here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_SCALE = 1_000_000_000
UNDEF_PRICE = np.iinfo(np.int64).max


def _raw_price(q: pd.DataFrame, side: str):
    name = side + "_raw"
    if name in q:
        source = q[name].to_numpy()
        if source.dtype.kind not in "iu":
            raise ValueError(f"{name} must retain native integer nanodollar prices")
        good = (source > -(UNDEF_PRICE // 2)) & (source < UNDEF_PRICE // 2)
        values = np.zeros(len(source), dtype=np.int64)
        values[good] = source[good].astype(np.int64)
        return values, good
    # Synthetic callers historically supplied dollar floats. Quantize each
    # quote leg before averaging; retain integer native columns for real data.
    source = pd.to_numeric(q[side], errors="coerce").to_numpy(dtype=np.float64)
    good = np.isfinite(source) & (np.abs(source) < UNDEF_PRICE / (2 * PRICE_SCALE))
    values = np.zeros(len(source), dtype=np.int64)
    values[good] = np.rint(source[good] * PRICE_SCALE).astype(np.int64)
    return values, good


def _states(q: pd.DataFrame, clock: str, *, quality: bool = False):
    ordered = q.sort_values(clock, kind="stable")
    raw_times = ordered[clock].to_numpy()
    if raw_times.dtype.kind not in "iu" or np.any(raw_times > np.iinfo(np.int64).max) or np.any(raw_times < 0):
        raise ValueError("Event times must be signed-range integer nanoseconds")
    times = raw_times.astype(np.int64, copy=False)
    bid, bid_ok = _raw_price(ordered, "bid")
    ask, ask_ok = _raw_price(ordered, "ask")
    valid = bid_ok & ask_ok
    if quality:
        valid &= (bid > 0) & (ask >= bid)
    if quality and "valid" in ordered:
        valid &= ordered["valid"].fillna(False).to_numpy(dtype=bool)
    midpoint = np.where(valid, bid, 0) + np.where(valid, ask, 0)
    segment = (ordered["segment"].to_numpy(np.int64) if quality and "segment" in ordered
               else np.cumsum(~valid, dtype=np.int64))
    return ordered, times, midpoint, valid, segment


def mid_changes_integer(q: pd.DataFrame, clock: str = "ts_recv", *, quality: bool = False):
    """Times and exact signed twice-midpoint changes in nanodollars."""
    _, times, midpoint, valid, segment = _states(q, clock, quality=quality)
    change = np.diff(midpoint)
    keep = valid[1:] & valid[:-1] & (segment[1:] == segment[:-1]) & (change != 0)
    return times[1:][keep], change[keep]


def mid_changes(q: pd.DataFrame, clock: str = "ts_recv"):
    """Existing API: native times and exact-derived midpoint dollar changes."""
    times, change = mid_changes_integer(q, clock)
    return times, change.astype(np.float64) / (2 * PRICE_SCALE)


def midpoint_diagnostics(q: pd.DataFrame, clock: str = "ts_recv") -> dict:
    """Count float-only phantom changes on the same input quote cohort."""
    ordered, _, midpoint, valid, segment = _states(q, clock)
    ordinary = ((ordered["bid"] + ordered["ask"]) / 2).to_numpy(np.float64)
    floats, integer = np.diff(ordinary), np.diff(midpoint)
    adjacent = valid[1:] & valid[:-1] & (segment[1:] == segment[:-1])
    float_moves = np.isfinite(floats) & (floats != 0)
    return {"float_midpoint_changes": int(float_moves.sum()),
            "integer_midpoint_changes": int((adjacent & (integer != 0)).sum()),
            "float_only_phantom_midpoint_changes": int((adjacent & float_moves & (integer == 0)).sum())}


def _aggregate_bins(times, increments, width):
    times, increments = np.asarray(times), np.asarray(increments, dtype=np.float64)
    if times.ndim != 1 or increments.ndim != 1 or times.shape != increments.shape:
        raise ValueError("Increment timestamps and values must have matching vector shapes")
    if times.dtype.kind not in "iu" or np.any(times < 0) or np.any(times > np.iinfo(np.int64).max):
        raise ValueError("Increment times must be signed-range integer nanoseconds")
    if not np.isfinite(increments).all():
        raise ValueError("Increment values must be finite")
    bins = times.astype(np.int64) // width
    order = np.argsort(bins, kind="stable")
    bins, increments = bins[order], increments[order]
    if not len(bins):
        return bins, increments
    unique, first = np.unique(bins, return_index=True)
    values = np.add.reduceat(increments, first)
    nonzero = values != 0
    return unique[nonzero], values[nonzero]


def crosscorr(tx, dx, ty, dy, dt_ns: int = 10_000, max_lag_ns: int = 2_000_000,
              chunk: int = 200_000) -> pd.DataFrame:
    """Sparse, bounded cross-products of aggregated fixed-bin increments.

    Bins have UTC-epoch origin. Positive lag means Y follows X. Missing bins
    are zero increments; callers enforce their intended exposure/quality
    masks. This is not the original rounded event-difference statistic.
    """
    if (not isinstance(dt_ns, (int, np.integer)) or dt_ns <= 0
            or not isinstance(max_lag_ns, (int, np.integer)) or max_lag_ns < 0
            or max_lag_ns % dt_ns or not isinstance(chunk, (int, np.integer)) or chunk <= 0):
        raise ValueError("Require positive integer bin/chunk and integral-bin nonnegative maximum lag")
    x, dx = _aggregate_bins(tx, dx, dt_ns)
    y, dy = _aggregate_bins(ty, dy, dt_ns)
    k = int(max_lag_ns // dt_ns)
    acc = np.zeros(2 * k + 1, dtype=np.float64)
    # Normalize each vector before cross-products, avoiding energy overflow.
    sx = float(np.max(np.abs(dx))) if len(dx) else 0.
    sy = float(np.max(np.abs(dy))) if len(dy) else 0.
    if sx > 0 and sy > 0:
        dx, dy = dx / sx, dy / sy
        dx, dy = dx / np.sqrt(np.dot(dx, dx)), dy / np.sqrt(np.dot(dy, dy))
        block = min(int(chunk), 20_000)
        for first in range(0, len(x), block):
            tx, vx = x[first:first + block], dx[first:first + block]
            lo, hi = np.searchsorted(y, tx - k, side="left"), np.searchsorted(y, tx + k, side="right")
            count = hi - lo
            cumulative = np.r_[0, np.cumsum(count)]
            start = 0
            while start < len(tx):
                stop = min(len(tx), max(start + 1, int(np.searchsorted(cumulative, cumulative[start] + 2_000_000, side="right") - 1)))
                counts = count[start:stop]
                total = int(counts.sum())
                if total:
                    xi = np.repeat(np.arange(start, stop), counts)
                    offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
                    yi = np.repeat(lo[start:stop], counts) + offsets
                    lag = y[yi] - tx[xi]
                    acc += np.bincount(lag + k, weights=vx[xi] * dy[yi], minlength=2 * k + 1)
                start = stop
    else:
        acc[:] = np.nan
    if np.any(np.abs(acc[np.isfinite(acc)]) > 1 + 1e-12):
        raise ArithmeticError("Aggregated-bin normalization violated its unit bound")
    result = pd.DataFrame({"lag_us": np.arange(-k, k + 1) * dt_ns / 1000, "corr": np.clip(acc, -1, 1)})
    result.attrs.update(normalization="uncentered_normalized_aggregated_bin_increments",
                        bin_ns=int(dt_ns), n_x_bins=len(x), n_y_bins=len(y))
    return result


def summarize_cc(cc: pd.DataFrame) -> dict:
    """Peak and squared positive/negative lead-lag ratio; undefined stays NaN."""
    values, lag = cc["corr"].to_numpy(float), cc["lag_us"].to_numpy(float)
    good = np.isfinite(values)
    if not good.any():
        return {"peak_lag_us": np.nan, "peak_corr": np.nan, "llr": np.nan}
    pos, neg = float(np.square(values[good & (lag > 0)]).sum()), float(np.square(values[good & (lag < 0)]).sum())
    i = int(np.flatnonzero(good)[np.argmax(values[good])])
    return {"peak_lag_us": float(lag[i]), "peak_corr": float(values[i]), "llr": pos / neg if neg > 0 else np.nan}


def pair_leadlag(quotes: pd.DataFrame, clock: str = "ts_recv", dt_ns: int = 10_000,
                 max_lag_ns: int = 2_000_000):
    """Corrected curves and summaries in the starter's pair layout."""
    curves, rows = [], []
    for symbol, one in quotes.groupby("symbol"):
        changes = {venue: mid_changes(group, clock) for venue, group in one.groupby("venue")}
        venues = sorted(changes)
        for i, x in enumerate(venues):
            for y in venues[i + 1:]:
                curve = crosscorr(*changes[x], *changes[y], dt_ns=dt_ns, max_lag_ns=max_lag_ns)
                curve["x"], curve["y"], curve["symbol"] = x, y, symbol
                curves.append(curve)
                rows.append({"symbol": symbol, "x": x, "y": y, **summarize_cc(curve)})
    return (pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()), pd.DataFrame(rows)


def _bounds(ordered, times):
    start = ordered.attrs.get("requested_start", ordered.attrs.get("requested_start_ns", ordered.attrs.get("start_ns")))
    end = ordered.attrs.get("requested_end", ordered.attrs.get("requested_end_ns", ordered.attrs.get("end_ns")))
    if start is None:
        start = int(times[0]) if len(times) else 0
    elif not isinstance(start, (int, np.integer)):
        start = pd.Timestamp(start).value
    if end is None:
        end = int(times[-1]) + 1 if len(times) else 0
    elif not isinstance(end, (int, np.integer)):
        end = pd.Timestamp(end).value
    return int(start), int(end)


def _complete_mask(times, valid, segments, event_times, early, late, bounds):
    if not len(times):
        return np.zeros(len(event_times), dtype=bool)
    left, right = event_times + early, event_times + late
    baseline = np.searchsorted(times, left, side="right") - 1
    safe = np.maximum(baseline, 0)
    good = (baseline >= 0) & (left >= bounds[0]) & (right < bounds[1]) & valid[safe]
    barriers = ~valid
    if len(barriers) > 1:
        barriers = barriers | np.r_[False, segments[1:] != segments[:-1]]
    bt = times[barriers]
    next_index = np.searchsorted(bt, left, side="right")
    stop = np.full(len(event_times), bounds[1], dtype=np.int64)
    have_next = next_index < len(bt)
    stop[have_next] = np.minimum(stop[have_next], bt[next_index[have_next]])
    return good & (right < stop)


def event_study(quotes: pd.DataFrame, leader: str, symbol: str, offsets_ns: np.ndarray,
                clock: str = "ts_recv", min_move: float = 0.0, *, support: str = "complete") -> pd.DataFrame:
    """Exact signed ratios with a numerical-only or complete-cohort policy.

    Legacy support retains per-offset cohorts and last-state extrapolation.
    Complete support fixes each venue's cohort over all offsets, requiring a
    strict prior baseline and continuous valid state. Ties are not coalesced;
    callers wanting that quality policy must normalize before calling. Signed
    ratios are not a literal absorbed fraction and need not lie in [0,1].
    """
    offsets = np.asarray(offsets_ns)
    if support not in {"legacy", "complete"} or offsets.ndim != 1 or not len(offsets) or offsets.dtype.kind not in "iu":
        raise ValueError("Require integer offsets and support='legacy' or 'complete'")
    if not np.isfinite(min_move) or min_move < 0:
        raise ValueError("min_move must be finite and nonnegative")
    offsets = offsets.astype(np.int64)
    one = quotes.loc[quotes["symbol"].eq(symbol)]
    source = one.loc[one["venue"].eq(leader)]
    source_ordered, lt, lm, lv, ls = _states(source, clock, quality=support == "complete")
    t, integer_change = mid_changes_integer(source, clock, quality=support == "complete")
    keep = np.abs(integer_change) / (2 * PRICE_SCALE) > min_move
    t, integer_change = t[keep], integer_change[keep]
    early, late = min(int(offsets.min()), -1), max(int(offsets.max()), -1)
    source_good = (_complete_mask(lt, lv, ls, t, early, late, _bounds(source_ordered, lt))
                   if support == "complete" else np.ones(len(t), bool))
    rows, cohort_counts = [], {}
    for venue, group in one.groupby("venue"):
        ordered, times, midpoint, valid, segment = _states(group, clock, quality=support == "complete")
        common = source_good.copy()
        if support == "complete":
            common &= _complete_mask(times, valid, segment, t, early, late, _bounds(ordered, times))
        selected, delta = t[common], integer_change[common]
        baseline = np.searchsorted(times, selected - 1, side="right") - 1
        cohort_counts[str(venue)] = len(selected)
        for offset in offsets:
            later = np.searchsorted(times, selected + offset, side="right") - 1
            good = (baseline >= 0) & (later >= 0)
            if len(times):
                good &= valid[np.maximum(baseline, 0)] & valid[np.maximum(later, 0)]
            else:
                good[:] = False
            numerator = midpoint[later[good]] - midpoint[baseline[good]]
            ratios = numerator / delta[good]
            rows.append({"venue": venue, "offset_us": int(offset) / 1000,
                         "share": float(ratios.mean()) if len(ratios) else np.nan, "n": int(good.sum())})
    result = pd.DataFrame(rows, columns=["venue", "offset_us", "share", "n"])
    result.attrs.update(support=support, exact_integer_midpoints=True, n_leader_moves=len(t),
                        cohort_counts=cohort_counts, fixed_offset_cohort=support == "complete")
    return result
