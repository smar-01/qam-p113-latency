"""Native-event quote reaction latency with explicit observation barriers.

The measurement follows ``smar-01/qam-p113-latency``: after a bid or ask
changes to P, find the follower's first later quote at or beyond P in the
direction of that change. Equal-time updates count as already there. A later
leader reversal does not cancel the original event. This is a price-matching
description, not a causal propagation-time estimate.

Unlike the starter analysis, matches never cross a quality discontinuity in
either feed. Inference must distinguish unmatched complete horizons from
unknown outcomes truncated by a barrier or the exclusive request end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class QuoteSeries:
    """One venue/symbol/clock, with conservative tie handling done by the loader.

    Prices are integer nanodollars. ``segments`` changes across every known
    observation discontinuity; invalid rows themselves are retained barriers.
    ``start_ns`` and ``end_ns`` delimit the exact requested observation window,
    with an exclusive end. Quiet valid quotes persist until a loader-inserted
    age barrier or another explicit discontinuity.
    """

    times_ns: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    valid: np.ndarray
    segments: np.ndarray
    start_ns: int
    end_ns: int
    _trees: dict = field(default_factory=dict, init=False, repr=False)
    _barrier_times: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        for name in ("times_ns", "bid", "ask", "segments"):
            values = np.asarray(getattr(self, name))
            if values.ndim != 1 or values.dtype.kind not in "iu":
                raise ValueError(f"{name} must be a one-dimensional integer array")
            if values.dtype.kind == "u" and np.any(values > np.iinfo(np.int64).max):
                raise ValueError(f"{name} exceeds signed nanosecond/integer range")
            setattr(self, name, values.astype(np.int64, copy=False))
        self.valid = np.asarray(self.valid, dtype=bool)
        if self.valid.ndim != 1 or any(len(getattr(self, name)) != len(self.times_ns)
                                     for name in ("bid", "ask", "valid", "segments")):
            raise ValueError("All quote arrays must be one-dimensional with the same length")
        if len(self.times_ns) and (np.any(self.times_ns < 0) or np.any(np.diff(self.times_ns) <= 0)):
            raise ValueError("Quote timestamps must be nonnegative and strictly increasing; coalesce ties first")
        if np.any(self.segments < 0):
            raise ValueError("Observation segment IDs must be nonnegative")
        if np.any(self.valid & ((self.bid <= 0) | (self.ask <= self.bid))):
            raise ValueError("Valid states require a positive, strictly uncrossed two-sided quote")
        self.start_ns, self.end_ns = int(self.start_ns), int(self.end_ns)
        if not 0 <= self.start_ns < self.end_ns <= np.iinfo(np.int64).max:
            raise ValueError("Require nonnegative start and a later signed-int64 exclusive end")
        boundary = ~self.valid
        if len(boundary) > 1:
            boundary = boundary | np.r_[False, self.segments[1:] != self.segments[:-1]]
        self._barrier_times = self.times_ns[boundary]

    def next_barrier(self, times: np.ndarray) -> np.ndarray:
        """First quality boundary strictly after each event, capped by request end."""
        index = np.searchsorted(self._barrier_times, times, side="right")
        result = np.full(len(times), self.end_ns, dtype=np.int64)
        found = index < len(self._barrier_times)
        result[found] = np.minimum(self._barrier_times[index[found]], self.end_ns)
        return result

    def threshold_index(self, side: str) -> "_ThresholdIndex":
        if side not in ("bid", "ask"):
            raise ValueError("side must be bid or ask")
        if side not in self._trees:
            self._trees[side] = _ThresholdIndex(getattr(self, side), self.valid)
        return self._trees[side]


class _ThresholdIndex:
    """Batched first-hit range searches using integer min/max segment trees.

    A query's interval is decomposed into at most 2 log2(n) complete tree
    nodes. Choose its leftmost qualifying node, then descend once. All work
    over events is vectorized; there is no Python loop per quote or match.
    """

    def __init__(self, prices: np.ndarray, valid: np.ndarray):
        self.length = len(prices)
        self.size = 1 << max(0, (self.length - 1).bit_length())
        low, high = np.iinfo(np.int64).min, np.iinfo(np.int64).max
        self.maximum = np.full(2 * self.size, low, dtype=np.int64)
        self.minimum = np.full(2 * self.size, high, dtype=np.int64)
        self.maximum[self.size:self.size + self.length] = np.where(valid, prices, low)
        self.minimum[self.size:self.size + self.length] = np.where(valid, prices, high)
        width = self.size // 2
        while width:
            self.maximum[width:2 * width] = np.maximum(self.maximum[2 * width:4 * width:2], self.maximum[2 * width + 1:4 * width:2])
            self.minimum[width:2 * width] = np.minimum(self.minimum[2 * width:4 * width:2], self.minimum[2 * width + 1:4 * width:2])
            width //= 2

    def first_hit(self, lo: np.ndarray, hi: np.ndarray, target: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """First i in [lo,hi) with price>=target for +1 or <=target for -1."""
        lo, hi, target, direction = (np.asarray(x, dtype=np.int64) for x in (lo, hi, target, direction))
        if not (lo.shape == hi.shape == target.shape == direction.shape) or lo.ndim != 1:
            raise ValueError("Threshold queries must have matching one-dimensional arrays")
        if np.any((lo < 0) | (hi < lo) | (hi > self.length)) or np.any(np.abs(direction) != 1):
            raise ValueError("Invalid threshold query bounds or direction")
        n = len(lo)
        left, right = lo + self.size, hi + self.size
        best_start = np.full(n, self.size, dtype=np.int64)
        best_node = np.zeros(n, dtype=np.int64)
        span, offset = 1, self.size
        while np.any(left < right):
            active = left < right
            for node, choose in ((left, active & ((left & 1) == 1)),
                                 (right - 1, active & ((right & 1) == 1))):
                idx = np.flatnonzero(choose)
                if not len(idx):
                    continue
                nd = node[idx]
                hit = np.where(direction[idx] > 0, self.maximum[nd] >= target[idx], self.minimum[nd] <= target[idx])
                starts = (nd - offset) * span
                keep = hit & (starts < best_start[idx])
                selected = idx[keep]
                best_start[selected] = starts[keep]
                best_node[selected] = nd[keep]
            left = (left + (left & 1)) // 2
            right = (right - (right & 1)) // 2
            span *= 2
            offset //= 2
        found = best_node > 0
        idx = np.flatnonzero(found)
        node = best_node[idx]
        while np.any(node < self.size):
            moving = node < self.size
            selected = idx[moving]
            child = 2 * node[moving]
            hit_left = np.where(direction[selected] > 0, self.maximum[child] >= target[selected], self.minimum[child] <= target[selected])
            node[moving] = child + ~hit_left
        result = np.full(n, -1, dtype=np.int64)
        result[idx] = node - self.size
        return result


def reaction_events(leader: QuoteSeries, follower: QuoteSeries, side: str = "bid", *,
                    max_horizon_ns: int = 100_000_000, min_move: int = 0) -> pd.DataFrame:
    """Match once at the maximum horizon, reusable at any shorter horizon.

    ``observed_ns`` is the largest complete integer-nanosecond follow-up
    before a quality boundary or exclusive request end, capped at the search
    horizon. Use ``observed_ns >= horizon_ns`` for complete-follow-up rate
    estimates; otherwise an early known match and an unknown nonmatch would
    enter different denominators. ``latency_ns == -1`` means no observed hit.
    """
    if side not in ("bid", "ask") or not isinstance(max_horizon_ns, (int, np.integer)) or max_horizon_ns <= 0:
        raise ValueError("Require side bid/ask and a positive integer horizon in nanoseconds")
    if not isinstance(min_move, (int, np.integer)) or min_move < 0:
        raise ValueError("min_move must be a nonnegative integer nanodollar amount")
    p = getattr(leader, side)
    move = p[1:] - p[:-1]
    keep = (leader.valid[1:] & leader.valid[:-1] & (leader.segments[1:] == leader.segments[:-1])
            & (np.abs(move) > min_move) & (leader.times_ns[1:] >= leader.start_ns)
            & (leader.times_ns[1:] < leader.end_ns))
    position = np.flatnonzero(keep) + 1
    t, target = leader.times_ns[position], p[position]
    direction = np.sign(move[keep]).astype(np.int8)
    n = len(t)
    state = np.full(n, "no_state", dtype=object)
    lag = np.full(n, -1, dtype=np.int64)
    observed = np.zeros(n, dtype=np.int64)
    reason = np.full(n, "end", dtype=object)
    if n and len(follower.times_ns):
        i0 = np.searchsorted(follower.times_ns, t, side="right") - 1
        has_state = (i0 >= 0) & (t >= follower.start_ns) & (t < follower.end_ns)
        has_state &= follower.valid[np.clip(i0, 0, len(follower.valid) - 1)]
        idx = np.flatnonzero(has_state)
        fp = getattr(follower, side)
        already = np.zeros(n, dtype=bool)
        already[idx] = np.where(direction[idx] > 0, fp[i0[idx]] >= target[idx], fp[i0[idx]] <= target[idx])
        state[already] = "already"
        behind = has_state & ~already
        state[behind] = "behind"
        leader_stop, follower_stop = leader.next_barrier(t), follower.next_barrier(t)
        stop = np.minimum(leader_stop, follower_stop)
        end = min(leader.end_ns, follower.end_ns)
        # A barrier at the request end is administratively censored, not a
        # known feed-quality event within the requested data.
        quality_stop = stop < end
        reason[:] = np.where(quality_stop, "quality", "end")
        available = np.maximum(0, stop - t - 1)
        observed[has_state] = np.minimum(available[has_state], max_horizon_ns)
        reason[available >= max_horizon_ns] = "horizon"
        idx = np.flatnonzero(behind)
        if len(idx):
            last_time = t[idx] + observed[idx]
            lo = i0[idx] + 1
            hi = np.searchsorted(follower.times_ns, last_time, side="right")
            hit = follower.threshold_index(side).first_hit(lo, hi, target[idx], direction[idx])
            matched = hit >= 0
            lag[idx[matched]] = follower.times_ns[hit[matched]] - t[idx[matched]]
    result = pd.DataFrame({"t": t, "price": target, "direction": direction,
                           "latency_ns": lag, "observed_ns": observed,
                           "terminal_reason": reason, "status": state})
    result.attrs["max_horizon_ns"] = int(max_horizon_ns)
    result.attrs["side"] = side
    return result


def classify_horizon(events: pd.DataFrame, horizon_ns: int) -> np.ndarray:
    """Observed outcomes, retaining quality/end truncation as distinct states."""
    if not isinstance(horizon_ns, (int, np.integer)) or horizon_ns <= 0:
        raise ValueError("horizon_ns must be a positive integer")
    if horizon_ns > events.attrs.get("max_horizon_ns", horizon_ns):
        raise ValueError("Requested horizon exceeds the original match search")
    status = events["status"].to_numpy(dtype=object).copy()
    behind = status == "behind"
    lag = events["latency_ns"].to_numpy(np.int64)
    matched = behind & (lag > 0) & (lag <= horizon_ns)
    status[behind] = "censored"
    truncated = behind & (events["observed_ns"].to_numpy(np.int64) < horizon_ns)
    reason = events["terminal_reason"].to_numpy()
    status[truncated & (reason == "quality")] = "quality_censored"
    status[truncated & (reason == "end")] = "end_censored"
    status[matched] = "matched"
    return status


def summarize_events(events: pd.DataFrame, horizon_ns: int) -> dict:
    """Complete-follow-up summaries plus transparent partial-observation counts.

    The primary latency quantiles use only behind-events with a full horizon
    available. ``n_observed_matches`` also reports early known matches from
    truncated follow-up; they do not silently enter the primary denominator.
    """
    status = classify_horizon(events, horizon_ns)
    behind = events["status"].to_numpy() == "behind"
    complete = behind & (events["observed_ns"].to_numpy(np.int64) >= horizon_ns)
    matched = (status == "matched") & complete
    values = events.loc[matched, "latency_ns"].to_numpy(np.int64) / 1_000
    eligible_count, match_count = int(complete.sum()), int(matched.sum())
    truncated = behind & ~complete
    reasons = events["terminal_reason"].to_numpy()
    result = {"horizon_ms": horizon_ns / 1e6, "n_events": len(events),
              "n_already": int(np.sum(status == "already")), "n_no_state": int(np.sum(status == "no_state")),
              "n_behind": int(behind.sum()), "n_complete_behind": eligible_count, "n_eligible": eligible_count,
              "n_matched": match_count, "n_observed_matches": int(np.sum(status == "matched")),
              "known_matches_truncated": int(np.sum((status == "matched") & truncated)),
              "n_censored": eligible_count - match_count,
              "n_quality_truncated": int(np.sum(truncated & (reasons == "quality"))),
              "n_right_truncated": int(np.sum(truncated & (reasons == "end"))),
              "n_quality_censored": int(np.sum(status == "quality_censored")),
              "n_end_censored": int(np.sum(status == "end_censored")),
              "match_rate": float(match_count / eligible_count) if eligible_count else np.nan}
    for name, q in (("p10_us", .1), ("median_us", .5), ("p90_us", .9)):
        result[name] = float(np.quantile(values, q)) if len(values) else np.nan
    result["mean_us"] = float(values.mean()) if len(values) else np.nan
    result["sum_latency_ns"] = int(events.loc[matched, "latency_ns"].sum())
    waiting_sum_ns = result["sum_latency_ns"] + (eligible_count - match_count) * horizon_ns
    result["restricted_wait_sum_ns"] = waiting_sum_ns
    result["restricted_mean_wait_us"] = waiting_sum_ns / eligible_count / 1_000 if eligible_count else np.nan
    return result


def _twice_mid(quotes: QuoteSeries) -> np.ndarray:
    if np.any(quotes.valid & (quotes.ask > np.iinfo(np.int64).max // 2)):
        raise ValueError("Integer twice-midpoint would exceed signed int64")
    return np.where(quotes.valid, quotes.bid, 0) + np.where(quotes.valid, quotes.ask, 0)


def _mid_changes(quotes: QuoteSeries) -> tuple[np.ndarray, np.ndarray]:
    midpoint = _twice_mid(quotes)
    change = midpoint[1:] - midpoint[:-1]
    keep = (quotes.valid[1:] & quotes.valid[:-1]
            & (quotes.segments[1:] == quotes.segments[:-1]) & (change != 0)
            & (quotes.times_ns[1:] >= quotes.start_ns) & (quotes.times_ns[1:] < quotes.end_ns))
    return quotes.times_ns[1:][keep], change[keep]


def _valid_intervals(quotes: QuoteSeries) -> tuple[np.ndarray, np.ndarray]:
    """Maximal valid intervals without merging across even zero-width resets."""
    n = len(quotes.times_ns)
    if not n:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    changed = quotes.segments[1:] != quotes.segments[:-1]
    first = quotes.valid & np.r_[True, ~quotes.valid[:-1] | changed]
    last = quotes.valid & np.r_[~quotes.valid[1:] | changed, True]
    starts_i, lasts_i = np.flatnonzero(first), np.flatnonzero(last)
    starts = np.maximum(quotes.times_ns[starts_i], quotes.start_ns)
    ends = np.full(len(lasts_i), quotes.end_ns, dtype=np.int64)
    have_next = lasts_i + 1 < n
    ends[have_next] = np.minimum(quotes.times_ns[lasts_i[have_next] + 1], quotes.end_ns)
    keep = starts < ends
    return starts[keep], ends[keep]


def _joint_intervals(a: QuoteSeries, b: QuoteSeries) -> tuple[np.ndarray, np.ndarray]:
    sa, ea = _valid_intervals(a)
    sb, eb = _valid_intervals(b)
    starts, ends = [], []
    i = j = 0
    while i < len(sa) and j < len(sb):
        start, end = max(int(sa[i]), int(sb[j])), min(int(ea[i]), int(eb[j]))
        if start < end:
            starts.append(start)
            ends.append(end)
        if ea[i] <= eb[j]:
            i += 1
        else:
            j += 1
    return np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)


def _inside_intervals(times: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    if not len(starts):
        return np.zeros(len(times), dtype=bool)
    index = np.searchsorted(starts, times, side="right") - 1
    return (index >= 0) & (times < ends[np.maximum(index, 0)])


def realized_volatility(quotes: dict[str, QuoteSeries], *, step_ns: int = 1_000_000_000,
                        window: int = 300, min_venues: int | None = None) -> pd.DataFrame:
    """Trailing volatility of the cross-venue median midpoint, every second.

    Each return requires a supported midpoint at both endpoints, and a full
    ``window`` of consecutive returns is required. The 75th-percentile cut is
    retrospective within this saved symbol/day window, not a tradable signal.
    """
    if not quotes:
        return pd.DataFrame(columns=["t", "mid", "n_venues", "rv", "regime"])
    if not isinstance(step_ns, (int, np.integer)) or step_ns <= 0 or not isinstance(window, int) or window < 2:
        raise ValueError("Require positive integer step_ns and an integer window >= 2")
    minimum = (len(quotes) + 1) // 2 if min_venues is None else min_venues
    if not isinstance(minimum, int) or not 1 <= minimum <= len(quotes):
        raise ValueError("min_venues must be between one and the number of supplied venues")
    start, end = max(q.start_ns for q in quotes.values()), min(q.end_ns for q in quotes.values())
    grid = np.arange(start - start % step_ns + step_ns, end, step_ns, dtype=np.int64)
    values = np.full((len(quotes), len(grid)), np.nan)
    for v, series in enumerate(quotes.values()):
        if not len(series.times_ns):
            continue
        i = np.searchsorted(series.times_ns, grid, side="right") - 1
        safe = np.maximum(i, 0)
        good = (i >= 0) & series.valid[safe] & (grid >= series.start_ns) & (grid < series.end_ns)
        values[v, good] = _twice_mid(series)[safe[good]] / 2e9
    support = np.isfinite(values).sum(axis=0)
    middle = np.full(len(grid), np.nan)
    supported = support >= minimum
    if supported.any():
        middle[supported] = np.nanmedian(values[:, supported], axis=0)
    returns = np.r_[np.nan, np.diff(np.log(middle))] if len(middle) else np.array([])
    rv = np.sqrt(pd.Series(returns ** 2).rolling(window, min_periods=window).sum().to_numpy())
    regime = np.full(len(grid), "unknown", dtype=object)
    finite = np.isfinite(rv)
    threshold = float(np.quantile(rv[finite], .75)) if finite.any() else np.nan
    regime[finite] = np.where(rv[finite] >= threshold, "volatile", "calm")
    result = pd.DataFrame({"t": grid, "mid": middle, "n_venues": support, "rv": rv, "regime": regime})
    result.attrs.update(quantile=.75, threshold=threshold, retrospective=True, window=window, step_ns=int(step_ns))
    return result


def event_study(leader: QuoteSeries, follower: QuoteSeries, offsets_ns: Sequence[int]) -> pd.DataFrame:
    """Average share of each leader midpoint move absorbed by the follower.

    Every offset uses the same leader-event cohort. Both feeds must remain
    continuously valid over the entire interval from the earliest offset
    (and the strict t-1 ns baseline) through the latest offset. The denominator
    is the leader's exact integer twice-midpoint change; it is not a regression.
    """
    offsets = np.asarray(offsets_ns)
    if offsets.ndim != 1 or not len(offsets) or offsets.dtype.kind not in "iu":
        raise ValueError("offsets_ns must be a nonempty integer vector")
    offsets = offsets.astype(np.int64, copy=False)
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("offsets_ns must be strictly increasing")
    t, change = _mid_changes(leader)
    starts, ends = _joint_intervals(leader, follower)
    early, late = min(int(offsets[0]), -1), max(int(offsets[-1]), -1)
    keep = _inside_intervals(t + early, starts, ends)
    if len(starts):
        interval = np.searchsorted(starts, t + early, side="right") - 1
        keep &= t + late < ends[np.maximum(interval, 0)]
    t, change = t[keep], change[keep]
    midpoint = _twice_mid(follower)
    base = np.searchsorted(follower.times_ns, t - 1, side="right") - 1
    rows = []
    for offset in offsets:
        future = np.searchsorted(follower.times_ns, t + offset, side="right") - 1
        absorbed = (midpoint[future] - midpoint[base]) / change if len(t) else np.array([])
        rows.append({"offset_us": int(offset) / 1_000, "share": float(absorbed.mean()) if len(absorbed) else np.nan,
                     "n": len(t)})
    result = pd.DataFrame(rows)
    result.attrs.update(n_total_leader_events=len(keep), n_common_cohort=int(keep.sum()),
                        fixed_common_cohort=True, baseline="strictly_before_leader_event")
    return result


def binned_crosscorr(leader: QuoteSeries, follower: QuoteSeries, *, bin_ns: int = 10_000,
                     max_lag_ns: int = 2_000_000) -> pd.DataFrame:
    """Sparse, bounded cross-products of aggregated fixed-bin mid increments.

    Bin origin is UTC epoch. Midpoint increments in the same bin are summed
    *before* the energy normalization. Positive lag means follower after
    leader. This is an uncentered normalized binned-increment cross-product,
    not a Pearson correlation. Quiet zero-increment bins need not be allocated.

    Both feeds must have continuous valid support max_lag_ns before/after each
    included increment. This omits quality boundaries and request edges rather
    than correlating across them. Each output is in [-1,1] by Cauchy-Schwarz.
    """
    if (not isinstance(bin_ns, (int, np.integer)) or bin_ns <= 0
            or not isinstance(max_lag_ns, (int, np.integer)) or max_lag_ns < 0
            or max_lag_ns % bin_ns):
        raise ValueError("Require a positive integer bin_ns and a nonnegative integer multiple max_lag_ns")
    starts, ends = _joint_intervals(leader, follower)
    starts, ends = starts + max_lag_ns, ends - max_lag_ns
    good_interval = starts < ends
    starts, ends = starts[good_interval], ends[good_interval]

    def increments(series):
        times, changes = _mid_changes(series)
        good = _inside_intervals(times, starts, ends)
        times, changes = times[good], changes[good]
        if not len(times):
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64), 0
        bins = times // bin_ns
        unique, index = np.unique(bins, return_index=True)
        summed = np.add.reduceat(changes.astype(np.float64), index)
        nonzero = summed != 0
        return unique[nonzero], summed[nonzero], len(times)

    x, dx, nx = increments(leader)
    y, dy, ny = increments(follower)
    k = max_lag_ns // bin_ns
    accumulator = np.zeros(2 * k + 1, dtype=np.float64)
    # Sparse bin pairs give exactly the dense-grid increment cross-product
    # without allocating a full 10-microsecond grid for a trading session.
    for first in range(0, len(x), 20_000):
        tx, vx = x[first:first + 20_000], dx[first:first + 20_000]
        lo, hi = np.searchsorted(y, tx - k, side="left"), np.searchsorted(y, tx + k, side="right")
        counts = hi - lo
        cumulative = np.r_[0, np.cumsum(counts)]
        a = 0
        while a < len(tx):
            b = min(len(tx), max(a + 1, int(np.searchsorted(cumulative, cumulative[a] + 2_000_000, side="right") - 1)))
            count = counts[a:b]
            total = int(count.sum())
            if total:
                ix = np.repeat(np.arange(a, b), count)
                offset = np.arange(total) - np.repeat(np.cumsum(count) - count, count)
                iy = np.repeat(lo[a:b], count) + offset
                lag = y[iy] - tx[ix]
                accumulator += np.bincount(lag + k, weights=vx[ix] * dy[iy], minlength=2 * k + 1)
            a = b
    energy = float(np.sqrt(np.dot(dx, dx) * np.dot(dy, dy)))
    corr = accumulator / energy if energy > 0 else np.full(len(accumulator), np.nan)
    if np.any(np.abs(corr[np.isfinite(corr)]) > 1 + 1e-12):
        raise ArithmeticError("Binned increment normalization violated its unit bound")
    result = pd.DataFrame({"lag_us": np.arange(-k, k + 1) * bin_ns / 1_000,
                           "corr": np.clip(corr, -1, 1)})
    result.attrs.update(bin_ns=int(bin_ns), max_lag_ns=int(max_lag_ns),
                        n_leader_increments=nx, n_follower_increments=ny,
                        n_leader_nonzero_bins=len(x), n_follower_nonzero_bins=len(y),
                        normalization="uncentered_normalized_binned_increment_crossproduct",
                        common_support_padding_ns=int(max_lag_ns))
    return result
