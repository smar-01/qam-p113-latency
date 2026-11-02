"""Calm vs volatile windows from rolling realized volatility.

Per symbol: a cross-venue median mid sampled every `step_ns` (default 1 s),
then a full trailing `window` of supported returns. Raw size-only updates must
be retained to measure quote age. Missing/flagged/ambiguous/stale intervals are
not filled. The quantile is a retrospective label, not a forecasting signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _states(group: pd.DataFrame, clock: str, max_age_ns: int | None,
            end_ns: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Completed quote states, preserving quality and freshness discontinuities."""
    raw = group[clock].to_numpy()
    if raw.dtype.kind not in "iu" or np.any(raw > np.iinfo(np.int64).max) or np.any(raw < 0):
        raise ValueError("Volatility requires defined integer nanosecond timestamps.")
    t = raw.astype(np.int64, copy=False)
    bid, ask = group.bid.to_numpy(float), group.ask.to_numpy(float)
    valid = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > bid)
    if "valid" in group:
        valid &= group.valid.to_numpy(bool)
    if "flags" in group:
        bits = group["flags"].to_numpy(np.uint64)
        bad = (bits & (32 | 8 | 4)) != 0
        included = ((bits & 128) != 0) | bad
        valid &= ~bad & ((bits & 128) != 0)
    else:
        included = np.ones(len(t), dtype=bool)
    keep = np.flatnonzero(included)
    t, bid, ask, valid = (a[keep] for a in (t, bid, ask, valid))
    if not len(t):
        return t, np.array([], float), valid, np.array([], np.int64)
    highest = np.maximum.accumulate(t)
    reversed_positions = np.flatnonzero(t[1:] < highest[:-1]) + 1
    for position in reversed_positions:
        valid[(t >= t[position]) & (t <= highest[position - 1])] = False
    state = pd.DataFrame({"time": t, "bid": bid, "ask": ask, "valid": valid})
    for name in ("bid_sz", "ask_sz"):
        if name in group:
            state[name] = group[name].to_numpy()[keep]
    grouped = state.groupby("time", sort=True)
    names = ["bid", "ask"] + [name for name in ("bid_sz", "ask_sz") if name in state]
    conflicts = grouped[names].nunique(dropna=False).gt(1).any(axis=1)
    states = grouped[names].first()
    states["valid"] = grouped.valid.all() & ~conflicts
    states = states.reset_index()
    t = states.time.to_numpy(np.int64)
    valid = states.valid.to_numpy(bool)
    mid = (states.bid.to_numpy(float) + states.ask.to_numpy(float)) / 2
    if max_age_ns is not None:
        deadline = t + max_age_ns + 1  # age == maximum remains observable
        next_time = np.r_[t[1:], end_ns]
        expires = valid & (deadline < next_time) & (deadline < end_ns)
        if expires.any():
            t = np.r_[t, deadline[expires]]
            mid = np.r_[mid, np.full(expires.sum(), np.nan)]
            valid = np.r_[valid, np.zeros(expires.sum(), bool)]
            order = np.argsort(t, kind="stable")
            t, mid, valid = t[order], mid[order], valid[order]
    segment = np.cumsum(~valid, dtype=np.int64)
    return t, np.where(valid, mid, np.nan), valid, segment


def realized_vol(quotes: pd.DataFrame, clock: str = "ts_recv", step_ns: int = 1_000_000_000,
                 window: int = 300, *, max_age_ms: float | None = 250,
                 min_venues: int | None = None, require_full_window: bool = True) -> pd.DataFrame:
    """Quality-aware volatility; use raw states, not a price-change-only table.

    ``max_age_ms=None`` explicitly disables the age rule. ``require_full_window``
    defaults to true; false is a labelled legacy-warmup sensitivity. In either
    mode bad states remain barriers and the selected clock is honored.
    """
    if clock not in {"ts_recv", "ts_event"}:
        raise ValueError("clock must be ts_recv or ts_event")
    if type(step_ns) is not int or step_ns <= 0 or type(window) is not int or window < 2:
        raise ValueError("Use a positive integer grid step and at least two returns.")
    if max_age_ms is not None and (not np.isfinite(max_age_ms) or max_age_ms <= 0):
        raise ValueError("max_age_ms must be positive and finite, or None")
    if quotes.attrs.get("price_changes_only") and max_age_ms is not None:
        raise ValueError("Freshness-aware volatility requires uncompressed raw size/state updates.")
    max_age_ns = None if max_age_ms is None else int(max_age_ms * 1e6)
    out = []
    for sym, qs in quotes.groupby("symbol"):
        t0 = int(quotes.attrs.get("requested_start_ns", qs[clock].min()))
        recorded_end = quotes.attrs.get("requested_end_ns")
        t1 = int(recorded_end) if recorded_end is not None else int(qs[clock].max()) + 1
        if clock == "ts_event":
            t1 = min(t1, min(int(g[clock].max()) for _, g in qs.groupby("venue")))
        grid = np.arange(t0 - t0 % step_ns + step_ns, t1, step_ns, dtype=np.int64)
        mids, continuous = [], []
        required = (qs.venue.nunique() + 1) // 2 if min_venues is None else min_venues
        if type(required) is not int or not 1 <= required <= qs.venue.nunique():
            raise ValueError("min_venues must be between one and the intended venue count")
        for _, g in qs.groupby("venue"):
            tv, mv, valid, segments = _states(g, clock, max_age_ns, t1)
            if not len(tv):
                mids.append(np.full(len(grid), np.nan))
                continuous.append(np.zeros(len(grid), bool))
                continue
            i = np.searchsorted(tv, grid, side="right") - 1
            safe = np.maximum(i, 0)
            good = (i >= 0) & valid[safe]
            mids.append(np.where(good, mv[safe], np.nan))
            uninterrupted = np.zeros(len(grid), bool)
            if len(grid) > 1:
                uninterrupted[1:] = good[:-1] & good[1:] & (segments[safe[:-1]] == segments[safe[1:]])
            continuous.append(uninterrupted)
        values = np.vstack(mids)
        support = np.isfinite(values).sum(axis=0)
        mid = np.full(len(grid), np.nan)
        supported = support >= required
        if supported.any():
            mid[supported] = np.nanmedian(values[:, supported], axis=0)
        r = np.r_[np.nan, np.diff(np.log(mid))]
        if not len(grid):
            r = np.array([], float)
        continuity = np.vstack(continuous).sum(axis=0)
        r[continuity < required] = np.nan
        minimum = window if require_full_window else max(2, window // 5)
        rv = pd.Series(r ** 2).rolling(window, min_periods=minimum).sum()
        out.append(pd.DataFrame({"symbol": sym, "t": grid, "rv": np.sqrt(rv.to_numpy()),
                                 "n_venues": support, "n_continuous_venues": continuity}))
    result = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["symbol", "t", "rv", "n_venues", "n_continuous_venues"])
    result.attrs.update(clock=clock, max_age_ms=max_age_ms, require_full_window=require_full_window,
                        retrospective_regime=True)
    return result


def label(rv: pd.DataFrame, q: float = 0.75) -> pd.DataFrame:
    rv = rv.copy()
    cut = rv.groupby("symbol")["rv"].transform(lambda s: s.quantile(q))
    rv["regime"] = np.where(rv["rv"] >= cut, "volatile", "calm")
    rv.loc[rv["rv"].isna(), "regime"] = "unknown"
    return rv


def tag(events: pd.DataFrame, labelled: pd.DataFrame, time_col: str = "t") -> pd.Series:
    """Regime of each event (by symbol), from the last grid point at or before it."""
    reg = pd.Series("unknown", index=events.index, dtype=object)
    for sym, idx in events.groupby("symbol").groups.items():
        lab = labelled[labelled["symbol"] == sym]
        if lab.empty:
            continue
        tg = lab["t"].to_numpy(np.int64)
        i = np.searchsorted(tg, events.loc[idx, time_col].to_numpy(np.int64), side="right") - 1
        r = lab["regime"].to_numpy()
        reg.loc[idx] = np.where(i >= 0, r[np.clip(i, 0, None)], "unknown")
    return reg
