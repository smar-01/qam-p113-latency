"""Local ColonyLab quality adapter for the corrected teammate command.

This optional layer deliberately depends on the enclosing ColonyLab package.
It reuses its independently validated native integer normalization and matching
engine instead of claiming that the starter matcher handles unknown support.
The numerical-only command does not import or use this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import leadlag
from .venues import BY_VENUE


def prepare(quotes: pd.DataFrame, clock: str, max_age_ms: float = 250):
    try:
        from colonylab.latency_data import normalize_frame
    except ImportError as exc:
        raise ImportError("Quality mode requires the enclosing ColonyLab package; install it or explicitly use --numerical-only.") from exc
    if quotes.attrs.get("price_changes_only"):
        raise ValueError("Quality mode requires raw size/state updates, not compressed price changes")
    start = quotes.attrs.get("requested_start_ns", quotes.attrs.get("start_ns"))
    end = quotes.attrs.get("requested_end_ns", quotes.attrs.get("end_ns"))
    start = int(quotes.ts_recv.min()) if start is None else int(start)
    end = int(quotes.ts_recv.max()) + 1 if end is None else int(end)
    prepared, diagnostics = {}, []
    for venue, group in quotes.groupby("venue", sort=True):
        bid, bid_ok = leadlag._raw_price(group, "bid")
        ask, ask_ok = leadlag._raw_price(group, "ask")
        frame = pd.DataFrame({
            "ts_recv": group.ts_recv.to_numpy(np.int64), "ts_event": group.ts_event.to_numpy(np.int64),
            "symbol": group.symbol.astype(str).to_numpy(),
            "bid_px_00": np.where(bid_ok, bid, leadlag.UNDEF_PRICE),
            "ask_px_00": np.where(ask_ok, ask, leadlag.UNDEF_PRICE),
            "bid_sz_00": group.bid_sz.to_numpy(np.int64) if "bid_sz" in group else 1,
            "ask_sz_00": group.ask_sz.to_numpy(np.int64) if "ask_sz" in group else 1,
            "flags": group["flags"].to_numpy(np.uint64) if "flags" in group else 128,
        })
        dataset = BY_VENUE[venue].dataset if venue in BY_VENUE else str(venue)
        request = {"dataset": dataset, "schema": "mbp-1", "symbols": sorted(frame.symbol.unique()),
                   "start": pd.Timestamp(start, tz="UTC").isoformat(),
                   "end": pd.Timestamp(end, tz="UTC").isoformat()}
        loaded, rows = normalize_frame(frame, request, clocks=[clock], max_age_ms=max_age_ms)
        for (symbol, _), series in loaded.items():
            prepared[(symbol, venue)] = series
        diagnostics.extend(dict(row, venue=venue) for row in rows)
    return prepared, pd.DataFrame(diagnostics)


def pair_latencies(prepared, *, horizon_ns: int, min_move: float = 0.0):
    from colonylab.latency import reaction_events, classify_horizon
    pieces = []
    symbols = sorted({symbol for symbol, _ in prepared})
    for symbol in symbols:
        venues = sorted(venue for sym, venue in prepared if sym == symbol)
        for leader in venues:
            for follower in venues:
                if leader == follower:
                    continue
                for side in ("bid", "ask"):
                    events = reaction_events(prepared[(symbol, leader)], prepared[(symbol, follower)], side,
                                             max_horizon_ns=horizon_ns, min_move=int(np.rint(min_move * 1e9)))
                    events["base_status"] = events.status
                    status = classify_horizon(events, horizon_ns)
                    truncated = events.base_status.eq("behind").to_numpy() & (events.observed_ns.to_numpy() < horizon_ns)
                    status[truncated & events.terminal_reason.eq("quality").to_numpy()] = "quality_censored"
                    status[truncated & events.terminal_reason.eq("end").to_numpy()] = "end_censored"
                    events["status"] = status
                    events["symbol"], events["leader"], events["follower"], events["side"] = symbol, leader, follower, side
                    pieces.append(events)
    result = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    result.attrs["horizon_ns"] = horizon_ns
    return result


def summarize(events: pd.DataFrame, by=("leader", "follower"), *, horizon_ns: int):
    from colonylab.latency import summarize_events
    rows = []
    for keys, group in events.groupby(list(by), sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        raw = group.copy()
        raw["status"] = raw.base_status
        raw.attrs["max_horizon_ns"] = horizon_ns
        rows.append({**dict(zip(by, keys)), **summarize_events(raw, horizon_ns)})
    return pd.DataFrame(rows)


def realized_vol(prepared):
    from colonylab.latency import realized_volatility
    frames = []
    for symbol in sorted({sym for sym, _ in prepared}):
        quotes = {venue: series for (sym, venue), series in prepared.items() if sym == symbol}
        one = realized_volatility(quotes, window=300)
        one["symbol"] = symbol
        frames.append(one)
    return pd.concat(frames, ignore_index=True)


def pair_leadlag(prepared):
    from colonylab.latency import binned_crosscorr
    curves, rows = [], []
    for symbol in sorted({sym for sym, _ in prepared}):
        venues = sorted(venue for sym, venue in prepared if sym == symbol)
        for i, first in enumerate(venues):
            for second in venues[i + 1:]:
                cc = binned_crosscorr(prepared[(symbol, first)], prepared[(symbol, second)])
                rows.append({"symbol": symbol, "x": first, "y": second, **leadlag.summarize_cc(cc)})
                cc["symbol"], cc["x"], cc["y"] = symbol, first, second
                curves.append(cc)
    return pd.concat(curves, ignore_index=True), pd.DataFrame(rows)


def event_study(prepared, leader: str, symbol: str, offsets_ns):
    from colonylab.latency import event_study as exact_study
    frames = []
    for sym, venue in sorted(prepared):
        if sym != symbol:
            continue
        one = exact_study(prepared[(symbol, leader)], prepared[(symbol, venue)], offsets_ns)
        one["venue"] = venue
        frames.append(one)
    return pd.concat(frames, ignore_index=True)
