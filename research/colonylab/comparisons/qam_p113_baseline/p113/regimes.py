"""Calm vs volatile windows from rolling realized volatility.

Per symbol: a cross-venue median mid sampled every `step_ns` (default 1 s), log returns,
rolling sum of squared returns over `window` samples (default 300 = 5 minutes) =
realized variance. A sample is 'volatile' when its realized vol is at or above the
`q` quantile of that symbol's day (default 0.75), else 'calm'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def realized_vol(quotes: pd.DataFrame, clock: str = "ts_recv", step_ns: int = 1_000_000_000,
                 window: int = 300) -> pd.DataFrame:
    out = []
    for sym, qs in quotes.groupby("symbol"):
        t0, t1 = qs[clock].min(), qs[clock].max()
        grid = np.arange(t0 - t0 % step_ns + step_ns, t1 + 1, step_ns, dtype=np.int64)
        mids = []
        for _, g in qs.groupby("venue"):
            g = g.sort_values(clock, kind="stable")
            tv = g[clock].to_numpy(np.int64)
            mv = ((g["bid"] + g["ask"]) / 2).to_numpy(np.float64)
            i = np.searchsorted(tv, grid, side="right") - 1
            mids.append(np.where(i >= 0, mv[np.clip(i, 0, None)], np.nan))
        with np.errstate(all="ignore"):
            mid = np.nanmedian(np.vstack(mids), axis=0)
        r = np.r_[np.nan, np.diff(np.log(mid))]
        rv = pd.Series(r ** 2).rolling(window, min_periods=max(2, window // 5)).sum()
        out.append(pd.DataFrame({"symbol": sym, "t": grid, "rv": np.sqrt(rv.to_numpy())}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["symbol", "t", "rv"])


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
