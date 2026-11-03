"""Lead-lag from mid-price changes, without putting the data on a fixed time grid.

Cross-correlation at lag l (bin width dt):
    C(l) = sum over pairs (i, j) with round((t_j - t_i)/dt) == l of dx_i * dy_j
           / sqrt(sum dx^2 * sum dy^2)
where dx are venue X's mid changes and dy venue Y's. This equals the correlation of the
two venues' binned mid increments, but only pairs within max_lag are visited, so it runs
at microsecond resolution on a full day. Positive lag = Y moves after X = X leads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mid_changes(q: pd.DataFrame, clock: str = "ts_recv"):
    """Times and sizes of non-zero mid changes for one venue + symbol."""
    q = q.sort_values(clock, kind="stable")
    mid = ((q["bid"] + q["ask"]) / 2).to_numpy(np.float64)
    t = q[clock].to_numpy(np.int64)
    dm = np.diff(mid)
    ok = np.isfinite(dm) & (dm != 0)
    return t[1:][ok], dm[ok]


def crosscorr(tx, dx, ty, dy, dt_ns: int = 10_000, max_lag_ns: int = 2_000_000,
              chunk: int = 200_000) -> pd.DataFrame:
    """C(l) for l in [-max_lag, max_lag] in steps of dt. Returns lag_us, corr."""
    K = int(max_lag_ns // dt_ns)
    acc = np.zeros(2 * K + 1)
    for s in range(0, len(tx), chunk):
        x_t, x_d = tx[s:s + chunk], dx[s:s + chunk]
        lo = np.searchsorted(ty, x_t - max_lag_ns, side="left")
        hi = np.searchsorted(ty, x_t + max_lag_ns, side="right")
        cnt = hi - lo
        if cnt.sum() == 0:
            continue
        xi = np.repeat(np.arange(len(x_t)), cnt)
        offs = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        yj = np.repeat(lo, cnt) + offs
        lag = np.rint((ty[yj] - x_t[xi]) / dt_ns).astype(np.int64)
        keep = np.abs(lag) <= K
        acc += np.bincount(lag[keep] + K, weights=(x_d[xi] * dy[yj])[keep], minlength=2 * K + 1)
    norm = np.sqrt((dx ** 2).sum() * (dy ** 2).sum())
    lags = np.arange(-K, K + 1) * dt_ns / 1e3
    return pd.DataFrame({"lag_us": lags, "corr": acc / norm if norm else np.nan})


def summarize_cc(cc: pd.DataFrame) -> dict:
    """Peak lag and the lead-lag ratio LLR = sum_{l>0} C^2 / sum_{l<0} C^2 (Huth & Abergel)."""
    c = cc["corr"].to_numpy()
    lag = cc["lag_us"].to_numpy()
    pos, neg = (c[lag > 0] ** 2).sum(), (c[lag < 0] ** 2).sum()
    i = int(np.nanargmax(c))
    return {"peak_lag_us": lag[i], "peak_corr": c[i], "llr": pos / neg if neg else np.inf}


def pair_leadlag(quotes: pd.DataFrame, clock: str = "ts_recv", dt_ns: int = 10_000,
                 max_lag_ns: int = 2_000_000):
    """Cross-correlation curves and summaries for every venue pair (X < Y), per symbol."""
    curves, rows = [], []
    for sym, qs in quotes.groupby("symbol"):
        mc = {v: mid_changes(g, clock) for v, g in qs.groupby("venue")}
        vs = sorted(mc)
        for i, x in enumerate(vs):
            for y in vs[i + 1:]:
                cc = crosscorr(*mc[x], *mc[y], dt_ns=dt_ns, max_lag_ns=max_lag_ns)
                cc["x"], cc["y"], cc["symbol"] = x, y, sym
                curves.append(cc)
                rows.append({"symbol": sym, "x": x, "y": y, **summarize_cc(cc)})
    return (pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()), pd.DataFrame(rows)


def event_study(quotes: pd.DataFrame, leader: str, symbol: str, offsets_ns: np.ndarray,
                clock: str = "ts_recv", min_move: float = 0.0) -> pd.DataFrame:
    """Average share of the leader's mid move that each venue has absorbed at each offset.

    For each leader mid change of size > min_move at time t: share_v(o) =
    (mid_v(t + o) - mid_v(t - 1ns)) / leader_move. 1.0 = fully caught up.
    """
    qs = quotes[quotes["symbol"] == symbol]
    t, dm = mid_changes(qs[qs["venue"] == leader], clock)
    big = np.abs(dm) > min_move
    t, dm = t[big], dm[big]
    rows = []
    for v, g in qs.groupby("venue"):
        g = g.sort_values(clock, kind="stable")
        tv = g[clock].to_numpy(np.int64)
        mv = ((g["bid"] + g["ask"]) / 2).to_numpy(np.float64)

        def mid_at(times):
            i = np.searchsorted(tv, times, side="right") - 1
            out = np.where(i >= 0, mv[np.clip(i, 0, None)], np.nan)
            return out

        base = mid_at(t - 1)
        for o in offsets_ns:
            share = (mid_at(t + o) - base) / dm
            rows.append({"venue": v, "offset_us": o / 1e3, "share": np.nanmean(share),
                         "n": int(np.isfinite(share).sum())})
    return pd.DataFrame(rows)
