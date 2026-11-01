"""Reaction latency: after a leader venue changes its best bid (or ask), how long until a
follower venue's best bid (or ask) reaches the same price?

Definition, per leader event on one side (bid or ask):
    leader moves its price from p0 to P at time t (direction d = sign(P - p0)).
    follower state at t:  if it already sits at or beyond P in direction d, the event is
                          'already' (the follower was not behind; excluded from latency).
    otherwise latency = first follower update time t' > t with follower price at or beyond P
                        in direction d, minus t; 'censored' if that does not happen within
                        `horizon_ns`.
Times use one clock column, `ts_recv` (Databento capture time, one clock for all venues)
or `ts_event` (each exchange's own matching-engine clock).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAT_COLS = ["leader", "follower", "symbol", "side", "t", "direction", "latency_ns", "status"]


def _side_events(times: np.ndarray, px: np.ndarray, min_move: float):
    """Leader price changes on one side: times, new prices, directions."""
    prev = np.r_[np.nan, px[:-1]]
    move = px - prev
    ok = np.isfinite(move) & (np.abs(move) > min_move)
    return times[ok], px[ok], np.sign(move[ok])


def reaction_latency(leader: pd.DataFrame, follower: pd.DataFrame, side: str = "bid",
                     clock: str = "ts_recv", horizon_ns: int = 10_000_000,
                     min_move: float = 0.0) -> pd.DataFrame:
    """Latency of `follower` matching each of `leader`'s price changes on `side`.

    Both frames hold one venue and one symbol. Returns t, direction, latency_ns, status.
    """
    L = leader.sort_values(clock, kind="stable")
    F = follower.sort_values(clock, kind="stable")
    t, P, d = _side_events(L[clock].to_numpy(np.int64), L[side].to_numpy(np.float64), min_move)
    tf = F[clock].to_numpy(np.int64)
    pf = F[side].to_numpy(np.float64)
    n, nf = len(t), len(tf)
    lat = np.full(n, np.nan)
    status = np.full(n, "censored", dtype=object)
    if n == 0 or nf == 0:
        return pd.DataFrame({"t": t, "direction": d, "latency_ns": lat, "status": status})

    def at_or_beyond(price, idx):
        p = price[idx]
        return np.where(d[idx] > 0, p >= P[idx], p <= P[idx])  # NaN compares False

    # follower state as of t (updates at the same nanosecond count as already there)
    i0 = np.searchsorted(tf, t, side="right") - 1
    has_state = i0 >= 0
    already = np.zeros(n, bool)
    idx = np.flatnonzero(has_state)
    already[idx] = at_or_beyond(pf[np.clip(i0, 0, None)], idx) if len(idx) else False
    status[already] = "already"

    # walk forward through follower updates, vectorised over unresolved events
    cur = i0 + 1
    active = np.flatnonzero(~already)
    while len(active):
        c = cur[active]
        inb = c < nf
        active, c = active[inb], c[inb]
        if not len(active):
            break
        dt = tf[c] - t[active]
        within = dt <= horizon_ns
        active, c, dt = active[within], c[within], dt[within]
        if not len(active):
            break
        hit = np.where(d[active] > 0, pf[c] >= P[active], pf[c] <= P[active])
        done = active[hit]
        lat[done] = dt[hit]
        status[done] = "matched"
        active = active[~hit]
        cur[active] += 1
    return pd.DataFrame({"t": t, "direction": d, "latency_ns": lat, "status": status})


def pair_latencies(quotes: pd.DataFrame, clock: str = "ts_recv", horizon_ns: int = 10_000_000,
                   venues: list[str] | None = None, min_move: float = 0.0) -> pd.DataFrame:
    """reaction_latency for every ordered venue pair, symbol and side. Long table."""
    venues = venues or sorted(quotes["venue"].unique())
    out = []
    for sym, qs in quotes.groupby("symbol"):
        by_v = {v: g for v, g in qs.groupby("venue") if v in venues}
        for a in by_v:
            for b in by_v:
                if a == b:
                    continue
                for side in ("bid", "ask"):
                    r = reaction_latency(by_v[a], by_v[b], side, clock, horizon_ns, min_move)
                    r.insert(0, "side", side)
                    r.insert(0, "symbol", sym)
                    r.insert(0, "follower", b)
                    r.insert(0, "leader", a)
                    out.append(r)
    if not out:
        return pd.DataFrame(columns=LAT_COLS)
    return pd.concat(out, ignore_index=True)[LAT_COLS]


def summarize(lat: pd.DataFrame, by: tuple[str, ...] = ("leader", "follower")) -> pd.DataFrame:
    """Per group: event counts, match rate and latency quantiles in microseconds."""
    by = list(by)

    def agg(g):
        m = g.loc[g["status"] == "matched", "latency_ns"] / 1e3
        behind = (g["status"] != "already").sum()
        return pd.Series({
            "n_events": len(g),
            "n_behind": behind,
            "n_matched": len(m),
            "match_rate": len(m) / behind if behind else np.nan,
            "p10_us": m.quantile(0.10) if len(m) else np.nan,
            "median_us": m.median() if len(m) else np.nan,
            "p90_us": m.quantile(0.90) if len(m) else np.nan,
            "mean_us": m.mean() if len(m) else np.nan,
        })

    return lat.groupby(by).apply(agg, include_groups=False).reset_index()


def latency_matrix(summary: pd.DataFrame, value: str = "median_us", min_matched: int = 20) -> pd.DataFrame:
    """Leader x follower matrix of `value`; cells with fewer than min_matched matches are NaN."""
    s = summary.copy()
    s.loc[s["n_matched"] < min_matched, value] = np.nan
    m = s.pivot(index="leader", columns="follower", values=value)
    order = sorted(set(m.index) | set(m.columns))
    return m.reindex(index=order, columns=order)


def leadership_share(summary: pd.DataFrame) -> pd.Series:
    """Information-share-style summary from the pairwise matrix.

    share(i) = sum_j matched(i->j) / sum_j [matched(i->j) + matched(j->i)].
    1.0 = venue i always moves first; 0.5 = no leadership either way. This is a counting
    measure, not Hasbrouck's VECM information share.
    """
    m = summary.pivot(index="leader", columns="follower", values="n_matched").fillna(0)
    order = sorted(set(m.index) | set(m.columns))
    m = m.reindex(index=order, columns=order, fill_value=0)
    led, followed = m.sum(axis=1), m.sum(axis=0)
    return (led / (led + followed)).rename("leadership_share").sort_values(ascending=False)
