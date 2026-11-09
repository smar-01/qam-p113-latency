"""Venue-by-venue view of one stock on one day (default: META around Meta's Muse launch).

For each venue this answers:
  1. Price path       How far the venue's midpoint sits from the cross-venue median, second by second.
  2. Inside quote     How much of the time the venue's bid/ask is the best in the market (NBBO),
                      over the whole session and during the day's biggest price moves.
  3. First mover      In each big move, when the venue's midpoint gets halfway to the new price,
                      and how far behind the first venue that was.
  4. Reaction latency The team's pairwise latency engine (p113/latency.py), split into big-move
                      windows versus the rest of the session.

Big moves are found from the data: the largest non-overlapping changes in the cross-venue median
midpoint over --move-window-s seconds. No announcement time is needed.

This reads local MBP-1 files only and never downloads. Get the data with pull.py first, e.g.
    python pull.py --symbols META --start 2026-09-09T08:00 --end 2026-09-10T00:00 --datasets all --cost-only
then
    python event_venues.py --day 2026-09-09 --session extended
    python event_venues.py --synthetic          # offline check; all data is synthetic
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from p113 import figures, latency  # noqa: E402
from p113.load import bbo_changes, find_raw, load_raw  # noqa: E402

ROOT = Path(__file__).resolve().parent
NY = "America/New_York"
SESSIONS = {"rth": ("09:30", "16:00"), "extended": ("04:00", "20:00"), "all": None}
F_LAST, BAD_FLAGS = 128, 32 | 8 | 4
SECOND = 1_000_000_000


# ----------------------------------------------------------------------------- loading
def day_files(data_dir: Path, day: str) -> list[Path]:
    """MBP-1 files pulled for one day (pull.py names them mbp-1_<start-date>_...)."""
    return [p for p in find_raw(data_dir) if p.name.startswith(f"mbp-1_{day}_")]


def session_bounds(day: str, session: str, q: pd.DataFrame) -> tuple[int, int]:
    """Session start/end in UTC nanoseconds. 'all' uses whatever the files cover."""
    if SESSIONS[session] is None:
        start = q.attrs.get("requested_start_ns", int(q["ts_recv"].min()))
        end = q.attrs.get("requested_end_ns") or int(q["ts_recv"].max()) + 1
        return int(start), int(end)
    a, b = SESSIONS[session]
    return pd.Timestamp(f"{day} {a}", tz=NY).value, pd.Timestamp(f"{day} {b}", tz=NY).value


def venue_states(q: pd.DataFrame, clock: str, max_spread_bps: float | None = None
                 ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per venue: sorted times and the bid/ask in force from each time (NaN = no usable quote).

    Rows that are not the last in their exchange packet are skipped (they are not a settled state).
    Flagged, one-sided, locked or crossed per-venue states become NaN rather than being bridged.
    With max_spread_bps, quotes wider than that are also NaN: a thin venue's far-away placeholder
    bid and ask have a midpoint that is not a real price. Same-timestamp rows keep the last one.
    """
    out = {}
    for venue, g in q.groupby("venue"):
        g = g.sort_values(clock, kind="stable")
        bid, ask = g["bid"].to_numpy(float), g["ask"].to_numpy(float)
        ok = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > bid)
        keep = np.ones(len(g), bool)
        if "valid" in g:
            ok &= g["valid"].to_numpy(bool)
        if max_spread_bps is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                ok &= (ask - bid) / ((ask + bid) / 2) * 1e4 <= max_spread_bps
        if "flags" in g:
            f = g["flags"].to_numpy(np.uint64)
            bad = (f & BAD_FLAGS) != 0
            keep = ((f & F_LAST) != 0) | bad
        t = g[clock].to_numpy(np.int64)[keep]
        bid, ask = np.where(ok, bid, np.nan)[keep], np.where(ok, ask, np.nan)[keep]
        last = np.r_[t[1:] != t[:-1], True] if len(t) else np.array([], bool)
        out[str(venue)] = (t[last], bid[last], ask[last])
    return out


def state_at(state, times: np.ndarray):
    t, bid, ask = state
    i = np.searchsorted(t, times, side="right") - 1
    j = np.maximum(i, 0)
    has = i >= 0
    if not len(t):
        nan = np.full(len(times), np.nan)
        return nan, nan.copy()
    return np.where(has, bid[j], np.nan), np.where(has, ask[j], np.nan)


# ----------------------------------------------------------------------------- 1. price paths
def grid_mids(states, grid: np.ndarray, min_venues: int | None = None):
    """Each venue's midpoint on the grid and the cross-venue median (consensus) midpoint."""
    venues = sorted(states)
    mids = np.vstack([np.mean(state_at(states[v], grid), axis=0) for v in venues]) if venues \
        else np.empty((0, len(grid)))
    need = min_venues or max(2, (len(venues) + 1) // 2)
    need = min(need, max(len(venues), 1))
    support = np.isfinite(mids).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        consensus = np.nanmedian(mids, axis=0) if venues else np.full(len(grid), np.nan)
    consensus[support < need] = np.nan
    return venues, mids, consensus


# ----------------------------------------------------------------------------- big moves
def find_moves(grid, consensus, window_s: int, n_moves: int, min_bps: float,
               exclude: list[tuple[int, int]] = ()) -> pd.DataFrame:
    """Largest non-overlapping consensus moves over `window_s` grid steps (1 s each).

    Windows touching any [lo, hi) interval in `exclude` (e.g. the opening and closing auctions)
    are skipped, so mechanical auction jumps do not crowd out the moves of interest.
    """
    cols = ["move", "start_ns", "end_ns", "p0", "p1", "move_bps"]
    k = int(window_s)
    if len(consensus) <= k:
        return pd.DataFrame(columns=cols)
    change = (consensus[k:] - consensus[:-k]) / consensus[:-k] * 1e4
    size = np.where(np.isfinite(change), np.abs(change), -1.0)
    for lo, hi in exclude:
        size[(grid[:-k] < hi) & (grid[k:] >= lo)] = -1.0
    starts = []
    for i in np.argsort(-size, kind="stable"):
        if len(starts) >= n_moves or size[i] < min_bps or size[i] < 0:
            break
        if any(abs(i - s) <= k for s in starts):  # windows [i, i+k] would overlap
            continue
        starts.append(int(i))
    starts.sort()
    rows = [{"start_ns": int(grid[i]), "end_ns": int(grid[i + k]), "p0": consensus[i],
             "p1": consensus[i + k], "move_bps": change[i]} for i in starts]
    moves = pd.DataFrame(rows, columns=cols[1:])
    moves.insert(0, "move", np.arange(1, len(moves) + 1))
    return moves


def in_moves(times: np.ndarray, moves: pd.DataFrame) -> np.ndarray:
    if moves.empty:
        return np.zeros(len(times), bool)
    starts, ends = moves["start_ns"].to_numpy(np.int64), moves["end_ns"].to_numpy(np.int64)
    i = np.searchsorted(starts, times, side="right") - 1
    return (i >= 0) & (times <= ends[np.maximum(i, 0)])


# ----------------------------------------------------------------------------- 3. first mover
def halfway_crossings(states, moves: pd.DataFrame, window_ns: int) -> pd.DataFrame:
    """Time each venue's midpoint first reaches halfway from the move's start to end price.

    status: 'crossed'; 'already' (past halfway when the move window opened, i.e. ahead; time is
    set to the window start); 'no_cross' (never got there by end + one more window).
    lag_us = venue's crossing time minus the earliest crossing among venues for that move.
    """
    rows = []
    for m in moves.itertuples():
        target, sign = m.p0 + 0.5 * (m.p1 - m.p0), np.sign(m.p1 - m.p0)
        for venue, (t, bid, ask) in states.items():
            lo = max(np.searchsorted(t, m.start_ns, side="right") - 1, 0)
            hi = np.searchsorted(t, m.end_ns + window_ns, side="right")
            seg_t, seg_mid = t[lo:hi], (bid[lo:hi] + ask[lo:hi]) / 2
            with np.errstate(invalid="ignore"):
                hit = (seg_mid - target) * sign >= 0
            if not hit.any():
                rows.append({"move": m.move, "venue": venue, "cross_ns": pd.NA, "status": "no_cross"})
                continue
            first = int(np.argmax(hit))
            already = seg_t[first] <= m.start_ns
            rows.append({"move": m.move, "venue": venue,
                         "cross_ns": int(max(seg_t[first], m.start_ns)),
                         "status": "already" if already else "crossed"})
    out = pd.DataFrame(rows, columns=["move", "venue", "cross_ns", "status"])
    # Integer nanoseconds: epoch times near 1.8e18 lose ~256 ns of precision as floats.
    out["cross_ns"] = out["cross_ns"].astype("Int64")
    if len(out):
        lag_ns = out["cross_ns"] - out.groupby("move")["cross_ns"].transform("min")
        out["lag_us"] = lag_ns.to_numpy(dtype=float, na_value=np.nan) / 1e3
        out["rank"] = out.groupby("move")["cross_ns"].rank(method="min").to_numpy(dtype=float, na_value=np.nan)
    return out


def first_mover_summary(cross: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        ok = g["status"] != "no_cross"
        lag = g.loc[ok, "lag_us"]
        return pd.Series({
            "n_moves": len(g), "n_reached_halfway": int(ok.sum()),
            "first_share": float((lag == 0).sum() / len(g)) if len(g) else np.nan,
            "median_rank": g.loc[ok, "rank"].median(),
            "median_lag_behind_first_us": lag.median(), "p90_lag_behind_first_us": lag.quantile(.9),
        })
    if cross.empty:
        return pd.DataFrame()
    return (cross.groupby("venue").apply(agg, include_groups=False)
            .sort_values("median_lag_behind_first_us").reset_index())


# ----------------------------------------------------------------------------- 2. inside quote
def nbbo_shares(states, start: int, end: int, moves: pd.DataFrame, chunk: int = 500_000) -> pd.DataFrame:
    """Time-weighted share of each venue's quoting time spent at the national best bid / offer.

    The NBBO here is built only from the venues loaded (no off-exchange or missing venues).
    Also reports the time-weighted average distance of each venue's bid/ask from the best, in cents,
    and how often the combined best bid/offer was locked or crossed.
    """
    venues = sorted(states)
    times = [s[0][(s[0] >= start) & (s[0] < end)] for s in states.values()]
    T = np.unique(np.concatenate([[start], *times])).astype(np.int64)
    dur = np.diff(np.r_[T, end]).astype(np.float64)
    move_mask = in_moves(T, moves)
    names = ["quoting_s", "at_bid_s", "at_ask_s", "at_both_s", "bid_gap_c_s", "ask_gap_c_s"]
    acc = {w: np.zeros((len(venues), len(names))) for w in ("session", "big_moves")}
    locked = {"session": 0.0, "big_moves": 0.0}
    span = {"session": 0.0, "big_moves": 0.0}
    for lo in range(0, len(T), chunk):
        tt, dd, mm = T[lo:lo + chunk], dur[lo:lo + chunk], move_mask[lo:lo + chunk]
        B = np.empty((len(venues), len(tt)))
        A = np.empty_like(B)
        for i, v in enumerate(venues):
            B[i], A[i] = state_at(states[v], tt)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            nbb, nbo = np.nanmax(B, axis=0), np.nanmin(A, axis=0)
        has = np.isfinite(nbb) & np.isfinite(nbo)
        for w, sel in (("session", np.ones(len(tt), bool)), ("big_moves", mm)):
            d = dd * sel
            locked[w] += float(d[has & (nbb >= nbo)].sum())
            span[w] += float(d[has].sum())
            for i in range(len(venues)):
                q = np.isfinite(B[i]) & np.isfinite(A[i])
                at_b, at_a = q & (B[i] == nbb), q & (A[i] == nbo)
                acc[w][i] += [d[q].sum(), d[at_b].sum(), d[at_a].sum(), d[at_b & at_a].sum(),
                              (d[q] * (nbb[q] - B[i][q]) * 100).sum(), (d[q] * (A[i][q] - nbo[q]) * 100).sum()]
    rows = []
    for w, a in acc.items():
        for i, v in enumerate(venues):
            quoting = a[i, 0]
            rows.append({"window": w, "venue": v, "quoting_seconds": quoting / SECOND,
                         "pct_time_at_best_bid": 100 * a[i, 1] / quoting if quoting else np.nan,
                         "pct_time_at_best_ask": 100 * a[i, 2] / quoting if quoting else np.nan,
                         "pct_time_at_both": 100 * a[i, 3] / quoting if quoting else np.nan,
                         "avg_bid_behind_best_cents": a[i, 4] / quoting if quoting else np.nan,
                         "avg_ask_behind_best_cents": a[i, 5] / quoting if quoting else np.nan,
                         "pct_time_nbbo_locked_or_crossed": 100 * locked[w] / span[w] if span[w] else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 4. reaction latency
def pair_latency_summaries(rq: pd.DataFrame, clock: str, horizon_ns: int, moves: pd.DataFrame):
    """The team's reaction-latency engine, summarized one venue pair at a time.

    Same numbers as latency.summarize(latency.pair_latencies(...)), but each pair's events are
    discarded after summarizing, so a full day does not hold tens of millions of rows in memory.
    Returns (overall summary, summary split into big_move / other windows, total pair-events).
    """
    overall, by_window, n_events = [], [], 0
    by_venue = {v: g for v, g in rq.groupby("venue")}
    for a in sorted(by_venue):
        for b in sorted(by_venue):
            if a == b:
                continue
            parts = []
            for side in ("bid", "ask"):
                r = latency.reaction_latency(by_venue[a], by_venue[b], side, clock, horizon_ns)
                r.insert(0, "side", side)
                parts.append(r)
            lat = pd.concat(parts, ignore_index=True)
            lat.insert(0, "follower", b)
            lat.insert(0, "leader", a)
            lat["window"] = np.where(in_moves(lat["t"].to_numpy(np.int64), moves), "big_move", "other")
            n_events += len(lat)
            if len(lat):
                overall.append(latency.summarize(lat))
                by_window.append(latency.summarize(lat, by=("leader", "follower", "window")))
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()  # noqa: E731
    return cat(overall), cat(by_window), n_events


# ----------------------------------------------------------------------------- activity
def activity(q: pd.DataFrame, clock: str, start: int, end: int, moves: pd.DataFrame) -> pd.DataFrame:
    t = q[clock].to_numpy(np.int64)
    q = q.assign(_big=in_moves(t, moves))
    seconds = (end - start) / SECOND
    move_seconds = float((moves["end_ns"] - moves["start_ns"]).sum()) / SECOND if len(moves) else 0.0
    rows = []
    for v, g in q.groupby("venue"):
        ok = (g["ask"] > g["bid"]) & (g["bid"] > 0)
        spread = (g["ask"] - g["bid"])[ok] * 100
        spread_bps = ((g["ask"] - g["bid"]) / ((g["ask"] + g["bid"]) / 2))[ok] * 1e4
        rows.append({"venue": v, "updates": len(g), "share_of_all_updates": len(g) / len(q),
                     "updates_per_s": len(g) / seconds if seconds else np.nan,
                     "updates_per_s_in_big_moves": g["_big"].sum() / move_seconds if move_seconds else np.nan,
                     "median_spread_cents": spread.median(), "median_spread_bps": spread_bps.median()})
    return pd.DataFrame(rows).sort_values("updates", ascending=False)


# ----------------------------------------------------------------------------- figures
def _local(ns):
    return pd.to_datetime(np.asarray(ns, dtype=np.int64), utc=True).tz_convert(NY).tz_localize(None)


def plot_price_paths(grid, venues, mids, consensus, moves, path: Path, title: str):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, height_ratios=[1, 1.3])
    x = _local(grid)
    top.plot(x, consensus, color="black", lw=1)
    top.set_ylabel("cross-venue median mid ($)")
    top.set_title(title)
    dev = pd.DataFrame((mids - consensus) * 100, index=venues).T
    smooth = dev.rolling(60, min_periods=10).mean()
    for v in venues:
        bottom.plot(x, smooth[v], lw=1, label=v)
    bottom.axhline(0, color="grey", lw=.8)
    bottom.set_ylabel("venue mid minus median (cents)\n60 s rolling mean")
    for ax in (top, bottom):
        for m in moves.itertuples():
            ax.axvspan(_local([m.start_ns])[0], _local([m.end_ns])[0], color="orange", alpha=.25, lw=0)
    bottom.legend(ncol=min(len(venues), 8), fontsize=8, loc="upper left")
    bottom.set_xlabel("New York time (orange = biggest moves)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_biggest_move(states, moves, cross, path: Path, window_ns: int, title: str):
    if moves.empty:
        return
    m = moves.loc[moves["move_bps"].abs().idxmax()]
    target = m.p0 + 0.5 * (m.p1 - m.p0)
    c = cross[(cross["move"] == m.move) & (cross["status"] != "no_cross")]
    fig, (wide, zoom) = plt.subplots(1, 2, figsize=(14, 5))
    first = c["cross_ns"].min() if len(c) else m.start_ns
    last = c["cross_ns"].max() if len(c) else m.end_ns
    pad = max(2e6, 0.2 * (last - first))
    zoom_lo, zoom_hi = first - pad, last + pad
    for v, (t, bid, ask) in sorted(states.items()):
        mid = (bid + ask) / 2
        for ax, lo, hi, origin, scale in ((wide, m.start_ns - window_ns / 2, m.end_ns + window_ns / 2, m.start_ns, 1e9),
                                          (zoom, zoom_lo, zoom_hi, first, 1e6)):
            i0 = max(np.searchsorted(t, lo, side="right") - 1, 0)
            i1 = np.searchsorted(t, hi, side="right")
            xs = np.r_[(t[i0:i1] - origin) / scale, (hi - origin) / scale]
            ys = np.r_[mid[i0:i1], mid[i1 - 1] if i1 > i0 else np.nan]
            ax.step(xs, ys, where="post", lw=1.2, label=v)
    for ax in (wide, zoom):
        ax.axhline(target, color="grey", ls="--", lw=.8)
        ax.set_ylabel("venue midpoint ($)")
    wide.set_xlabel("seconds from move start")
    wide.set_title(f"{title}\nbiggest move: {m.move_bps:+.1f} bps starting {_local([m.start_ns])[0]:%H:%M:%S}")
    zoom.set_xlabel("milliseconds from first venue reaching halfway (dashed line)")
    zoom.set_title("zoom: order in which venues reach the halfway price")
    zoom.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_nbbo(shares: pd.DataFrame, path: Path, title: str):
    s = shares.assign(pct=(shares["pct_time_at_best_bid"] + shares["pct_time_at_best_ask"]) / 2)
    names = {"session": "whole session", "big_moves": "during biggest moves"}
    p = s.pivot(index="venue", columns="window", values="pct")
    p = p[[c for c in names if c in p]].sort_values("session", ascending=False).rename(columns=names)
    ax = p.plot.bar(figsize=(9, 4.5), rot=0, color=["#4c72b0", "#dd8452"][:p.shape[1]])
    ax.set_ylabel("% of quoting time at best bid / ask\n(average of the two sides)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_first_mover(summary: pd.DataFrame, path: Path, title: str):
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(summary["venue"], summary["median_lag_behind_first_us"], color="#55a868")
    for i, r in enumerate(summary.itertuples()):
        ax.text(i, r.median_lag_behind_first_us, f"first {100 * r.first_share:.0f}%", ha="center",
                va="bottom", fontsize=8)
    ax.set_ylabel("median µs behind the first venue\nto reach halfway in a big move")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- driver
def run(quotes: pd.DataFrame, out: Path, **kwargs) -> dict:
    """Write all tables, figures, summary.md and run_info.json for one symbol-day."""
    with warnings.catch_warnings():  # all-NaN slices (venues not quoting) are expected and handled
        warnings.simplefilter("ignore", RuntimeWarning)
        return _run(quotes, out, **kwargs)


def _run(quotes: pd.DataFrame, out: Path, *, symbol: str, day: str, session: str, clocks: list[str],
         move_window_s: int, n_moves: int, min_move_bps: float, horizon_ns: int,
         max_spread_bps: float | None = 25.0, min_venues: int = 3, skip_open_close_min: float = 5) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    q_all = quotes[quotes["symbol"] == symbol]
    if q_all.empty:
        raise ValueError(f"No {symbol} rows; symbols present: {sorted(quotes['symbol'].unique())}")
    q_all.attrs.update(quotes.attrs)
    start, end = session_bounds(day, session, q_all)
    in_session = (q_all["ts_recv"] >= start) & (q_all["ts_recv"] < end)
    q = q_all[in_session]
    if q.empty:
        raise ValueError(f"No {symbol} quotes inside the {session} session of {day}")
    window_ns = move_window_s * SECOND
    info = {"symbol": symbol, "day": day, "session": session, "start_ny": str(_local([start])[0]),
            "end_ny": str(_local([end])[0]), "venues": sorted(q["venue"].unique()), "clocks": clocks,
            "move_window_s": move_window_s, "n_moves_requested": n_moves, "min_move_bps": min_move_bps,
            "latency_horizon_ms": horizon_ns / 1e6, "rows_in_session": int(len(q)),
            "max_spread_bps_for_midpoints": max_spread_bps, "min_venues_for_consensus": min_venues,
            "skip_open_close_min": skip_open_close_min, "by_clock": {}}
    exclude = []
    if skip_open_close_min > 0:
        span = int(skip_open_close_min * 60 * SECOND)
        for hhmm, before in (("09:30", 0), ("16:00", 60 * SECOND)):
            at = pd.Timestamp(f"{day} {hhmm}", tz=NY).value
            exclude.append((at - before, at + span))
    report = [f"# {symbol} across venues, {day} ({session} session)\n",
              f"Venues: {', '.join(info['venues'])}. Rows: {len(q):,}.\n"]
    for clock in clocks:
        # Full file, so each venue's opening state is known. All quotes for the inside-quote shares;
        # only reasonably tight quotes for midpoint-based measures (price gaps, first mover).
        states = venue_states(q_all, clock)
        tight = venue_states(q_all, clock, max_spread_bps)
        grid = np.arange(start - start % SECOND + SECOND, end, SECOND, dtype=np.int64)
        venues, mids, consensus = grid_mids(tight, grid, min_venues)
        moves = find_moves(grid, consensus, move_window_s, n_moves, min_move_bps, exclude)
        moves.assign(start_ny=_local(moves["start_ns"]).strftime("%H:%M:%S") if len(moves) else []) \
            .to_csv(out / f"big_moves_{clock}.csv", index=False)
        dev = pd.DataFrame((mids - consensus).T * 100, columns=venues)
        dev.insert(0, "time_ny", _local(grid))
        dev.insert(1, "consensus_mid", consensus)
        dev.to_csv(out / f"venue_vs_median_cents_{clock}.csv", index=False)
        dev_summary = pd.DataFrame({
            "venue": venues,
            "median_cents_vs_median": np.nanmedian(dev[venues].to_numpy(), axis=0),
            "mean_abs_cents_vs_median": np.nanmean(np.abs(dev[venues].to_numpy()), axis=0),
            "mean_abs_cents_in_big_moves": [np.nanmean(np.abs(dev.loc[in_moves(grid, moves), v]))
                                            if len(moves) else np.nan for v in venues]})
        dev_summary.to_csv(out / f"venue_deviation_summary_{clock}.csv", index=False)

        cross = halfway_crossings(tight, moves, window_ns)
        cross.to_csv(out / f"halfway_crossings_{clock}.csv", index=False)
        fm = first_mover_summary(cross)
        fm.to_csv(out / f"first_mover_{clock}.csv", index=False)

        shares = nbbo_shares(states, start, end, moves)
        shares.to_csv(out / f"nbbo_time_share_{clock}.csv", index=False)
        act = activity(q, clock, start, end, moves)
        act.to_csv(out / f"venue_activity_{clock}.csv", index=False)

        overall, by_window, n_pair_events = pair_latency_summaries(bbo_changes(q), clock, horizon_ns, moves)
        overall.to_csv(out / f"latency_summary_{clock}.csv", index=False)
        by_window.to_csv(out / f"latency_summary_by_window_{clock}.csv", index=False)
        matrix = latency.latency_matrix(overall)
        matrix.to_csv(out / f"latency_matrix_{clock}.csv")
        lead = pd.DataFrame({"session": latency.leadership_share(overall)})
        big = by_window[by_window["window"] == "big_move"]
        if len(big):
            lead["big_moves"] = latency.leadership_share(big)
        # Venues whose typical quote is tight. A venue that rarely matches anyone's price can look
        # like a 'leader' because it is seldom counted as following; this view leaves those out.
        core = act.loc[act["median_spread_bps"] <= (max_spread_bps or np.inf), "venue"].tolist()
        info.setdefault("core_venues", core)
        if len(core) >= 2:
            sub = lambda d: d[d["leader"].isin(core) & d["follower"].isin(core)]  # noqa: E731
            lead["session_core_venues"] = latency.leadership_share(sub(overall))
            if len(big):
                lead["big_moves_core_venues"] = latency.leadership_share(sub(big))
        lead = lead.sort_values(lead.columns[-1], ascending=False)
        lead.to_csv(out / f"leadership_share_{clock}.csv")

        label = f"{symbol} {day}, {clock}"
        plot_price_paths(grid, venues, mids, consensus, moves, out / f"price_paths_{clock}.png", label)
        plot_biggest_move(tight, moves, cross, out / f"biggest_move_{clock}.png", window_ns, label)
        plot_nbbo(shares, out / f"nbbo_time_share_{clock}.png", f"time at the inside quote, {label}")
        plot_first_mover(fm, out / f"first_mover_{clock}.png", f"who reaches the new price first, {label}")
        figures.heatmap(matrix, out / f"latency_heatmap_{clock}.png", f"conditional matched latency (µs), {label}")

        info["by_clock"][clock] = {"big_moves_found": int(len(moves)), "latency_pair_events": int(n_pair_events)}
        sess = shares[shares["window"] == "session"].set_index("venue")
        report += [f"\n## Clock: {clock}\n",
                   f"Big moves found: {len(moves)} (>= {min_move_bps} bps over {move_window_s} s; "
                   f"first {skip_open_close_min:g} min after the 09:30 open and around the 16:00 close skipped).\n",
                   f"\nMidpoint measures use only quotes with spread <= {max_spread_bps} bps. "
                   f"Core (typically tight) venues: {', '.join(core)}.\n",
                   "\n### Biggest moves\n```\n" + moves.drop(columns=["start_ns", "end_ns"])
                   .assign(start_ny=_local(moves["start_ns"]).strftime("%H:%M:%S") if len(moves) else [])
                   .round(3).to_string(index=False) + "\n```\n",
                   "\n### Who reaches the new price first (big moves)\n```\n" + fm.round(2).to_string(index=False) + "\n```\n",
                   "\n### Time at the inside quote (whole session vs big moves)\n```\n" +
                   shares.pivot(index="venue", columns="window",
                                values=["pct_time_at_best_bid", "pct_time_at_best_ask", "avg_bid_behind_best_cents"])
                   .round(2).to_string() + "\n```\n",
                   f"\nCombined NBBO locked or crossed: {sess['pct_time_nbbo_locked_or_crossed'].iloc[0]:.3f}% of the session.\n",
                   "\n### Leadership share (reaction-latency engine)\n```\n" + lead.round(3).to_string() + "\n```\n",
                   "\n### Venue activity\n```\n" + act.round(3).to_string(index=False) + "\n```\n"]
        print(f"[{clock}] {len(moves)} big moves, {n_pair_events:,} pair-events; outputs in {out}", flush=True)
    report.append(
        "\n## Reading these numbers\n"
        "- Big moves are picked from the data and are retrospective; they are not tied to a news timestamp.\n"
        "- 'First to halfway' and leadership count who moved first, not who caused the move.\n"
        "- The NBBO is built only from the venues loaded here; it is not the official SIP NBBO.\n"
        "- ts_recv includes each feed's path to Databento; ts_event uses each exchange's own clock.\n"
        "- One day, one symbol: descriptive only, no confidence intervals.\n"
        "- Outside 09:30-16:00 some venues leave old quotes standing, so the combined best bid/offer is\n"
        "  often locked or crossed; read pre-market and after-hours inside-quote shares with care.\n")
    (out / "summary.md").write_text("".join(report), encoding="utf-8")
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def synthetic_quotes() -> pd.DataFrame:
    """Known answer: XNAS leads; others follow after fixed lags, doubled in the volatile middle."""
    from p113.synthetic import make_quotes
    us = 1_000
    return make_quotes(symbol="META", seconds=1_800, volatile_lag_mult=2.0,
                       lags_ns={"XNYS": 50 * us, "ARCX": 80 * us, "BATS": 150 * us, "EDGX": 160 * us,
                                "IEXG": 400 * us},
                       feed_ns={"XNYS": 20 * us, "BATS": 30 * us})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic", action="store_true", help="offline run on generated data")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--day", help="YYYY-MM-DD, the start date used in pull.py (e.g. 2026-09-09)")
    p.add_argument("--symbol", default="META")
    p.add_argument("--session", choices=list(SESSIONS), default="rth",
                   help="rth 09:30-16:00 NY, extended 04:00-20:00 NY, all = whatever was downloaded")
    p.add_argument("--clock", choices=["ts_recv", "ts_event", "both"], default="both")
    p.add_argument("--move-window-s", type=int, default=10, help="length of a 'big move' window")
    p.add_argument("--n-moves", type=int, default=20, help="how many of the biggest moves to study")
    p.add_argument("--min-move-bps", type=float, default=10., help="ignore moves smaller than this")
    p.add_argument("--horizon-ms", type=float, default=10., help="reaction-latency matching horizon")
    p.add_argument("--max-spread-bps", type=float, default=25.,
                   help="ignore a venue's midpoint while its spread is wider than this (0 = no limit)")
    p.add_argument("--min-venues", type=int, default=3, help="venues needed for the cross-venue median")
    p.add_argument("--skip-open-close-min", type=float, default=5.,
                   help="skip big moves in the first N minutes after 09:30 and around 16:00 (0 = keep)")
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    if (a.move_window_s < 1 or a.n_moves < 1 or a.min_move_bps < 0 or a.horizon_ms <= 0
            or a.max_spread_bps < 0 or a.min_venues < 1 or a.skip_open_close_min < 0):
        p.error("Use a window >= 1 s, n-moves >= 1, min-venues >= 1, a positive horizon, "
                "and nonnegative min-move, spread and skip values")
    if a.synthetic:
        quotes = synthetic_quotes()
        symbol, session = "META", "all"
        day = str(_local([quotes["ts_recv"].min()])[0].date())
        out = a.out or ROOT / "outputs" / "event_synthetic"
    else:
        if not a.day:
            p.error("--day is required for real data (or use --synthetic)")
        files = day_files(a.data, a.day)
        if not files:
            p.error(f"No mbp-1_{a.day}_*.dbn.zst files under {a.data}/<dataset>/; pull them with pull.py first")
        quotes = load_raw(files, keep_all=True, quality=False)
        symbol, session, day = a.symbol, a.session, a.day
        out = a.out or ROOT / "outputs" / f"event_{symbol}_{day}_{session}"
    clocks = ["ts_recv", "ts_event"] if a.clock == "both" else [a.clock]
    return run(quotes, out, symbol=symbol, day=day, session=session, clocks=clocks,
               move_window_s=a.move_window_s, n_moves=a.n_moves, min_move_bps=a.min_move_bps,
               horizon_ns=int(a.horizon_ms * 1e6), max_spread_bps=a.max_spread_bps or None,
               min_venues=a.min_venues, skip_open_close_min=a.skip_open_close_min)


if __name__ == "__main__":
    main()
