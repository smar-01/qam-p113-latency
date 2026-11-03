"""Run the corrected local teammate study without downloading data.

Default quality mode uses the enclosing ColonyLab normalization/matcher on
raw state updates. --numerical-only explicitly retains the starter reaction
cohort while correcting midpoint arithmetic and binned normalization. Study
one aligned request interval per invocation; do not bridge multiple dates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from p113 import figures, latency, leadlag, regimes
from p113.load import bbo_changes, find_raw, load_raw

ROOT = Path(__file__).resolve().parent


def _legacy_vol(quotes, clock):
    """Starter-style quote carrying/warmup, explicitly on the selected clock."""
    out = []
    for symbol, one in quotes.groupby("symbol"):
        start, end = int(one[clock].min()), int(one[clock].max())
        grid = np.arange(start - start % 1_000_000_000 + 1_000_000_000, end + 1, 1_000_000_000, dtype=np.int64)
        mids = []
        for _, group in one.groupby("venue"):
            group = group.sort_values(clock, kind="stable")
            t = group[clock].to_numpy(np.int64)
            mid = ((group.bid + group.ask) / 2).to_numpy(float)
            i = np.searchsorted(t, grid, side="right") - 1
            mids.append(np.where(i >= 0, mid[np.maximum(i, 0)], np.nan))
        with np.errstate(all="ignore"):
            mid = np.nanmedian(np.vstack(mids), axis=0)
        r = np.r_[np.nan, np.diff(np.log(mid))]
        rv = np.sqrt(pd.Series(r ** 2).rolling(300, min_periods=60).sum().to_numpy())
        out.append(pd.DataFrame({"symbol": symbol, "t": grid, "rv": rv}))
    return pd.concat(out, ignore_index=True)


def run(quotes: pd.DataFrame, out: Path, leader: str, clocks: list[str], horizon_ns: int,
        vol_q: float, min_move: float, *, numerical_only: bool = False,
        event_support: str | None = None, max_quote_age_ms: float = 250):
    out.mkdir(parents=True, exist_ok=True)
    support = event_support or ("legacy" if numerical_only else "complete")
    if not numerical_only and support != "complete":
        raise ValueError("Quality mode requires complete event-study support")
    if not numerical_only and min_move != 0:
        raise ValueError("The validated quality event-study adapter currently requires --min-move 0; numerical-only mode supports a custom threshold")
    if not numerical_only and quotes.attrs.get("price_changes_only"):
        raise ValueError("Quality mode needs uncompressed raw state updates")
    legacy_quotes = bbo_changes(quotes) if numerical_only else None
    metadata = {"mode": "numerical_only" if numerical_only else "quality_controlled",
                "event_study_support": support, "clocks": clocks, "horizon_ns": horizon_ns,
                "max_quote_age_ms": None if numerical_only else max_quote_age_ms,
                "volatility": "selected-clock starter-style carrying/min60 warmup" if numerical_only else "selected-clock quality states/full300-return warmup",
                "reaction_latency": "unchanged starter cohort" if numerical_only else "ColonyLab native integer/full-horizon quality matcher",
                "note": "All single-window results are descriptive; no date-block inference is computed by this command."}
    (out / "analysis_mode.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for clock in clocks:
        if numerical_only:
            rv = regimes.label(_legacy_vol(legacy_quotes, clock), q=vol_q)
            lat = latency.pair_latencies(legacy_quotes, clock=clock, horizon_ns=horizon_ns, min_move=min_move)
            summ = latency.summarize(lat)
            curves, ll = leadlag.pair_leadlag(legacy_quotes, clock=clock)
            symbols = sorted(legacy_quotes.symbol.unique())
        else:
            from p113 import quality
            prepared, diagnostics = quality.prepare(quotes, clock, max_quote_age_ms)
            diagnostics.to_csv(out / f"quote_quality_{clock}.csv", index=False)
            rv = regimes.label(quality.realized_vol(prepared), q=vol_q)
            lat = quality.pair_latencies(prepared, horizon_ns=horizon_ns, min_move=min_move)
            summ = quality.summarize(lat, horizon_ns=horizon_ns)
            quality.summarize(lat, by=("symbol", "leader", "follower"), horizon_ns=horizon_ns).to_csv(out / f"latency_summary_by_symbol_{clock}.csv", index=False)
            curves, ll = quality.pair_leadlag(prepared)
            symbols = sorted({symbol for symbol, _ in prepared})
        rv.to_csv(out / f"realized_vol_{clock}.csv", index=False)
        lat["regime"] = regimes.tag(lat, rv)
        lat.to_parquet(out / f"latency_events_{clock}.parquet", index=False)
        summ.to_csv(out / f"latency_summary_{clock}.csv", index=False)
        matrix = latency.latency_matrix(summ)
        matrix.to_csv(out / f"latency_matrix_{clock}.csv")
        latency.leadership_share(summ).to_csv(out / f"leadership_share_{clock}.csv")
        by_reg = (latency.summarize(lat, by=("leader", "follower", "regime")) if numerical_only else
                  quality.summarize(lat, by=("leader", "follower", "regime"), horizon_ns=horizon_ns))
        by_reg.to_csv(out / f"latency_summary_by_regime_{clock}.csv", index=False)
        curves.to_csv(out / f"leadlag_curves_{clock}.csv", index=False)
        ll.to_csv(out / f"leadlag_summary_{clock}.csv", index=False)
        figures.heatmap(matrix, out / f"heatmap_{clock}.png", f"conditional matched latency (µs), {clock}")
        figures.distributions(lat, leader, out / f"distributions_{leader}_{clock}.png")
        figures.regime_bars(by_reg, leader, out / f"regimes_{leader}_{clock}.png")
        figures.leadlag(curves, leader, out / f"leadlag_{leader}_{clock}.png")
        offsets = np.r_[-100, 0, np.unique(np.logspace(0, 4, 25).astype(int))] * 1000
        for symbol in symbols:
            es = (leadlag.event_study(legacy_quotes, leader, symbol, offsets, clock=clock, min_move=min_move, support=support)
                  if numerical_only else quality.event_study(prepared, leader, symbol, offsets))
            es.to_csv(out / f"event_study_{leader}_{symbol}_{clock}.csv", index=False)
            figures.event_study(es, leader, symbol, out / f"event_study_{leader}_{symbol}_{clock}.png")
        print(f"[{clock}] {metadata['mode']}: {len(lat):,} pair-events; outputs in {out}", flush=True)
    return metadata


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--leader", default="XNAS")
    p.add_argument("--clock", choices=["ts_recv", "ts_event", "both"], default="both")
    p.add_argument("--horizon-ms", type=float, default=10.)
    p.add_argument("--vol-q", type=float, default=.75)
    p.add_argument("--min-move", type=float, default=0.)
    p.add_argument("--numerical-only", action="store_true", help="Retain the starter reaction cohort; disable quality filtering")
    p.add_argument("--event-support", choices=["legacy", "complete"], default=None)
    p.add_argument("--max-quote-age-ms", type=float, default=250.)
    a = p.parse_args(argv)
    if a.horizon_ms <= 0 or not 0 < a.vol_q < 1 or a.min_move < 0:
        p.error("Use a positive horizon, volatility quantile in (0,1), and nonnegative minimum move")
    if a.synthetic:
        from p113.synthetic import make_quotes
        quotes = make_quotes(volatile_lag_mult=2., feed_ns={"XNYS": 20_000, "BATS": 30_000})
        out = a.out or ROOT / "outputs" / "synthetic"
    else:
        files = find_raw(a.data)
        if not files:
            p.error(f"No local MBP-1 files under {a.data}; this command never downloads data")
        quotes = load_raw(files, keep_all=True, quality=False)
        out = a.out or ROOT / "outputs" / "real"
    clocks = ["ts_recv", "ts_event"] if a.clock == "both" else [a.clock]
    return run(quotes, out, a.leader, clocks, int(a.horizon_ms * 1e6), a.vol_q, a.min_move,
               numerical_only=a.numerical_only, event_support=a.event_support, max_quote_age_ms=a.max_quote_age_ms)


if __name__ == "__main__":
    main()
