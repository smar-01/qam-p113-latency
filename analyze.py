"""Standalone numerical corrections on local MBP-1 or synthetic quotes.

Preserves the starter bid/ask reaction cohort. --event-support complete adds
fixed observed midpoint-response cohorts; it does not add quality-aware latency.
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


def run(quotes: pd.DataFrame, out: Path, leader: str, clocks: list[str], horizon_ns: int,
        vol_q: float, min_move: float, *, event_support: str = "legacy",
        max_quote_age_ms: float = 250):
    """Keep reaction events unchanged; calculate volatility from raw states."""
    if event_support not in {"legacy", "complete"}:
        raise ValueError("event_support must be legacy or complete")
    if quotes.attrs.get("price_changes_only"):
        raise ValueError("The driver requires raw size/state updates for freshness-aware volatility")
    if (not np.isfinite(max_quote_age_ms) or max_quote_age_ms <= 0 or horizon_ns <= 0
            or not 0 < vol_q < 1 or not np.isfinite(min_move) or min_move < 0):
        raise ValueError("Use positive age/horizon, quantile in (0,1), and finite nonnegative minimum move")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    reaction_quotes = bbo_changes(quotes)  # Preserve receive-order starter selection for both clocks.
    metadata = {"mode": "numerical_only", "clocks": clocks, "horizon_ns": int(horizon_ns),
        "event_study_support": event_support,
        "reaction_latency": "unchanged starter bid/ask cohort; no freshness or full-follow-up filter",
        "midpoint_arithmetic": "native integer twice-midpoint numerator and denominator",
        "leadlag": "uncentered normalized cross-products of UTC-epoch fixed-bin midpoint increments",
        "volatility": "raw size/state updates, selected clock, full 300-return warmup and quote support",
        "volatility_max_quote_age_ms": float(max_quote_age_ms), "latency_max_quote_age_ms": None,
        "quality_aware_reaction_latency": False,
        "note": "Single-window descriptive results; volatility support does not qualify latency observations."}
    (out / "analysis_mode.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for clock in clocks:
        rv = regimes.label(regimes.realized_vol(quotes, clock=clock, max_age_ms=max_quote_age_ms), q=vol_q)
        rv.to_csv(out / f"realized_vol_{clock}.csv", index=False)
        lat = latency.pair_latencies(reaction_quotes, clock=clock, horizon_ns=horizon_ns, min_move=min_move)
        lat["regime"] = regimes.tag(lat, rv)
        lat.to_parquet(out / f"latency_events_{clock}.parquet", index=False)
        summary = latency.summarize(lat)
        summary.to_csv(out / f"latency_summary_{clock}.csv", index=False)
        matrix = latency.latency_matrix(summary)
        matrix.to_csv(out / f"latency_matrix_{clock}.csv")
        latency.leadership_share(summary).to_csv(out / f"leadership_share_{clock}.csv")
        by_regime = latency.summarize(lat, by=("leader", "follower", "regime"))
        by_regime.to_csv(out / f"latency_summary_by_regime_{clock}.csv", index=False)
        curves, peaks = leadlag.pair_leadlag(reaction_quotes, clock=clock)
        curves.to_csv(out / f"leadlag_curves_{clock}.csv", index=False)
        peaks.to_csv(out / f"leadlag_summary_{clock}.csv", index=False)
        figures.heatmap(matrix, out / f"heatmap_{clock}.png", f"conditional matched latency (µs), {clock}")
        figures.distributions(lat, leader, out / f"distributions_{leader}_{clock}.png")
        figures.regime_bars(by_regime, leader, out / f"regimes_{leader}_{clock}.png")
        figures.leadlag(curves, leader, out / f"leadlag_{leader}_{clock}.png")
        offsets = np.r_[-100, 0, np.unique(np.logspace(0, 4, 25).astype(int))] * 1000
        event_quotes = quotes if event_support == "complete" else reaction_quotes
        for symbol in sorted(quotes.symbol.unique()):
            response = leadlag.event_study(event_quotes, leader, symbol, offsets, clock=clock,
                                           min_move=min_move, support=event_support)
            response.to_csv(out / f"event_study_{leader}_{symbol}_{clock}.csv", index=False)
            figures.event_study(response, leader, symbol, out / f"event_study_{leader}_{symbol}_{clock}.png")
        print(f"[{clock}] numerical_only / event-support={event_support}: {len(lat):,} pair-events; outputs in {out}", flush=True)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--leader", default="XNAS")
    parser.add_argument("--clock", choices=["ts_recv", "ts_event", "both"], default="both")
    parser.add_argument("--horizon-ms", type=float, default=10.)
    parser.add_argument("--vol-q", type=float, default=.75)
    parser.add_argument("--min-move", type=float, default=0.)
    parser.add_argument("--event-support", choices=["legacy", "complete"], default="legacy")
    parser.add_argument("--max-quote-age-ms", type=float, default=250., help="Freshness for volatility only; does not filter latency")
    parser.add_argument("--numerical-only", action="store_true", help="Acknowledge the default standalone numerical-only mode")
    args = parser.parse_args(argv)
    if (not np.isfinite(args.horizon_ms) or args.horizon_ms <= 0 or not 0 < args.vol_q < 1
            or not np.isfinite(args.min_move) or args.min_move < 0
            or not np.isfinite(args.max_quote_age_ms) or args.max_quote_age_ms <= 0):
        parser.error("Use a positive horizon/quote age, quantile in (0,1), and finite nonnegative minimum move")
    if args.synthetic:
        from p113.synthetic import make_quotes
        quotes = make_quotes(volatile_lag_mult=2., feed_ns={"XNYS": 20_000, "BATS": 30_000})
        out = args.out or ROOT / "outputs" / "synthetic"
    else:
        files = find_raw(args.data)
        if not files:
            parser.error(f"No local MBP-1 files under {args.data}; this command never downloads data")
        quotes = load_raw(files, keep_all=True, quality=False)
        out = args.out or ROOT / "outputs" / "real"
    clocks = ["ts_recv", "ts_event"] if args.clock == "both" else [args.clock]
    return run(quotes, out, args.leader, clocks, int(args.horizon_ms * 1e6), args.vol_q, args.min_move,
               event_support=args.event_support, max_quote_age_ms=args.max_quote_age_ms)


if __name__ == "__main__":
    main()
