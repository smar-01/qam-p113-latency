"""Run the full study on pulled data (data/raw) or on synthetic data (--synthetic).

    python analyze.py --synthetic            # no key, no data needed: writes outputs/synthetic/
    python analyze.py                        # every data/raw/*/mbp-1_*.dbn.zst -> outputs/real/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from p113 import figures, latency, leadlag, regimes
from p113.load import bbo_changes, find_raw, load_raw

ROOT = Path(__file__).resolve().parent


def run(quotes: pd.DataFrame, out: Path, leader: str, clocks: list[str], horizon_ns: int,
        vol_q: float, min_move: float):
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(quotes):,} quote changes, venues {sorted(quotes['venue'].unique())}, "
          f"symbols {sorted(quotes['symbol'].unique())}")

    rv = regimes.label(regimes.realized_vol(quotes), q=vol_q)
    rv.to_csv(out / "realized_vol.csv", index=False)

    for clock in clocks:
        lat = latency.pair_latencies(quotes, clock=clock, horizon_ns=horizon_ns, min_move=min_move)
        lat["regime"] = regimes.tag(lat, rv)
        lat.to_parquet(out / f"latency_events_{clock}.parquet", index=False)

        summ = latency.summarize(lat)
        summ.to_csv(out / f"latency_summary_{clock}.csv", index=False)
        mat = latency.latency_matrix(summ)
        mat.to_csv(out / f"latency_matrix_{clock}.csv")
        latency.leadership_share(summ).to_csv(out / f"leadership_share_{clock}.csv")
        by_reg = latency.summarize(lat, by=("leader", "follower", "regime"))
        by_reg.to_csv(out / f"latency_summary_by_regime_{clock}.csv", index=False)

        figures.heatmap(mat, out / f"heatmap_{clock}.png", f"median reaction latency (µs), {clock}")
        figures.distributions(lat, leader, out / f"distributions_{leader}_{clock}.png")
        figures.regime_bars(by_reg, leader, out / f"regimes_{leader}_{clock}.png")

        curves, ll = leadlag.pair_leadlag(quotes, clock=clock)
        curves.to_csv(out / f"leadlag_curves_{clock}.csv", index=False)
        ll.to_csv(out / f"leadlag_summary_{clock}.csv", index=False)
        figures.leadlag(curves, leader, out / f"leadlag_{leader}_{clock}.png")

        offsets = np.r_[-100, 0, np.unique(np.logspace(0, 4, 25).astype(int))] * 1_000  # -100 µs .. 10 ms
        for sym in sorted(quotes["symbol"].unique()):
            es = leadlag.event_study(quotes, leader, sym, offsets, clock=clock, min_move=min_move)
            es.to_csv(out / f"event_study_{leader}_{sym}_{clock}.csv", index=False)
            figures.event_study(es, leader, sym, out / f"event_study_{leader}_{sym}_{clock}.png")

        print(f"\n[{clock}] median latency (µs), leader rows -> follower columns")
        print(mat.round(1).to_string())
        print(f"\n[{clock}] lead-lag peaks\n{ll.round(3).to_string(index=False)}")
    print(f"\noutputs in {out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--leader", default="XNAS", help="venue whose moves the per-leader plots follow")
    p.add_argument("--clock", choices=["ts_recv", "ts_event", "both"], default="both")
    p.add_argument("--horizon-ms", type=float, default=10.0, help="give up matching after this long")
    p.add_argument("--vol-q", type=float, default=0.75, help="realized-vol quantile that starts 'volatile'")
    p.add_argument("--min-move", type=float, default=0.0, help="only leader moves larger than this ($)")
    a = p.parse_args(argv)

    if a.synthetic:
        from p113.synthetic import make_quotes

        quotes = bbo_changes(make_quotes(volatile_lag_mult=2.0, feed_ns={"XNYS": 20_000, "BATS": 30_000}))
        out = a.out or ROOT / "outputs" / "synthetic"
    else:
        files = find_raw(a.data)
        if not files:
            raise SystemExit(f"no mbp-1 files under {a.data}; run pull.py first (or use --synthetic)")
        quotes = load_raw(files)
        out = a.out or ROOT / "outputs" / "real"
    clocks = ["ts_recv", "ts_event"] if a.clock == "both" else [a.clock]
    run(quotes, out, a.leader, clocks, int(a.horizon_ms * 1e6), a.vol_q, a.min_move)


if __name__ == "__main__":
    main()
