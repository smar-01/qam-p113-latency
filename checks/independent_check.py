"""Recompute reaction latency from the raw files with a plain loop, sharing no code with
p113/latency.py, and report data health. Regular hours only (09:30-16:00 ET)."""
import sys
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd

DAY = sys.argv[1] if len(sys.argv) > 1 else "2026-10-15"
ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"
HORIZON = 10_000_000


def load(ds, sym):
    f = next((ROOT / ds).glob(f"mbp-1_{DAY}_*.dbn.zst"))
    d = db.DBNStore.from_file(f).to_df(price_type="float", pretty_ts=False, map_symbols=True)
    d = d[d.symbol == sym].reset_index()
    t = pd.to_datetime(d.ts_recv, utc=True).dt.tz_convert("America/New_York")
    d = d[(t.dt.time >= pd.Timestamp("09:30").time()) & (t.dt.time < pd.Timestamp("16:00").time())]
    return d[["ts_recv", "bid_px_00", "ask_px_00"]].dropna()


def lat(a, b):
    ta, pa = a.ts_recv.astype("int64").values, a.bid_px_00.values
    tb, pb = b.ts_recv.astype("int64").values, b.bid_px_00.values
    out, j, n_behind = [], 0, 0
    for i in range(1, len(pa)):
        if pa[i] <= pa[i - 1]:  # upticks in the bid only
            continue
        while j + 1 < len(tb) and tb[j + 1] <= ta[i]:
            j += 1
        if tb[j] > ta[i] or pb[j] >= pa[i]:
            continue
        n_behind += 1
        k = j + 1
        while k < len(tb) and tb[k] - ta[i] <= HORIZON:
            if pb[k] >= pa[i]:
                out.append((tb[k] - ta[i]) / 1000)
                break
            k += 1
    return n_behind, np.array(out)


for sym in ["AAPL", "SPY"]:
    q = {v: load(ds, sym) for v, ds in {"XNAS": "XNAS.ITCH", "BATS": "BATS.PITCH", "EDGX": "EDGX.PITCH", "IEXG": "IEXG.TOPS"}.items()}
    for v, d in q.items():
        crossed = (d.bid_px_00 >= d.ask_px_00).mean()
        print(f"{sym} {v}: rows {len(d):,}, locked/crossed {crossed:.3%}")
    for a, b in [("XNAS", "BATS"), ("BATS", "EDGX"), ("XNAS", "IEXG")]:
        n, x = lat(q[a], q[b])
        print(f"{sym} {a}->{b} bid upticks, ts_recv, RTH: behind {n:,}, matched {len(x):,}, median {np.median(x):.1f} us, p10 {np.percentile(x,10):.1f}, p90 {np.percentile(x,90):.1f}")
