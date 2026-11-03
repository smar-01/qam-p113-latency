"""Calm vs volatile inside regular hours only (09:30-16:00 ET), so the split can't just be
open market vs pre/post-market. Volatile = top quarter of each symbol's regular-hours seconds
by the 5-minute realized vol in realized_vol.csv. Second cut drops the first and last 5 minutes."""
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "outputs"


def hm(ns):
    return pd.to_datetime(ns, utc=True).dt.tz_convert("America/New_York").dt.strftime("%H:%M")


rows = []
for day in sys.argv[1:]:
    rv = pd.read_csv(OUT / day / "realized_vol.csv").dropna(subset=["rv"])
    rv = rv[hm(rv.t).between("09:30", "15:59")].copy()
    rv["sec"] = rv.t // 1_000_000_000
    rv["volatile"] = rv.rv >= rv.groupby("symbol").rv.transform(lambda s: s.quantile(0.75))
    for c in ["ts_recv", "ts_event"]:
        e = pd.read_parquet(OUT / day / f"latency_events_{c}.parquet",
                            columns=["leader", "follower", "symbol", "t", "latency_ns", "status"])
        e["hm"] = hm(e.t)
        e = e[e.hm.between("09:30", "15:59")]
        e["sec"] = e.t // 1_000_000_000
        e = e.merge(rv[["symbol", "sec", "volatile"]], on=["symbol", "sec"], how="inner")
        for label, g in [("all regular hours", e), ("excl first/last 5 min", e[e.hm.between("09:35", "15:54")])]:
            m = g[g.status == "matched"].groupby(["leader", "follower", "volatile"]).latency_ns.median().div(1e3).unstack()
            beh = g[g.status != "already"].groupby("volatile").status.apply(lambda s: (s == "matched").mean())
            rows.append({"day": day, "clock": c, "cut": label, "calm_us": round(m[False].median(), 1),
                         "volatile_us": round(m[True].median(), 1),
                         "pairs_slower_in_volatile": f"{(m[True] > m[False]).sum()}/{len(m)}",
                         "match_calm": round(beh[False], 3), "match_volatile": round(beh[True], 3)})
res = pd.DataFrame(rows)
print(res.to_string(index=False))
res.to_csv(OUT.parent / "results" / "rth_regimes.csv", index=False)
