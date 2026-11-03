"""Load Databento MBP-1 files into one tidy quote table.

Output columns (one row per top-of-book change on one venue):
    ts_recv, ts_event : int64 nanoseconds since epoch (UTC)
    venue, symbol     : str
    bid, ask          : float64 dollars (NaN when that side is empty)
    bid_sz, ask_sz    : int64
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

UNDEF_PRICE = 9223372036854775807  # databento_dbn.UNDEF_PRICE
QUOTE_COLS = ["ts_recv", "ts_event", "venue", "symbol", "bid", "ask", "bid_sz", "ask_sz"]


def _px(col: pd.Series) -> np.ndarray:
    raw = col.to_numpy(dtype=np.int64)
    out = raw.astype(np.float64) / 1e9
    out[raw == UNDEF_PRICE] = np.nan
    return out


def from_store(store) -> pd.DataFrame:
    """DBNStore (MBP-1 or BBO schema) -> quote table."""
    from p113.venues import venue_for_publisher

    df = store.to_df(price_type="fixed", pretty_ts=False, map_symbols=True).reset_index()
    if df.empty:
        return pd.DataFrame(columns=QUOTE_COLS)
    pubs = {p: venue_for_publisher(p) for p in df["publisher_id"].unique()}
    sym = df["symbol"] if "symbol" in df and df["symbol"].notna().all() else df["instrument_id"].astype(str)
    q = pd.DataFrame({
        "ts_recv": df["ts_recv"].to_numpy(dtype=np.int64),
        "ts_event": df["ts_event"].to_numpy(dtype=np.int64),
        "venue": df["publisher_id"].map(pubs).astype(str),
        "symbol": sym.astype(str),
        "bid": _px(df["bid_px_00"]),
        "ask": _px(df["ask_px_00"]),
        "bid_sz": df["bid_sz_00"].to_numpy(dtype=np.int64),
        "ask_sz": df["ask_sz_00"].to_numpy(dtype=np.int64),
    })
    return q


def bbo_changes(q: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where a venue's best bid or ask PRICE changed (sizes ignored)."""
    q = q.sort_values(["venue", "symbol", "ts_recv"], kind="stable")
    g = q.groupby(["venue", "symbol"], sort=False)
    prev_bid, prev_ask = g["bid"].shift(), g["ask"].shift()
    changed = ~((q["bid"] == prev_bid) | (q["bid"].isna() & prev_bid.isna())) | \
              ~((q["ask"] == prev_ask) | (q["ask"].isna() & prev_ask.isna()))
    return q[changed].reset_index(drop=True)


def load_raw(paths: list[Path]) -> pd.DataFrame:
    import databento as db

    parts = [from_store(db.DBNStore.from_file(p)) for p in paths]
    if not parts:
        return pd.DataFrame(columns=QUOTE_COLS)
    return bbo_changes(pd.concat(parts, ignore_index=True))


def find_raw(data_dir: Path, schema: str = "mbp-1") -> list[Path]:
    return sorted(Path(data_dir).glob(f"*/{schema}_*.dbn.zst"))
