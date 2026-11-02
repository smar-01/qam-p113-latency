"""Load Databento MBP-1 files into one tidy quote table.

Output columns (one row per top-of-book change on one venue):
    ts_recv, ts_event : int64 nanoseconds since epoch (UTC)
    venue, symbol     : str
    bid, ask          : float64 dollars (NaN when that side is empty), unchanged for A/B compatibility
    bid_raw, ask_raw  : original int64 nanodollar prices, including the DBN undefined sentinel
    flags, valid      : retained quality metadata; default loading does not filter the legacy cohort
    bid_sz, ask_sz    : int64
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

UNDEF_PRICE = 9223372036854775807  # databento_dbn.UNDEF_PRICE
F_LAST, BAD_FLAGS = 128, 32 | 8 | 4
QUOTE_COLS = ["ts_recv", "ts_event", "venue", "symbol", "bid", "ask", "bid_sz", "ask_sz",
              "bid_raw", "ask_raw", "flags", "valid"]


def _px(col: pd.Series) -> np.ndarray:
    raw = col.to_numpy(dtype=np.int64)
    out = raw.astype(np.float64) / 1e9
    out[raw == UNDEF_PRICE] = np.nan
    return out


def from_store(store, *, quality: bool = False) -> pd.DataFrame:
    """DBNStore -> raw states, retaining exact prices and flags.

    ``quality=False`` keeps the original float bid/ask and row cohort exactly.
    Numerical-only comparisons should keep this setting. ``quality=True`` marks
    unsafe states as missing rather than dropping them and bridging a gap.
    The reaction matcher does not enforce full observation-barrier latency;
    marking invalid rows here does not add that policy to its estimates.
    """
    from .venues import venue_for_publisher

    df = store.to_df(price_type="fixed", pretty_ts=False, map_symbols=True).reset_index()
    if df.empty:
        return pd.DataFrame(columns=QUOTE_COLS)
    pubs = {p: venue_for_publisher(p) for p in df["publisher_id"].unique()}
    sym = df["symbol"] if "symbol" in df and df["symbol"].notna().all() else df["instrument_id"].astype(str)
    bid_raw = df["bid_px_00"].to_numpy(dtype=np.int64)
    ask_raw = df["ask_px_00"].to_numpy(dtype=np.int64)
    flags = df["flags"].to_numpy(dtype=np.uint64) if "flags" in df else np.full(len(df), F_LAST, dtype=np.uint64)
    valid = (((flags & F_LAST) != 0) & ((flags & BAD_FLAGS) == 0) & (bid_raw > 0)
             & (ask_raw > bid_raw) & (ask_raw != UNDEF_PRICE)
             & (df["bid_sz_00"].to_numpy() > 0) & (df["ask_sz_00"].to_numpy() > 0))
    q = pd.DataFrame({
        "ts_recv": df["ts_recv"].to_numpy(dtype=np.int64),
        "ts_event": df["ts_event"].to_numpy(dtype=np.int64),
        "venue": df["publisher_id"].map(pubs).astype(str),
        "symbol": sym.astype(str),
        "bid": _px(df["bid_px_00"]),
        "ask": _px(df["ask_px_00"]),
        "bid_sz": df["bid_sz_00"].to_numpy(dtype=np.int64),
        "ask_sz": df["ask_sz_00"].to_numpy(dtype=np.int64),
        "bid_raw": bid_raw, "ask_raw": ask_raw, "flags": flags, "valid": valid,
    })
    if quality:
        q.loc[~q.valid, ["bid", "ask"]] = np.nan
    q.attrs.update(price_changes_only=False, quality_filtered=quality,
                   requested_start_ns=int(store.metadata.start),
                   requested_end_ns=int(store.metadata.end) if store.metadata.end is not None else None)
    return q


def bbo_changes(q: pd.DataFrame, *, preserve_quality: bool = False,
                clock: str = "ts_recv") -> pd.DataFrame:
    """Legacy price-change selection, with explicit optional barrier retention.

    Default selection is unchanged so original and numerical-corrected code
    can receive exactly the same rows. Freshness-aware volatility must use
    uncompressed raw states: size-only updates refresh observed-state age.
    """
    attrs = dict(q.attrs)
    q = q.sort_values(["venue", "symbol", clock], kind="stable")
    g = q.groupby(["venue", "symbol"], sort=False)
    prev_bid, prev_ask = g["bid"].shift(), g["ask"].shift()
    changed = ~((q["bid"] == prev_bid) | (q["bid"].isna() & prev_bid.isna())) | \
              ~((q["ask"] == prev_ask) | (q["ask"].isna() & prev_ask.isna()))
    if preserve_quality and "valid" in q:
        previous_valid = g["valid"].shift().fillna(False)
        changed |= ~q.valid | ~previous_valid
    result = q[changed].reset_index(drop=True)
    result.attrs.update(attrs, price_changes_only=True)
    return result


def load_raw(paths: list[Path], *, keep_all: bool = True, quality: bool = False) -> pd.DataFrame:
    import databento as db

    parts = [from_store(db.DBNStore.from_file(p), quality=quality) for p in paths]
    if not parts:
        return pd.DataFrame(columns=QUOTE_COLS)
    bounds = {(part.attrs.get("requested_start_ns"), part.attrs.get("requested_end_ns")) for part in parts}
    if len(bounds) != 1:
        raise ValueError("Analyze one aligned requested interval per run; do not carry quotes across dates.")
    result = pd.concat(parts, ignore_index=True)
    result.attrs.update(parts[0].attrs)
    return result if keep_all else bbo_changes(result, preserve_quality=quality)


def find_raw(data_dir: Path, schema: str = "mbp-1") -> list[Path]:
    return sorted(Path(data_dir).glob(f"*/{schema}_*.dbn.zst"))
