"""Regression checks for the corrected teammate loader, cache and volatility."""
from __future__ import annotations

import json

import databento as db
import databento_dbn as dbn
import numpy as np
import pandas as pd
import pytest


from p113 import load, regimes
import pull


def test_query_path_distinguishes_symbols_times_and_limits(monkeypatch, tmp_path):
    monkeypatch.setattr(pull, "DATA", tmp_path)
    def path(symbols=("AAPL",), start="2026-10-15T14:00:00Z", limit=None):
        return pull.out_path("XNAS.ITCH", "mbp-1", start, "2026-10-15T15:00:00Z", symbols, limit=limit)
    assert len({path(), path(("SPY",)), path(start="2026-10-15T14:10:00Z"), path(limit=100)}) == 4
    assert path(("AAPL", "SPY")) == path(("SPY", "AAPL"))


def test_unverified_or_changed_cache_is_not_silently_reused(tmp_path):
    request = pull.request_identity("XNAS.ITCH", "mbp-1", "2026-10-15", "2026-10-16", ["AAPL"])
    path = tmp_path / "sample.dbn.zst"
    assert not pull.cache_reusable(path, request)
    path.write_bytes(b"complete-source")
    with pytest.raises(ValueError, match="completion receipt"):
        pull.cache_reusable(path, request)
    pull._record_complete(path, request)
    assert pull.cache_reusable(path, request)
    other = dict(request, symbols=["SPY"])
    with pytest.raises(ValueError, match="request, bytes or completion"):
        pull.cache_reusable(path, other)
    path.write_bytes(b"tampered-source")
    with pytest.raises(ValueError, match="request, bytes or completion"):
        pull.cache_reusable(path, request)


def test_default_float_cohort_and_original_integer_prices_preserved():
    t = 1_790_000_000_000_000_000
    meta = dbn.Metadata(dataset="XNAS.ITCH", start=t, end=t + 10**9,
                        stype_in=dbn.SType.RAW_SYMBOL, stype_out=dbn.SType.INSTRUMENT_ID,
                        schema=dbn.Schema.MBP_1, symbols=["AAPL"])
    levels = dbn.BidAskPair(bid_px=100_040_000_000, ask_px=100_060_000_000,
                           bid_sz=100, ask_sz=100, bid_ct=1, ask_ct=1)
    row = dbn.MBP1Msg(publisher_id=2, instrument_id=1, ts_event=t + 1, ts_recv=t + 2,
                     price=100_040_000_000, size=100, action=dbn.Action.ADD, side=dbn.Side.BID,
                     depth=0, flags=132, ts_in_delta=0, sequence=1, levels=levels)
    store = db.DBNStore.from_bytes(bytes(meta.encode()) + bytes(row))
    q = load.from_store(store)
    assert q.bid.iloc[0] == 100.04 and q.ask.iloc[0] == 100.06
    assert q.bid_raw.iloc[0] == 100_040_000_000 and q.ask_raw.iloc[0] == 100_060_000_000
    assert q["flags"].iloc[0] == 132 and not q.valid.iloc[0]
    assert q.ts_recv.iloc[0] == t + 2 and q.ts_event.iloc[0] == t + 1
    filtered = load.from_store(store, quality=True)
    assert filtered[["bid", "ask"]].isna().all().all()
    assert filtered.bid_raw.iloc[0] == 100_040_000_000


def raw_quotes(seconds=6, spacing_ms=100):
    ns = np.arange(0, seconds * 1_000_000_000 + 1, spacing_ms * 1_000_000, dtype=np.int64)
    frame = pd.DataFrame({"ts_recv": ns, "ts_event": ns, "venue": "XNAS", "symbol": "SYN",
                          "bid": 100 + np.floor(ns / 1e9) * .01, "ask": 100.02 + np.floor(ns / 1e9) * .01,
                          "bid_sz": 100, "ask_sz": 100, "flags": 128, "valid": True})
    frame.attrs.update(requested_start_ns=0, requested_end_ns=seconds * 1_000_000_000 + 1)
    return frame


def test_full_warmup_and_selected_clock_are_explicit():
    q = raw_quotes()
    rv = regimes.realized_vol(q, window=3)
    assert rv.rv.iloc[:3].isna().all()
    assert rv.rv.iloc[3:].notna().all()
    shifted = q.copy()
    shifted["ts_event"] += 2_000_000_000
    event = regimes.realized_vol(shifted, window=3, clock="ts_event")
    assert event.rv.first_valid_index() > rv.rv.first_valid_index()
    assert event.attrs["clock"] == "ts_event"


def test_stale_or_flagged_intervals_are_not_forward_filled():
    q = raw_quotes()
    q = q.loc[~q.ts_recv.between(1_500_000_000, 2_500_000_000)].copy()
    rv = regimes.realized_vol(q, window=2)
    assert rv.loc[rv.t.eq(2_000_000_000), "n_venues"].iloc[0] == 0
    assert rv.loc[rv.t.eq(3_000_000_000), "n_continuous_venues"].iloc[0] == 0
    flagged = raw_quotes()
    flagged.loc[flagged.ts_recv.eq(2_500_000_000), "flags"] = 132
    out = regimes.realized_vol(flagged, window=2)
    assert out.loc[out.t.eq(3_000_000_000), "n_continuous_venues"].iloc[0] == 0


def test_size_only_updates_refresh_age_and_compression_is_not_silently_used():
    q = raw_quotes()
    q["bid"], q["ask"] = 100., 100.02
    rv = regimes.realized_vol(q, window=2)
    assert (rv.n_venues == 1).all()
    compressed = load.bbo_changes(q)
    with pytest.raises(ValueError, match="uncompressed"):
        regimes.realized_vol(compressed, window=2)


def test_conflicting_timestamp_is_a_barrier_not_last_row_wins():
    q = raw_quotes()
    conflict = q.loc[q.ts_recv.eq(2_500_000_000)].copy()
    conflict["bid"] += .01
    q = pd.concat([q, conflict], ignore_index=True).sort_values("ts_recv", kind="stable")
    rv = regimes.realized_vol(q, window=2)
    assert rv.loc[rv.t.eq(3_000_000_000), "n_continuous_venues"].iloc[0] == 0


def test_no_key_still_stops_before_any_api_call(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    monkeypatch.setattr(pull, "DATA", tmp_path / "raw")
    monkeypatch.setattr(db, "Historical", lambda *a, **kw: pytest.fail("API client created without key"))
    assert pull.main(["--cost-only", "--datasets", "XNAS.ITCH"]) == 2
    assert not (tmp_path / "raw").exists()
