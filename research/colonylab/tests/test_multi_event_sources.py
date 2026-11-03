"""Exact fragment partitions and native quote continuity without new pulls."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import databento as db
import numpy as np
import pandas as pd
import pytest

from colonylab import multi_event_sources as sources


@pytest.fixture
def sample(tmp_path, monkeypatch):
    start = pd.Timestamp("2025-07-31T20:00:00Z").value
    seam, end = start + 5_000_000_000, start + 10_000_000_000
    group = {"start": pd.Timestamp(start, tz="UTC").isoformat(), "end": pd.Timestamp(end, tz="UTC").isoformat()}
    protocol = {"datasets": ["XNAS.ITCH", "ARCX.PILLAR"], "quote_age_ms": [1000, 6000], "clocks": ["ts_recv", "ts_event"]}
    entries, stores = [], {}
    def frame_for(first, last, n, rows=2):
        receives = np.array([first + 1_000_000, last - 1_000_000], np.int64)[:rows]
        bid = 200_000_000_000 + np.arange(len(receives), dtype=np.int64) * 10_000_000 + n * 20_000_000
        return pd.DataFrame({"ts_recv": receives, "ts_event": receives - 500_000,
            "symbol": np.full(len(receives), "AAPL", dtype=object), "flags": np.full(len(receives), 128, dtype=np.uint8),
            "bid_px_00": bid, "ask_px_00": bid + 10_000_001,
            "bid_sz_00": np.full(len(receives), 100, dtype=np.uint32),
            "ask_sz_00": np.full(len(receives), 200, dtype=np.uint32)}).set_index("ts_recv")
    for dataset in protocol["datasets"]:
        for n, (first, last) in enumerate([(start, seam), (seam, end)]):
            request = {"dataset": dataset, "schema": "mbp-1", "symbols": ["AAPL"], "stype_in": "raw_symbol",
                "start": pd.Timestamp(first, tz="UTC").isoformat(), "end": pd.Timestamp(last, tz="UTC").isoformat(), "limit": 100}
            raw = tmp_path / (dataset + str(n) + ".dbn.zst")
            raw.write_bytes((dataset + str(n)).encode())
            metadata = SimpleNamespace(dataset=dataset, schema="mbp-1", start=first, end=last, limit=100,
                                       stype_in="raw_symbol", symbols=["AAPL"], partial=[], not_found=[])
            store = SimpleNamespace(metadata=metadata, frame=frame_for(first, last, n))
            def to_df(store=store, **kwargs):
                assert kwargs == {"price_type": "fixed", "pretty_ts": False, "map_symbols": True}
                return store.frame.copy()
            store.to_df = to_df
            stores[str(raw)] = store
            entries.append({"request": request, "complete": True, "analyzable": True, "status": "complete",
                "rows": len(store.frame), "data_path": str(raw), "file_size_bytes": raw.stat().st_size,
                "content_sha256": sources.file_sha256(raw)})
    monkeypatch.setattr(db.DBNStore, "from_file", lambda raw: stores[str(raw)])
    return SimpleNamespace(start=start, seam=seam, end=end, group=group, protocol=protocol,
                           entries=entries, stores=stores, frame_for=frame_for)


def decode(sample):
    return sources.decode_group(sample.entries, sample.group, sample.protocol)


def make_empty(sample, index):
    entry = sample.entries[index]
    store = sample.stores[entry["data_path"]]
    store.frame = store.frame.iloc[:0].copy()
    entry.update(rows=0, analyzable=False)


def test_unordered_fragments_are_verified_and_normalized_as_one_integer_union(sample):
    sample.entries.reverse()
    paths, facts = decode(sample)
    assert len(paths) == len(facts) == 2 * 2 * 2
    for dataset in sample.protocol["datasets"]:
        path = paths[(dataset, 6000, "ts_recv")]
        assert path.start_ns == sample.start and path.end_ns == sample.end
        assert path.price2.dtype == np.int64 and path.price2[path.valid][0] == 400_010_000_001
        assert (path.price2[path.valid] % 2 == 1).all()
    for row in facts:
        assert row["raw_records"] == 4 and row["source_available"] and row["source_analyzable"]
        assert row["source_chunk_count"] == len(row["source_hashes"]) == 2
        assert [piece["start_ns"] for piece in row["source_fragments"]] == [sample.start, sample.seam]
        assert row["virtual_union_request"]["limit"] == 200 and row["virtual_union_has_dbn_metadata"] is False


def test_exchange_events_before_receive_fragment_seam_survive_union_normalization(sample):
    entry = sample.entries[1]
    store = sample.stores[entry["data_path"]]
    frame = store.frame.reset_index()
    frame.loc[0, "ts_recv"] = sample.seam + 100
    frame.loc[0, "ts_event"] = sample.seam - 400
    store.frame = frame.set_index("ts_recv")
    paths, _ = decode(sample)
    assert sample.seam - 400 in paths[("XNAS.ITCH", 6000, "ts_event")].times_ns
    assert sample.seam + 100 in paths[("XNAS.ITCH", 6000, "ts_recv")].times_ns


@pytest.mark.parametrize("corruption", ["gap", "overlap", "duplicate", "outside", "missing_venue"])
def test_bad_partition_is_rejected_before_decoding_any_raw_source(sample, monkeypatch, corruption):
    if corruption == "gap":
        sample.entries[1]["request"]["start"] = pd.Timestamp(sample.seam + 1, tz="UTC").isoformat()
    elif corruption == "overlap":
        sample.entries[1]["request"]["start"] = pd.Timestamp(sample.seam - 1, tz="UTC").isoformat()
    elif corruption == "duplicate":
        sample.entries.append(sample.entries[0])
    elif corruption == "outside":
        sample.entries[0]["request"]["start"] = pd.Timestamp(sample.start - 1, tz="UTC").isoformat()
    else:
        sample.entries = sample.entries[:2]
    monkeypatch.setattr(db.DBNStore, "from_file", lambda _: pytest.fail("Invalid registry must be rejected before raw decoding"))
    with pytest.raises(ValueError, match="partition|outside|Missing source fragment"):
        decode(sample)


@pytest.mark.parametrize("status", ["unavailable", "failed"])
def test_unknown_fragment_quarantines_entire_venue_without_bridging(sample, status):
    sample.entries[1].update(complete=False, analyzable=False, status=status, availability_reason="known_provider_gap")
    paths, facts = decode(sample)
    for age in sample.protocol["quote_age_ms"]:
        for clock in sample.protocol["clocks"]:
            assert len(paths[("XNAS.ITCH", age, clock)].times_ns) == 0
            assert len(paths[("ARCX.PILLAR", age, clock)].times_ns) > 0
    xnas = [row for row in facts if row["dataset"] == "XNAS.ITCH"]
    assert all(not row["source_available"] and not row["source_analyzable"] and "known_provider_gap" in row["reason"] for row in xnas)


def test_complete_empty_initial_fragment_does_not_future_fill_an_initial_quote(sample):
    make_empty(sample, 0)
    paths, facts = decode(sample)
    path = paths[("XNAS.ITCH", 6000, "ts_recv")]
    assert path.times_ns[0] > sample.seam and not np.any(path.times_ns <= sample.seam)
    assert all(row["source_available"] and row["source_analyzable"] and row["raw_records"] == 2
               for row in facts if row["dataset"] == "XNAS.ITCH")


def test_complete_empty_tail_keeps_natural_age_barrier_instead_of_resetting_or_filling(sample):
    make_empty(sample, 1)
    paths, _ = decode(sample)
    path = paths[("XNAS.ITCH", 1000, "ts_recv")]
    last_observation = sample.seam - 1_000_000
    expiry = last_observation + 1_000_000_000
    assert expiry in path.times_ns and not path.valid[np.searchsorted(path.times_ns, expiry)]
    assert path.end_ns == sample.end
    longer = paths[("XNAS.ITCH", 6000, "ts_recv")]
    assert longer.valid[-1]  # Complete no-update interval can remain fresh under the stated policy.


def test_all_complete_empty_fragments_produce_no_observed_quote_path(sample):
    make_empty(sample, 0)
    make_empty(sample, 1)
    paths, facts = decode(sample)
    assert all(len(path.times_ns) == 0 for (dataset, _, _), path in paths.items() if dataset == "XNAS.ITCH")
    assert all(row["source_available"] and not row["source_analyzable"] and row["raw_records"] == 0
               and row["reason"] == "all_source_fragments_empty_no_initial_quote_state"
               for row in facts if row["dataset"] == "XNAS.ITCH")


@pytest.mark.parametrize("corruption", ["bytes", "metadata", "row_count", "float_prices"])
def test_each_source_identity_and_integer_record_contract_is_independently_checked(sample, corruption):
    entry = sample.entries[1]
    store = sample.stores[entry["data_path"]]
    if corruption == "bytes":
        Path(entry["data_path"]).write_bytes(b"changed")
    elif corruption == "metadata":
        store.metadata.end += 1
    elif corruption == "row_count":
        entry["rows"] += 1
    else:
        store.frame["bid_px_00"] = store.frame["bid_px_00"].astype(float)
    with pytest.raises(ValueError, match="bytes changed|metadata interval|row count|must remain integer"):
        decode(sample)


@pytest.mark.parametrize("edge", ["before_start", "at_end"])
def test_raw_receive_rows_cannot_escape_their_own_fragment_even_inside_union(sample, edge):
    entry = sample.entries[1 if edge == "before_start" else 0]
    store = sample.stores[entry["data_path"]]
    frame = store.frame.reset_index()
    frame.loc[0, "ts_recv"] = sample.seam - 1 if edge == "before_start" else sample.seam
    store.frame = frame.set_index("ts_recv")
    with pytest.raises(ValueError, match="outside its exact interval"):
        decode(sample)
