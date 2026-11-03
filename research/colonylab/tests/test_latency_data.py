"""Quality barriers and source identities for native event-time latency inputs."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from colonylab.latency_data import _clock_integers, _metadata_check, load_study, normalize_frame
from colonylab.network_wide_data import requests_for


BASE = pd.Timestamp("2026-08-03T14:00:00Z").value


def request(end=10_000):
    return {"dataset": "XNAS.ITCH", "schema": "mbp-1", "symbols": ["AAPL"], "stype_in": "raw_symbol",
            "start": pd.Timestamp(BASE, tz="UTC").isoformat(),
            "end": pd.Timestamp(BASE + end, tz="UTC").isoformat(), "limit": 1000}


def frame(times, *, bids=None, asks=None, sizes=None, flags=None, event=None, symbols=None):
    n = len(times)
    return pd.DataFrame({
        "ts_recv": np.asarray(times, dtype=np.uint64) + np.uint64(BASE),
        "ts_event": np.asarray(event if event is not None else times, dtype=np.uint64) + np.uint64(BASE),
        "symbol": symbols if symbols is not None else ["AAPL"] * n,
        "bid_px_00": np.asarray(bids if bids is not None else [100_000_000_001] * n, dtype=np.int64),
        "ask_px_00": np.asarray(asks if asks is not None else [100_010_000_001] * n, dtype=np.int64),
        "bid_sz_00": sizes if sizes is not None else [100] * n,
        "ask_sz_00": [100] * n, "flags": flags if flags is not None else [128] * n,
    })


def get(f, **kwargs):
    series, quality = normalize_frame(f, request(), clocks=("ts_recv",), **kwargs)
    return series[("AAPL", "ts_recv")], quality[0]


def test_integer_prices_keep_nanodollar_differences():
    s, q = get(frame([100, 200], bids=[100_000_000_001, 100_000_000_002]))
    assert s.bid.tolist() == [100_000_000_001, 100_000_000_002]
    assert s.times_ns.dtype == np.int64
    assert q["valid_state_records"] == 2


def test_bad_nonfinal_state_and_recovery_keep_barriers():
    s, q = get(frame([100, 200, 300, 400], flags=[128, 4, 0, 128]))
    assert (s.times_ns - BASE).tolist() == [100, 200, 400]
    assert s.valid.tolist() == [True, False, True]
    assert s.segments.tolist() == [0, 1, 1]
    assert q["intermediate_records_omitted"] == 1
    assert q["maybe_bad_book_records"] == 1


def test_duplicate_conflicting_size_is_ambiguous_and_cannot_bridge():
    s, q = get(frame([100, 200, 200, 300], sizes=[100, 100, 200, 100]))
    assert s.valid.tolist() == [True, False, True]
    assert q["ambiguous_timestamp_groups"] == 1
    assert q["duplicate_timestamp_records"] == 1


def test_identical_ties_collapse_without_quality_loss():
    s, q = get(frame([100, 100, 200]))
    assert q["ambiguous_timestamp_groups"] == 0
    assert q["duplicate_timestamp_records"] == 1
    assert s.valid.all()


@pytest.mark.parametrize("flag", [4, 8, 32])
def test_existing_native_quality_flags_are_barriers(flag):
    s, _ = get(frame([100, 200, 300], flags=[128, flag | 128, 128]))
    assert s.valid.tolist() == [True, False, True]


def test_crossed_locked_and_missing_quotes_are_invalid():
    s, _ = get(frame([100, 200, 300, 400], bids=[101, 100, 0, 100], asks=[100, 100, 101, 101]))
    assert s.valid.tolist() == [False, False, False, True]


def test_clock_reversal_invalidates_entire_reordered_interval():
    f = frame([100, 200, 300, 400, 500], event=[100, 300, 200, 250, 500],
              bids=[100, 101, 102, 103, 104], asks=[110] * 5)
    result, quality = normalize_frame(f, request())
    s = result[("AAPL", "ts_event")]
    assert (s.times_ns - BASE).tolist() == [100, 200, 250, 300]
    assert s.valid.tolist() == [True, False, False, False]
    q = next(x for x in quality if x["clock"] == "ts_event")
    assert q["timestamp_reversals"] == 1
    assert q["clock_reversal_barrier_records"] == 3


def test_stale_boundary_and_valid_recovery_are_retained():
    s, q = get(frame([100, 2000], bids=[100, 101], asks=[110, 110]), max_age_ms=0.001)
    assert (s.times_ns - BASE).tolist() == [100, 1100, 2000, 3000]
    assert s.valid.tolist() == [True, False, True, False]
    assert s.segments.tolist() == [0, 1, 1, 2]
    assert q["stale_barriers_inserted"] == 2


def test_size_only_updates_refresh_age_before_price_compression():
    s, q = get(frame([100, 900, 1700], sizes=[100, 200, 300]), max_age_ms=0.001)
    assert (s.times_ns - BASE).tolist() == [100, 2700]
    assert s.valid.tolist() == [True, False]
    assert q["stale_barriers_inserted"] == 1


def test_exchange_clock_end_is_conservative_but_receive_end_is_request_end():
    result, quality = normalize_frame(frame([100, 200, 300], event=[90, 190, 290]), request())
    assert result[("AAPL", "ts_recv")].end_ns == BASE + 10_000
    assert result[("AAPL", "ts_event")].end_ns == BASE + 290
    assert next(q for q in quality if q["clock"] == "ts_event")["chosen_clock_end_ns"] == BASE + 290


def test_clock_order_is_checked_per_symbol_not_interleaved_stream():
    f = frame([100, 200, 300, 400], symbols=["AAPL", "MSFT", "AAPL", "MSFT"], event=[90, 80, 290, 280])
    req = request()
    req["symbols"] = ["AAPL", "MSFT"]
    _, quality = normalize_frame(f, req)
    assert all(row["timestamp_reversals"] == 0 for row in quality)


def test_undefined_uint64_timestamp_does_not_wrap_to_negative():
    values, good = _clock_integers(np.array([BASE, np.iinfo(np.uint64).max], dtype=np.uint64))
    assert values.tolist() == [BASE, 0]
    assert good.tolist() == [True, False]


def test_unplaceable_exchange_timestamp_creates_neighbor_barriers():
    f = frame([100, 200, 300, 400, 500])
    f.loc[2, "ts_event"] = np.iinfo(np.uint64).max
    result, quality = normalize_frame(f, request())
    s = result[("AAPL", "ts_event")]
    assert s.valid.tolist() == [True, False, False]
    assert next(row for row in quality if row["clock"] == "ts_event")["invalid_timestamp_records"] == 1


def test_receive_bounds_are_checked_before_normalization():
    with pytest.raises(ValueError, match="half-open"):
        get(frame([100, 10_000]))


def test_metadata_must_match_exact_request():
    req = request()
    metadata = SimpleNamespace(schema="mbp-1", dataset="XNAS.ITCH", start=BASE, end=BASE + 10_000,
                               limit=1000, stype_in="raw_symbol", symbols=["AAPL"], partial=[], not_found=[])
    _metadata_check(SimpleNamespace(metadata=metadata), req)
    metadata.end += 1
    with pytest.raises(ValueError, match="interval"):
        _metadata_check(SimpleNamespace(metadata=metadata), req)


@pytest.fixture
def source_study(tmp_path, monkeypatch):
    import databento as db

    config = {
        "name": "latency-test", "datasets": ["XNAS.ITCH"], "symbols": ["AAPL"], "schemas": ["mbp-1"],
        "training_days": ["2026-08-03"], "held_out_days": ["2026-08-04"],
        "start_time_utc": "14:00:00", "minutes": 1, "workers": 1,
        "record_limit_per_request": 1000, "max_billable_bytes_per_request": 100000,
        "billing_assumed_included": True, "data_dir": "data/test", "out_dir": "out/test",
    }
    config_path = tmp_path / "test.toml"
    config_path.write_text("\n".join(f"{key} = {json.dumps(value).lower() if isinstance(value, bool) else json.dumps(value)}"
                                     for key, value in config.items()), encoding="utf-8")
    entries, metadata_by_path, hashes = [], {}, []
    for index, req in enumerate(requests_for(config)):
        path = tmp_path / config["data_dir"] / str(index) / "sample.dbn.zst"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"raw-dbn-example")
        entry = {"request": req, "complete": True, "record_limit_reached": False,
                 "rows": 2, "file_size_bytes": path.stat().st_size, "data_path": str(path)}
        entries.append(entry)
        (path.parent / "fetch.json").write_text(json.dumps(entry), encoding="utf-8")
        (path.parent / "stream_complete.json").write_text(json.dumps({"request": req}), encoding="utf-8")
        metadata_by_path[str(path)] = SimpleNamespace(
            schema="mbp-1", dataset=req["dataset"], start=pd.Timestamp(req["start"]).value,
            end=pd.Timestamp(req["end"]).value, limit=req["limit"], stype_in="raw_symbol",
            symbols=req["symbols"], partial=[], not_found=[])
        hashes.append({"path": str(path.relative_to(tmp_path)),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest_path = tmp_path / config["out_dir"] / "data_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest = {"complete": True, "configuration": config,
                "configuration_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                "requests": entries}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "research-archive-manifest.json").write_text(json.dumps({"files": hashes}), encoding="utf-8")
    monkeypatch.setattr(db.DBNStore, "from_file", lambda path: SimpleNamespace(metadata=metadata_by_path[str(path)]))
    return tmp_path, config_path, manifest_path, manifest


def test_study_checks_full_grid_receipts_metadata_and_original_hash(source_study):
    root, config_path, _, manifest = source_study
    _, checked = load_study(config_path, root)
    assert len(checked) == 2
    assert all(row["source_hash_verified_against"] == "research_archive_manifest" for row in checked)
    assert checked[0]["content_sha256"] == hashlib.sha256(b"raw-dbn-example").hexdigest()


def test_changed_bytes_of_same_length_fail_original_source_hash(source_study):
    root, config_path, _, manifest = source_study
    from pathlib import Path
    Path(manifest["requests"][0]["data_path"]).write_bytes(b"RAW-dbn-example")
    with pytest.raises(ValueError, match="source hash has changed"):
        load_study(config_path, root)


def test_missing_venue_date_fails_before_source_decode(source_study):
    root, config_path, manifest_path, manifest = source_study
    manifest["requests"].pop()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="request grid"):
        load_study(config_path, root)


def test_wrong_stream_marker_is_not_reused_as_complete(source_study):
    root, config_path, _, manifest = source_study
    from pathlib import Path
    entry = manifest["requests"][0]
    marker = Path(entry["data_path"]).parent / "stream_complete.json"
    marker.write_text(json.dumps({"request": dict(entry["request"], symbols=["MSFT"])}), encoding="utf-8")
    with pytest.raises(ValueError, match="stream marker"):
        load_study(config_path, root)


def test_capped_manifest_source_is_rejected(source_study):
    root, config_path, manifest_path, manifest = source_study
    manifest["requests"][0]["record_limit_reached"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete or capped"):
        load_study(config_path, root)
