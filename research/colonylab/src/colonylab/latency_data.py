"""Offline, source-bound native MBP-1 inputs for reaction-latency analysis.

The loader deliberately does not use the network's time grids or its midpoint
caches: neither preserves the best-bid/best-ask event times needed here.  It
retains the existing quote-quality barriers while keeping integer DBN prices.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .network_wide_data import read_wide_config, requests_for, validate_config

if TYPE_CHECKING:
    from .latency import QuoteSeries

F_LAST, F_SNAPSHOT, F_BAD_TS_RECV, F_MAYBE_BAD_BOOK = 128, 32, 8, 4
UNDEF_PRICE = np.iinfo(np.int64).max
INT64_MAX = np.iinfo(np.int64).max


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signature(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _metadata_check(store, request: dict) -> None:
    metadata = store.metadata
    _require(str(metadata.schema) == "mbp-1", "Latency requires native MBP-1 records.")
    _require(str(metadata.dataset) == request["dataset"], "DBN dataset differs from the exact request.")
    _require(int(metadata.start) == pd.Timestamp(request["start"]).value and
             metadata.end is not None and int(metadata.end) == pd.Timestamp(request["end"]).value,
             "DBN metadata interval differs from the exact request.")
    _require(int(metadata.limit or 0) == request["limit"], "DBN record limit differs from the exact request.")
    _require(str(metadata.stype_in) == "raw_symbol" and set(metadata.symbols) == set(request["symbols"]),
             "DBN symbol metadata differs from the exact request.")
    _require(not metadata.partial and not metadata.not_found,
             "Partial or unresolved DBN symbols cannot enter latency analysis.")


def _expected_hashes(root: Path, study_out: Path) -> dict[str, tuple[str, str]]:
    expected = {}
    archive_path = root / "research-archive-manifest.json"
    if archive_path.is_file():
        archive = json.loads(archive_path.read_text(encoding="utf-8"))
        for row in archive.get("files", []):
            if str(row.get("path", "")).endswith("sample.dbn.zst"):
                expected[str((root / row["path"]).resolve())] = (row["sha256"], "research_archive_manifest")
    provenance_path = study_out / "analysis_provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        for row in provenance.get("sources", []):
            source = str(Path(row["data_path"]).resolve())
            if source in expected:
                _require(expected[source][0] == row["content_sha256"],
                         "Saved study provenance disagrees with the research archive hash.")
            expected[source] = (row["content_sha256"], "study_provenance_and_archive" if source in expected
                                else "study_provenance")
    return expected


def load_study(config_path: str | Path, root: str | Path) -> tuple[dict, list[dict]]:
    """Validate an entire saved request grid and freshly hash every source.

    No API clients or credentials are used.  The configuration's historical
    billing preferences do not alter its acquisition identity.
    """
    import databento as db

    root = Path(root).resolve()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = read_wide_config(config_path)
    study_out = root / config["out_dir"]
    manifest_path = study_out / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("complete") is True, "Latency requires a complete saved acquisition manifest.")
    stored = validate_config(manifest["configuration"])
    _require(manifest.get("configuration_sha256") == _signature(stored),
             "Acquisition manifest configuration hash is inconsistent.")
    ignore = {"billing_assumed_included", "max_cost_usd"}
    _require({k: v for k, v in stored.items() if k not in ignore} ==
             {k: v for k, v in config.items() if k not in ignore},
             "Analysis configuration differs from the acquired study.")
    entries = manifest["requests"]
    expected = {_signature(request) for request in requests_for(config)}
    actual = [_signature(entry["request"]) for entry in entries]
    _require(len(actual) == len(set(actual)) and set(actual) == expected,
             "Saved request grid has missing, duplicate or unexpected venue/date requests.")
    known_hashes = _expected_hashes(root, study_out)
    checked = []
    for entry in sorted(entries, key=lambda row: (row["request"]["start"], row["request"]["dataset"])):
        request = entry["request"]
        _require(entry.get("complete") is True and not entry.get("record_limit_reached"),
                 "Incomplete or capped DBN sources cannot enter latency analysis.")
        _require(0 < entry.get("rows", 0) < request["limit"], "Invalid or capped saved record count.")
        path = Path(entry["data_path"])
        if not path.is_absolute():
            path = root / path
        # Restored snapshots may retain the old checkout's absolute path.
        if not path.is_file():
            from .network_data import cache_directory
            path = cache_directory(root / config["data_dir"], request) / "sample.dbn.zst"
        _require(path.is_file(), f"Saved DBN source is missing: {path}")
        directory = path.parent
        receipt = json.loads((directory / "fetch.json").read_text(encoding="utf-8"))
        marker = json.loads((directory / "stream_complete.json").read_text(encoding="utf-8"))
        _require(receipt.get("request") == request and marker.get("request") == request,
                 "Source receipt/stream marker differs from the exact saved request.")
        _require(receipt.get("complete") is True and not receipt.get("record_limit_reached") and
                 receipt.get("rows") == entry["rows"], "Receipt is incomplete, capped or has a different record count.")
        size = path.stat().st_size
        _require(size == entry["file_size_bytes"] == receipt["file_size_bytes"],
                 "Raw DBN size differs from its completed receipt.")
        _metadata_check(db.DBNStore.from_file(path), request)
        digest = file_sha256(path)
        prior = known_hashes.get(str(path.resolve()))
        # The old absolute path can still key original provenance after restore.
        if prior is None:
            prior = known_hashes.get(str(Path(entry["data_path"]).resolve()))
        if prior is not None:
            _require(digest == prior[0], f"Saved raw source hash has changed: {path}")
        checked.append(dict(entry, data_path=str(path.resolve()), content_sha256=digest,
                            source_hash_verified_against=prior[1] if prior else "new_content_identity",
                            acquisition_manifest=str(manifest_path.resolve())))
    return config, checked


def _clock_integers(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mask uint64 DBN undefined timestamps before a potentially wrapping cast."""
    raw = np.asarray(values)
    if raw.dtype.kind in "ui":
        good = (raw >= 0) & (raw <= INT64_MAX)
        result = np.zeros(len(raw), dtype=np.int64)
        result[good] = raw[good].astype(np.int64)
        return result, good
    numeric = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(numeric) & (numeric >= 0) & (numeric <= INT64_MAX) & (numeric == np.floor(numeric))
    result = np.zeros(len(raw), dtype=np.int64)
    result[good] = numeric[good].astype(np.int64)
    return result, good


def _normalize_symbol(frame: pd.DataFrame, request: dict, symbol: str, clock: str,
                      max_age_ns: int) -> tuple["QuoteSeries", dict]:
    from .latency import QuoteSeries

    start = int(pd.Timestamp(request["start"]).value)
    request_end = int(pd.Timestamp(request["end"]).value)
    time, timestamp_ok = _clock_integers(frame[clock].to_numpy())
    flags = frame["flags"].to_numpy(dtype=np.uint64)
    bad_flags = (flags & (F_SNAPSHOT | F_BAD_TS_RECV | F_MAYBE_BAD_BOOK)) != 0
    last = (flags & F_LAST) != 0
    included = last | bad_flags | ~timestamp_ok
    bid = frame["bid_px_00"].to_numpy(dtype=np.int64)
    ask = frame["ask_px_00"].to_numpy(dtype=np.int64)
    bs = frame["bid_sz_00"].to_numpy(dtype=np.int64)
    az = frame["ask_sz_00"].to_numpy(dtype=np.int64)
    valid = (bid > 0) & (ask > bid) & (ask != UNDEF_PRICE) & (bs > 0) & (az > 0) & ~bad_flags & timestamp_ok
    valid &= last
    # An unplaceable clock record makes its adjacent observed states barriers;
    # silently dropping it would incorrectly preserve an uninterrupted episode.
    bad_time_positions = np.flatnonzero(included & ~timestamp_ok)
    for offset in (-1, 1):
        adjacent = bad_time_positions + offset
        adjacent = adjacent[(adjacent >= 0) & (adjacent < len(valid))]
        valid[adjacent] = False
        included[adjacent] = True
    observed = included & timestamp_ok
    observed_times = time[observed]
    prior_max = np.maximum.accumulate(observed_times) if len(observed_times) else observed_times
    reversed_records = np.r_[False, observed_times[1:] < prior_max[:-1]] if len(observed_times) else np.zeros(0, bool)
    lo = observed_times[reversed_records]
    hi = np.r_[0, prior_max[:-1]][reversed_records]
    actual_reversals = int((np.diff(observed_times) < 0).sum())
    end = min(request_end, int(observed_times.max())) if clock == "ts_event" and len(observed_times) else request_end
    end = max(start, end)
    in_bounds = (time >= start) & (time < end)
    keep = observed & in_bounds
    t, b, a, bv, av, v = (array[keep] for array in (time, bid, ask, bs, az, valid))
    order = np.argsort(t, kind="stable")
    t, b, a, bv, av, v = (array[order] for array in (t, b, a, bv, av, v))
    reversal_mask = np.zeros(len(t), dtype=bool)
    if len(lo) and len(t):
        changes = np.zeros(len(t) + 1, dtype=np.int64)
        np.add.at(changes, np.searchsorted(t, lo, side="left"), 1)
        np.add.at(changes, np.searchsorted(t, hi, side="right"), -1)
        reversal_mask = np.cumsum(changes[:-1]) > 0
        v &= ~reversal_mask
    groups = np.r_[0, np.flatnonzero(np.diff(t) != 0) + 1] if len(t) else np.array([], dtype=np.int64)
    ambiguous = 0
    duplicates = len(t) - len(groups)
    if len(t):
        lengths = np.diff(np.r_[groups, len(t)])
        first = np.repeat(groups, lengths)
        conflict = (b != b[first]) | (a != a[first]) | (bv != bv[first]) | (av != av[first])
        conflicts = np.logical_or.reduceat(conflict, groups)
        ambiguous = int(conflicts.sum())
        group_bad = np.logical_or.reduceat(~v | conflict, groups)
        t, b, a = t[groups], b[groups], a[groups]
        v = ~group_bad
    # Valid size-only records refresh observed-state age before price compression.
    expiry = t + max_age_ns
    following = np.r_[t[1:], end] if len(t) else np.array([], dtype=np.int64)
    expires = v & (expiry < following) & (expiry < end)
    expiry_times = expiry[expires]
    if len(expiry_times):
        t = np.r_[t, expiry_times]
        b = np.r_[b, np.zeros(len(expiry_times), dtype=np.int64)]
        a = np.r_[a, np.zeros(len(expiry_times), dtype=np.int64)]
        v = np.r_[v, np.zeros(len(expiry_times), dtype=bool)]
        sort = np.argsort(t, kind="stable")
        t, b, a, v = t[sort], b[sort], a[sort], v[sort]
    b = np.where(v, b, 0).astype(np.int64)
    a = np.where(v, a, 0).astype(np.int64)
    segments = np.cumsum(~v, dtype=np.int64)
    states_before_compression = len(t)
    if len(t):
        retain = np.r_[True, (~v[1:]) | (~v[:-1]) | (segments[1:] != segments[:-1]) |
                        (b[1:] != b[:-1]) | (a[1:] != a[:-1])]
        t, b, a, v, segments = (array[retain] for array in (t, b, a, v, segments))
    series = QuoteSeries(times_ns=t, bid=b, ask=a, valid=v, segments=segments,
                         start_ns=start, end_ns=end)
    quality = {
        "dataset": request["dataset"], "date": request["start"][:10], "symbol": symbol, "clock": clock,
        "raw_records": len(frame), "snapshot_records": int(((flags & F_SNAPSHOT) != 0).sum()),
        "bad_receive_time_records": int(((flags & F_BAD_TS_RECV) != 0).sum()),
        "maybe_bad_book_records": int(((flags & F_MAYBE_BAD_BOOK) != 0).sum()),
        "intermediate_records_omitted": int((~included).sum()), "invalid_timestamp_records": int((~timestamp_ok).sum()),
        "timestamp_reversals": actual_reversals, "records_below_prior_clock_max": int(reversed_records.sum()),
        "clock_reversal_barrier_records": int(reversal_mask.sum()),
        "removed_outside_request": int((observed & ~((time >= start) & (time < request_end))).sum()),
        "exchange_boundary_records_removed": int((observed & (time >= end) & (time < request_end)).sum()),
        "duplicate_timestamp_records": duplicates, "ambiguous_timestamp_groups": ambiguous,
        "stale_barriers_inserted": len(expiry_times), "max_age_ms": max_age_ns / 1e6,
        "state_records_before_price_compression": states_before_compression, "state_records": len(t),
        "invalid_state_records": int((~v).sum()), "valid_state_records": int(v.sum()),
        "requested_start_ns": start, "requested_end_ns": request_end, "chosen_clock_end_ns": end,
    }
    return series, quality


def normalize_frame(frame: pd.DataFrame, request: dict, *, clocks=("ts_recv", "ts_event"),
                    max_age_ms: float = 1000) -> tuple[dict[tuple[str, str], "QuoteSeries"], list[dict]]:
    """Normalize one exact raw request without changing any source files."""
    required = {"ts_recv", "ts_event", "symbol", "flags", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"}
    _require(required <= set(frame.columns), "Native MBP-1 inputs lack required timestamps, flags or quote fields.")
    _require(np.isfinite(max_age_ms) and max_age_ms > 0, "Quote maximum age must be positive and finite.")
    _require(set(clocks) <= {"ts_recv", "ts_event"} and len(clocks) == len(set(clocks)), "Unknown or repeated latency clock.")
    frame = frame.copy()
    frame["symbol"] = frame["symbol"].map(lambda x: x.decode() if isinstance(x, bytes) else str(x))
    _require(set(frame.symbol.unique()) == set(request["symbols"]), "Raw DBN symbol set differs from the requested set.")
    time, ok = _clock_integers(frame.ts_recv.to_numpy())
    start, end = pd.Timestamp(request["start"]).value, pd.Timestamp(request["end"]).value
    _require(ok.all() and (time >= start).all() and (time < end).all(),
             "Raw receive timestamps fall outside their exact half-open request interval.")
    result, quality = {}, []
    for symbol in request["symbols"]:
        selected = frame.loc[frame.symbol.eq(symbol)].reset_index(drop=True)
        for clock in clocks:
            series, facts = _normalize_symbol(selected, request, symbol, clock, int(max_age_ms * 1e6))
            result[(symbol, clock)] = series
            quality.append(facts)
    return result, quality


def load_entry(entry: dict, clocks=("ts_recv", "ts_event"), max_age_ms: float = 1000):
    """Decode an integrity-checked source and retain native integer quote states."""
    import databento as db

    path = Path(entry["data_path"])
    _require(path.stat().st_size == entry["file_size_bytes"], "Raw source size changed after manifest validation.")
    if entry.get("content_sha256"):
        _require(file_sha256(path) == entry["content_sha256"],
                 "Raw source hash changed after manifest validation.")
    store = db.DBNStore.from_file(path)
    _metadata_check(store, entry["request"])
    frame = store.to_df(price_type="fixed", pretty_ts=False, map_symbols=True).reset_index()
    _require(0 < len(frame) == entry["rows"] < entry["request"]["limit"],
             "Decoded raw DBN count differs from its completed receipt or reaches its cap.")
    return normalize_frame(frame, entry["request"], clocks=clocks, max_age_ms=max_age_ms)
