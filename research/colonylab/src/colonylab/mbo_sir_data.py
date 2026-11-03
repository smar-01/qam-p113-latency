"""Exact-grid, resumable full-depth MBO acquisition for the SIRS study.

Historical access is assumed included at the user's instruction. No monetary
metadata is requested. Record/byte bounds prevent incomplete or oversized
windows, and the complete stream is retained before receipt validation.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import tomllib
import traceback
from typing import Callable
from uuid import uuid4

import numpy as np
import pandas as pd

from .config import safe_error
from .data import client
from .network_wide_data import NATIVE_DATASETS


CONFIG_KEYS = {
    "name", "datasets", "symbol", "market_dataset", "market_symbol", "training_days",
    "held_out_days", "download_start_utc", "session_start_utc", "session_end_utc",
    "validation_start_utc", "validation_minutes", "anchor_step_ms", "data_dir", "out_dir",
    "workers", "record_limit_mbo", "record_limit_aux", "max_bytes_mbo", "billing_assumed_included",
}
METADATA_WORKERS = 4
DOWNLOAD_MAX_ATTEMPTS = 3
BILLING_ASSUMPTION = "User reports all historical access is included; no cost metadata or monetary ceiling applied."


def _signature(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Retain the same durable source while bounded Windows locks expire."""
    for attempt in range(12):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(.1 * 2 ** attempt, 1.0))


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    _replace_with_retry(pending, path)


def _release_failed_stream_frames(exc: Exception) -> None:
    """Release SDK writers retained by completed transfer exception frames.

    SDK 0.88 closes its local output writer only after a successful transfer.
    On failure that writer remains in the traceback frame, preventing Windows
    from renaming its prefix. Clearing completed frames releases those local
    resources without forced collection or changing the original exception.
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        traceback.clear_frames(exc.__traceback__)
        exc = exc.__cause__ or exc.__context__


def validate_config(config: dict) -> dict:
    if set(config) != CONFIG_KEYS:
        raise ValueError("MBO SIRS configuration requires exactly: " + ", ".join(sorted(CONFIG_KEYS)))
    result = dict(config)
    if not isinstance(result["name"], str) or not result["name"].strip():
        raise ValueError("name must be a nonempty string.")
    if result["billing_assumed_included"] is not True:
        raise ValueError("This explicitly included-access study requires billing_assumed_included=true.")
    datasets = result["datasets"]
    if (not isinstance(datasets, list) or not 2 <= len(datasets) <= len(NATIVE_DATASETS)
            or len(set(datasets)) != len(datasets) or not set(datasets) <= NATIVE_DATASETS):
        raise ValueError("datasets must be two or more unique supported native MBO exchange feeds.")
    if result["market_dataset"] not in NATIVE_DATASETS:
        raise ValueError("market_dataset must be a supported native exchange feed.")
    for key in ("symbol", "market_symbol"):
        if not isinstance(result[key], str) or not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,15}", result[key]):
            raise ValueError("Use one explicit uppercase symbol per stream, never ALL_SYMBOLS.")
        if result[key] == "ALL_SYMBOLS":
            raise ValueError("ALL_SYMBOLS is disabled.")
    if result["symbol"] == result["market_symbol"]:
        raise ValueError("The market-control symbol must differ from the target symbol.")
    for key in ("training_days", "held_out_days"):
        values = result[key]
        if (not isinstance(values, list) or not 1 <= len(values) <= 60
                or any(not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) for value in values)
                or len(set(values)) != len(values) or values != sorted(values)):
            raise ValueError(f"{key} requires unique chronological explicit ISO dates.")
        for value in values:
            try:
                stamp = pd.Timestamp(value)
            except ValueError as exc:
                raise ValueError("Invalid study date.") from exc
            if stamp.weekday() > 4:
                raise ValueError("Study dates must be exchange weekdays.")
        result[key] = list(values)
    if max(result["training_days"]) >= min(result["held_out_days"]):
        raise ValueError("All training dates must precede all disjoint held-out dates.")
    for key in ("download_start_utc", "session_start_utc", "session_end_utc", "validation_start_utc"):
        value = result[key]
        if not isinstance(value, str) or not re.fullmatch(r"\d{2}:\d{2}:\d{2}", value):
            raise ValueError(f"{key} must be HH:MM:SS UTC.")
        try:
            pd.Timestamp("2026-06-01T" + value + "Z")
        except ValueError as exc:
            raise ValueError(f"Invalid {key}.") from exc
    if not result["download_start_utc"] <= result["session_start_utc"] < result["session_end_utc"]:
        raise ValueError("Download must start no later than the session, and end after it starts.")
    for key, low, high in (("workers", 1, 8), ("record_limit_mbo", 1, 50_000_000),
                          ("record_limit_aux", 1, 5_000_000), ("max_bytes_mbo", 1, 3_000_000_000),
                          ("validation_minutes", 1, 60), ("anchor_step_ms", 1, 1000)):
        if type(result[key]) is not int or not low <= result[key] <= high:
            raise ValueError(f"{key} must be an integer from {low} to {high}.")
    day = result["training_days"][0]
    validation_start = pd.Timestamp(f"{day}T{result['validation_start_utc']}Z")
    validation_end = validation_start + pd.Timedelta(minutes=result["validation_minutes"])
    session_start = pd.Timestamp(f"{day}T{result['session_start_utc']}Z")
    session_end = pd.Timestamp(f"{day}T{result['session_end_utc']}Z")
    if validation_start < session_start or validation_end > session_end:
        raise ValueError("The independent MBP10 validation window must be inside the trading session.")
    for key in ("data_dir", "out_dir"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"{key} must be an explicit directory.")
    return result


def read_config(path: Path) -> dict:
    return validate_config(tomllib.loads(Path(path).read_text(encoding="utf-8")))


def build_requests(config: dict) -> list[dict]:
    config = validate_config(config)
    requests = []
    for day in config["training_days"] + config["held_out_days"]:
        full_start = pd.Timestamp(f"{day}T{config['download_start_utc']}Z")
        end = pd.Timestamp(f"{day}T{config['session_end_utc']}Z")
        validation_start = pd.Timestamp(f"{day}T{config['validation_start_utc']}Z")
        for dataset in config["datasets"]:
            for schema, start, stop, limit in (
                ("mbo", full_start, end, config["record_limit_mbo"]),
                ("mbp-10", validation_start, validation_start + pd.Timedelta(minutes=config["validation_minutes"]), config["record_limit_aux"]),
            ):
                requests.append({"dataset": dataset, "schema": schema, "symbols": [config["symbol"]],
                                 "stype_in": "raw_symbol", "start": start.isoformat(), "end": stop.isoformat(), "limit": limit})
        requests.append({"dataset": config["market_dataset"], "schema": "bbo-1s", "symbols": [config["market_symbol"]],
                         "stype_in": "raw_symbol", "start": pd.Timestamp(f"{day}T{config['session_start_utc']}Z").isoformat(),
                         "end": end.isoformat(), "limit": config["record_limit_aux"]})
    return requests


def cache_directory(data_dir: Path, request: dict) -> Path:
    return Path(data_dir) / f"{request['start'][:10]}_{request['dataset']}_{request['schema']}_{_signature(request)[:16]}"


def _read_cached(data_dir: Path, request: dict) -> dict | None:
    directory = cache_directory(data_dir, request)
    receipt, marker, raw = directory / "fetch.json", directory / "stream_complete.json", directory / "sample.dbn.zst"
    if not receipt.exists():
        return None
    entry = json.loads(receipt.read_text(encoding="utf-8"))
    if entry.get("request") != request:
        raise ValueError("Exact request differs from the cached MBO SIRS receipt.")
    if entry.get("record_limit_reached") or entry.get("rows", 0) >= request["limit"]:
        raise ValueError("A capped DBN cannot be reused as a complete study window.")
    if not entry.get("complete"):
        return None
    if (not raw.is_file() or not marker.is_file()
            or json.loads(marker.read_text(encoding="utf-8")).get("request") != request
            or entry.get("file_size_bytes") != raw.stat().st_size or entry.get("rows", 0) <= 0):
        raise ValueError("Completed cache is missing its matching raw file, stream marker, or receipt size.")
    return dict(entry, cache_reused=True, data_path=str(raw.resolve()))


def _quote_one(request: dict, data_dir: Path, api_factory: Callable, progress: Callable) -> dict:
    cached = _read_cached(data_dir, request)
    if cached:
        return dict(cached, estimate_reused=True)
    if request["schema"] != "mbo":
        return {"request": request, "expected_records_without_limit": None,
                "estimated_billable_bytes_without_limit": None, "metadata_sizing_skipped": True,
                "cost_check_skipped": True, "cache_reused": False}
    full = {key: value for key, value in request.items() if key != "limit"}
    for attempt in range(3):
        try:
            api = api_factory()
            count = int(api.metadata.get_record_count(**full))
            size = int(api.metadata.get_billable_size(**full))
            if count < 0 or size < 0:
                raise ValueError("Invalid negative record/byte metadata.")
            return {"request": request, "expected_records_without_limit": count,
                    "estimated_billable_bytes_without_limit": size,
                    "metadata_count_may_overreport": True, "cost_check_skipped": True,
                    "metadata_sizing_skipped": False, "cache_reused": False}
        except Exception as exc:
            transient = any(token in safe_error(exc).lower() for token in
                            ("timeout", "connection", "502", "503", "504", "429", "temporar"))
            if not transient or attempt == 2:
                raise
            progress(f"Retrying metadata {request['start'][:10]} {request['dataset']} ({attempt + 2}/3).")
            time.sleep(1 + 2 * attempt)
    raise AssertionError("Unreachable metadata retry state.")


def _manifest_identity(config: dict, requests: list[dict], config_path: Path | None) -> dict:
    return {"created_utc": _now(), "configuration": config, "configuration_sha256": _signature(config),
            "configuration_path": str(Path(config_path).resolve()) if config_path else None,
            "expected_request_count": len(requests),
            "request_grid_sha256": _signature({"requests": requests}),
            "download_workers": config["workers"], "metadata_workers": METADATA_WORKERS,
            "billing_assumption": BILLING_ASSUMPTION, "cost_check_skipped": True,
            "estimated_cost_usd": None, "complete": False}


def plan_study(
    config: dict, config_path: Path | None = None, *, progress: Callable[[str], None] = print,
    api_factory: Callable | None = None,
) -> dict:
    """Size every full MBO request before retrieving any time series.

    Metadata progress is checkpointed by the exact frozen configuration. Auxiliary
    windows deliberately do not issue count, size or cost metadata requests.
    """
    config = validate_config(config)
    api_factory = api_factory or client
    requests = build_requests(config)
    data_dir, out_dir = Path(config["data_dir"]), Path(config["out_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "metadata_progress.json"
    quoted: dict[str, dict] = {}
    if progress_path.exists():
        stored = json.loads(progress_path.read_text(encoding="utf-8"))
        if stored.get("configuration_sha256") == _signature(config):
            expected = {_signature(request) for request in requests}
            for entry in stored.get("requests", []):
                token = _signature(entry["request"])
                if token not in expected or token in quoted:
                    raise ValueError("Metadata progress contains a duplicate or foreign request.")
                if entry["request"]["schema"] == "mbo":
                    for key in ("expected_records_without_limit", "estimated_billable_bytes_without_limit"):
                        if type(entry.get(key)) is not int or entry[key] < 0:
                            raise ValueError("Persisted MBO metadata has an invalid size estimate.")
                quoted[token] = entry
    # Refresh local completion independently of a previous metadata checkpoint.
    for request in requests:
        cached = _read_cached(data_dir, request)
        if cached:
            quoted[_signature(request)] = dict(cached, estimate_reused=True)
    pending = [request for request in requests if _signature(request) not in quoted]
    progress(f"Sizing {sum(request['schema'] == 'mbo' for request in requests)} full MBO windows; "
             f"{len(quoted)} exact requests already checkpointed; auxiliary sizing and all cost checks skipped.")
    failures = []
    state = _manifest_identity(config, requests, config_path)
    state.update(requests=list(quoted.values()), failures=failures)
    with ThreadPoolExecutor(max_workers=METADATA_WORKERS) as pool:
        jobs = {pool.submit(_quote_one, request, data_dir, api_factory, progress): request for request in pending}
        for future in as_completed(jobs):
            request = jobs[future]
            try:
                entry = future.result()
                quoted[_signature(request)] = entry
                if request["schema"] == "mbo":
                    progress(f"MBO size {request['start'][:10]} {request['dataset']}: "
                             f"{entry['expected_records_without_limit']:,} records; "
                             f"{entry['estimated_billable_bytes_without_limit'] / 1e6:.1f} MB raw estimate.")
            except Exception as exc:
                failures.append({"request": request, "error": safe_error(exc)})
            state["requests"] = list(quoted.values())
            _atomic_json(progress_path, state)
    if failures:
        raise ValueError(f"Free sizing failed for {len(failures)} requests; checkpoints preserved; no downloads performed.")
    entries = [quoted[_signature(request)] for request in requests]
    oversized = [entry for entry in entries if entry["request"]["schema"] == "mbo" and
                 (entry["expected_records_without_limit"] >= config["record_limit_mbo"]
                  or entry["estimated_billable_bytes_without_limit"] > config["max_bytes_mbo"])]
    estimated_total = sum(entry["estimated_billable_bytes_without_limit"] or 0 for entry in entries)
    remaining = sum((entry["estimated_billable_bytes_without_limit"] or 0) for entry in entries if not entry.get("cache_reused"))
    free = shutil.disk_usage(data_dir.resolve()).free
    reserve = 20_000_000_000
    plan = _manifest_identity(config, requests, config_path)
    plan.update(requests=entries, estimated_total_mbo_billable_bytes=estimated_total,
                estimated_remaining_mbo_billable_bytes=remaining, free_disk_bytes=free,
                feature_disk_reserve_bytes=reserve, disk_space_sufficient=free >= remaining + reserve,
                oversized_requests=[entry["request"] for entry in oversized],
                ready_to_download=not oversized and free >= remaining + reserve,
                sizing_note="Conservative raw record/byte estimates; successful uncapped SDK stream and full DBN parsing establish completion.")
    _atomic_json(out_dir / "download_plan.json", plan)
    if oversized:
        raise ValueError("At least one full MBO window exceeds its record/byte safety bound; exact dates preserved and no downloads performed.")
    if not plan["disk_space_sufficient"]:
        raise ValueError("Insufficient free disk for remaining raw estimates plus the 20 GB feature reserve; no downloads performed.")
    return plan


def _instrument_mapping(metadata, request: dict) -> tuple[set[int], dict]:
    symbol = request["symbols"][0]
    mappings = getattr(metadata, "mappings", {})
    intervals = mappings.get(symbol, [])
    identifiers = set()
    display = []
    day = request["start"][:10]
    for interval in intervals:
        start, end = str(interval["start_date"]), str(interval["end_date"])
        display.append({"start_date": start, "end_date": end, "symbol": str(interval["symbol"])})
        if start <= day < end:
            identifiers.add(int(interval["symbol"]))
    if not identifiers:
        raise ValueError("DBN metadata cannot resolve the requested symbol to instrument IDs on this date.")
    return identifiers, {symbol: display}


def _export_stream(store, entry: dict, directory: Path, *, recovered: bool = False,
                   write_receipt: bool = True) -> dict:
    """Fully parse DBN in bounded NumPy chunks; never materialize a DataFrame."""
    request = entry["request"]
    schema = str(getattr(store, "schema", request["schema"]))
    if schema != request["schema"]:
        raise ValueError("DBN schema differs from its exact request.")
    if str(getattr(store.metadata, "dataset", request["dataset"])) != request["dataset"]:
        raise ValueError("DBN dataset differs from its exact native-venue request.")
    identifiers, mappings = _instrument_mapping(store.metadata, request)
    total = 0
    actual_ids: set[int] = set()
    record_types: set[int] = set()
    publisher_ids: set[int] = set()
    receive_min = receive_max = event_min = event_max = None
    columns: tuple[str, ...] | None = None
    outside = 0
    start_ns, end_ns = pd.Timestamp(request["start"]).value, pd.Timestamp(request["end"]).value
    chunks = store.to_ndarray(count=1_000_000)
    for chunk in chunks:
        if chunk.size == 0:
            continue
        if chunk.ndim != 1 or not chunk.dtype.names:
            raise ValueError("DBN parser yielded an invalid structured record chunk.")
        if columns is None:
            columns = chunk.dtype.names
        elif columns != chunk.dtype.names:
            raise ValueError("DBN changed record layout within a historical stream.")
        if not {"instrument_id", "rtype", "ts_event"} <= set(chunk.dtype.names):
            raise ValueError("DBN records lack their identity or event timestamp header.")
        total += len(chunk)
        actual_ids.update(map(int, np.unique(chunk["instrument_id"])))
        record_types.update(map(int, np.unique(chunk["rtype"])))
        if "publisher_id" in chunk.dtype.names:
            publisher_ids.update(map(int, np.unique(chunk["publisher_id"])))
        if not actual_ids <= identifiers:
            raise ValueError("DBN contains records for an instrument outside the requested symbol mapping.")
        events = chunk["ts_event"].astype(np.uint64, copy=False)
        emin, emax = int(events.min()), int(events.max())
        event_min, event_max = (emin if event_min is None else min(event_min, emin),
                                emax if event_max is None else max(event_max, emax))
        if "ts_recv" in chunk.dtype.names:
            receives = chunk["ts_recv"].astype(np.uint64, copy=False)
            rmin, rmax = int(receives.min()), int(receives.max())
            receive_min, receive_max = (rmin if receive_min is None else min(receive_min, rmin),
                                        rmax if receive_max is None else max(receive_max, rmax))
        else:
            receives = events
        # Keep all raw records, including flagged snapshots or bad clocks. The
        # replay layer must treat invalid records as barriers. Bar data can
        # expose a stale last receive timestamp, so its event clock is audited.
        check_clock = events if request["schema"] == "bbo-1s" else receives
        clock_bad = (check_clock < start_ns) | (check_clock >= end_ns)
        if "flags" in chunk.dtype.names:
            clock_bad &= (chunk["flags"].astype(np.uint8) & (32 | 8)) == 0
        outside += int(np.count_nonzero(clock_bad))
    # SDK 0.88's NDArrayStreamIterator warns, drops trailing incomplete record
    # bytes, and sets _close_on_next=True. Treat that parser state as a hard
    # completeness failure. This explicit per-iterator test avoids changing
    # process-global warning filters while eight streams decode concurrently.
    if getattr(chunks, "_close_on_next", False):
        raise ValueError("DBN parser found trailing incomplete record bytes; raw file preserved as incomplete.")
    raw = directory / "sample.dbn.zst"
    capped = total >= request["limit"]
    expected = entry.get("expected_records_without_limit")
    result = dict(entry, request_sha256=_signature(request), rows=total,
                  complete=total > 0 and not capped, record_limit_reached=capped,
                  cache_reused=False, recovered_from_local_dbn=recovered,
                  data_path=str(raw.resolve()), file_size_bytes=raw.stat().st_size,
                  schema=schema, columns=list(columns or ()), instrument_ids=sorted(actual_ids),
                  requested_symbol_mapping=mappings, record_types=sorted(record_types),
                  publisher_ids=sorted(publisher_ids),
                  receive_timestamp_min_ns=receive_min, receive_timestamp_max_ns=receive_max,
                  event_timestamp_min_ns=event_min, event_timestamp_max_ns=event_max,
                  unflagged_records_outside_requested_clock_range=outside,
                  metadata_record_count_difference=(total - expected) if expected is not None else None,
                  fewer_rows_than_metadata=(total < expected) if expected is not None else None,
                  downloaded_utc=_now(), cost_check_skipped=True,
                  estimated_cost_usd=None, incremental_estimated_cost_usd=None)
    if write_receipt:
        _atomic_json(directory / "fetch.json", result)
    if not result["complete"]:
        raise ValueError("DBN was empty or reached the explicit record safety cap; it is preserved and excluded from analysis.")
    return result


def _recover_completed(directory: Path, request: dict) -> Path | None:
    marker_path = directory / "stream_complete.json"
    final = directory / "sample.dbn.zst"
    if not marker_path.exists():
        if final.exists():
            # Preserve an unconfirmed prefix under a unique name before an
            # included-access retry; never silently treat it as complete.
            _replace_with_retry(final, directory / f"interrupted.{uuid4().hex}.dbn.zst")
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("request") != request:
        raise ValueError("Successful stream marker belongs to another request.")
    if final.exists():
        return final
    name = marker.get("download_file")
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("Successful stream marker has an invalid local filename.")
    pending = directory / name
    if not pending.is_file():
        raise ValueError("Successful stream marker has no preserved DBN file.")
    _replace_with_retry(pending, final)
    return final


def _download_one(entry: dict, data_dir: Path, api_factory: Callable, progress: Callable) -> dict:
    import databento as db
    request = entry["request"]
    cached = _read_cached(data_dir, request)
    if cached:
        return cached
    directory = cache_directory(data_dir, request)
    request_path = directory / "request.json"
    if request_path.exists():
        if json.loads(request_path.read_text(encoding="utf-8")) != request:
            raise ValueError("Existing raw directory has a different exact request; all files preserved.")
    else:
        if (directory / "sample.dbn.zst").exists() or (directory / "stream_complete.json").exists():
            raise ValueError("Existing raw DBN has no exact request provenance; preserved without reuse.")
        _atomic_json(request_path, request)
    recovered = _recover_completed(directory, request)
    if recovered is not None:
        return _export_stream(db.DBNStore.from_file(recovered), entry, directory, recovered=True)
    attempts_path = directory / "download_attempts.json"
    history = json.loads(attempts_path.read_text(encoding="utf-8")).get("attempts", []) if attempts_path.exists() else []
    for attempt in range(DOWNLOAD_MAX_ATTEMPTS):
        pending = directory / f"stream.{uuid4().hex}.dbn.zst"
        started = _now()
        try:
            api_factory().timeseries.get_range(**request, path=pending)
        except Exception as exc:
            original_error = safe_error(exc)
            _release_failed_stream_frames(exc)
            preserved = None
            preservation_error = None
            preserved_in_place = False
            if pending.exists():
                failed = directory / f"interrupted.{uuid4().hex}.dbn.zst"
                try:
                    _replace_with_retry(pending, failed)
                    preserved = failed.name
                except OSError as preservation_exc:
                    # An external lock may outlive the bounded retry. Keep this
                    # unique source intact and continue with a new filename;
                    # the transfer error remains the actual failure reason.
                    preserved = pending.name
                    preserved_in_place = True
                    preservation_error = safe_error(preservation_exc)
            history.append({"started_utc": started, "finished_utc": _now(), "success": False,
                            "error": original_error, "source_filename": pending.name,
                            "preserved_prefix": preserved, "preserved_in_place": preserved_in_place,
                            "preservation_error": preservation_error})
            try:
                _atomic_json(attempts_path, {"request": request, "attempts": history})
            except Exception as checkpoint_exc:
                exc.add_note("Download attempt checkpoint failed: " + safe_error(checkpoint_exc))
                raise exc from None
            if attempt == DOWNLOAD_MAX_ATTEMPTS - 1:
                raise
            progress(f"Retrying included-access stream {request['start'][:10]} {request['dataset']} "
                     f"{request['schema']} ({attempt + 2}/{DOWNLOAD_MAX_ATTEMPTS}); partial bytes preserved.")
            time.sleep(1 + 2 * attempt)
            continue
        # Commit successful termination before any record decoding/export. A
        # crash either side of the atomic file rename remains recoverable.
        _atomic_json(directory / "stream_complete.json", {
            "request": request, "request_sha256": _signature(request),
            "download_file": pending.name, "stream_finished_utc": _now(),
        })
        history.append({"started_utc": started, "finished_utc": _now(), "success": True,
                        "download_file": pending.name})
        _atomic_json(attempts_path, {"request": request, "attempts": history})
        final = directory / "sample.dbn.zst"
        _replace_with_retry(pending, final)
        return _export_stream(db.DBNStore.from_file(final), entry, directory)
    raise AssertionError("Unreachable stream retry state.")


def fetch_study(
    config: dict, config_path: Path | None = None, *, progress: Callable[[str], None] = print,
    api_factory: Callable | None = None, download_workers: int | None = None,
) -> dict:
    """Download the exact grid; an optional transport override leaves it fixed.

    The runtime worker count does not alter research configuration, request
    identities, metadata concurrency or reusable completed streams. Both the
    configured and actual transport counts are recorded in plan/manifest.
    """
    config = validate_config(config)
    actual_workers = config["workers"] if download_workers is None else download_workers
    if type(actual_workers) is not int or not 1 <= actual_workers <= 32:
        raise ValueError("Runtime download_workers must be an integer from one to 32.")
    api_factory = api_factory or client
    plan = plan_study(config, config_path, progress=progress, api_factory=api_factory)
    data_dir, out_dir = Path(config["data_dir"]), Path(config["out_dir"])
    plan.update(download_workers=actual_workers, configured_download_workers=config["workers"],
                download_workers_override=download_workers)
    _atomic_json(out_dir / "download_plan.json", plan)
    # Create request directories serially: concurrent recursive OneDrive mkdir
    # has produced spurious permission errors in this workspace previously.
    for entry in plan["requests"]:
        cache_directory(data_dir, entry["request"]).mkdir(parents=True, exist_ok=True)
    manifest = dict(plan, requests=[], complete=False, downloaded_rows=0, time_series_downloaded=False)
    manifest_path = out_dir / "data_manifest.json"
    _atomic_json(manifest_path, manifest)
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=actual_workers) as pool:
        jobs = {pool.submit(_download_one, entry, data_dir, api_factory, progress): entry for entry in plan["requests"]}
        for future in as_completed(jobs):
            entry = jobs[future]
            try:
                result = future.result()
            except Exception as exc:
                result = dict(entry, complete=False, error=safe_error(exc))
                directory = cache_directory(data_dir, entry["request"])
                receipt_path, raw = directory / "fetch.json", directory / "sample.dbn.zst"
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    if receipt.get("request") == entry["request"]:
                        result.update(receipt, complete=False, error=safe_error(exc))
                if raw.exists():
                    result.update(data_path=str(raw.resolve()), file_size_bytes=raw.stat().st_size,
                                  successful_stream_marker_present=(directory / "stream_complete.json").exists())
            results[_signature(entry["request"])] = result
            manifest["requests"] = [results[_signature(item["request"])] for item in plan["requests"]
                                    if _signature(item["request"]) in results]
            manifest["downloaded_rows"] = sum(item.get("rows", 0) for item in manifest["requests"])
            manifest["time_series_downloaded"] = any(item.get("rows", 0) for item in manifest["requests"])
            _atomic_json(manifest_path, manifest)
            request = entry["request"]
            progress(f"{'Cached' if result.get('cache_reused') else 'Fetched'} {request['start'][:10]} "
                     f"{request['dataset']} {request['schema']}: {result.get('rows', 0):,} records; "
                     f"complete={result.get('complete', False)} ({len(results)}/{len(plan['requests'])}).")
    manifest["complete"] = (len(results) == plan["expected_request_count"] and
                            all(entry.get("complete", False) for entry in results.values()))
    manifest["finished_utc"] = _now()
    _atomic_json(manifest_path, manifest)
    if not manifest["complete"]:
        raise ValueError("Full-depth MBO SIRS grid is incomplete; successful streams and interrupted prefixes preserved; analysis must not start.")
    return manifest


def repair_failed_study(
    config: dict, *, progress: Callable[[str], None] = print,
    api_factory: Callable | None = None, workers: int = 4,
) -> dict:
    """Repair only completed failed futures recorded in a manifest snapshot.

    The main acquisition can continue: queued/active requests absent from the
    snapshot are never submitted, and its manifest/plan are never written.
    Completion here applies only to the selected repair requests, not the study.
    """
    config = validate_config(config)
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Repair workers must be an integer from one to eight.")
    data_dir, out_dir = Path(config["data_dir"]), Path(config["out_dir"])
    requests = build_requests(config)
    expected = {_signature(request): request for request in requests}
    plan_bytes = (out_dir / "download_plan.json").read_bytes()
    manifest_bytes = (out_dir / "data_manifest.json").read_bytes()
    plan, manifest = json.loads(plan_bytes), json.loads(manifest_bytes)
    for label, document in (("plan", plan), ("manifest", manifest)):
        if (document.get("configuration") != config
                or document.get("configuration_sha256") != _signature(config)
                or document.get("expected_request_count") != len(requests)
                or document.get("request_grid_sha256") != _signature({"requests": requests})):
            raise ValueError(f"Repair {label} differs from the exact study configuration/request grid.")
    if (not plan.get("ready_to_download") or plan.get("oversized_requests")
            or [entry.get("request") for entry in plan.get("requests", [])] != requests):
        raise ValueError("Repair requires the complete, ready, exact-grid download plan.")
    plan_entries = {_signature(entry["request"]): entry for entry in plan["requests"]}
    selected = []
    seen = set()
    recorded = manifest.get("requests")
    if not isinstance(recorded, list):
        raise ValueError("Repair manifest has no completed-future request list.")
    for entry in recorded:
        if not isinstance(entry, dict) or not isinstance(entry.get("request"), dict):
            raise ValueError("Repair manifest contains an invalid request entry.")
        token = _signature(entry["request"])
        if token in seen or token not in expected or entry["request"] != expected[token]:
            raise ValueError("Repair manifest contains a duplicate or foreign request.")
        seen.add(token)
        if type(entry.get("complete")) is not bool:
            raise ValueError("Repair manifest entry lacks a completed-future status.")
        if entry["complete"] is False:
            if not isinstance(entry.get("error"), str) or not entry["error"].strip():
                raise ValueError("Repair requires an explicit recorded error for each failed future.")
            selected.append((token, entry))
    # Preserve frozen grid order, irrespective of completion order in a snapshot.
    selected.sort(key=lambda item: list(expected).index(item[0]))
    actual_workers = min(workers, len(selected))
    report = {
        "created_utc": _now(), "configuration": config,
        "configuration_sha256": _signature(config),
        "request_grid_sha256": _signature({"requests": requests}),
        "expected_study_request_count": len(requests),
        "manifest_snapshot_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "plan_snapshot_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "manifest_recorded_completed_futures": len(recorded),
        "selected_failed_request_count": len(selected),
        "selected_requests": [entry["request"] for _, entry in selected],
        "requested_repair_workers": workers, "repair_workers": actual_workers,
        "scope": "Only completed failed futures in the manifest snapshot; completion does not establish whole-study completion.",
        "main_manifest_modified": False, "download_plan_modified": False,
        "complete": not selected, "requests": [], "cost_check_skipped": True,
    }
    report_path = out_dir / "repair_report.json"
    _atomic_json(report_path, report)
    if selected:
        api_factory = api_factory or client
        results = {}
        with ThreadPoolExecutor(max_workers=actual_workers) as pool:
            jobs = {pool.submit(_download_one, plan_entries[token], data_dir, api_factory, progress): (token, failed)
                    for token, failed in selected}
            for future in as_completed(jobs):
                token, failed = jobs[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = dict(plan_entries[token], complete=False, error=safe_error(exc))
                result = dict(result, original_manifest_error=failed["error"])
                results[token] = result
                report["requests"] = [results[key] for key, _ in selected if key in results]
                _atomic_json(report_path, report)
                request = failed["request"]
                progress(f"Repair {request['start'][:10]} {request['dataset']} {request['schema']}: "
                         f"complete={result.get('complete', False)} ({len(results)}/{len(selected)} selected failures).")
        report["complete"] = len(results) == len(selected) and all(entry.get("complete") is True for entry in results.values())
    report["finished_utc"] = _now()
    _atomic_json(report_path, report)
    return report


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_study(config: dict, *, progress: Callable[[str], None] = print, workers: int = 4) -> dict:
    """Reparse and hash every completed raw file, independently of acquisition.

    This is useful when acquisition began before a parser integrity check was
    added. It issues no API request, changes no download receipt, and fails on
    a missing/extra/duplicate request, byte/row mismatch or incomplete record.
    """
    import databento as db
    config = validate_config(config)
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Audit workers must be an integer from one to eight.")
    data_dir, out_dir = Path(config["data_dir"]), Path(config["out_dir"])
    manifest_path = out_dir / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = build_requests(config)
    expected_tokens = [_signature(request) for request in expected]
    entries = manifest.get("requests", [])
    actual_tokens = [_signature(entry["request"]) for entry in entries]
    if (not manifest.get("complete") or manifest.get("configuration") != config
            or manifest.get("configuration_sha256") != _signature(config)
            or manifest.get("request_grid_sha256") != _signature({"requests": expected})
            or len(actual_tokens) != len(set(actual_tokens)) or set(actual_tokens) != set(expected_tokens)):
        raise ValueError("Raw audit requires the complete exact configuration and request grid, without holes or duplicates.")
    report = {"created_utc": _now(), "complete": False, "configuration_sha256": _signature(config),
              "request_grid_sha256": manifest["request_grid_sha256"], "expected_request_count": len(expected),
              "validator_code_sha256": _file_sha256(Path(__file__)), "sdk_version": getattr(db, "__version__", None),
              "method": "Independent full DBN NumPy-chunk parse, explicit truncation state check and compressed-byte SHA256.",
              "requests": [], "failures": []}
    report_path = out_dir / "raw_audit.json"
    _atomic_json(report_path, report)
    def inspect(entry):
        request = entry["request"]
        if not entry.get("complete") or entry.get("record_limit_reached"):
            raise ValueError("Raw manifest contains an incomplete or capped request.")
        cached = _read_cached(data_dir, request)
        if cached is None or cached["rows"] != entry["rows"] or cached["file_size_bytes"] != entry["file_size_bytes"]:
            raise ValueError("Raw receipt and completed acquisition manifest disagree.")
        directory = cache_directory(data_dir, request)
        raw = directory / "sample.dbn.zst"
        if Path(entry["data_path"]).resolve() != raw.resolve():
            raise ValueError("Manifest data path differs from the exact request's canonical cache file.")
        before = raw.stat()
        parsed = _export_stream(db.DBNStore.from_file(raw), entry, directory, write_receipt=False)
        if parsed["rows"] != entry["rows"]:
            raise ValueError("Independently parsed raw record count differs from the acquisition receipt.")
        for key in ("schema", "instrument_ids", "record_types", "receive_timestamp_min_ns", "receive_timestamp_max_ns",
                    "event_timestamp_min_ns", "event_timestamp_max_ns"):
            if key in entry and parsed[key] != entry[key]:
                raise ValueError(f"Independently parsed raw {key} differs from its acquisition receipt.")
        sha256 = _file_sha256(raw)
        after = raw.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise ValueError("Raw bytes changed during their independent audit.")
        return {"request": request, "request_sha256": _signature(request), "data_path": str(raw.resolve()),
                "rows": parsed["rows"], "file_size_bytes": after.st_size, "sha256": sha256,
                "schema": parsed["schema"], "instrument_ids": parsed["instrument_ids"],
                "record_types": parsed["record_types"], "publisher_ids": parsed["publisher_ids"],
                "unflagged_records_outside_requested_clock_range": parsed["unflagged_records_outside_requested_clock_range"],
                "complete": True}
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(inspect, entry): entry for entry in entries}
        for future in as_completed(jobs):
            entry = jobs[future]
            try:
                item = future.result()
                results[_signature(entry["request"])] = item
            except Exception as exc:
                report["failures"].append({"request": entry["request"], "error": safe_error(exc)})
            report["requests"] = [results[token] for token in expected_tokens if token in results]
            _atomic_json(report_path, report)
            request = entry["request"]
            progress(f"Raw integrity audit {request['start'][:10]} {request['dataset']} {request['schema']}: "
                     f"{len(results)}/{len(expected)} passed; {len(report['failures'])} failures.")
    report["complete"] = len(results) == len(expected) and not report["failures"]
    report["finished_utc"] = _now()
    report["total_rows"] = sum(entry["rows"] for entry in results.values())
    report["total_compressed_bytes"] = sum(entry["file_size_bytes"] for entry in results.values())
    _atomic_json(report_path, report)
    if not report["complete"]:
        raise ValueError("Independent full-grid raw integrity audit failed; all source files preserved and fitting must not start.")
    return report
