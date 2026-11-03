"""Bounded native MBP-1 acquisition for explicit corporate-event windows.

Planning uses metadata only. Streaming has no automatic retry: an uncertain
attempt is preserved and blocks another retrieval of the same request. A
durable successful-stream marker permits local decoding recovery without a
second potentially billable request.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
from uuid import uuid4

import numpy as np
import pandas as pd

from .config import safe_error
from .data import client
from .network_wide_data import NATIVE_DATASETS


VERSION = 1
MAX_RECORDS = 10_000_000
MAX_BILLABLE_BYTES_PER_REQUEST = 1_000_000_000
MAX_RAW_BYTES_PER_REQUEST = 1_000_000_000
METADATA_MAX_AGE_SECONDS = 24 * 60 * 60
REQUEST_KEYS = {"dataset", "schema", "symbols", "stype_in", "start", "end", "limit"}


class UnavailableError(ValueError):
    """Known source availability/quality exclusion, distinct from API failure."""


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    # OneDrive and antivirus can briefly hold an existing destination open.
    # Retry only the local commit of these same bytes, never an API stream.
    for attempt in range(10):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(.05 * 2 ** attempt, .5))


def _paths(root, data_dir, out_dir):
    root = Path(root).resolve()
    _require(root.is_dir(), "Event acquisition root must be an existing workspace")
    result = []
    for value in (data_dir, out_dir):
        path = Path(value)
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        _require(path.is_relative_to(root) and path != root, "Event acquisition paths must remain inside the workspace")
        _require(not any(part in {".git", ".agents", ".codex"} for part in path.relative_to(root).parts),
                 "Event acquisition cannot write into workspace control directories")
        result.append(path)
    _require(result[0] != result[1], "Raw data and event output directories must differ")
    return root, *result


def _validate_requests(requests):
    _require(isinstance(requests, list) and requests, "Use a nonempty explicit request list")
    checked, seen = [], set()
    for request in requests:
        _require(isinstance(request, dict) and set(request) == REQUEST_KEYS, "Event requests require the exact MBP-1 request fields")
        _require(request["dataset"] in NATIVE_DATASETS and request["schema"] == "mbp-1"
                 and request["symbols"] == ["AAPL"] and request["stype_in"] == "raw_symbol",
                 "Event acquisition requires one native AAPL MBP-1 stream")
        _require(type(request["limit"]) is int and 1 <= request["limit"] <= MAX_RECORDS,
                 "Event record limit must be between 1 and 10,000,000")
        _require(isinstance(request["start"], str) and isinstance(request["end"], str), "Use explicit ISO request times")
        start, end = pd.Timestamp(request["start"]), pd.Timestamp(request["end"])
        _require(start.tzinfo is not None and end.tzinfo is not None
                 and pd.Timedelta(0) < end - start <= pd.Timedelta(hours=4), "Event windows need explicit timezones and at most four hours")
        key = (request["dataset"], start.value, end.value, request["limit"])
        _require(key not in seen, "Duplicate event request")
        seen.add(key)
        checked.append(dict(request, symbols=["AAPL"]))
    return checked


def _workers(workers):
    _require(type(workers) is int and 1 <= workers <= 8, "Event workers must be between one and eight")


def cache_directory(data_dir, request):
    base = Path(data_dir).resolve()
    # Keep the complete identity hash while leaving room for atomic marker
    # filenames under Windows' legacy MAX_PATH limit. Already fetched pilot
    # folders retain their original location to prevent another retrieval.
    token = _signature(request)
    canonical = base / token
    legacy = base / (request["start"][:10] + "_" + request["dataset"] + "_" + token)
    path = (legacy if legacy.exists() and not canonical.exists() else canonical).resolve()
    _require(path.is_relative_to(base), "Raw cache resolves outside its workspace data directory")
    return path


def _identity(requests):
    return {"version": VERSION, "requests_sha256": _signature(requests), "max_records": MAX_RECORDS,
            "max_billable_bytes_per_request": MAX_BILLABLE_BYTES_PER_REQUEST,
            "max_raw_bytes_per_request": MAX_RAW_BYTES_PER_REQUEST}


def _days(request):
    start = pd.Timestamp(request["start"]).tz_convert("UTC").normalize()
    last = (pd.Timestamp(request["end"]).tz_convert("UTC") - pd.Timedelta(nanoseconds=1)).normalize()
    return [stamp.date().isoformat() for stamp in pd.date_range(start, last, freq="D")]


def _local_state(data_dir, request):
    directory = cache_directory(data_dir, request)
    if not directory.exists():
        return None
    request_path = directory / "request.json"
    if request_path.exists():
        _require(json.loads(request_path.read_text()) == request, "Raw cache request identity differs")
    else:
        _require(not any(directory.iterdir()), "Raw cache has files without an exact request identity")
        return None
    marker_path, receipt_path = directory / "stream_complete.json", directory / "fetch.json"
    raw = directory / "sample.dbn.zst"
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text())
        _require(marker.get("request") == request, "Successful stream marker belongs to a different request")
        if not raw.is_file():
            name = marker.get("download_file")
            _require(isinstance(name, str) and Path(name).name == name, "Successful stream marker has an invalid filename")
            raw = directory / name
        _require(raw.is_file() and raw.stat().st_size == marker["file_size_bytes"]
                 and _sha(raw) == marker["content_sha256"], "Completed raw stream bytes differ from their durable marker")
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            _require(receipt.get("request") == request, "Raw receipt request differs")
            if receipt.get("complete") is True:
                _require(not receipt.get("record_limit_reached") and 0 <= receipt["rows"] < request["limit"]
                         and receipt["file_size_bytes"] == marker["file_size_bytes"]
                         and receipt["content_sha256"] == marker["content_sha256"], "Completed raw receipt differs from its source")
                return dict(receipt, data_path=str(raw), cache_reused=True, incremental_estimated_cost_usd=0., status="cached")
        return {"request": request, "status": "recoverable", "recoverable_path": str(raw),
                "incremental_estimated_cost_usd": 0., "estimated_cost_usd": marker.get("estimated_cost_usd"),
                "cache_reused": False}
    # Even an attempt with no retained bytes may have incurred a charge. Never
    # silently repeat it; an explicit repair decision must precede another pull.
    if any(path.name != "request.json" for path in directory.iterdir()):
        return {"request": request, "status": "blocked_interrupted", "complete": False,
                "error": "An uncertain previous stream is preserved; automatic retrieval retry is disabled",
                "incremental_estimated_cost_usd": None, "cache_reused": False}
    return None


def _catalog(dataset, requests):
    api = client()
    days = sorted({day for request in requests for day in _days(request)})
    schemas = api.metadata.list_schemas(dataset=dataset)
    available = api.metadata.get_dataset_range(dataset=dataset)
    rows = api.metadata.get_dataset_condition(dataset=dataset, start_date=days[0], end_date=days[-1])
    conditions = {row["date"]: row for row in rows if row["date"] in days}
    _require(len(conditions) == len([row for row in rows if row["date"] in days]), "Duplicate dataset condition dates")
    return {"dataset": dataset, "schemas": schemas, "range": available, "conditions": conditions,
            "quoted_utc": _now()}


def _availability(request, catalog):
    if "mbp-1" not in catalog["schemas"]:
        raise UnavailableError("Native MBP-1 schema is unavailable")
    span = catalog["range"].get("schema", {}).get("mbp-1")
    if not (isinstance(span, dict) and pd.Timestamp(span["start"]) <= pd.Timestamp(request["start"])
             < pd.Timestamp(request["end"]) <= pd.Timestamp(span["end"])):
        raise UnavailableError("Request lies outside entitled MBP-1 schema range")
    conditions = [catalog["conditions"].get(day) for day in _days(request)]
    if not all(row is not None and row.get("condition") == "available" for row in conditions):
        raise UnavailableError("Requested dataset date is absent or marked degraded, pending or missing")
    return conditions


def _unavailable(request, catalog, exc):
    return {"request": request, "status": "unavailable", "complete": False, "analyzable": False,
            "availability_reason": safe_error(exc), "incremental_estimated_cost_usd": 0.,
            "cache_reused": False, "dataset_conditions": [catalog["conditions"].get(day) for day in _days(request)],
            "schema_range": catalog["range"].get("schema", {}).get("mbp-1")}


def _estimate(request):
    api = client()
    full = {key: value for key, value in request.items() if key != "limit"}
    count = api.metadata.get_record_count(**full)
    size = api.metadata.get_billable_size(**full)
    cost = float(api.metadata.get_cost(**full))
    _require(isinstance(count, int) and not isinstance(count, bool) and count >= 0
             and isinstance(size, int) and not isinstance(size, bool) and size >= 0
             and math.isfinite(cost) and cost >= 0, "Metadata count/size/cost is invalid")
    fits = count < request["limit"] and size <= MAX_BILLABLE_BYTES_PER_REQUEST
    return {"request": request, "status": "ready" if fits else "blocked_size", "cache_reused": False,
            "expected_records_without_limit": count, "estimated_billable_bytes_without_limit": size,
            "estimated_cost_usd": cost, "incremental_estimated_cost_usd": cost,
            "metadata_count_may_overreport": True, "quoted_utc": _now()}


def plan_requests(requests, root, data_dir, out_dir, workers=3, progress=print):
    """Metadata-only preflight; failures are returned explicitly, never billed."""
    requests = _validate_requests(requests)
    _workers(workers)
    root, data_dir, out_dir = _paths(root, data_dir, out_dir)
    identity = _identity(requests)
    progress_path = out_dir / "metadata_progress.json"
    saved = {}
    if progress_path.is_file():
        candidate = json.loads(progress_path.read_text())
        if candidate.get("identity") == identity:
            age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(candidate["created_utc"])).total_seconds()
            if 0 <= age <= METADATA_MAX_AGE_SECONDS:
                saved = candidate
    state = {"identity": identity, "created_utc": saved.get("created_utc", _now()),
             "catalogs": saved.get("catalogs", {}), "requests": saved.get("requests", {})}
    datasets = sorted({request["dataset"] for request in requests})
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_catalog, dataset, [r for r in requests if r["dataset"] == dataset]): dataset
                   for dataset in datasets if dataset not in state["catalogs"] or "error" in state["catalogs"][dataset]}
        for future in as_completed(futures):
            dataset = futures[future]
            try:
                state["catalogs"][dataset] = future.result()
            except Exception as exc:
                state["catalogs"][dataset] = {"dataset": dataset, "error": safe_error(exc)}
            _atomic_json(progress_path, state)
    entries, pending = {}, []
    for request in requests:
        token = _signature(request)
        catalog = state["catalogs"][request["dataset"]]
        try:
            _require("error" not in catalog, catalog.get("error", "Dataset metadata failed"))
            conditions = _availability(request, catalog)
            local = _local_state(data_dir, request)
            prior = state["requests"].get(token)
            if local is not None:
                entries[token] = dict(local, dataset_conditions=conditions)
            elif prior is not None and prior.get("status") == "ready":
                entries[token] = dict(prior, dataset_conditions=conditions, metadata_quote_reused=True)
            else:
                pending.append(request)
        except UnavailableError as exc:
            entries[token] = _unavailable(request, catalog, exc)
        except Exception as exc:
            entries[token] = {"request": request, "status": "metadata_error", "error": safe_error(exc),
                              "incremental_estimated_cost_usd": None, "complete": False}
    progress(f"Event metadata: {len(requests)} exact streams; {len(pending)} require fresh count/byte/cost quotes.")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_estimate, request): request for request in pending}
        for n, future in enumerate(as_completed(futures), 1):
            request = futures[future]
            try:
                entry = future.result()
                entry["dataset_conditions"] = _availability(request, state["catalogs"][request["dataset"]])
            except Exception as exc:
                entry = {"request": request, "status": "metadata_error", "error": safe_error(exc),
                         "incremental_estimated_cost_usd": None, "complete": False}
            entries[_signature(request)] = entry
            state["requests"][_signature(request)] = entry
            _atomic_json(progress_path, state)
            if n % 10 == 0 or n == len(pending):
                progress(f"Event metadata quoted {n}/{len(pending)} pending streams.")
    ordered = [entries[_signature(request)] for request in requests]
    ready = all(row["status"] in {"ready", "cached", "recoverable", "unavailable"} for row in ordered)
    costs = [row.get("incremental_estimated_cost_usd") for row in ordered]
    total = sum(costs) if all(value is not None for value in costs) else None
    plan = {"identity": identity, "created_utc": _now(), "requests": ordered,
            "catalogs": state["catalogs"], "expected_request_count": len(requests),
            "ready_to_download": ready, "time_series_downloaded": False,
            "unavailable_requests": sum(row["status"] == "unavailable" for row in ordered),
            "incremental_estimated_cost_usd": total,
            "expected_total_records": sum(row.get("expected_records_without_limit", row.get("rows", 0)) for row in ordered),
            "estimated_total_billable_bytes": sum(row.get("estimated_billable_bytes_without_limit", 0) for row in ordered),
            "note": "Metadata estimates may overreport non-10-minute windows; actual billing follows bytes sent. "
                    "No automatic paid retries. Empty completed streams are not analyzable."}
    _atomic_json(out_dir / "download_plan.json", plan)
    progress(f"Event metadata complete: ready={ready}; incremental estimated USD={total}.")
    return plan


def _parse_stream(raw, request):
    """Audit every record in bounded chunks, preserving native integer fields."""
    import databento as db

    store = db.DBNStore.from_file(raw)
    metadata = store.metadata
    start, end = pd.Timestamp(request["start"]).value, pd.Timestamp(request["end"]).value
    _require(str(metadata.dataset) == request["dataset"] and str(metadata.schema) == "mbp-1"
             and int(metadata.start) == start and metadata.end is not None and int(metadata.end) == end
             and int(metadata.limit or 0) == request["limit"] and str(metadata.stype_in) == "raw_symbol"
             and set(metadata.symbols) == {"AAPL"}, "DBN metadata differs from the exact event request")
    mapped = set()
    days = _days(request)
    for interval in getattr(metadata, "mappings", {}).get("AAPL", []):
        if any(str(interval["start_date"]) <= day < str(interval["end_date"]) for day in days):
            mapped.add(int(interval["symbol"]))
    rows, invalid_clock, flagged = 0, 0, 0
    raw_bytes, outside = 0, 0
    ids, publishers = set(), set()
    minimum, maximum = None, None
    required = {"ts_recv", "ts_event", "instrument_id", "publisher_id", "rtype", "flags",
                "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"}
    chunks = store.to_ndarray(count=100_000)
    for chunk in chunks:
        _require(chunk.ndim == 1 and chunk.dtype.names and required <= set(chunk.dtype.names), "Malformed MBP-1 record layout")
        if not len(chunk):
            continue
        rows += len(chunk)
        raw_bytes += chunk.nbytes
        _require(rows < request["limit"] and raw_bytes <= MAX_BILLABLE_BYTES_PER_REQUEST,
                 "Decoded stream reaches its record/byte safety cap")
        _require(np.all(chunk["rtype"] == 1), "Stream contains a non-MBP-1 record")
        ids.update(map(int, np.unique(chunk["instrument_id"])))
        publishers.update(map(int, np.unique(chunk["publisher_id"])))
        _require(ids <= mapped and not metadata.partial and not metadata.not_found, "Unresolved or unrequested DBN instrument")
        clock = chunk["ts_recv"].astype(np.uint64, copy=False)
        flags = chunk["flags"].astype(np.uint8, copy=False)
        bad = (flags & (32 | 8 | 4)) != 0
        flagged += int(bad.sum())
        valid_clock = clock <= np.iinfo(np.int64).max
        invalid_clock += int((~valid_clock).sum())
        outside += int(np.sum(((clock < start) | (clock >= end) | ~valid_clock) & ~bad))
        if valid_clock.any():
            lo, hi = int(clock[valid_clock].min()), int(clock[valid_clock].max())
            minimum, maximum = lo if minimum is None else min(minimum, lo), hi if maximum is None else max(maximum, hi)
    _require(not getattr(chunks, "_close_on_next", False), "DBN parser found an incomplete trailing record")
    _require(outside == 0, "Unflagged records fall outside the requested receive-time interval")
    return {"rows": rows, "analyzable": rows > 0, "record_limit_reached": False,
            "decoded_record_bytes": raw_bytes, "instrument_ids": sorted(ids), "publisher_ids": sorted(publishers),
            "receive_timestamp_min_ns": minimum, "receive_timestamp_max_ns": maximum,
            "invalid_receive_clock_records": invalid_clock, "quality_flagged_records": flagged,
            "unflagged_records_outside_requested_clock_range": outside}


def _finish_local(entry, directory, raw):
    request = entry["request"]
    marker = json.loads((directory / "stream_complete.json").read_text())
    _require(marker["request"] == request and raw.stat().st_size == marker["file_size_bytes"]
             and _sha(raw) == marker["content_sha256"], "Successful stream identity changed before decoding")
    final = directory / "sample.dbn.zst"
    if raw != final:
        raw.replace(final)
    audit = _parse_stream(final, request)
    result = {**entry, **audit, "status": "complete", "complete": True, "stream_complete": True,
              "data_path": str(final), "file_size_bytes": final.stat().st_size,
              "content_sha256": marker["content_sha256"], "downloaded_utc": marker["stream_finished_utc"],
              "estimated_cost_usd": marker.get("estimated_cost_usd"), "cost_check_skipped": False}
    result.pop("recoverable_path", None)
    _atomic_json(directory / "fetch.json", result)
    return result


def _fetch_one(entry, data_dir, allocated_cost):
    request = entry["request"]
    local = _local_state(data_dir, request)
    if local and local["status"] == "cached":
        return local
    directory = cache_directory(data_dir, request)
    if local and local["status"] == "recoverable":
        return _finish_local(entry, directory, Path(local["recoverable_path"]))
    _require(local is None, "Uncertain previous stream blocks automatic retrieval retry")
    # Recheck the exact full-window cost immediately before each potential
    # charge. A price increase requires a new whole-grid budget allocation.
    api = client()
    full = {key: value for key, value in request.items() if key != "limit"}
    current = float(api.metadata.get_cost(**full))
    _require(math.isfinite(current) and 0 <= current <= allocated_cost, "Current stream estimate exceeds its whole-grid allocation")
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_json(directory / "request.json", request)
    # Exclusive creation prevents simultaneous fetch invocations from issuing
    # the same charge. This marker is never removed after an uncertain attempt.
    with (directory / "attempt_started.json").open("x", encoding="utf-8") as handle:
        json.dump({"request": request, "started_utc": _now(), "estimated_cost_usd": current}, handle)
    pending = directory / ("stream." + uuid4().hex + ".dbn.zst")
    try:
        api.timeseries.get_range(**request, path=pending)
    except Exception as exc:
        error = safe_error(exc)
        traceback.clear_frames(exc.__traceback__)
        _atomic_json(directory / "failed_attempt.json", {"request": request, "error": error,
                     "preserved_prefix": pending.name if pending.exists() else None, "finished_utc": _now()})
        raise ValueError(error) from None
    _require(pending.is_file(), "Successful SDK stream did not preserve a raw file")
    _require(pending.stat().st_size <= MAX_RAW_BYTES_PER_REQUEST, "Raw stream exceeds its compressed-byte safety cap")
    _atomic_json(directory / "stream_complete.json", {"request": request, "download_file": pending.name,
                 "file_size_bytes": pending.stat().st_size, "content_sha256": _sha(pending),
                 "stream_finished_utc": _now(), "estimated_cost_usd": current})
    return _finish_local(dict(entry, incremental_estimated_cost_usd=current), directory, pending)


def fetch_requests(plan, root, data_dir, out_dir, max_cost_usd, workers=3, progress=print):
    """Fetch a fully quoted grid once; preserve uncertainty instead of retrying."""
    _workers(workers)
    _require(isinstance(max_cost_usd, (int, float)) and not isinstance(max_cost_usd, bool)
             and math.isfinite(max_cost_usd) and max_cost_usd >= 0, "Use a finite nonnegative event acquisition budget")
    root, data_dir, out_dir = _paths(root, data_dir, out_dir)
    entries = plan.get("requests", [])
    requests = _validate_requests([entry["request"] for entry in entries])
    _require(plan.get("identity") == _identity(requests) and plan.get("ready_to_download") is True,
             "Fetch requires the complete exact ready metadata plan")
    refreshed_by_request, allocations, pending = {}, {}, []
    # Complete all fresh size/cost checks before starting any stream. Local
    # completed streams need no API access and incur no incremental charge.
    for entry in entries:
        request = entry["request"]
        catalog = plan["catalogs"][request["dataset"]]
        try:
            _availability(request, catalog)
        except UnavailableError as exc:
            refreshed_by_request[_signature(request)] = _unavailable(request, catalog, exc)
            allocations[_signature(request)] = 0.
            continue
        _require(entry["status"] in {"ready", "cached", "recoverable"}, "Plan availability status is inconsistent")
        local = _local_state(data_dir, request)
        if local:
            _require(local["status"] in {"cached", "recoverable"}, "Preserved uncertain stream requires explicit repair")
            refreshed_by_request[_signature(request)] = dict(entry, **local)
            allocations[_signature(request)] = 0.
        else:
            age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(plan["created_utc"])).total_seconds()
            _require(0 <= age <= METADATA_MAX_AGE_SECONDS, "Event metadata plan is stale; replan before retrieval")
            pending.append(entry)
    failures = []
    progress(f"Event fetch preflight: refreshing {len(pending)} available uncached stream quotes.")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_estimate, entry["request"]): entry for entry in pending}
        for n, future in enumerate(as_completed(futures), 1):
            entry = futures[future]
            try:
                current = future.result()
                _require(current["status"] == "ready", "Fresh full-window sizing exceeds event safety bounds")
                token = _signature(entry["request"])
                refreshed_by_request[token] = dict(entry, **current)
                allocations[token] = current["estimated_cost_usd"]
            except Exception as exc:
                failures.append(safe_error(exc))
            if n % 20 == 0 or n == len(pending):
                progress(f"Event fetch preflight refreshed {n}/{len(pending)} stream quotes.")
    _require(not failures, "Fresh event preflight failed: " + (failures[0] if failures else ""))
    refreshed = [refreshed_by_request[_signature(request)] for request in requests]
    total = sum(allocations.values())
    _require(total <= max_cost_usd, f"Fresh event estimate USD {total:.6f} exceeds ceiling USD {max_cost_usd:.6f}")
    unavailable = {_signature(entry["request"]): entry for entry in refreshed if entry["status"] == "unavailable"}
    manifest = {"identity": plan["identity"], "created_utc": _now(), "requests": list(unavailable.values()),
                "expected_request_count": len(requests), "authorized_estimate_ceiling_usd": float(max_cost_usd),
                "fresh_incremental_estimated_cost_usd": total, "complete": False,
                "time_series_downloaded": False, "automatic_paid_retries": False}
    _atomic_json(out_dir / "data_manifest.json", manifest)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, entry, data_dir, allocations[_signature(entry["request"])]): entry
                   for entry in refreshed if entry["status"] != "unavailable"}
        results = dict(unavailable)
        for n, future in enumerate(as_completed(futures), 1):
            entry = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                directory = cache_directory(data_dir, entry["request"])
                result = dict(entry, status="failed", complete=False, analyzable=False, error=safe_error(exc),
                    stream_attempted=(directory / "attempt_started.json").is_file(),
                    successful_stream_marker_present=(directory / "stream_complete.json").is_file(),
                    preserved_raw_files=[str(path) for path in directory.glob("*.dbn.zst")])
            results[_signature(entry["request"])] = result
            manifest["requests"] = [results[_signature(request)] for request in requests if _signature(request) in results]
            _atomic_json(out_dir / "data_manifest.json", manifest)
            progress(f"Event streams {n}/{len(requests)}: {entry['request']['start'][:10]} "
                     f"{entry['request']['dataset']} {result['status']}; rows={result.get('rows', 0)}.")
    manifest["requests"] = [results[_signature(request)] for request in requests]
    manifest.update(complete=all(row.get("complete") is True for row in manifest["requests"]),
                    acquisition_complete_for_available_requests=all(row.get("complete") is True for row in manifest["requests"]
                                                                  if row["status"] != "unavailable"),
                    unavailable_requests=len(unavailable),
                    downloaded_rows=sum(row.get("rows", 0) for row in manifest["requests"]),
                    analyzable_requests=sum(bool(row.get("analyzable")) for row in manifest["requests"]),
                    time_series_downloaded=any(row.get("rows", 0) > 0 or row.get("stream_attempted", False)
                                               or row.get("stream_complete", False) for row in manifest["requests"]), finished_utc=_now())
    _atomic_json(out_dir / "data_manifest.json", manifest)
    return manifest
