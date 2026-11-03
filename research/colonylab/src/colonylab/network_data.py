"""Bounded, resumable downloads for an aligned exchange-network pilot.

MBO is used for order-flow event counts only.  It is
deliberately not reconstructed as a queue from a mid-session starting point.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import safe_error, write_json
from .data import client

DATASETS = ("XNAS.ITCH", "XNYS.PILLAR", "BATS.PITCH")
SCHEMAS = ("mbp-1", "mbo")
SYMBOLS = ("AAPL", "MSFT", "SPY")
DAYS = ("2026-10-21", "2026-10-22", "2026-10-23")
MAX_RECORDS = 1_200_000
MAX_REQUEST_BYTES = 75_000_000


def requests_for(minutes: int = 1) -> list[dict]:
    if type(minutes) is not int or not 1 <= minutes <= 15:
        raise ValueError("Aligned pilot windows must be from 1 to 15 minutes.")
    result = []
    for day in DAYS:
        start = pd.Timestamp(f"{day}T14:00:00Z")
        end = start + pd.Timedelta(minutes=minutes)
        for dataset in DATASETS:
            for schema in SCHEMAS:
                result.append({"dataset": dataset, "schema": schema,
                               "symbols": list(SYMBOLS), "stype_in": "raw_symbol",
                               "start": start.isoformat(), "end": end.isoformat(),
                               "limit": MAX_RECORDS})
    return result


def cache_directory(data_dir: Path, request: dict) -> Path:
    token = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:12]
    day = request["start"][:10]
    return data_dir / f"{day}_{request['dataset']}_{request['schema']}_{token}"


def _read_cached(data_dir: Path, request: dict) -> dict | None:
    directory = cache_directory(data_dir, request)
    path = directory / "fetch.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("request") != request:
        raise ValueError("Request mismatch in cached network manifest.")
    if result.get("record_limit_reached") or result.get("rows", 0) >= request["limit"]:
        raise ValueError("Cached network request reached its row cap and cannot be reused as complete.")
    if result.get("complete") and (directory / "sample.dbn.zst").exists():
        return dict(result, cache_reused=True)
    return None


def _estimate(request: dict, data_dir: Path) -> dict:
    cached = _read_cached(data_dir, request)
    if cached:
        return dict(cached, estimate_reused=True, incremental_estimated_cost_usd=0.0)
    api = client()
    # Counts and byte size deliberately omit the safety limit: it must not hide
    # a truncated window when we decide whether this aligned design will fit.
    full_request = {k: v for k, v in request.items() if k != "limit"}
    count = int(api.metadata.get_record_count(**full_request))
    billable_bytes = int(api.metadata.get_billable_size(**full_request))
    cost = float(api.metadata.get_cost(**request))
    if count < 0 or billable_bytes < 0 or not math.isfinite(cost) or cost < 0:
        raise ValueError("Databento returned an invalid network-pilot estimate.")
    return {"request": request, "expected_records_without_limit": count,
            "estimated_billable_bytes_without_limit": billable_bytes,
            "estimated_cost_usd": cost, "incremental_estimated_cost_usd": cost,
            "metadata_count_may_overreport": True,
            "cache_reused": False}


def plan_network(data_dir: Path, out_dir: Path, *, minutes: int = 1,
                 budget: float = 0.50, workers: int = 3,
                 progress: Callable[[str], None] = print) -> dict:
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("Budget must be finite and nonnegative.")
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Workers must be an integer from 1 to 4.")
    requests = requests_for(minutes)
    progress(f"Estimating {len(requests)} aligned {minutes}-minute requests before downloads.")
    entries = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(_estimate, request, data_dir): request for request in requests}
        for future in as_completed(jobs):
            request = jobs[future]
            entry = future.result()
            entries.append(entry)
            progress(f"Estimate {request['start'][:10]} {request['dataset']} {request['schema']}: "
                     f"{entry['expected_records_without_limit']:,} records, "
                     f"${entry['incremental_estimated_cost_usd']:.6f} incremental")
    entries.sort(key=lambda entry: (entry["request"]["start"], entry["request"]["dataset"], entry["request"]["schema"]))
    too_big = [entry for entry in entries
               if entry["expected_records_without_limit"] >= MAX_RECORDS
               or entry["estimated_billable_bytes_without_limit"] > MAX_REQUEST_BYTES]
    # If a single feed exceeds the safety bound, shorten EVERY window together.
    if too_big and minutes > 1:
        progress("At least one request exceeds the pilot size bound; replanning all venues and dates at one minute.")
        return plan_network(data_dir, out_dir, minutes=1, budget=budget,
                            workers=workers, progress=progress)
    total = sum(entry["incremental_estimated_cost_usd"] for entry in entries)
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(),
            "datasets": list(DATASETS), "schemas": list(SCHEMAS), "symbols": list(SYMBOLS),
            "days": list(DAYS), "window_minutes": minutes, "window_start_utc": "14:00:00",
            "authorized_estimate_ceiling_usd": budget,
            "incremental_estimated_cost_usd": total,
            "original_estimated_cost_usd": sum(entry["estimated_cost_usd"] for entry in entries),
            "estimated_total_billable_bytes": sum(entry["estimated_billable_bytes_without_limit"] for entry in entries),
            "expected_total_records": sum(entry["expected_records_without_limit"] for entry in entries),
            "record_limit_per_request": MAX_RECORDS, "max_billable_bytes_per_request": MAX_REQUEST_BYTES,
            "ready_to_download": total <= budget and not too_big,
            "time_series_downloaded": False, "requests": entries,
            "note": "Cost is a metadata estimate, not an invoice or a hard billing guarantee. "
                    "Databento metadata counts can overreport ranges not aligned to multiples of 10 minutes; "
                    "these counts and byte sizes are conservative sizing estimates, not exact expected rows.",
            "metadata_documentation": "https://databento.com/docs/api-historical/metadata/metadata-get-record-count?lang=python"}
    write_json(out_dir / "download_plan.json", plan)
    progress(f"Total incremental estimate ${total:.6f}; {plan['expected_total_records']:,} records; "
             f"{plan['estimated_total_billable_bytes'] / 1e6:.2f} MB billable bytes.")
    if too_big:
        raise ValueError("At least one one-minute feed exceeds the pilot size bound; no download performed.")
    if total > budget:
        raise ValueError(f"Estimated ${total:.6f} exceeds pilot ceiling ${budget:.6f}; no download performed.")
    return plan


def _export_entry(store, entry: dict, directory: Path, *, recovered: bool = False) -> dict:
    frame = store.to_df(price_type="float", pretty_ts=True, map_symbols=True)
    request = entry["request"]
    rows = len(frame)
    capped = rows >= request["limit"]
    expected = entry["expected_records_without_limit"]
    timestamps = pd.to_datetime(frame.index, utc=True)
    event_times = pd.to_datetime(frame["ts_event"], utc=True) if "ts_event" in frame else timestamps
    # Metadata rounds non-10-minute ranges and is not an exact completeness
    # count. Completion means the SDK stream ended successfully, the DBN parsed,
    # and the explicit safety cap was not reached. Preserve the difference.
    short = rows < expected
    result = dict(entry, rows=rows, complete=not capped and rows > 0,
                  record_limit_reached=capped, fewer_rows_than_metadata=short,
                  cache_reused=False, recovered_from_local_dbn=recovered,
                  data_path=str((directory / "sample.dbn.zst").resolve()),
                  file_size_bytes=(directory / "sample.dbn.zst").stat().st_size,
                  columns=list(frame.columns),
                  receive_timestamp_min=str(timestamps.min()) if rows else None,
                  receive_timestamp_max=str(timestamps.max()) if rows else None,
                  event_timestamp_min=str(event_times.min()) if rows else None,
                  event_timestamp_max=str(event_times.max()) if rows else None,
                  downloaded_utc=datetime.now(timezone.utc).isoformat())
    write_json(directory / "fetch.json", result)
    return result


def _download_entry(entry: dict, data_dir: Path) -> dict:
    request = entry["request"]
    cached = _read_cached(data_dir, request)
    if cached:
        return cached
    directory = cache_directory(data_dir, request)
    directory.mkdir(parents=True, exist_ok=True)
    dbn_path = directory / "sample.dbn.zst"
    if dbn_path.exists():
        # Recover an interrupted export without another billable retrieval.
        request_path = directory / "request.json"
        if not request_path.exists() or json.loads(request_path.read_text(encoding="utf-8")) != request:
            raise ValueError("Existing network DBN has no matching request; it has been preserved.")
        marker_path = directory / "stream_complete.json"
        if not marker_path.exists() or json.loads(marker_path.read_text(encoding="utf-8")).get("request") != request:
            raise ValueError("Existing network DBN has no durable successful-stream marker; "
                             "partial data has been preserved and no new retrieval was attempted.")
        import databento as db
        return _export_entry(db.DBNStore.from_file(dbn_path), entry, directory, recovered=True)
    write_json(directory / "request.json", request)
    # All cost estimates and size checks completed before reaching this call.
    store = client().timeseries.get_range(**request, path=dbn_path)
    marker_temp = directory / "stream_complete.json.tmp"
    write_json(marker_temp, {"request": request,
               "stream_finished_utc": datetime.now(timezone.utc).isoformat()})
    marker_temp.replace(directory / "stream_complete.json")
    return _export_entry(store, entry, directory)


def fetch_network(data_dir: Path, out_dir: Path, *, minutes: int = 1,
                  budget: float = 0.50, workers: int = 3,
                  progress: Callable[[str], None] = print) -> dict:
    plan = plan_network(data_dir, out_dir, minutes=minutes, budget=budget,
                        workers=workers, progress=progress)
    manifest = dict(plan, requests=[], complete=False, time_series_downloaded=False)
    write_json(out_dir / "data_manifest.json", manifest)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(_download_entry, entry, data_dir): entry for entry in plan["requests"]}
        for future in as_completed(jobs):
            entry = jobs[future]
            try:
                result = future.result()
            except Exception as exc:
                result = dict(entry, complete=False, error=safe_error(exc))
            manifest["requests"].append(result)
            manifest["time_series_downloaded"] |= bool(result.get("rows"))
            write_json(out_dir / "data_manifest.json", manifest)
            request = result["request"]
            progress(f"{'Cached' if result.get('cache_reused') else 'Fetched'} "
                     f"{request['start'][:10]} {request['dataset']} {request['schema']}: "
                     f"{result.get('rows', 0):,} rows; complete={result.get('complete', False)}")
    manifest["requests"].sort(key=lambda entry: (entry["request"]["start"], entry["request"]["dataset"], entry["request"]["schema"]))
    manifest["complete"] = all(entry.get("complete", False) for entry in manifest["requests"])
    manifest["downloaded_rows"] = sum(entry.get("rows", 0) for entry in manifest["requests"])
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(out_dir / "data_manifest.json", manifest)
    if not manifest["complete"]:
        raise ValueError("Aligned download is incomplete. Inspect data_manifest.json; analysis was not started.")
    return manifest


def load_network_frames(manifest_path: Path) -> dict[tuple[str, str, str], pd.DataFrame]:
    import databento as db
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError("Only a complete aligned network manifest can be analyzed.")
    frames = {}
    for entry in manifest["requests"]:
        request = entry["request"]
        if not entry.get("complete") or entry.get("record_limit_reached"):
            raise ValueError("Incomplete/capped request cannot enter the network panel.")
        frame = db.DBNStore.from_file(entry["data_path"]).to_df(
            price_type="float", pretty_ts=True, map_symbols=True)
        if len(frame) >= request["limit"]:
            raise ValueError("Actual cached DBN has reached its row cap and cannot enter the panel.")
        if len(frame) != entry["rows"]:
            raise ValueError("Cached DBN row count differs from its completed manifest.")
        frame.attrs.update(requested_start=request["start"], requested_end=request["end"])
        key = (request["dataset"], request["schema"], request["start"][:10])
        if key in frames:
            raise ValueError("Duplicate dataset/schema/day in network manifest.")
        frames[key] = frame
    expected = {(dataset, schema, day) for dataset in DATASETS for schema in SCHEMAS for day in DAYS}
    if set(frames) != expected:
        raise ValueError("Network manifest does not contain every planned venue/schema/day.")
    return frames
