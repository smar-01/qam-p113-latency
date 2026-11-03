"""Configured, bounded acquisition for the wider historical venue replication.

The complete requested grid is sized before any time-series pull. Price checks
are optional when the user reports that historical access is included.
Oversized requests stop the run; no dates, symbols, or minutes are dropped.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
import tomllib
from typing import Callable
from uuid import uuid4

import pandas as pd

from .config import safe_error, write_json
from . import network_data
from .network_data import _download_entry, _estimate, _read_cached, cache_directory


CONFIG_KEYS = {
    "name", "datasets", "symbols", "schemas", "training_days", "held_out_days",
    "start_time_utc", "minutes", "max_cost_usd", "workers",
    "record_limit_per_request", "max_billable_bytes_per_request", "data_dir", "out_dir",
}
OPTIONAL_CONFIG_KEYS = {"billing_assumed_included", "max_cost_usd"}
BILLING_ASSUMPTION = "User reports historical access is included; billing was not independently verified."
NATIVE_DATASETS = {
    "XNAS.ITCH", "XNYS.PILLAR", "BATS.PITCH", "ARCX.PILLAR", "EDGX.PITCH",
    "BATY.PITCH", "EDGA.PITCH", "EPRL.DOM", "MEMX.MEMOIR", "XASE.PILLAR",
    "XBOS.ITCH", "XCHI.PILLAR", "XPSX.ITCH",
}


def validate_config(config: dict) -> dict:
    """Validate an explicit native-venue MBP1 study before opening a client."""
    required = CONFIG_KEYS - OPTIONAL_CONFIG_KEYS
    if not required <= set(config) or set(config) - (CONFIG_KEYS | OPTIONAL_CONFIG_KEYS):
        raise ValueError("Wide configuration requires: " + ", ".join(sorted(required)))
    result = dict(config)
    if type(result.get("billing_assumed_included", False)) is not bool:
        raise ValueError("billing_assumed_included must be boolean.")
    if not isinstance(result["name"], str) or not result["name"].strip():
        raise ValueError("The study requires a nonempty name.")
    for key, maximum in (("datasets", 13), ("symbols", 10), ("training_days", 60), ("held_out_days", 60)):
        values = result[key]
        if not isinstance(values, list) or not 1 <= len(values) <= maximum:
            raise ValueError(f"{key} requires an explicit nonempty list of at most {maximum} entries.")
        if any(not isinstance(value, str) or not value or value != value.strip() for value in values):
            raise ValueError(f"{key} entries must be nonempty strings without surrounding whitespace.")
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate {key} entries are prohibited.")
        result[key] = list(values)
    if not set(result["datasets"]) <= NATIVE_DATASETS:
        raise ValueError("Wide study datasets must be supported native exchange MBP1 feeds.")
    if result["schemas"] != ["mbp-1"]:
        raise ValueError("The wider frozen design supports exactly schemas=['mbp-1'].")
    if any(value.upper() == "ALL_SYMBOLS" or not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,15}", value)
           for value in result["symbols"]):
        raise ValueError("Use explicit uppercase ticker symbols; ALL_SYMBOLS is disabled.")
    for key in ("training_days", "held_out_days"):
        for value in result[key]:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                raise ValueError("Every study date must be an explicit ISO date.")
            try:
                date = pd.Timestamp(value)
            except ValueError as exc:
                raise ValueError("Invalid explicit study date.") from exc
            if date.weekday() > 4:
                raise ValueError("Study dates must be exchange weekdays.")
        if result[key] != sorted(result[key]):
            raise ValueError(f"{key} must be sorted chronologically.")
    if set(result["training_days"]) & set(result["held_out_days"]):
        raise ValueError("Training and held-out dates must be disjoint.")
    if max(result["training_days"]) >= min(result["held_out_days"]):
        raise ValueError("Every training date must precede every held-out date.")
    if not isinstance(result["start_time_utc"], str) or not re.fullmatch(r"\d{2}:\d{2}:\d{2}", result["start_time_utc"]):
        raise ValueError("start_time_utc requires an explicit HH:MM:SS UTC time.")
    for key, low, high in (("minutes", 1, 60), ("workers", 1, 4),
                          ("record_limit_per_request", 1, 5_000_000),
                          ("max_billable_bytes_per_request", 1, 500_000_000)):
        if type(result[key]) is not int or not low <= result[key] <= high:
            raise ValueError(f"{key} must be an integer from {low} to {high}.")
    budget = result.get("max_cost_usd")
    if budget is None and not result.get("billing_assumed_included", False):
        raise ValueError("max_cost_usd is required when billing_assumed_included is false.")
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, (float, int)) or not math.isfinite(budget) or budget < 0):
        raise ValueError("max_cost_usd must be finite and nonnegative.")
    for key in ("data_dir", "out_dir"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"{key} requires an explicit directory.")
    try:
        start = pd.Timestamp(f"{result['training_days'][0]}T{result['start_time_utc']}Z")
    except ValueError as exc:
        raise ValueError("Invalid start_time_utc.") from exc
    if (start + pd.Timedelta(minutes=result["minutes"])).date() != start.date():
        raise ValueError("Requested aligned window must remain within its explicit date.")
    return result


def read_wide_config(path: Path) -> dict:
    return validate_config(tomllib.loads(path.read_text(encoding="utf-8")))


def requests_for(config: dict) -> list[dict]:
    config = validate_config(config)
    requests = []
    for day in config["training_days"] + config["held_out_days"]:
        start = pd.Timestamp(f"{day}T{config['start_time_utc']}Z")
        for dataset in config["datasets"]:
            requests.append({"dataset": dataset, "schema": "mbp-1",
                             "symbols": list(config["symbols"]), "stype_in": "raw_symbol",
                             "start": start.isoformat(),
                             "end": (start + pd.Timedelta(minutes=config["minutes"])).isoformat(),
                             "limit": config["record_limit_per_request"]})
    return requests


def _signature(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def _study_design(config: dict) -> dict:
    """Billing preferences may change without altering persisted study data."""
    return {key: value for key, value in config.items() if key not in OPTIONAL_CONFIG_KEYS}


def _included_cost_policy(entry: dict) -> dict:
    return dict(entry, estimated_cost_usd=None, incremental_estimated_cost_usd=None,
                cost_check_skipped=True, billing_assumption=BILLING_ASSUMPTION)


def _original_estimate(entries: list[dict]) -> float | None:
    values = [entry.get("estimated_cost_usd") for entry in entries]
    return None if any(value is None for value in values) else sum(values)


def _wide_estimate(request: dict, data_dir: Path, *, billing_assumed_included: bool) -> dict:
    if not billing_assumed_included:
        return _estimate(request, data_dir)
    cached = _read_cached(data_dir, request)
    if cached:
        return _included_cost_policy(dict(cached, estimate_reused=True))
    api = network_data.client()
    full_request = {key: value for key, value in request.items() if key != "limit"}
    count = int(api.metadata.get_record_count(**full_request))
    billable_bytes = int(api.metadata.get_billable_size(**full_request))
    if count < 0 or billable_bytes < 0:
        raise ValueError("Databento returned an invalid wider-study size estimate.")
    return _included_cost_policy({
        "request": request, "expected_records_without_limit": count,
        "estimated_billable_bytes_without_limit": billable_bytes,
        "metadata_count_may_overreport": True, "cache_reused": False,
    })


def _estimate_with_retry(request: dict, data_dir: Path, progress: Callable[[str], None], *,
                         billing_assumed_included: bool = False) -> dict:
    for attempt in range(3):
        try:
            return _wide_estimate(request, data_dir, billing_assumed_included=billing_assumed_included)
        except Exception as exc:
            message = safe_error(exc)
            transient = any(token in message for token in ("504", "503", "502", "429", "ConnectionError", "Timeout"))
            if not transient or attempt == 2:
                raise
            progress(f"Retrying transient metadata failure for {request['start'][:10]} {request['dataset']} (attempt {attempt + 2}/3).")
            time.sleep(1 + 2 * attempt)


def _recent_metadata(progress_path: Path, config: dict, requests: list[dict]) -> list[dict]:
    if not progress_path.exists():
        return []
    stored = json.loads(progress_path.read_text(encoding="utf-8"))
    if stored.get("configuration_sha256") != _signature(config):
        return []
    created = datetime.fromisoformat(stored["created_utc"])
    if created.tzinfo is None or not 0 <= (datetime.now(timezone.utc) - created).total_seconds() <= 3600:
        return []
    expected = {json.dumps(request, sort_keys=True) for request in requests}
    entries = stored.get("requests", [])
    actual = [json.dumps(entry["request"], sort_keys=True) for entry in entries]
    if len(set(actual)) != len(actual) or not set(actual) <= expected:
        raise ValueError("Persisted free metadata progress has an invalid request grid.")
    for entry in entries:
        fields = ["expected_records_without_limit", "estimated_billable_bytes_without_limit"]
        if not config.get("billing_assumed_included", False):
            fields += ["estimated_cost_usd", "incremental_estimated_cost_usd"]
        elif not entry.get("cost_check_skipped") or any(
                entry.get(field) is not None for field in ("estimated_cost_usd", "incremental_estimated_cost_usd")):
            raise ValueError("Persisted metadata has inconsistent included-access policy.")
        for field in fields:
            if field == "estimated_cost_usd" and entry[field] is None and entry.get("cache_reused"):
                # A download made under included-access mode has no price quote.
                # Its existing local file still requires zero new retrieval.
                continue
            if not isinstance(entry[field], (int, float)) or not math.isfinite(entry[field]) or entry[field] < 0:
                raise ValueError("Persisted free metadata progress contains invalid estimates.")
    return entries


def plan_wide_network(config: dict, data_dir: Path, out_dir: Path, *,
                      config_path: Path | None = None,
                      progress: Callable[[str], None] = print) -> dict:
    config = validate_config(config)
    included = config.get("billing_assumed_included", False)
    requests = requests_for(config)
    progress(f"Estimating all {len(requests)} aligned {config['minutes']}-minute MBP1 requests before retrieval.")
    progress_path = out_dir / "metadata_progress.json"
    entries = _recent_metadata(progress_path, config, requests)
    quoted_utc = (json.loads(progress_path.read_text(encoding="utf-8"))["created_utc"]
                  if entries else datetime.now(timezone.utc).isoformat())
    # Completed local DBNs need no new retrieval even when the original free
    # quote remains recent. Preserve its original cost but reserve zero more.
    for index, entry in enumerate(entries):
        cached = _read_cached(data_dir, entry["request"])
        if cached:
            reused = dict(cached, estimate_reused=True, incremental_estimated_cost_usd=0.0)
            entries[index] = _included_cost_policy(reused) if included else reused
    quoted = {json.dumps(entry["request"], sort_keys=True) for entry in entries}
    pending = [request for request in requests if json.dumps(request, sort_keys=True) not in quoted]
    failures = []
    if entries:
        progress(f"Reusing {len(entries)} exact recent free metadata quotes; {len(pending)} remain.")
    state = {"created_utc": quoted_utc,
             "configuration_sha256": _signature(config), "configuration": config,
             "requests": entries, "failures": failures}
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        jobs = {pool.submit(_estimate_with_retry, request, data_dir, progress,
                            billing_assumed_included=included): request for request in pending}
        for future in as_completed(jobs):
            request = jobs[future]
            try:
                entry = future.result()
            except Exception as exc:
                failures.append({"request": request, "error": safe_error(exc)})
                write_json(progress_path, state)
                continue
            entries.append(entry)
            write_json(progress_path, state)
            request = entry["request"]
            progress(f"Estimate {request['start'][:10]} {request['dataset']}: "
                     f"{entry['expected_records_without_limit']:,} records; " +
                     ("cost check skipped (user reports included access)" if included else
                      f"${entry['incremental_estimated_cost_usd']:.6f} incremental"))
    if failures:
        raise ValueError(f"Metadata failed for {len(failures)} configured requests; {len(entries)} successful quotes "
                         "are preserved in metadata_progress.json. No time series was downloaded. Retry the same command.")
    entries.sort(key=lambda entry: (entry["request"]["start"], entry["request"]["dataset"]))
    oversized = [entry for entry in entries
                 if entry["expected_records_without_limit"] >= config["record_limit_per_request"]
                 or entry["estimated_billable_bytes_without_limit"] > config["max_billable_bytes_per_request"]]
    total = None if included else sum(entry["incremental_estimated_cost_usd"] for entry in entries)
    plan = {
        "created_utc": quoted_utc,
        "configuration": config, "configuration_sha256": _signature(config),
        "configuration_source": str(config_path.resolve()) if config_path else None,
        "datasets": config["datasets"], "schemas": config["schemas"], "symbols": config["symbols"],
        "training_days": config["training_days"], "held_out_days": config["held_out_days"],
        "days": config["training_days"] + config["held_out_days"],
        "window_minutes": config["minutes"], "window_start_utc": config["start_time_utc"],
        "authorized_estimate_ceiling_usd": None if included else config["max_cost_usd"],
        "cost_check_skipped": included,
        "billing_assumption": BILLING_ASSUMPTION if included else None,
        "incremental_estimated_cost_usd": total,
        "original_estimated_cost_usd": None if included else _original_estimate(entries),
        "expected_total_records": sum(entry["expected_records_without_limit"] for entry in entries),
        "estimated_total_billable_bytes": sum(entry["estimated_billable_bytes_without_limit"] for entry in entries),
        "ready_to_download": (included or total <= config["max_cost_usd"]) and not oversized,
        "time_series_downloaded": False, "requests": entries,
        "note": ("Cost checks were skipped on the user's included-access assumption. " if included else
                "Metadata cost is an estimate, not an invoice or hard billing guarantee. ") +
                "Counts and byte sizes omit the retrieval cap. No aligned window is shortened or omitted.",
        "metadata_documentation": "https://databento.com/docs/api-historical/metadata/metadata-get-record-count?lang=python",
    }
    write_json(out_dir / "download_plan.json", plan)
    cost_text = "Cost check skipped (user reports included access)" if included else f"All-request estimate: ${total:.6f}"
    progress(f"{cost_text}; {plan['expected_total_records']:,} records; "
             f"{plan['estimated_total_billable_bytes'] / 1e6:.2f} MB billable bytes.")
    if oversized:
        names = ", ".join(f"{e['request']['start'][:10]} {e['request']['dataset']}" for e in oversized)
        raise ValueError(f"Wide requests exceed their configured size bound ({names}); no download performed and no window shortened.")
    if not included and total > config["max_cost_usd"]:
        raise ValueError(f"Estimated ${total:.6f} exceeds configured ceiling ${config['max_cost_usd']:.6f}; no download performed.")
    return plan


def _prepare_download_directories(data_dir: Path, requests: list[dict]) -> None:
    """Create shared parents serially; tolerate brief Windows sync contention."""
    for directory in [data_dir] + [cache_directory(data_dir, request) for request in requests]:
        for attempt in range(3):
            try:
                directory.mkdir(parents=True, exist_ok=True)
                break
            except PermissionError:
                if attempt == 2:
                    raise
                time.sleep(0.05 * (attempt + 1))


def recover_interrupted_wide_downloads(config: dict, data_dir: Path, manifest_path: Path) -> dict:
    """Preserve failed stream prefixes so an explicit later fetch can retry.

    This helper performs no retrieval. It is restricted to the user's included
    access setting and recognized transient failures. Every candidate is fully
    validated before any same-directory atomic rename. Completed caches and
    streams with successful durable markers are retained in place.
    """
    config = validate_config(config)
    if not config.get("billing_assumed_included", False):
        raise ValueError("Interrupted-stream recovery requires explicit billing_assumed_included=true.")
    data_root = Path(data_dir).resolve()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    stored = validate_config(manifest["configuration"])
    if stored != config or manifest.get("configuration_sha256") != _signature(stored):
        raise ValueError("Recovery manifest must match the exact configured study and checksum.")
    expected = {json.dumps(request, sort_keys=True) for request in requests_for(config)}
    entries = manifest.get("requests", [])
    actual = [json.dumps(entry["request"], sort_keys=True) for entry in entries]
    if len(set(actual)) != len(actual) or not set(actual) <= expected:
        raise ValueError("Recovery manifest must contain only unique requests from the exact configured grid.")
    report = {"prepared_for_retry": [], "export_recovery": [],
              "skipped_complete_cache": [], "no_partial_file": []}
    pending = []

    def checked_path(path: Path, directory: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(data_root) or resolved.parent != directory:
            raise ValueError("Recovery paths must resolve inside their exact request cache and data directory.")
        return resolved

    for entry in entries:
        if entry.get("complete"):
            continue
        request = entry["request"]
        error = str(entry.get("error", "")).lower()
        transient = any(token in error for token in (
            "response ended prematurely", "timeout", "connectionerror", "502", "503", "504"))
        if not transient:
            raise ValueError("Recovery refused a nontransient or unexplained failed request; existing files were preserved.")
        directory = cache_directory(data_root, request).resolve()
        if not directory.is_relative_to(data_root) or directory == data_root:
            raise ValueError("Recovery cache directory resolves outside the intended data directory.")
        dbn = checked_path(directory / "sample.dbn.zst", directory)
        if "data_path" in entry and checked_path(Path(entry["data_path"]), directory) != dbn:
            raise ValueError("Failed manifest data_path differs from the exact request cache path.")
        request_path = checked_path(directory / "request.json", directory)
        if not request_path.exists() or json.loads(request_path.read_text(encoding="utf-8")) != request:
            raise ValueError("Recovery requires an exact matching request.json; existing files were preserved.")
        fetch_path = checked_path(directory / "fetch.json", directory)
        receipt = json.loads(fetch_path.read_text(encoding="utf-8")) if fetch_path.exists() else None
        if receipt is not None and receipt.get("request") != request:
            raise ValueError("Recovery fetch receipt does not match its exact request.")
        if receipt is not None and receipt.get("complete"):
            report["skipped_complete_cache"].append(request)
            continue
        marker = checked_path(directory / "stream_complete.json", directory)
        if marker.exists():
            marker_value = json.loads(marker.read_text(encoding="utf-8"))
            if marker_value.get("request") != request:
                raise ValueError("Recovery found a successful-stream marker for another request; files were preserved.")
            if not dbn.exists():
                raise ValueError("Successful-stream marker has no DBN; files were preserved.")
            report["export_recovery"].append(request)
            continue
        if not dbn.exists() and receipt is None:
            report["no_partial_file"].append(request)
            continue
        pending.append((request, dbn if dbn.exists() else None, fetch_path if receipt is not None else None))

    # No filesystem mutation occurs until every failed candidate passes checks.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for request, dbn, receipt in pending:
        suffix = stamp + "." + uuid4().hex[:12]
        preserved = []
        for source, name in ((dbn, f"sample.interrupted.{suffix}.dbn.zst"),
                             (receipt, f"fetch.interrupted.{suffix}.json")):
            if source is not None:
                destination = source.with_name(name)
                checked_path(destination, source.parent)
                source.rename(destination)
                preserved.append(str(destination))
        report["prepared_for_retry"].append({"request": request, "preserved_paths": preserved})
    return report


def fetch_wide_network(config: dict, data_dir: Path, out_dir: Path, *,
                       config_path: Path | None = None,
                       plan: dict | None = None,
                       download_workers: int | None = None,
                       progress: Callable[[str], None] = print) -> dict:
    config = validate_config(config)
    if download_workers is not None and (type(download_workers) is not int or not 1 <= download_workers <= 16):
        raise ValueError("download_workers must be an integer from 1 to 16.")
    transport_workers = config["workers"] if download_workers is None else download_workers
    included = config.get("billing_assumed_included", False)
    if plan is None:
        plan = plan_wide_network(config, data_dir, out_dir, config_path=config_path, progress=progress)
    else:
        # An explicitly supplied recent metadata quote avoids repeating all
        # free metadata calls between a reviewed plan and the immediate pull.
        # The original quote and every aligned request remain unchanged.
        if plan.get("configuration") != config or plan.get("configuration_sha256") != _signature(config):
            raise ValueError("Supplied download plan differs from the frozen configuration.")
        created = datetime.fromisoformat(plan["created_utc"])
        if created.tzinfo is None or not 0 <= (datetime.now(timezone.utc) - created).total_seconds() <= 3600:
            raise ValueError("Supplied metadata quote must be timezone-aware and no more than one hour old.")
        expected = {json.dumps(request, sort_keys=True) for request in requests_for(config)}
        actual = [json.dumps(entry["request"], sort_keys=True) for entry in plan["requests"]]
        if len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != expected:
            raise ValueError("Supplied plan does not contain the exact frozen request grid.")
        for entry in plan["requests"]:
            if not 0 <= entry["expected_records_without_limit"] < config["record_limit_per_request"]:
                raise ValueError("Supplied plan exceeds its record bound.")
            if not 0 <= entry["estimated_billable_bytes_without_limit"] <= config["max_billable_bytes_per_request"]:
                raise ValueError("Supplied plan exceeds its byte bound.")
            if included:
                if not entry.get("cost_check_skipped") or any(
                        entry.get(field) is not None for field in ("estimated_cost_usd", "incremental_estimated_cost_usd")):
                    raise ValueError("Supplied plan has inconsistent included-access policy.")
            else:
                for field in ("estimated_cost_usd", "incremental_estimated_cost_usd"):
                    if field == "estimated_cost_usd" and entry[field] is None and entry.get("cache_reused"):
                        continue
                    if not math.isfinite(entry[field]) or entry[field] < 0:
                        raise ValueError("Supplied plan has invalid cost metadata.")
        if not plan.get("ready_to_download"):
            raise ValueError("Supplied plan is not ready for retrieval.")
        if included:
            if not plan.get("cost_check_skipped") or plan.get("incremental_estimated_cost_usd") is not None:
                raise ValueError("Supplied plan has inconsistent included-access policy.")
            progress("Reusing the exact recent all-request size metadata; cost check skipped.")
        else:
            total = sum(entry["incremental_estimated_cost_usd"] for entry in plan["requests"])
            if total > config["max_cost_usd"] or not math.isclose(
                    total, plan["incremental_estimated_cost_usd"], abs_tol=1e-9):
                raise ValueError("Supplied plan fails the configured budget check.")
            progress(f"Reusing the exact recent all-request metadata quote: ${total:.6f}.")
    manifest = dict(plan, requests=[], complete=False, time_series_downloaded=False,
                    download_workers=transport_workers)
    write_json(out_dir / "data_manifest.json", manifest)
    _prepare_download_directories(data_dir, [entry["request"] for entry in plan["requests"]])
    with ThreadPoolExecutor(max_workers=transport_workers) as pool:
        jobs = {pool.submit(_download_entry, entry, data_dir): entry for entry in plan["requests"]}
        for future in as_completed(jobs):
            entry = jobs[future]
            try:
                result = future.result()
            except Exception as exc:
                result = dict(entry, complete=False, error=safe_error(exc))
            if included:
                result = _included_cost_policy(result)
            manifest["requests"].append(result)
            manifest["time_series_downloaded"] |= bool(result.get("rows"))
            write_json(out_dir / "data_manifest.json", manifest)
            request = result["request"]
            progress(f"{'Cached' if result.get('cache_reused') else 'Fetched'} "
                     f"{request['start'][:10]} {request['dataset']}: "
                     f"{result.get('rows', 0):,} rows; complete={result.get('complete', False)}")
    manifest["requests"].sort(key=lambda entry: (entry["request"]["start"], entry["request"]["dataset"]))
    manifest["complete"] = len(manifest["requests"]) == len(plan["requests"]) and all(
        entry.get("complete", False) for entry in manifest["requests"])
    manifest["downloaded_rows"] = sum(entry.get("rows", 0) for entry in manifest["requests"])
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(out_dir / "data_manifest.json", manifest)
    if not manifest["complete"]:
        raise ValueError("Wide aligned download is incomplete. Inspect data_manifest.json; analysis was not started.")
    return manifest


def load_wide_frames(manifest_path: Path, *, expected_config: dict | None = None) -> dict[tuple[str, str, str], pd.DataFrame]:
    """Load exactly the persisted requested grid; reject altered or capped panels."""
    import databento as db

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError("Only a complete wider network manifest can be analyzed.")
    config = validate_config(manifest["configuration"])
    if manifest.get("configuration_sha256") != _signature(config):
        raise ValueError("Stored wide configuration checksum does not match its provenance.")
    if expected_config is not None and _study_design(config) != _study_design(validate_config(expected_config)):
        raise ValueError("Requested analysis configuration differs from the downloaded frozen study.")
    expected = {json.dumps(request, sort_keys=True) for request in requests_for(config)}
    entries = manifest["requests"]
    actual = [json.dumps(entry["request"], sort_keys=True) for entry in entries]
    if len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Wide manifest must contain exactly every configured venue/day request, with no extras or duplicates.")
    frames = {}
    for entry in entries:
        request = entry["request"]
        if not entry.get("complete") or entry.get("record_limit_reached"):
            raise ValueError("Incomplete or capped requests cannot enter the wider panel.")
        frame = db.DBNStore.from_file(entry["data_path"]).to_df(
            price_type="float", pretty_ts=True, map_symbols=True)
        if not 0 < len(frame) < request["limit"] or len(frame) != entry["rows"]:
            raise ValueError("Actual wider DBN is empty, capped, or inconsistent with its completed manifest.")
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None or frame.index.hasnans:
            raise ValueError("Wide DBN requires valid timezone-aware receive timestamps.")
        start, end = pd.Timestamp(request["start"]), pd.Timestamp(request["end"])
        if (frame.index < start).any() or (frame.index >= end).any():
            raise ValueError("Wide DBN has receive timestamps outside its requested half-open window.")
        if "symbol" not in frame or set(frame["symbol"].dropna().unique()) != set(request["symbols"]):
            raise ValueError("Wide DBN does not contain exactly the requested symbol set.")
        frame.attrs.update(requested_start=request["start"], requested_end=request["end"])
        frames[(request["dataset"], request["schema"], request["start"][:10])] = frame
    return frames
