"""Bounded, resumable checks of historical US-equity data access.

The catalog describes products, not an account's permission to download them.
Only the tiny time-series requests below test actual retrieval. They deliberately
do not attempt a full day of order-book reconstruction or estimate market share.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import date, datetime, time, timedelta, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
from threading import RLock
import time as time_module
from typing import Any

import pandas as pd

from .config import ROOT, safe_error, write_json
from .data import client


US_EQUITY_PREFIXES = {
    "ARCX", "BATS", "BATY", "EDGA", "EDGX", "EPRL", "IEXG", "MEMX",
    "OCEA", "XASE", "XBOS", "XCHI", "XCIS", "XNAS", "XNYS", "XPSX",
    "LTSE", "FINN", "FINY", "EQUS", "DBEQ",
}
PROBE_SCHEMAS = ("mbo", "mbp-1", "trades", "ohlcv-1d")
MAX_AUDIT_BUDGET_USD = 0.10


class BudgetLedger:
    """Atomically reserve estimates before any concurrent billable request."""

    def __init__(self, budget: float):
        self.budget = budget
        self.reserved = 0.0
        self.lock = RLock()

    def reserve(self, estimate: float) -> bool:
        with self.lock:
            if self.reserved + estimate > self.budget:
                return False
            self.reserved += estimate
            return True


def select_equity_datasets(datasets: list[str]) -> list[str]:
    """Keep equity exchange, ATS, TRF and composite feeds; exclude options."""
    return sorted({d for d in datasets if d.split(".", 1)[0] in US_EQUITY_PREFIXES})


def feed_scope(dataset: str) -> str:
    prefix = dataset.split(".", 1)[0]
    if prefix in {"EQUS", "DBEQ"}:
        return "composite_feed_not_an_execution_venue"
    if prefix in {"FINN", "FINY"}:
        return "trade_reporting_facility_not_exchange_book"
    if prefix == "OCEA":
        return "overnight_ats"
    if dataset == "XNAS.BASIC" or dataset.endswith(".NLS"):
        return "combined_exchange_and_off_exchange_reporting_check_product_definition"
    return "individual_exchange_feed"


def make_probe_request(dataset: str, schema: str, day: date, symbols: list[str],
                       intraday_time: str = "14:00:00") -> dict:
    if schema not in PROBE_SCHEMAS:
        raise ValueError("Unsupported audit schema")
    if not 1 <= len(symbols) <= 3 or any(
        not isinstance(s, str) or not s or s != s.strip()
        or s.upper() == "ALL_SYMBOLS" or "," in s for s in symbols
    ):
        raise ValueError("Audit needs one to three explicit ticker symbols")
    clock = time.fromisoformat(intraday_time)
    if clock.tzinfo is not None or clock.microsecond:
        raise ValueError("Intraday clock must be a whole second, interpreted in UTC")
    if schema == "ohlcv-1d":
        start = f"{day.isoformat()}T00:00:00Z"
        end = f"{(day + timedelta(days=1)).isoformat()}T00:00:00Z"
        selected_symbols = symbols
        limit = 100
    else:
        instant = datetime.combine(day, clock, tzinfo=timezone.utc)
        start = instant.isoformat().replace("+00:00", "Z")
        end = (instant + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
        selected_symbols = symbols[:1]
        limit = 2000 if schema == "mbo" else 500
    return {
        "dataset": dataset, "schema": schema, "symbols": selected_symbols,
        "start": start, "end": end, "stype_in": "raw_symbol", "limit": limit,
    }


def schema_availability(dataset_range: dict, request: dict) -> dict:
    limits = dataset_range.get("schema", {}).get(request["schema"], dataset_range)
    result = {"available_start": limits.get("start"), "available_end": limits.get("end")}
    try:
        result["requested_interval_in_range"] = bool(
            pd.Timestamp(limits["start"]) <= pd.Timestamp(request["start"])
            and pd.Timestamp(request["end"]) <= pd.Timestamp(limits["end"])
        )
    except (KeyError, TypeError, ValueError):
        result["requested_interval_in_range"] = None
    return result


def summarize_store(store: Any, request: dict, publishers: list[dict]) -> dict:
    frame = store.to_df(price_type="float", pretty_ts=True, map_symbols=True)
    known = {int(p["publisher_id"]): p for p in publishers if "publisher_id" in p}
    result: dict[str, Any] = {
        "records": len(frame), "returned_dataset": str(store.dataset),
        "returned_schema": str(store.schema), "columns": list(frame.columns),
        "record_limit_reached": len(frame) >= request["limit"],
        "first_timestamp": None, "last_timestamp": None,
        "timestamp_basis": None, "publisher_ids": [], "publisher_mappings": [],
    }
    if len(frame):
        ts = frame.index if isinstance(frame.index, pd.DatetimeIndex) else None
        if ts is None and "ts_event" in frame:
            ts = pd.to_datetime(frame["ts_event"], utc=True)
        if ts is not None:
            result.update(first_timestamp=str(ts.min()), last_timestamp=str(ts.max()),
                          timestamp_basis=frame.index.name or "ts_event")
    if "publisher_id" in frame:
        ids = sorted(int(x) for x in frame["publisher_id"].dropna().unique())
        result["publisher_ids"] = ids
        result["publisher_mappings"] = [known.get(i, {"publisher_id": i}) for i in ids]
    if "action" in frame:
        result["action_counts"] = {str(k): int(v) for k, v in frame["action"].value_counts().items()}
        result["trade_action_records"] = int((frame["action"] == "T").sum())
    if "flags" in frame:
        flags = pd.to_numeric(frame["flags"], errors="coerce").fillna(0).astype("uint64")
        # DBN RecordFlags.F_SNAPSHOT = 32; snapshots initialize a book and are
        # not contemporaneous order additions or executions.
        result["snapshot_records"] = int(((flags & 32) != 0).sum())
        result["flag_counts"] = {str(k): int(v) for k, v in flags.value_counts().items()}
    if request["schema"] == "ohlcv-1d":
        cols = [c for c in ("symbol", "publisher_id", "volume", "open", "close") if c in frame]
        records = frame[cols].reset_index().to_dict(orient="records")
        # JSON serialization below handles Timestamp and numpy scalars.
        result["daily_bars"] = records
        result["volume_interpretation"] = (
            "Product-defined daily trade volume. Composite publisher IDs do not "
            "automatically identify original execution venues; do not sum overlapping feeds."
        )
    return result


def _request_paths(data_dir: Path, request: dict) -> tuple[Path, Path]:
    digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:16]
    stem = f"{request['dataset'].replace('.', '_')}_{request['schema']}_{digest}"
    return data_dir / f"{stem}.dbn.zst", data_dir / f"{stem}.json"


def _replace_report_file(source: Path, target: Path) -> None:
    """Allow short Windows/OneDrive locks without exposing a partial report."""
    for attempt in range(10):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time_module.sleep(0.05 * (attempt + 1))


def _save_reports(result: dict, out_dir: Path) -> None:
    report_path = out_dir / "databento_access.json"
    report_tmp = out_dir / "databento_access.json.tmp"
    write_json(report_tmp, result)
    _replace_report_file(report_tmp, report_path)
    rows = []
    bars = []
    for dataset, feed in result["feeds"].items():
        for probe in feed.get("probes", []):
            flat = {"dataset": dataset, "feed_scope": feed["feed_scope"]}
            flat.update({k: v for k, v in probe.items() if k not in {"request", "daily_bars"}})
            flat["symbols"] = ",".join(probe["request"]["symbols"])
            flat["start"] = probe["request"]["start"]
            flat["end"] = probe["request"]["end"]
            flat["limit"] = probe["request"]["limit"]
            for k, v in list(flat.items()):
                if isinstance(v, (dict, list)):
                    flat[k] = json.dumps(v, default=str)
            rows.append(flat)
            for bar in probe.get("daily_bars", []):
                bars.append(dict(dataset=dataset, feed_scope=feed["feed_scope"], **bar))
    for name, values in (("databento_access.csv", rows), ("databento_access_daily_volumes.csv", bars)):
        columns = list(dict.fromkeys(k for row in values for k in row))
        csv_tmp = out_dir / f"{name}.tmp"
        with csv_tmp.open("w", newline="", encoding="utf-8") as file:
            if columns:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                writer.writerows(values)
        _replace_report_file(csv_tmp, out_dir / name)


def audit_access(*, day: date = date(2026, 10, 23), budget: float = MAX_AUDIT_BUDGET_USD,
                 symbols: list[str] | None = None, data_dir: Path = ROOT / "data/access_audit",
                 out_dir: Path = ROOT / "out/access", api: Any = None,
                 datasets: list[str] | None = None, intraday_time: str = "14:00:00",
                 workers: int = 1, progress: Any = print) -> dict:
    """Query catalog, then make cost-checked bounded requests for US-equity feeds.

    Existing DBNs are decoded locally and reused, without repeating a download.
    A failed retrieval reserves its estimate too, since a partial response may
    already have delivered data. Estimates are not actual billing guarantees.
    """
    if not math.isfinite(budget) or not 0 <= budget <= MAX_AUDIT_BUDGET_USD:
        raise ValueError(f"Audit budget must be between $0 and ${MAX_AUDIT_BUDGET_USD:.2f}")
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Workers must be an integer from 1 to 4")
    symbols = ["AAPL", "MSFT", "SPY"] if symbols is None else symbols
    make_probe_request("XNAS.ITCH", "mbo", day, symbols, intraday_time)  # validate before API
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    injected_api = api is not None
    api = client() if api is None else api
    catalog = list(api.metadata.list_datasets())
    publishers = list(api.metadata.list_publishers())
    selected = select_equity_datasets(catalog)
    if datasets is not None:
        if any(d not in selected for d in datasets):
            raise ValueError("Requested dataset is absent from the US-equity catalog selection")
        selected = sorted(set(datasets))
    # Put the feeds needed for a first cross-venue network ahead of redundant
    # products. Stable order is retained within the rest of the catalog.
    priority = {d: i for i, d in enumerate(("XNAS.ITCH", "XNYS.PILLAR", "IEXG.TOPS", "MEMX.MEMOIR"))}
    selected.sort(key=lambda d: (priority.get(d, 100), d))
    result: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "session_date": str(day),
        "intraday_time_utc": intraday_time,
        "workers": workers,
        "catalog_datasets": catalog, "publishers": publishers, "selected_datasets": selected,
        "authorized_estimated_budget_usd": budget, "estimated_budget_reserved_usd": 0.0,
        "notes": [
            "Catalog/schema/range availability is separate from download permission.",
            "A successful nonempty get_range confirms retrieval only for this tested request/date/schema.",
            "Metadata estimates are not invoices or a guarantee that historical access is free.",
            "MBO may be dominated by initial snapshots; a capped probe is not a complete event window.",
            "Publisher IDs identify dataset/venue combinations, not named firms or necessarily original execution venues in composites.",
            "Daily bars follow each product's trade and session conventions; reconcile before constructing market shares.",
        ],
        "feeds": {},
    }
    ledger = BudgetLedger(budget)
    report_lock = RLock()

    def publish_feed(dataset: str, feed: dict) -> None:
        # Workers own their local dictionaries. Only complete deep copies enter
        # the report, and reservation snapshots are saved before get_range.
        with report_lock:
            result["feeds"][dataset] = deepcopy(feed)
            with ledger.lock:
                result["estimated_budget_reserved_usd"] = ledger.reserved
            _save_reports(result, out_dir)

    _save_reports(result, out_dir)

    def audit_feed(dataset: str) -> None:
        # Production workers have independent SDK clients and HTTP state.
        # An explicitly injected test API remains shared for deterministic tests.
        feed_api = api if injected_api or workers == 1 else client()
        feed: dict[str, Any] = {"feed_scope": feed_scope(dataset), "probes": []}
        publish_feed(dataset, feed)
        try:
            feed["schemas"] = list(feed_api.metadata.list_schemas(dataset=dataset))
            feed["range"] = feed_api.metadata.get_dataset_range(dataset=dataset)
            feed["metadata_status"] = "received"
        except Exception as exc:
            feed.update(metadata_status="error", metadata_error=safe_error(exc))
            publish_feed(dataset, feed)
            progress(f"{dataset}: metadata error")
            return
        for schema in PROBE_SCHEMAS:
            request = make_probe_request(dataset, schema, day, symbols, intraday_time)
            probe: dict[str, Any] = {
                "schema": schema, "request": request, "cost_status": "not_requested",
                "retrieval_status": "not_attempted", "support_status": "supported" if schema in feed["schemas"] else "unsupported",
                **schema_availability(feed["range"], request),
            }
            feed["probes"].append(probe)
            if probe["support_status"] == "unsupported":
                continue
            if probe["requested_interval_in_range"] is False:
                probe["retrieval_status"] = "outside_available_range"
                continue
            dbn_path, manifest_path = _request_paths(data_dir, request)
            try:
                # Only a matching provenance file makes a cache eligible.
                cached = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
                if cached and cached.get("request") == request and dbn_path.exists():
                    import databento as db
                    store = db.DBNStore.from_file(dbn_path)
                    probe.update(summarize_store(store, request, publishers))
                    probe.update(retrieval_status="cached_nonempty" if probe["records"] else "cached_empty",
                                 cost_status="cached_no_new_download", data_path=str(dbn_path.resolve()),
                                 original_estimated_cost_usd=cached.get("estimated_cost_usd"))
                    continue
                if dbn_path.exists():
                    probe["retrieval_status"] = "blocked_unverified_existing_file"
                    continue
                estimated = float(feed_api.metadata.get_cost(**request))
                if not math.isfinite(estimated) or estimated < 0:
                    raise ValueError("Invalid Databento cost estimate")
                probe.update(cost_status="estimated", estimated_cost_usd=estimated)
                if not ledger.reserve(estimated):
                    probe["retrieval_status"] = "blocked_by_budget"
                    continue
                provenance = {
                    "request": request, "estimated_cost_usd": estimated,
                    "authorized_total_estimated_budget_usd": budget,
                    "requested_utc": datetime.now(timezone.utc).isoformat(),
                    "retrieval_status": "started",
                }
                write_json(manifest_path, provenance)
                publish_feed(dataset, feed)
                try:
                    store = feed_api.timeseries.get_range(**request, path=dbn_path)
                    probe.update(summarize_store(store, request, publishers))
                    probe.update(retrieval_status="downloaded_nonempty" if probe["records"] else "downloaded_empty",
                                 data_path=str(dbn_path.resolve()), file_bytes=dbn_path.stat().st_size)
                    provenance.update(retrieval_status=probe["retrieval_status"], records=probe["records"],
                                      completed_utc=datetime.now(timezone.utc).isoformat())
                except Exception as exc:
                    probe.update(retrieval_status="error", retrieval_error=safe_error(exc))
                    provenance.update(retrieval_status="error", retrieval_error=probe["retrieval_error"])
                write_json(manifest_path, provenance)
            except Exception as exc:
                if probe["cost_status"] == "not_requested":
                    probe.update(cost_status="error", cost_error=safe_error(exc))
                else:
                    probe.update(retrieval_status="error", retrieval_error=safe_error(exc))
            finally:
                publish_feed(dataset, feed)
                progress(f"{dataset} {schema}: {probe['retrieval_status']}; rows={probe.get('records', 0)}")
        publish_feed(dataset, feed)

    if workers == 1:
        for dataset in selected:
            audit_feed(dataset)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="feed-access") as pool:
            # Iterating propagates unexpected worker failures to the caller.
            list(pool.map(audit_feed, selected))
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    result["retrieval_status_counts"] = dict(Counter(
        p["retrieval_status"] for f in result["feeds"].values() for p in f["probes"]
    ))
    _save_reports(result, out_dir)
    return result
