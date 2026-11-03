from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import math

import pandas as pd

from .config import ROOT, load_key, read_request, safe_error, write_json


def client():
    import databento as db
    return db.Historical(key=load_key())


def doctor(output: Path) -> dict:
    api = client()
    datasets = api.metadata.list_datasets()
    result = {
        "key_configured": True,
        "catalogue_received": True,
        "authenticated_account_check": False,
        "datasets_in_catalogue": datasets,
        "note": "Catalogue/schema listings are not proof of historical-download entitlement.",
        "feeds": {},
        "time_series_downloaded": False,
    }
    try:
        jobs = api.batch.list_jobs(since=datetime.now(timezone.utc) - timedelta(days=1), short=True)
        result["authenticated_account_check"] = True
        result["recent_batch_job_count"] = len(jobs)
    except Exception as exc:
        result["account_check_error"] = safe_error(exc)
    candidates = ["EQUS.MINI", "XNAS.ITCH", "XNYS.PILLAR", "ARCX.PILLAR", "BATS.PITCH", "EDGX.PITCH"]
    for dataset in candidates:
        if dataset not in datasets:
            continue
        entry = {}
        try:
            entry["schemas"] = api.metadata.list_schemas(dataset=dataset)
            entry["range"] = api.metadata.get_dataset_range(dataset=dataset)
        except Exception as exc:
            entry["error"] = safe_error(exc)
        result["feeds"][dataset] = entry
    write_json(output, result)
    return result


def estimate(config_path: Path, output: Path) -> dict:
    request = read_request(config_path)
    api = client()
    schemas = api.metadata.list_schemas(dataset=request["dataset"])
    if request["schema"] not in schemas:
        raise ValueError(f"{request['dataset']} does not support {request['schema']}.")
    cost = float(api.metadata.get_cost(**request))
    if not math.isfinite(cost) or cost < 0:
        raise ValueError("Databento returned an invalid cost estimate.")
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "request": request,
        "estimated_cost_usd": cost,
        "time_series_downloaded": False,
        "warning": "Estimate is not a billing guarantee. Actual billing follows bytes sent. No download has occurred.",
    }
    write_json(output, result)
    return result


def fetch(plan_path: Path, max_cost_usd: float, output: Path) -> dict:
    if not math.isfinite(max_cost_usd) or max_cost_usd < 0:
        raise ValueError("Budget must be finite and nonnegative.")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    request = plan["request"]
    # Revalidate even a hand-edited plan before any paid call.
    allowed = {"dataset", "schema", "symbols", "start", "end", "limit", "stype_in"}
    if set(request) != allowed or request["stype_in"] != "raw_symbol":
        raise ValueError("Plan has invalid request fields. Generate it with the plan command.")
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "validate.toml"
        fields = {k: v for k, v in request.items() if k != "stype_in"}
        path.write_text("\n".join(f"{k} = {json.dumps(v)}" for k, v in fields.items()), encoding="utf-8")
        read_request(path)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "sample.dbn.zst").exists() and not (output / "fetch.json").exists() and not (output / "sample.csv").exists():
        if not (output / "request.json").exists():
            raise ValueError("Existing download has no request manifest; use analyze on the local file or a fresh output directory.")
        try:
            saved = json.loads((output / "request.json").read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ValueError("Existing download has an unreadable request manifest; no new download was attempted.") from exc
        if not isinstance(saved, dict) or not {"request", "max_cost_usd", "estimated_cost_usd"}.issubset(saved):
            raise ValueError("Existing download has an invalid request manifest; no new download was attempted.")
        if saved["request"] != request:
            raise ValueError("Cached download belongs to a different request.")
        import databento as db
        store = db.DBNStore.from_file(output / "sample.dbn.zst")
        return _export_download(store, output, request, saved["max_cost_usd"], saved["estimated_cost_usd"], recovered=True)
    if any((output / name).exists() for name in ("sample.dbn.zst", "fetch.json", "sample.csv")):
        raise ValueError("Output already has a download. Use a fresh output directory to avoid repeat billing or overwriting.")
    api = client()
    current_cost = float(api.metadata.get_cost(**request))
    if not math.isfinite(current_cost) or current_cost < 0 or current_cost > max_cost_usd:
        raise ValueError(f"Download blocked: current estimate ${current_cost:.6f} exceeds authorized budget ${max_cost_usd:.6f}.")
    # The sole billable operation. Keep parameters identical to the estimate.
    write_json(output / "request.json", {"request": request, "max_cost_usd": max_cost_usd, "estimated_cost_usd": current_cost})
    store = api.timeseries.get_range(**request, path=output / "sample.dbn.zst")
    return _export_download(store, output, request, max_cost_usd, current_cost)


def _export_download(store, output: Path, request: dict, max_cost_usd: float, current_cost: float, *, recovered: bool = False) -> dict:
    frame = store.to_df(price_type="float", pretty_ts=True, map_symbols=True)
    frame.to_csv(output / "sample.csv", index=True)
    result = {
        "request": request,
        "estimated_cost_usd": current_cost,
        "authorized_budget_usd": max_cost_usd,
        "rows": len(frame),
        "record_limit_reached": len(frame) >= request["limit"],
        "dataset": str(store.dataset),
        "schema": str(store.schema),
        "columns": list(frame.columns),
        "data_path": str((output / "sample.dbn.zst").resolve()),
        "recovered_from_local_cache": recovered,
        "note": "Estimated cost, not an invoice. A record-limited sample may end before the requested end time.",
    }
    write_json(output / "fetch.json", result)
    return result


def load_file(path: Path) -> tuple[pd.DataFrame, str | None, str | None]:
    if ".dbn" in path.suffixes:
        import databento as db
        store = db.DBNStore.from_file(path)
        return store.to_df(price_type="float", pretty_ts=True, map_symbols=True), str(store.dataset), str(store.schema)
    return pd.read_csv(path, keep_default_na=False), None, None
