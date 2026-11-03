"""Offline ADF/KPSS diagnostics on fixed, unbroken native price paths.

This module reads the audited event study, but never modifies it or obtains
new market data. Selection depends on quote support, never a test outcome.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from .aapl_event_trial import dump, file_sha, signature
from .multi_event_sources import decode_group


CODE = ("src/colonylab/stationarity.py", "src/colonylab/stationarity_pipeline.py",
        "scripts/run_stationarity.py", "src/colonylab/multi_event_sources.py",
        "src/colonylab/latency_data.py", "src/colonylab/event_midpoints.py",
        "src/colonylab/latency.py", "src/colonylab/aapl_event_trial.py")
FILES = ("protocol.json", "requests.csv", "runs.csv", "tests.csv", "series.npz", "results.json")


def series_identity(times, prices):
    values = np.asarray([times, prices], dtype="<i8")
    return "s" + hashlib.sha256(values.tobytes()).hexdigest()


def validate_source_study(root, source):
    """Require the precise previously audited inputs and unchanged computation."""
    audit = json.loads((source / "result_audit.json").read_text())
    if audit.get("passed") is not True or audit.get("status") != "passed":
        raise ValueError("Stationarity diagnostics require a passed source-study audit")
    for name, digest in audit["output_sha256"].items():
        if file_sha(source / name) != digest:
            raise ValueError(f"Audited source-study artifact changed: {name}")
    for row in audit["computation_code"]:
        if file_sha(root / row["path"]) != row["sha256"]:
            raise ValueError(f"Audited source-study computation changed: {row['path']}")
    protocol = json.loads((source / "protocol.json").read_text())
    manifest = json.loads((source / "data_manifest.json").read_text())
    if not manifest.get("complete") or file_sha(source / "protocol.json") != audit["protocol_sha256"]:
        raise ValueError("Incomplete acquisition or changed audited source protocol")
    if len({signature(row["request"]) for row in manifest["requests"]}) != len(manifest["requests"]):
        raise ValueError("Duplicate source requests")
    expected = {signature(row) for row in protocol["requests"]}
    if {signature(row["request"]) for row in manifest["requests"]} != expected:
        raise ValueError("Source acquisition differs from its frozen request grid")
    return protocol, manifest


def extract_group(task):
    from threadpoolctl import threadpool_limits
    from .stationarity import sample_contiguous
    threadpool_limits(limits=1)
    group, source_protocol, protocol = task["group"], task["source_protocol"], task["protocol"]
    paths, _ = decode_group(task["entries"], group, source_protocol)
    requests, arrays = [], {}
    for trial in task["trials"]:
        anchor = pd.Timestamp(trial["anchor_utc"]).value
        for window in protocol["windows"]:
            first = anchor + window["start_minute"] * 60_000_000_000
            stop = anchor + window["end_minute"] * 60_000_000_000
            for (venue, age, clock), path in sorted(paths.items()):
                sampled = sample_contiguous(path, first, stop, step_ns=protocol["step_ns"])
                times, prices = sampled.pop("times_ns"), sampled.pop("price2")
                sampled.pop("segments")
                token = series_identity(times, prices)
                values = np.array([times, prices], dtype=np.int64)
                if token in arrays and not np.array_equal(arrays[token], values):
                    raise ValueError("Diagnostic series identity collision")
                arrays[token] = values
                context = {"source_group_id": group["group_id"], "trial_id": trial["trial_id"],
                    "event_id": trial["event_id"], "event_class": trial["event_type"],
                    "bundle_id": trial["bundle_id"], "role": trial["role"],
                    "anchor_utc": trial["anchor_utc"], "window": window["name"],
                    "venue": venue, "clock": clock, "age_ms": age, "start_ns": first, "end_ns": stop}
                requests.append({"request_id": signature(context), **context, "series_id": token, **sampled,
                    "longer_run_600_points": len(times) >= protocol["longer_run_points"]})
    return {"group_id": group["group_id"], "requests": requests, "arrays": arrays}


def _test_series(task):
    from threadpoolctl import threadpool_limits
    from .stationarity import test_pairs
    threadpool_limits(limits=1)
    token, prices, minimum, alpha = task
    return [{"series_id": token, **row} for row in test_pairs(prices, min_observations=minimum, alpha=alpha)]


def csv_rows(path, rows):
    converted = [{key: json.dumps(value, sort_keys=True, allow_nan=False) if isinstance(value, (dict, list)) else value
                  for key, value in row.items()} for row in rows]
    frame = pd.DataFrame(converted)
    # A missing run endpoint must not force other nanosecond timestamps through
    # float64. Construct nullable integers directly from the original values.
    for name in frame.columns:
        if name.endswith("_ns"):
            frame[name] = pd.array([row.get(name) for row in converted], dtype="Int64")
    frame.to_csv(path, index=False)


def summarize(requests, tests):
    joined = pd.DataFrame(requests).merge(pd.DataFrame(tests), on="series_id", validate="many_to_many")
    rows = []
    for key, frame in joined.groupby(["clock", "age_ms", "transformation", "regression"], sort=True):
        unique = frame.drop_duplicates(["series_id", "transformation", "regression"])
        counts = unique.classification.value_counts().to_dict()
        rows.append({"clock": str(key[0]), "age_ms": int(key[1]),
            "transformation": str(key[2]), "regression": str(key[3]),
            "planned_windows": len(frame), "unique_series": len(unique),
            "tested": int(unique.status.eq("tested").sum()),
            "skipped": int(unique.status.ne("tested").sum()),
            "classification_counts": {str(k): int(v) for k, v in counts.items()},
            "adf_holm_rejections": int((unique.adf_pvalue_holm < .05).sum())})
    return rows


def analyze(root, config_path, out_dir, *, workers=3, progress=print):
    root, config_path, out = Path(root).resolve(), Path(config_path).resolve(), Path(out_dir).resolve()
    started = time.perf_counter()
    config = json.loads(config_path.read_text())
    # The method is intentionally fixed before examining its p-values.
    expected = json.loads((root / "aapl_stationarity_protocol.json").read_text())
    if config != expected or config["minimum_observations"] != 300 or config["step_ns"] != 1_000_000_000:
        raise ValueError("Use the frozen one-second, 300-observation diagnostic protocol")
    source = (root / config["source_study_dir"]).resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("Use a separate diagnostic output directory")
    source_protocol, manifest = validate_source_study(root, source)
    inputs = [{"path": str(path.resolve()), "sha256": file_sha(path)} for path in
              (config_path, source / "protocol.json", source / "data_manifest.json",
               source / "result_audit.json", source / "analysis_provenance.json")]
    protocol = {**config, "source_study_dir": str(source),
        "source_protocol_sha256": file_sha(source / "protocol.json"),
        "source_manifest_sha256": file_sha(source / "data_manifest.json"),
        "source_audit_sha256": file_sha(source / "result_audit.json"),
        "datasets": source_protocol["datasets"], "clocks": source_protocol["clocks"],
        "quote_age_ms": source_protocol["quote_age_ms"]}
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "protocol.json", protocol)
    code = [{"path": str((root / name).resolve()), "sha256": file_sha(root / name)} for name in CODE]
    tasks = []
    for group in source_protocol["source_groups"]:
        tokens = set(group["request_sha256"])
        entries = [row for row in manifest["requests"] if signature(row["request"]) in tokens]
        if len(entries) != len(tokens):
            raise ValueError("A source group is missing raw fragments")
        tasks.append({"group": group, "entries": entries, "source_protocol": source_protocol,
            "protocol": protocol, "trials": [row for row in source_protocol["trials"] if row["trial_id"] in group["trial_ids"]]})
    requests, arrays = [], {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(extract_group, task) for task in tasks]
        for n, future in enumerate(as_completed(futures), 1):
            result = future.result()
            requests.extend(result["requests"])
            for token, values in result["arrays"].items():
                if token in arrays and not np.array_equal(arrays[token], values):
                    raise ValueError("Cross-group series identity collision")
                arrays[token] = values
            progress(f"Sampled {n}/{len(tasks)} source groups: {result['group_id']}", flush=True)
    requests.sort(key=lambda row: row["request_id"])
    if len({row["request_id"] for row in requests}) != len(requests):
        raise ValueError("Duplicate diagnostic windows")
    expected_windows = len(source_protocol["trials"]) * len(protocol["windows"]) * len(protocol["datasets"]) * len(protocol["clocks"]) * len(protocol["quote_age_ms"])
    if len(requests) != expected_windows:
        raise ValueError("Diagnostic window registry is incomplete")
    csv_rows(out / "requests.csv", requests)
    arrays = dict(sorted(arrays.items()))
    np.savez_compressed(out / "series.npz", **arrays)
    runs = [{"series_id": token, "n_price_points": values.shape[1],
        "start_ns": int(values[0, 0]) if values.shape[1] else None,
        "end_ns": int(values[0, -1]) if values.shape[1] else None,
        "duration_seconds": float((values[0, -1] - values[0, 0]) / 1e9) if values.shape[1] else 0.}
        for token, values in arrays.items()]
    csv_rows(out / "runs.csv", runs)
    tests = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_test_series, (token, values[1], protocol["minimum_observations"], protocol["alpha"]))
                   for token, values in arrays.items()]
        for n, future in enumerate(as_completed(futures), 1):
            tests.extend(future.result())
            if n % 100 == 0 or n == len(futures):
                progress(f"Diagnosed {n}/{len(futures)} unique sampled price runs", flush=True)
    from .stationarity import adjust_adf_holm
    tests = adjust_adf_holm(sorted(tests, key=lambda row: (row["series_id"], row["transformation"], row["regression"])))
    csv_rows(out / "tests.csv", tests)
    results = {"status": "completed", "counts": {"source_groups": len(tasks),
        "diagnostic_windows": len(requests), "unique_series": len(arrays), "unique_test_pairs": len(tests),
        "tested_pairs": sum(row["status"] == "tested" for row in tests),
        "skipped_pairs": sum(row["status"] != "tested" for row in tests),
        "source_raw_requests": len(manifest["requests"]),
        "source_raw_records": sum(row["rows"] for row in manifest["requests"])},
        "protocol": protocol, "summary": summarize(requests, tests), "limitations": protocol["limitations"]}
    dump(out / "results.json", results)
    for row in inputs + code:
        if file_sha(row["path"]) != row["sha256"]:
            raise ValueError("Input or code changed during stationarity analysis")
    dump(out / "analysis_provenance.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "statsmodels", "databento")},
        "inputs": inputs, "computation_code": code, "outputs": {name: file_sha(out / name) for name in FILES},
        "no_new_market_data_requests": True, "source_study_modified": False})
    return results
