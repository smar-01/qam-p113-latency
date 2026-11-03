"""Native MBP-1 midpoint measurements for the frozen multi-event registry."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .aapl_event_trial import (audit_matches, dump, file_sha, observed_seconds,
                              signature, venue_metrics, window_events)
from .event_midpoints import NativeMidpoints, match_events
from .latency import summarize_events
from .latency_data import _metadata_check, normalize_frame
from .multi_event_stats import META, summarize_multi_events
from .multi_event_sources import decode_group

DAY_FILES = ("pair_metrics.csv", "venue_metrics.csv", "metrics.csv", "quality.csv", "matching_audit.csv")
CODE = ("src/colonylab/latency.py", "src/colonylab/latency_data.py", "src/colonylab/event_midpoints.py",
        "src/colonylab/aapl_event_trial.py", "src/colonylab/multi_event_sources.py", "src/colonylab/multi_event_pipeline.py")


def empty_path(start, end):
    return NativeMidpoints(np.array([], np.int64), np.array([], np.int64), np.array([], bool),
                           np.array([], np.int64), start, end)


def decode_sources(entries, group, protocol):
    return decode_group(entries, group, protocol)


def aggregate_metrics(pair_frame, venue_frame, trial, protocol, clock, age):
    """Require the same complete venue/pair universe for every headline row."""
    result = []
    metadata = {name: trial.get(name) for name in META}
    expected_pairs = len(protocol["datasets"]) * (len(protocol["datasets"]) - 1)
    intended_pairs = {(a, b) for a in protocol["datasets"] for b in protocol["datasets"] if a != b}
    for window in protocol["windows"]:
        name = window["name"]
        venues = venue_frame.loc[venue_frame.window.eq(name) & venue_frame.shock_threshold_bp.eq(protocol["primary_shock_threshold_bp"])]
        reasons = []
        if venues.duplicated("venue").any():
            raise ValueError("Duplicate venue identities in metric aggregation")
        if set(venues.venue) != set(protocol["datasets"]):
            reasons.append("missing_venue_rows")
        bad_coverage = (~np.isfinite(venues.coverage_fraction.to_numpy(float))
                        | venues.coverage_fraction.lt(protocol["minimum_coverage_fraction"])
                        | venues.coverage_fraction.gt(1))
        low = venues.loc[bad_coverage, "venue"].tolist()
        if low:
            reasons.append("insufficient_book_coverage:" + ",".join(sorted(low)))
        observable = not reasons
        for horizon in protocol["horizons_ms"]:
            pairs = pair_frame.loc[pair_frame.window.eq(name) & pair_frame.horizon_ms.eq(horizon)]
            pair_reasons = list(reasons)
            if pairs.duplicated(["leader", "follower"]).any():
                raise ValueError("Duplicate directed pair identities in metric aggregation")
            if set(pairs[["leader", "follower"]].itertuples(index=False, name=None)) != intended_pairs:
                pair_reasons.append("missing_directed_pairs")
            rate_reasons = list(pair_reasons)
            if pairs.n_eligible.le(0).any():
                rate_reasons.append("no_complete_eligible_events_for_pair")
            sparse = pairs.n_matched.lt(protocol["minimum_matched_events"]) | pairs.median_us.isna()
            if sparse.any():
                pair_reasons.append("insufficient_pair_matches:" + str(int(sparse.sum())))
            supported = not pair_reasons
            rate_supported = not rate_reasons
            eligible, matched = int(pairs.n_eligible.sum()), int(pairs.n_matched.sum())
            values = {"latency_us": (float(pairs.median_us.mean()) if supported else np.nan, supported, ";".join(pair_reasons)),
                      "match_rate": (matched / eligible if rate_supported and eligible else np.nan,
                                     rate_supported and bool(eligible), ";".join(rate_reasons)),
                      "shock_rate": (float(venues.clustered_shocks_per_minute.mean()) if observable else np.nan,
                                     observable, ";".join(reasons))}
            for metric, (value, complete, reason) in values.items():
                result.append({**metadata, "clock": clock, "age_ms": age, "window": name,
                               "horizon_ms": horizon, "metric": metric, "value": value,
                               "complete": complete, "reason": reason,
                               "expected_pairs": expected_pairs, "eligible_pair_events": eligible,
                               "matched_pair_events": matched,
                               "minimum_venue_coverage": float(venues.coverage_fraction.min()) if len(venues) else 0.})
    return result


def process_group(task):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    started = time.perf_counter()
    directory, protocol, group = Path(task["directory"]), task["protocol"], task["group"]
    entries = task["entries"]
    # Hash even a reused checkpoint's input sources before returning it.
    for entry in entries:
        if entry.get("complete") and entry.get("data_path"):
            if file_sha(entry["data_path"]) != entry["content_sha256"]:
                raise ValueError("Acquired bytes changed before group analysis")
    marker_path = directory / "complete.json"
    if marker_path.is_file():
        old = json.loads(marker_path.read_text())
        if old["identity"] == task["identity"] and all(file_sha(directory / name) == old["files"][name] for name in DAY_FILES):
            return {"group_id": group["group_id"], "reused": True}
    paths, quality = decode_sources(entries, group, protocol)
    pairs, venues, metrics, audits = [], [], [], []
    trials = [row for row in protocol["trials"] if row["trial_id"] in group["trial_ids"]]
    for clock in protocol["clocks"]:
        for age in protocol["quote_age_ms"]:
            local_pairs, local_venues = [], []
            windows_by_trial = {}
            for trial in trials:
                anchor = pd.Timestamp(trial["anchor_utc"]).value
                windows = [{**row, "start_ns": anchor + row["start_minute"] * 60_000_000_000,
                            "end_ns": anchor + row["end_minute"] * 60_000_000_000} for row in protocol["windows"]]
                windows_by_trial[trial["trial_id"]] = windows
                for dataset in protocol["datasets"]:
                    rows = venue_metrics(paths[(dataset, age, clock)], windows, protocol)
                    local_venues.extend({"trial_id": trial["trial_id"], "clock": clock, "age_ms": age,
                                         "venue": dataset, **row} for row in rows)
            horizon = int(max(protocol["horizons_ms"]) * 1e6)
            for leader in protocol["datasets"]:
                a = paths[(leader, age, clock)]
                for follower in protocol["datasets"]:
                    if follower == leader:
                        continue
                    b = paths[(follower, age, clock)]
                    events = match_events(a, b, max_horizon_ns=horizon)
                    events.attrs.update(clock=clock, metric=protocol["measurement"])
                    audited = audit_matches(a, b, events, horizon)
                    audits.append({"source_group_id": group["group_id"], "clock": clock, "age_ms": age,
                                   "leader": leader, "follower": follower, **audited})
                    for trial in trials:
                        for window in windows_by_trial[trial["trial_id"]]:
                            selected = window_events(events, window["start_ns"], window["end_ns"])
                            for h in protocol["horizons_ms"]:
                                row = summarize_events(selected, int(h * 1e6))
                                local_pairs.append({"trial_id": trial["trial_id"], "clock": clock, "age_ms": age,
                                                    "leader": leader, "follower": follower, "window": window["name"], **row})
            pair_df, venue_df = pd.DataFrame(local_pairs), pd.DataFrame(local_venues)
            for trial in trials:
                metrics.extend(aggregate_metrics(pair_df.loc[pair_df.trial_id.eq(trial["trial_id"])],
                    venue_df.loc[venue_df.trial_id.eq(trial["trial_id"])], trial, protocol, clock, age))
            pairs.extend(local_pairs)
            venues.extend(local_venues)
    for name, frame in zip(DAY_FILES, [pd.DataFrame(pairs), pd.DataFrame(venues), pd.DataFrame(metrics),
                                      pd.DataFrame(quality), pd.DataFrame(audits)]):
        frame.to_csv(directory / name, index=False)
    dump(marker_path, {"identity": task["identity"], "files": {name: file_sha(directory / name) for name in DAY_FILES},
                       "elapsed_seconds": time.perf_counter() - started})
    return {"group_id": group["group_id"], "reused": False, "trials": len(trials),
            "seconds": round(time.perf_counter() - started, 2)}


def analyze(root, out_dir, *, workers=2, progress=print, available_only=False):
    root, out = Path(root).resolve(), Path(out_dir).resolve()
    protocol = json.loads((out / "protocol.json").read_text())
    manifest = json.loads((out / "data_manifest.json").read_text())
    acquisition_done = bool(manifest.get("acquisition_complete_for_available_requests", manifest.get("complete")))
    if acquisition_done and any(not row.get("complete") and row.get("status") != "unavailable" for row in manifest["requests"]):
        raise ValueError("Acquisition completion claim conflicts with an incomplete available source")
    if not acquisition_done and not available_only:
        raise ValueError("Finish every available acquisition before calculating event results")
    expected_requests = protocol.get("requests") or [{"dataset": dataset, "schema": "mbp-1", "symbols": ["AAPL"], "stype_in": "raw_symbol",
                          "start": group["start"], "end": group["end"], "limit": protocol["record_limit_per_request"]}
                         for group in protocol["source_groups"] for dataset in protocol["datasets"]]
    actual_requests = [signature(row["request"]) for row in manifest["requests"]]
    expected_tokens = {signature(row) for row in expected_requests}
    if (len(actual_requests) != len(set(actual_requests)) or not set(actual_requests) <= expected_tokens
            or (acquisition_done and set(actual_requests) != expected_tokens)):
        raise ValueError("Acquired request registry differs from the exact frozen source grid")
    code = [{"path": name, "sha256": file_sha(root / name)} for name in CODE]
    tasks = []
    for group in protocol["source_groups"]:
        group_tokens = set(group.get("request_sha256", []))
        entries = [row for row in manifest["requests"] if signature(row["request"]) in group_tokens] if group_tokens else [
            row for row in manifest["requests"] if row["request"]["start"] == group["start"] and row["request"]["end"] == group["end"]]
        expected_count = len(group_tokens) or len(protocol["datasets"])
        if available_only and (len(entries) < expected_count
                               or any(not row.get("complete") and row.get("status") != "unavailable" for row in entries)):
            continue
        if len(entries) != expected_count or {row["request"]["dataset"] for row in entries} != set(protocol["datasets"]):
            raise ValueError("Acquisition registry is missing a planned venue request")
        directory = out / "groups" / group["group_id"]
        directory.mkdir(parents=True, exist_ok=True)
        identity = {"group": group, "protocol_sha256": signature(protocol), "code": code,
                    "sources": [{"request": row["request"], "sha256": row.get("content_sha256"),
                                 "rows": row.get("rows"), "status": row.get("status"), "complete": row.get("complete")} for row in entries]}
        tasks.append({"group": group, "entries": entries, "protocol": protocol,
                      "directory": str(directory), "identity": identity})
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for n, future in enumerate(as_completed([executor.submit(process_group, task) for task in tasks]), 1):
            progress(f"Analyzed {n}/{len(tasks)} source windows: {future.result()}", flush=True)
    if not acquisition_done:
        return {"status": "partial_checkpoints_only", "completed_source_windows": len(tasks),
                "planned_source_windows": len(protocol["source_groups"]), "event_results_published": False}
    for name in DAY_FILES:
        frames = []
        for task in tasks:
            directory = Path(task["directory"])
            marker = json.loads((directory / "complete.json").read_text())
            if marker["identity"] != task["identity"] or file_sha(directory / name) != marker["files"][name]:
                raise ValueError("Source-group checkpoint identity changed")
            frames.append(pd.read_csv(directory / name))
        pd.concat(frames, ignore_index=True).to_csv(out / name, index=False)
    metrics = pd.read_csv(out / "metrics.csv")
    results = summarize_multi_events(metrics, protocol)
    for key in ("event_changes", "event_excess", "class_summary", "class_comparisons"):
        pd.DataFrame(results[key]).to_csv(out / (key + ".csv"), index=False)
    audit = pd.read_csv(out / "matching_audit.csv")
    summary = {"status": "completed", "measurement": protocol["measurement"],
               "event_anchors": sum(row["role"] == "event" for row in protocol["trials"]),
               "event_bundles": len({row["bundle_id"] for row in protocol["trials"] if row["role"] == "event"}),
               "control_trials": sum(row["role"] == "control" for row in protocol["trials"]),
               "source_windows": len(tasks), "raw_requests": len(manifest["requests"]),
               "raw_records": sum(row.get("rows", 0) for row in manifest["requests"]),
               "matching_audit_cases": int(audit.sampled_cases.sum()), "matching_mismatches": int(audit.mismatches.sum()),
               "results": results, "limitations": protocol["limitations"]}
    dump(out / "summary.json", summary)
    dump(out / "analysis_provenance.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
         "protocol_sha256": file_sha(out / "protocol.json"), "data_manifest_sha256": file_sha(out / "data_manifest.json"),
         "computation_code": code + [{"path": "src/colonylab/multi_event_stats.py", "sha256": file_sha(root / "src/colonylab/multi_event_stats.py")}],
         "outputs": {name: file_sha(out / name) for name in (*DAY_FILES, "summary.json", "event_changes.csv", "event_excess.csv", "class_summary.csv", "class_comparisons.csv")}})
    return summary
