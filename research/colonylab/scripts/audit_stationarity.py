"""Independently audit saved price-path stationarity diagnostics, entirely offline.

Recomputes every persisted ADF/KPSS pair and global Holm adjustment; rebuilds
selection from representative native source groups without rerunning the main
event study. The separate result_audit.json records this bounded scope.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILES = {"protocol.json", "requests.csv", "runs.csv", "tests.csv", "series.npz", "results.json"}
SPECS = (("log_level", "c"), ("log_level", "ct"), ("log_return", "c"))
CONTEXT = ("source_group_id", "trial_id", "event_id", "event_class", "bundle_id", "role", "anchor_utc",
           "window", "venue", "clock", "age_ms", "start_ns", "end_ns")
COMPUTATION = {"src/colonylab/stationarity.py", "src/colonylab/stationarity_pipeline.py", "scripts/run_stationarity.py",
               "src/colonylab/multi_event_sources.py", "src/colonylab/latency_data.py", "src/colonylab/event_midpoints.py",
               "src/colonylab/latency.py", "src/colonylab/aapl_event_trial.py"}


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON {value}: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def series_token(values):
    return "s" + hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def same(actual, expected, context):
    if expected is None:
        require(actual is None or actual is pd.NA or isinstance(actual, float) and np.isnan(actual), context + ": expected missing value")
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), context + ": object keys differ")
        for key in expected:
            same(actual[key], expected[key], context + "." + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), context + ": list differs")
        for n, (left, right) in enumerate(zip(actual, expected)):
            same(left, right, context + f"[{n}]")
    elif isinstance(expected, (bool, np.bool_)):
        require(isinstance(actual, (bool, np.bool_)) and bool(actual) == bool(expected), context + ": boolean differs")
    elif isinstance(expected, str):
        if expected == "" and (actual is None or isinstance(actual, float) and np.isnan(actual)):
            actual = ""
        require(actual == expected, context + ": text differs")
    elif isinstance(expected, (int, np.integer)):
        require(actual is not None and np.isfinite(actual) and int(actual) == expected and actual == expected,
                context + ": integer differs")
    else:
        require(actual is not None and np.isfinite(actual) and np.isclose(actual, expected, rtol=1e-9, atol=1e-10),
                context + ": numeric value differs")


def independent_selection(path, start, end, step):
    """Direct native-state scan; prefix barriers prevent hidden gap bridging."""
    grid = np.arange(start, end, step, dtype=np.int64)
    t, p = np.asarray(path.times_ns), np.asarray(path.price2)
    valid, segments = np.asarray(path.valid, bool), np.asarray(path.segments)
    require(t.dtype.kind in "iu" and p.dtype.kind in "iu" and segments.dtype.kind in "iu", "Native selection needs integer arrays")
    require(len(t) == len(p) == len(valid) == len(segments) and (np.diff(t) > 0).all(), "Invalid native selection path")
    best, current, runs, valid_count, previous_position, previous_grid = [], [], 0, 0, None, None
    barriers = (~valid) | np.r_[False, segments[1:] != segments[:-1]] if len(t) else np.array([], bool)
    prefix = np.cumsum(barriers)
    for n, anchor in enumerate(grid):
        position = int(np.searchsorted(t, anchor, side="left")) - 1
        good = position >= 0 and path.start_ns <= anchor < path.end_ns and valid[position]
        if good:
            require(p[position] > 0, "Nonpositive supported native price")
            valid_count += 1
            joins = (bool(current) and previous_grid == n - 1 and segments[position] == segments[previous_position]
                     and prefix[position] == prefix[previous_position])
            if not joins:
                if len(current) > len(best):
                    best = current
                current, runs = [], runs + 1
            current.append((int(anchor), int(p[position])))
            previous_position, previous_grid = position, n
        else:
            if len(current) > len(best):
                best = current
            current, previous_position, previous_grid = [], None, None
    if len(current) > len(best):
        best = current
    values = np.array(best, dtype=np.int64).T if best else np.empty((2, 0), np.int64)
    return values, {"total_grid_points": len(grid), "valid_grid_points": valid_count, "run_count": runs,
        "longest_points": values.shape[1], "run_start_ns": int(values[0, 0]) if best else None,
        "run_end_ns": int(values[0, -1]) if best else None,
        "fraction_window": len(best) / len(grid), "valid_fraction_window": valid_count / len(grid)}


def recompute_pairs(prices, minimum=300, alpha=.05):
    """Call statsmodels directly; do not call the pipeline's testing helper."""
    from statsmodels.tsa.stattools import adfuller, kpss

    prices = np.asarray(prices)
    require(prices.ndim == 1 and prices.dtype.kind in "iu" and (prices > 0).all(), "Persisted prices are not positive native integers")
    require(not (prices.dtype.kind == "u" and (prices > np.iinfo(np.int64).max).any()), "Persisted prices exceed signed range")
    prices = prices.astype(np.int64, copy=False)
    if len(prices):
        relative = (prices - int(prices[0])).astype(float) / float(prices[0])
        log = np.empty(len(prices), float)
        near = relative > -.5
        log[near] = np.log1p(relative[near])
        log[~near] = np.log(prices[~near].astype(float)) - math.log(int(prices[0]))
    else:
        log = np.array([], float)
    output = []
    for transformation, regression in SPECS:
        x = log.copy() if transformation == "log_level" else np.diff(log)
        row = {"transformation": transformation, "regression": regression, "n_observations": len(x),
            "min_observations": minimum, "alpha": alpha, "status": "skipped", "skip_reason": "",
            "adf_stat": None, "adf_pvalue": None, "adf_maxlag": None, "adf_lags": None, "adf_nobs": None,
            "adf_critical": None, "kpss_stat": None, "kpss_pvalue": None, "kpss_pvalue_relation": None,
            "kpss_lags": None, "kpss_critical": None, "adf_reject_nominal": None, "kpss_reject_nominal": None,
            "warnings": [], "classification": "not_tested"}
        if len(x) < minimum:
            row["skip_reason"] = "too_short"
        elif not np.isfinite(x).all():
            row["skip_reason"] = "nonfinite_transformation"
        elif np.ptp(x) == 0:
            row["skip_reason"] = "constant_series"
        elif np.ptp(x) <= 64 * np.finfo(float).eps:
            row["skip_reason"] = "numerically_degenerate_series"
        else:
            x -= np.mean(x)
            row["adf_maxlag"] = min(math.ceil(12 * (len(x) / 100) ** .25), len(x) // 4)
            try:
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    a = adfuller(x, maxlag=row["adf_maxlag"], regression=regression, autolag="BIC")
                row["warnings"].extend({"test": "adf", "category": w.category.__name__, "message": str(w.message)} for w in captured)
                require(np.isfinite(a[0]) and np.isfinite(a[1]), "ADF returned a nonfinite statistic")
                row.update(adf_stat=float(a[0]), adf_pvalue=float(a[1]), adf_lags=int(a[2]), adf_nobs=int(a[3]),
                           adf_critical={k: float(v) for k, v in a[4].items()})
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    k = kpss(x, regression=regression, nlags="auto")
                row["warnings"].extend({"test": "kpss", "category": w.category.__name__, "message": str(w.message)} for w in captured)
                require(np.isfinite(k[0]) and np.isfinite(k[1]), "KPSS returned a nonfinite statistic")
                message = " ".join(str(w.message).lower() for w in captured)
                relation = ("<=" if "smaller" in message or k[1] <= .01 and k[0] > k[3]["1%"] else
                            ">=" if "greater" in message or "larger" in message or k[1] >= .1 and k[0] < k[3]["10%"] else "=")
                ar = bool(a[1] <= alpha)
                kr = (True if k[1] <= alpha else None) if relation == "<=" else (
                     False if k[1] > alpha else None) if relation == ">=" else bool(k[1] <= alpha)
                row.update(status="tested", kpss_stat=float(k[0]), kpss_pvalue=float(k[1]),
                    kpss_pvalue_relation=relation, kpss_lags=int(k[2]), kpss_critical={c: float(v) for c, v in k[3].items()},
                    adf_reject_nominal=ar, kpss_reject_nominal=kr,
                    classification="inconclusive" if kr is None else "stationarity_compatible" if ar and not kr else
                                   "unit_root_compatible" if not ar and kr else "conflicting" if ar and kr else "inconclusive")
            except Exception as exc:
                row.update(status="test_error", skip_reason=f"{type(exc).__name__}: {exc}", classification="not_tested")
        output.append(row)
    return output


def independent_holm(rows, alpha):
    result = [dict(row) for row in rows]
    eligible = [n for n, row in enumerate(result) if row.get("adf_pvalue") is not None and np.isfinite(row["adf_pvalue"])]
    ordered = sorted(eligible, key=lambda n: result[n]["adf_pvalue"])
    scaled = np.array([(len(ordered) - rank) * result[index]["adf_pvalue"] for rank, index in enumerate(ordered)])
    adjusted = np.minimum(1., np.maximum.accumulate(scaled)) if len(scaled) else scaled
    mapping = dict(zip(ordered, adjusted))
    for n, row in enumerate(result):
        p = float(mapping[n]) if n in mapping else None
        row.update(adf_pvalue_holm=p, adf_holm_family_size=len(eligible), adf_reject_holm=bool(p <= alpha) if p is not None else None)
    return result


def table(path):
    columns = pd.read_csv(path, nrows=0).columns
    frame = pd.read_csv(path, dtype={column: "Int64" for column in columns if column.endswith("_ns")})
    for column in ("adf_critical", "kpss_critical", "warnings"):
        if column in frame:
            frame[column] = frame[column].map(lambda x: json.loads(x) if isinstance(x, str) else None)
    return frame


def validate_bindings(root, out):
    protocol, provenance = read_json(out / "protocol.json"), read_json(out / "analysis_provenance.json")
    source = Path(protocol["source_study_dir"]).resolve()
    require(source != out and source not in out.parents and out not in source.parents, "Diagnostic audit must remain separate from its source study")
    require(set(provenance["outputs"]) == FILES, "Diagnostic output hash registry differs")
    for name, digest in provenance["outputs"].items():
        require(sha(out / name) == digest, "Diagnostic output bytes changed: " + name)
    for family in ("inputs", "computation_code"):
        paths = [row["path"] for row in provenance[family]]
        require(len(paths) == len(set(paths)), "Duplicate diagnostic provenance paths")
        for row in provenance[family]:
            require(sha(row["path"]) == row["sha256"], "Diagnostic input/code bytes changed: " + row["path"])
    require({Path(row["path"]).resolve().relative_to(root).as_posix() for row in provenance["computation_code"]} == COMPUTATION,
            "Diagnostic computation dependency registry differs")
    required_inputs = {str((source / name).resolve()) for name in ("protocol.json", "data_manifest.json", "result_audit.json", "analysis_provenance.json")}
    require(required_inputs <= {row["path"] for row in provenance["inputs"]}, "Source-study provenance bindings are incomplete")
    require(len(provenance["inputs"]) == len(required_inputs) + 1, "Diagnostic input registry has unexpected extra or missing files")
    extra = next(row["path"] for row in provenance["inputs"] if row["path"] not in required_inputs)
    same(read_json(extra), read_json(root / "aapl_stationarity_protocol.json"), "frozen stationarity configuration")
    for label, name in (("source_protocol_sha256", "protocol.json"), ("source_manifest_sha256", "data_manifest.json"), ("source_audit_sha256", "result_audit.json")):
        require(protocol[label] == sha(source / name), "Diagnostic source binding differs: " + label)
    source_protocol = read_json(source / "protocol.json")
    expected_protocol = {**read_json(extra), "source_study_dir": str(source),
        "source_protocol_sha256": sha(source / "protocol.json"), "source_manifest_sha256": sha(source / "data_manifest.json"),
        "source_audit_sha256": sha(source / "result_audit.json"),
        **{key: source_protocol[key] for key in ("datasets", "clocks", "quote_age_ms")}}
    same(protocol, expected_protocol, "frozen diagnostic protocol")
    audit = read_json(source / "result_audit.json")
    require(audit.get("passed") is True and audit.get("status") == "passed", "Source-study audit did not pass")
    for name, digest in audit["output_sha256"].items():
        require(sha(source / name) == digest, "Previously audited study output changed: " + name)
    for row in audit["computation_code"]:
        require(sha(root / row["path"]) == row["sha256"], "Previously audited computation changed")
    for package, expected in provenance["packages"].items():
        require(importlib.metadata.version(package) == expected, "Diagnostic package version changed: " + package)
    require(provenance["no_new_market_data_requests"] is True and provenance["source_study_modified"] is False,
            "Diagnostic isolation declarations differ")
    return protocol, provenance, source, source_protocol, read_json(source / "data_manifest.json")


def audit_tables(out, protocol, source_protocol, progress=print):
    requests, runs, tests = table(out / "requests.csv"), table(out / "runs.csv"), table(out / "tests.csv")
    for name in ("datasets", "clocks", "quote_age_ms"):
        same(protocol[name], source_protocol[name], "source diagnostic dimensions " + name)
    require(not requests.request_id.duplicated().any() and not runs.series_id.duplicated().any(), "Duplicate diagnostic request/run identities")
    require(not tests.duplicated(["series_id", "transformation", "regression"]).any(), "Duplicate diagnostic test pairs")
    trials = {row["trial_id"]: row for row in source_protocol["trials"]}
    groups = {row["group_id"]: row for row in source_protocol["source_groups"]}
    expected = set()
    for trial in trials.values():
        group_ids = [g for g, row in groups.items() if trial["trial_id"] in row["trial_ids"]]
        require(len(group_ids) == 1, "Trial must bind exactly one source group")
        for window in protocol["windows"]:
            anchor = pd.Timestamp(trial["anchor_utc"]).value
            for venue in protocol["datasets"]:
                for clock in protocol["clocks"]:
                    for age in protocol["quote_age_ms"]:
                        context = {"source_group_id": group_ids[0], "trial_id": trial["trial_id"], "event_id": trial["event_id"],
                            "event_class": trial["event_type"], "bundle_id": trial["bundle_id"], "role": trial["role"],
                            "anchor_utc": trial["anchor_utc"], "window": window["name"], "venue": venue, "clock": clock,
                            "age_ms": age, "start_ns": anchor + window["start_minute"] * 60_000_000_000,
                            "end_ns": anchor + window["end_minute"] * 60_000_000_000}
                        expected.add(signature(context))
    require(set(requests.request_id) == expected and len(requests) == len(expected), "Diagnostic window grid differs from frozen trials")
    for row in requests.to_dict("records"):
        require(signature({key: row[key] for key in CONTEXT}) == row["request_id"], "Diagnostic request metadata differs from its identity")
    recomputed = []
    with np.load(out / "series.npz", allow_pickle=False) as arrays:
        require(set(arrays.files) == set(requests.series_id) == set(runs.series_id), "Persisted series key registry differs")
        run_rows = runs.set_index("series_id")
        require(set(tests[["series_id", "transformation", "regression"]].itertuples(index=False, name=None))
                == {(sid, kind, trend) for sid in arrays.files for kind, trend in SPECS}, "Unique test-pair grid differs")
        for n, sid in enumerate(sorted(arrays.files), 1):
            values = arrays[sid]
            require(values.ndim == 2 and values.shape[0] == 2 and values.dtype.kind == "i" and values.dtype.itemsize == 8,
                    "Persisted series is not a two-row signed-int64 array")
            require(series_token(values) == sid, "Persisted series content identity differs")
            times, prices = values
            require((prices > 0).all() and (np.diff(times) == protocol["step_ns"]).all(), "Persisted run bridges a grid gap or has nonpositive prices")
            length = len(times)
            run = run_rows.loc[sid]
            for key, expected_value in {"n_price_points": length, "start_ns": int(times[0]) if length else None,
                "end_ns": int(times[-1]) if length else None, "duration_seconds": (int(times[-1]) - int(times[0])) / 1e9 if length else 0.}.items():
                same(run[key], expected_value, f"run {sid}.{key}")
            for row in requests.loc[requests.series_id.eq(sid)].to_dict("records"):
                total = len(range(int(row["start_ns"]), int(row["end_ns"]), protocol["step_ns"]))
                for key, expected_value in {"total_grid_points": total, "longest_points": length,
                    "run_start_ns": int(times[0]) if length else None, "run_end_ns": int(times[-1]) if length else None,
                    "fraction_window": length / total, "valid_fraction_window": row["valid_grid_points"] / total,
                    "longer_run_600_points": length >= protocol["longer_run_points"]}.items():
                    same(row[key], expected_value, f"request {row['request_id']}.{key}")
                require(length <= row["valid_grid_points"] <= total and 0 <= row["run_count"] <= row["valid_grid_points"], "Invalid sampled support counts")
                require(not length or row["start_ns"] <= times[0] <= times[-1] < row["end_ns"], "Persisted run escapes its request window")
            recomputed.extend({"series_id": sid, **row} for row in recompute_pairs(prices, protocol["minimum_observations"], protocol["alpha"]))
            if n % 250 == 0 or n == len(arrays.files):
                progress(f"Independently recomputed {n}/{len(arrays.files)} saved price runs", flush=True)
    recomputed = independent_holm(recomputed, protocol["alpha"])
    actual = {(row["series_id"], row["transformation"], row["regression"]): row for row in tests.to_dict("records")}
    for row in recomputed:
        saved = actual[(row["series_id"], row["transformation"], row["regression"])]
        require(set(saved) == set(row), "Persisted paired-test columns differ")
        for key, expected_value in row.items():
            same(saved[key], expected_value, f"paired test {row['series_id']}/{row['transformation']}/{row['regression']}.{key}")
    results = read_json(out / "results.json")
    counts = {"source_groups": len(groups), "diagnostic_windows": len(requests), "unique_series": len(runs),
              "unique_test_pairs": len(tests), "tested_pairs": int(tests.status.eq("tested").sum()),
              "skipped_pairs": int(tests.status.ne("tested").sum())}
    for key, value in counts.items():
        same(results["counts"][key], value, "result count " + key)
    same(results["protocol"], protocol, "result protocol")
    require(results["status"] == "completed", "Diagnostic analysis is incomplete")
    summaries = []
    joined = requests.merge(tests, on="series_id", validate="many_to_many")
    for key, frame in joined.groupby(["clock", "age_ms", "transformation", "regression"], sort=True):
        unique = frame.drop_duplicates(["series_id", "transformation", "regression"])
        summaries.append({**dict(zip(("clock", "age_ms", "transformation", "regression"), key)),
            "planned_windows": len(frame), "unique_series": len(unique), "tested": int(unique.status.eq("tested").sum()),
            "skipped": int(unique.status.ne("tested").sum()),
            "classification_counts": {str(k): int(v) for k, v in unique.classification.value_counts().to_dict().items()},
            "adf_holm_rejections": int((unique.adf_pvalue_holm < .05).sum())})
    same(results["summary"], summaries, "diagnostic summary")
    return requests, counts, len([row for row in recomputed if row["adf_pvalue"] is not None])


def audit_native_samples(root, requests, protocol, source_protocol, manifest, limit=3):
    sys.path.insert(0, str(root / "src"))
    from colonylab.multi_event_sources import decode_group

    groups = source_protocol["source_groups"]
    preferred = ["2022-06-06", "2026-06-08", "2025-07-31"]
    ordered = []
    for day in preferred:
        ordered.extend(group for group in groups if group["day"] == day and group not in ordered)
    ordered.extend(group for group in sorted(groups, key=lambda row: row["group_id"]) if group not in ordered)
    selected, checks = ordered[:limit], 0
    with np.load(Path(protocol["_out_dir"]) / "series.npz", allow_pickle=False) as arrays:
        for group in selected:
            entries = [row for row in manifest["requests"] if signature(row["request"]) in set(group["request_sha256"])]
            paths, _ = decode_group(entries, group, source_protocol)
            for row in requests.loc[requests.source_group_id.eq(group["group_id"])].to_dict("records"):
                path = paths[(row["venue"], row["age_ms"], row["clock"])]
                values, metadata = independent_selection(path, int(row["start_ns"]), int(row["end_ns"]), protocol["step_ns"])
                require(np.array_equal(values, arrays[row["series_id"]]), "Independent native selection differs: " + row["request_id"])
                for key, value in metadata.items():
                    same(row[key], value, "native selection " + row["request_id"] + "." + key)
                checks += 1
    return {"sampled_source_groups": [g["group_id"] for g in selected], "independent_selection_windows": checks,
            "independent_selection_mismatches": 0}


def audit(root, out, source_groups=3, progress=print):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    protocol, provenance, source, source_protocol, manifest = validate_bindings(root, out)
    raw_count, raw_bytes = 0, 0
    for row in manifest["requests"]:
        if row.get("complete") and row.get("data_path"):
            path = Path(row["data_path"])
            require(path.stat().st_size == row["file_size_bytes"] and sha(path) == row["content_sha256"], "Raw source bytes changed")
            raw_count, raw_bytes = raw_count + 1, raw_bytes + path.stat().st_size
    requests, counts, family = audit_tables(out, protocol, source_protocol, progress)
    results = read_json(out / "results.json")
    same(results["counts"]["source_raw_requests"], len(manifest["requests"]), "Source request count")
    same(results["counts"]["source_raw_records"], sum(row.get("rows", 0) for row in manifest["requests"]), "Source record count")
    same(results["limitations"], protocol["limitations"], "Diagnostic limitations")
    sampled = audit_native_samples(root, requests, {**protocol, "_out_dir": str(out)}, source_protocol, manifest, source_groups)
    for row in provenance["inputs"] + provenance["computation_code"]:
        require(sha(row["path"]) == row["sha256"], "Input/code changed during diagnostic audit")
    for name, digest in provenance["outputs"].items():
        require(sha(out / name) == digest, "Diagnostic output changed during audit")
    return {"passed": True, "status": "passed", **counts, **sampled,
        "all_unique_test_pairs_independently_recomputed": True, "global_adf_holm_family_size": family,
        "raw_files_freshly_hashed": raw_count, "raw_bytes_freshly_hashed": raw_bytes,
        "source_study_audit_sha256": sha(source / "result_audit.json"),
        "input_hashes": provenance["inputs"], "computation_code": provenance["computation_code"],
        "output_sha256": {**provenance["outputs"], "analysis_provenance.json": sha(out / "analysis_provenance.json")},
        "scope": "All persisted run identities, paired ADF/KPSS numeric results/warnings/classifications, global Holm values and summaries; "
                 "all input/output/source hashes and existing source-study audit binding; independent native selection only for the named source-group sample.",
        "source_study_modified": False, "new_market_data_requests": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--source-groups", type=int, default=3)
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.out_dir or root / "out/aapl_stationarity").resolve()
    require(args.source_groups > 0, "Audit at least one representative native source group")
    # Refuse an invalid destination before the exception handler can write a
    # failed audit over an existing source-study audit.
    destination_protocol = read_json(out / "protocol.json")
    require("source_study_dir" in destination_protocol, "Destination is not a stationarity diagnostic directory")
    source = Path(destination_protocol["source_study_dir"]).resolve()
    require(out != source and source not in out.parents and out not in source.parents,
            "Diagnostic audit output must remain separate from its source study")
    started = time.perf_counter()
    try:
        result = audit(root, out, args.source_groups)
    except Exception as exc:
        result = {"passed": False, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    result.update(created_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=round(time.perf_counter() - started, 3),
                  audit_script_sha256=sha(Path(__file__)))
    temporary = out / "result_audit.json.partial"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(out / "result_audit.json")
    print(json.dumps({key: result[key] for key in ("status", "elapsed_seconds", "error") if key in result}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
