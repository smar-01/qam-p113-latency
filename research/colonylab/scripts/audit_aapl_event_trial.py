"""Audit saved-data AAPL event outputs, identities and descriptive contrasts.

Uses local files only. Rehashes every referenced raw/cache file without loading
pickles or contacting a provider, checks the original reconstruction validation,
then independently rebuilds all date-level tables using summarize_trials.
The prior MBO reconstruction audit remains the raw-record semantic check; this
audit does not claim to repeat a complete raw MBO reconstruction.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DAY_FILES = ("pair_metrics.csv", "venue_metrics.csv", "price_profile.csv", "matching_audit.csv")
DAY_CODE = ("src/colonylab/event_midpoints.py", "src/colonylab/latency.py",
            "src/colonylab/aapl_event_trial.py")
ALL_CODE = (*DAY_CODE, "src/colonylab/event_trial_stats.py")
TABLES = ("latency_daily", "shock_daily", "leadership_daily", "daily_changes", "contrasts")
SEMANTIC_FAILURES = (
    "duplicate_adds", "unmatched_cancels", "cancel_identity_mismatches", "oversize_cancels",
    "unmatched_modifies", "unknown_modify_invalidations", "modify_upserts", "modify_side_mismatches",
    "snapshot_without_clear", "invalid_snapshot_actions", "invalid_order_fields", "invalid_fill_fields",
    "invalid_trade_fields", "unknown_actions", "non_native_records", "unmatched_fills", "invalid_timestamps",
)
FORBIDDEN_INFERENCE = re.compile(
    r"^(?:p_?values?|pvalue|ci\d*|ci_.*|confidence_.*|.*_confidence_interval|bootstrap_.*|.*_p_value)$",
    re.IGNORECASE,
)


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def read_json(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value {value}: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_grid(frame, columns, expected, context):
    require(set(columns) <= set(frame), f"{context}: missing identity columns")
    require(not frame.duplicated(columns).any(), f"{context}: duplicate identities")
    observed = set(frame[columns].itertuples(index=False, name=None))
    require(observed == expected,
            f"{context}: {len(expected - observed)} missing and {len(observed - expected)} extra identities")


def same_frame(actual, expected, context):
    try:
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True),
                                      check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-9)
    except AssertionError as error:
        raise ValueError(f"{context}: {str(error)[:1200]}") from error


def serialized_frame(records):
    # Match the documented CSV boundary, including empty strings and nested
    # exclusion metadata, rather than comparing CSV strings to Python lists.
    return pd.read_csv(StringIO(pd.DataFrame(records).to_csv(index=False)))


def same_value(actual, expected, context):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"{context}: object keys differ")
        for key in expected:
            same_value(actual[key], expected[key], f"{context}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"{context}: list length differs")
        for index, (left, right) in enumerate(zip(actual, expected)):
            same_value(left, right, f"{context}[{index}]")
    elif expected is None or isinstance(expected, (str, bool)):
        require(actual == expected and (expected is None or type(actual) is type(expected)), f"{context}: value differs")
    elif isinstance(expected, (int, float, np.integer, np.floating)):
        require(actual is not None and not isinstance(actual, bool)
                and np.isfinite(float(actual)) and np.isfinite(float(expected))
                and np.isclose(actual, expected, rtol=1e-10, atol=1e-9), f"{context}: number differs")
    else:
        require(actual == expected, f"{context}: value differs")


def no_inference_fields(value, context="results"):
    if isinstance(value, dict):
        for key, child in value.items():
            require(not FORBIDDEN_INFERENCE.fullmatch(key), f"{context}: prohibited inferential field {key}")
            no_inference_fields(child, f"{context}.{key}")
    elif isinstance(value, list):
        for child in value:
            no_inference_fields(child, context)


def audit_calendar(protocol, calendar):
    require(protocol["market_data_policy"] == calendar["market_data_policy"] == "saved_data_only_no_downloads",
            "Study must use saved market data only")
    events = calendar["events"]
    by_id = {row["id"]: row for row in events}
    require(len(by_id) == len(events), "Duplicate calendar event identifiers")
    for event in events:
        require(event.get("source_urls") and all(url.startswith(("https://www.apple.com/", "https://developer.apple.com/",
                "https://www.sec.gov/")) for url in event["source_urls"]), "Calendar source is not an approved primary source")
        if event["timestamp_precision"] == "date_only":
            require(event["event_time_utc"] is None and event["analysis_role"] != "primary_intraday_event",
                    "Date-only event has an invented time or measured-event role")
        if event["coverage_status"] != "covered_native_midpoints":
            require(event["analysis_role"] != "primary_intraday_event", "Uncovered event is labelled measured")
    anchor = by_id[protocol["event_id"]]
    require(anchor["id"] == "aapl_wwdc26_keynote_20260608" and anchor["date"] == "2026-06-08"
            and anchor["event_time_utc"] == "2026-06-08T17:00:00Z"
            and anchor["timestamp_nature"] == "scheduled_start"
            and anchor["coverage_status"] == "covered_native_midpoints", "Verified scheduled WWDC anchor differs")
    require(anchor["event_time_utc"] == protocol["event_day"] + "T" + protocol["anchor_time_utc"] + "Z",
            "Protocol anchor differs from primary-source calendar")
    require(len([event for event in events if event["analysis_role"] == "primary_intraday_event"]) == 1,
            "Single-event case study includes additional measured corporate events")
    require(by_id["aapl_wwdc26_coannouncements_20260608"]["bundle_id"] == anchor["bundle_id"],
            "WWDC co-announcements are not grouped into one information bundle")
    known_times = {"aapl_wwdc26_state_of_union_20260608": "2026-06-08T20:00:00Z",
                   "aapl_september_event_20260909": "2026-09-09T17:00:00Z",
                   "aapl_q3_earnings_call_20260730": "2026-07-30T21:00:00Z",
                   "aapl_q4_earnings_call_20261102": "2026-11-02T22:00:00Z"}
    for identity, expected in known_times.items():
        require(by_id[identity]["event_time_utc"] == expected, f"Verified source timestamp differs: {identity}")
        require(by_id[identity]["coverage_status"] != "covered_native_midpoints", f"Uncovered event marked measured: {identity}")
    require(by_id["aapl_10q_q3_fy2026_20260731"]["sec_acceptance_clock_raw"] == "2026-07-31 06:01:02",
            "SEC index acceptance clock differs")
    require(by_id["aapl_q4_earnings_call_20261102"].get("temporal_status_at_catalog_asof") == "scheduled_not_yet_started"
            and pd.Timestamp(calendar["as_of_utc"]) < pd.Timestamp(known_times["aapl_q4_earnings_call_20261102"]),
            "Catalog improperly treats its then-future earnings call as observed")
    require(set(protocol["excluded_event_week"]).isdisjoint(protocol["reference_days"]), "WWDC week contaminates references")
    require(len(protocol["primary_reference_days"]) == 6
            and all(pd.Timestamp(day).weekday() == 0 for day in protocol["primary_reference_days"])
            and len(protocol["reference_days"]) == 25, "Prespecified reference-date cohorts differ")
    return by_id


def audit_sources(root, out, protocol, provenance, days, code):
    archive = read_json(root / "research-archive-manifest.json")
    archive_rows = {row["path"].replace("\\", "/"): row for row in archive["files"]}
    replay = read_json(root / "out/mbo_sir/replay_manifest.json")
    acquisition = read_json(root / "out/mbo_sir/data_manifest.json")
    previous_audit = read_json(root / "out/mbo_sir/result_audit.json")
    require(replay.get("complete") is True and acquisition.get("complete") is True
            and (previous_audit.get("passed") is True or previous_audit.get("status") == "passed"),
            "Original acquisition/replay audit is incomplete or failed")
    replay_rows = {(row["day"], row["dataset"]): row for row in replay["tasks"]}
    acquisition_rows = {signature(row["request"]): row for row in acquisition["requests"]}
    reconstruction_code = signature({name: file_sha(root / "src/colonylab" / name)
                                     for name in ("mbo_book.py", "mbo_replay.py")})
    checked, totals = {}, {"cache_files": 0, "raw_files": 0, "cache_bytes": 0, "raw_bytes": 0,
                          "native_grid_anchors_checked": 0, "native_grid_eligible_anchors_matched": 0,
                          "native_grid_healthy_unknown_midpoint_anchors": 0}

    def verify_bytes(path, expected_hash, kind):
        path = path.resolve()
        require(path.is_relative_to(root), f"Input lies outside project: {path}")
        relative = path.relative_to(root).as_posix()
        archived = archive_rows.get(relative)
        require(archived is not None and archived["sha256"] == expected_hash, f"Original archive hash differs: {relative}")
        require(path.stat().st_size == archived["bytes"], f"Original archive size differs: {relative}")
        if path not in checked:
            checked[path] = file_sha(path)
            totals[kind + "_files"] += 1
            totals[kind + "_bytes"] += path.stat().st_size
        require(checked[path] == expected_hash, f"Input bytes changed: {relative}")

    sources, checkpoint_frames = [], {name: [] for name in DAY_FILES}
    intended = set(protocol["all_venues"])
    for day in days:
        directory = out / "days" / day
        marker = read_json(directory / "complete.json")
        facts = read_json(directory / "source_facts.json")
        expected_identity = {"day": day, "protocol_sha256": signature(protocol), "code": code,
                             "calendar_sha256": file_sha(root / protocol["event_calendar"])}
        require(marker["identity"] == expected_identity, f"Checkpoint identity differs: {day}")
        require(marker["source_facts_sha256"] == signature(facts), f"Source-facts checkpoint hash differs: {day}")
        require(set(marker["files"]) == set(DAY_FILES), f"Checkpoint output manifest differs: {day}")
        require(len(facts) == len(intended) and {row["dataset"] for row in facts} == intended,
                f"Source facts lack exact venue/day grid: {day}")
        for name in DAY_FILES:
            require(file_sha(directory / name) == marker["files"][name], f"Checkpoint output hash differs: {day}/{name}")
            checkpoint_frames[name].append(pd.read_csv(directory / name))
        anchor = pd.Timestamp(day + "T" + protocol["anchor_time_utc"] + "Z").value
        start = anchor + protocol["outer_start_minute"] * 60_000_000_000
        end = anchor + protocol["outer_end_minute"] * 60_000_000_000
        for fact in facts:
            venue = fact["dataset"]
            task = replay_rows[(day, venue)]
            original = task["provenance"]
            require(fact["day"] == day and fact["observation_start_ns"] == start and fact["observation_end_ns"] == end,
                    f"Source fact observation window differs: {day}/{venue}")
            require(fact["replay_code_sha256"] == original["code"] == reconstruction_code,
                    f"Original replay code differs: {day}/{venue}")
            require(fact["book_validation"] == task["validation"] and task["validation"].get("passed") is True
                    and not any(task["diagnostics"].get(name, 0) for name in SEMANTIC_FAILURES),
                    f"Original native reconstruction validation fails: {day}/{venue}")
            cache = Path(fact["cache_path"])
            require(cache.resolve() == (root / task["path"]).resolve() and cache.stat().st_size == fact["cache_bytes"],
                    f"Source fact cache identity differs: {day}/{venue}")
            verify_bytes(cache, fact["cache_sha256"], "cache")
            require(fact["metric"] == "native_receive_midpoint_matching" and fact["clock"] == "ts_recv"
                    and fact["native_receive_times_only"] is True and fact["bid_ask_events_available"] is False,
                    f"Source fact measurement label differs: {day}/{venue}")
            require(fact["grid_price_mismatches"] == 0 and fact["price2_recovery_max_residual"] < .01
                    and fact["jump2_recovery_max_residual"] < .01
                    and fact["grid_healthy_anchors"] == fact["grid_eligible_anchors_matched"] + fact["grid_healthy_unknown_midpoint_anchors"],
                    f"Native midpoint/grid quality facts fail: {day}/{venue}")
            for source, target in (("grid_anchors_checked", "native_grid_anchors_checked"),
                                   ("grid_eligible_anchors_matched", "native_grid_eligible_anchors_matched"),
                                   ("grid_healthy_unknown_midpoint_anchors", "native_grid_healthy_unknown_midpoint_anchors")):
                totals[target] += int(fact[source])
            require({row["schema"] for row in fact["raw_sources"]} == set(original["requests"]),
                    f"Raw source schemas differ: {day}/{venue}")
            for raw in fact["raw_sources"]:
                request, schema = raw["request"], raw["schema"]
                entry = acquisition_rows[signature(request)]
                require(request == original["requests"][schema] and request["schema"] == schema
                        and request["dataset"] == venue and request["symbols"] == ["AAPL"]
                        and request["start"][:10] == day and entry.get("complete") is True
                        and not entry.get("record_limit_reached") and raw["rows"] == entry["rows"]
                        and 0 < raw["rows"] < request["limit"] and raw["sha256"] == original["sources"][schema],
                        f"Raw source request/provenance differs: {day}/{venue}/{schema}")
                path = Path(raw["path"])
                receipt = read_json(path.parent / "fetch.json")
                termination = read_json(path.parent / "stream_complete.json")
                require(receipt.get("request") == termination.get("request") == request
                        and receipt.get("complete") is True and not receipt.get("record_limit_reached")
                        and receipt["rows"] == entry["rows"]
                        and receipt["file_size_bytes"] == entry["file_size_bytes"] == path.stat().st_size,
                        f"Raw source receipt/termination differs: {day}/{venue}/{schema}")
                verify_bytes(path, raw["sha256"], "raw")
        sources.extend(facts)
    require(provenance["sources"] == sources, "Aggregate source facts differ from date checkpoints")
    frames = {}
    require(set(provenance["output_sha256"]) == set(DAY_FILES), "Aggregate output hash manifest differs")
    for name in DAY_FILES:
        require(file_sha(out / name) == provenance["output_sha256"][name], f"Aggregate output hash differs: {name}")
        frames[name] = pd.read_csv(out / name)
        same_frame(frames[name], pd.concat(checkpoint_frames[name], ignore_index=True), f"Aggregate vs date checkpoints: {name}")
    return frames, totals, len(sources)


def audit(root, out):
    sys.path.insert(0, str(root / "src"))
    from colonylab.event_trial_stats import PAIR_REQUIRED, VENUE_REQUIRED, summarize_trials

    protocol = read_json(out / "protocol.json")
    calendar = read_json(out / "event_calendar.json")
    provenance = read_json(out / "provenance.json")
    summary = read_json(out / "summary.json")
    require(protocol == read_json(root / "aapl_event_trial.json"), "Output protocol differs from current frozen protocol")
    require(calendar == read_json(root / protocol["event_calendar"]), "Output calendar differs from frozen source catalog")
    audit_calendar(protocol, calendar)
    require(provenance["protocol_sha256"] == file_sha(out / "protocol.json")
            and provenance["calendar_sha256"] == file_sha(out / "event_calendar.json"), "Protocol/calendar provenance hashes differ")
    all_code = [{"path": name, "sha256": file_sha(root / name)} for name in ALL_CODE]
    require(provenance["computation_code"] == all_code, "Computation code is incomplete or changed")
    days = sorted(set([protocol["event_day"], *protocol["reference_days"]]))
    frames, totals, source_count = audit_sources(root, out, protocol, provenance, days, all_code[:-1])
    pair, venue = frames["pair_metrics.csv"], frames["venue_metrics.csv"]
    require(PAIR_REQUIRED <= set(pair) and VENUE_REQUIRED <= set(venue), "Pair/venue output schema is incomplete")
    venues = protocol["all_venues"]
    windows = [row["name"] for row in protocol["windows"]]
    exact_grid(pair, ["day", "window", "horizon_ms", "leader", "follower"],
               {(day, window, horizon, a, b) for day in days for window in windows for horizon in protocol["horizons_ms"]
                for a in venues for b in venues if a != b}, "Pair output")
    exact_grid(venue, ["day", "window", "venue", "shock_threshold_bp"],
               {(day, window, name, threshold) for day in days for window in windows for name in venues
                for threshold in protocol["shock_thresholds_bp"]}, "Venue output")
    require(pair.metric.eq("native_receive_midpoint_matching").all() and pair.clock.eq("ts_recv").all(),
            "Pair output mislabels the native receive-midpoint measurement")
    for left, right in ((pair.n_events, pair.n_already + pair.n_no_state + pair.n_behind),
                        (pair.n_behind, pair.n_eligible + pair.n_quality_truncated + pair.n_right_truncated),
                        (pair.n_matched + pair.known_matches_truncated, pair.n_observed_matches)):
        require(np.array_equal(left, right), "Pair event exclusion/conservation counts do not reconcile")
    for window in protocol["windows"]:
        selected = venue.loc[venue.window.eq(window["name"])]
        require(np.allclose(selected.scheduled_seconds, (window["end_minute"] - window["start_minute"]) * 60),
                "Venue scheduled duration differs from fixed window")
    clustered = np.divide(venue.shocks_clustered.to_numpy(float) * 60, venue.observed_seconds.to_numpy(float),
                          out=np.full(len(venue), np.nan), where=venue.observed_seconds.to_numpy(float) > 0)
    require("clustered_shocks_per_minute" in venue and
            np.allclose(venue.clustered_shocks_per_minute, clustered, equal_nan=True), "Clustered shock rates do not reconcile")
    matches = frames["matching_audit.csv"]
    exact_grid(matches, ["day", "leader", "follower"], {(day, a, b) for day in days for a in venues for b in venues if a != b},
               "Independent matching audit")
    require(matches.mismatches.eq(0).all() and (matches.sampled_cases >= 0).all()
            and (matches.sampled_cases <= matches.leader_events).all(), "Independent first-passage audit failed")
    profile = frames["price_profile.csv"]
    expected_profile = {(day, int(pd.Timestamp(day + "T" + protocol["anchor_time_utc"] + "Z").value + minute))
                        for day in days for minute in np.arange(protocol["outer_start_minute"] * 60_000_000_000,
                            protocol["outer_end_minute"] * 60_000_000_000, protocol["price_path_grid_seconds"] * 1_000_000_000)}
    exact_grid(profile, ["day", "time_ns"], expected_profile, "Descriptive price profile")
    require(profile.n_venues.between(0, len(protocol["primary_venues"])).all(), "Price profile venue support is impossible")
    require(summary["status"] == "completed" and summary["event_count"] == 1
            and summary["observed_dates"] == len(days) and summary["source_cache_tasks"] == source_count
            and summary["reference_days"] == len(protocol["reference_days"])
            and summary["primary_reference_days"] == len(protocol["primary_reference_days"])
            and summary["independent_matching_cases"] == int(matches.sampled_cases.sum())
            and summary["independent_matching_mismatches"] == 0, "Summary counters differ from audited outputs")
    coverage = pd.read_csv(out / "event_coverage.csv")
    coverage_keys = ("id", "event_class", "label", "date", "event_time_utc", "timestamp_nature", "coverage_status", "analysis_role")
    same_frame(coverage, serialized_frame([{key: row.get(key) for key in coverage_keys} for row in calendar["events"]]),
               "Calendar coverage decisions")
    recomputed = summarize_trials(pair, venue, protocol)
    no_inference_fields(summary["results"])
    same_value(summary["results"], recomputed, "Recomputed summary")
    for name in TABLES:
        table = pd.read_csv(out / (name + ".csv"))
        no_inference_fields({column: None for column in table.columns}, name)
        same_frame(table, serialized_frame(recomputed[name]), f"Recomputed {name}")
    for entry in all_code:
        require(file_sha(root / entry["path"]) == entry["sha256"], "Computation code changed while auditing")
    audited_outputs = (*DAY_FILES, "protocol.json", "event_calendar.json", "event_coverage.csv",
                       "summary.json", "provenance.json", *(name + ".csv" for name in TABLES))
    return {"passed": True, "status": "passed", "event_count": 1, "dates": len(days),
            "source_cache_tasks": source_count, **totals,
            "source_files_freshly_hashed": totals["raw_files"], "cache_files_freshly_hashed": totals["cache_files"],
            "independent_statistics_reconstructed": True,
            "pair_rows": len(pair), "venue_rows": len(venue), "independent_matching_cases": int(matches.sampled_cases.sum()),
            "independent_matching_mismatches": 0, "recomputed_tables": list(TABLES),
            "protocol_sha256": provenance["protocol_sha256"], "calendar_sha256": provenance["calendar_sha256"],
            "computation_code": all_code,
            "output_sha256": {name: file_sha(out / name) for name in audited_outputs},
            "scope": "Fresh input-byte, archive, receipt, reconstruction-validation, checkpoint, output-schema and descriptive-summary audit; no network requests and no raw-MBO re-reconstruction.",
            "inferential_interpretation": "One scheduled keynote; descriptive date-level contrasts only. No p-values, confidence intervals or event-class comparison."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.out_dir if args.out_dir is not None else root / "out/aapl_event_trial").resolve()
    started = time.perf_counter()
    try:
        result = audit(root, out)
    except Exception as error:
        result = {"passed": False, "status": "failed", "error": f"{type(error).__name__}: {error}"}
    result.update(created_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=round(time.perf_counter() - started, 3),
                  audit_script_sha256=file_sha(Path(__file__)))
    out.mkdir(parents=True, exist_ok=True)
    temporary = out / "result_audit.json.partial"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(out / "result_audit.json")
    print(json.dumps({key: result[key] for key in ("status", "elapsed_seconds", "error") if key in result}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
