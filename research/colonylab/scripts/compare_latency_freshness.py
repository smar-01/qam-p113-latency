"""Compare completed latency checkpoints differing only in quote maximum age.

This is limited descriptive sensitivity on the intersection of completed dates,
not a new full-study estimate or an identical-event causal comparison.  Existing
checkpoint outputs, protocols, computation fingerprints and source identities
are verified before rows are joined.  No source files are changed or fetched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROW_ID = ["study", "phase", "date", "clock", "symbol", "leader", "follower", "horizon_ms", "regime"]
GROUP_ID = ["study", "phase", "clock", "symbol", "horizon_ms"]


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _request_key(request):
    return json.dumps(request, sort_keys=True)


def _source_keys(sources, *, provenance=False):
    result = []
    for source in sources:
        digest = source["content_sha256"] if provenance else source["sha256"]
        require(isinstance(digest, str) and len(digest) == 64, "Invalid saved source SHA-256.")
        require(type(source["rows"]) is int and source["rows"] > 0, "Invalid saved source row count.")
        result.append((_request_key(source["request"]), digest, source["rows"]))
    require(len(result) == len(set(result)), "Duplicate saved source identities.")
    return set(result)


def _compare_code(first, second, *, allow_driver_change):
    left = {row["path"]: row["sha256"] for row in first}
    right = {row["path"]: row["sha256"] for row in second}
    require(set(left) == set(right), "Freshness runs have different computation-code file sets.")
    for path in ("src/colonylab/latency.py", "src/colonylab/latency_data.py"):
        require(path in left and left[path] == right[path], "Freshness matching/quote-loader code fingerprints differ.")
    changes = [{"path": path, "primary_sha256": left[path], "reference_sha256": right[path]}
               for path in sorted(left) if left[path] != right[path]]
    if changes:
        require(allow_driver_change and {row["path"] for row in changes} == {"src/colonylab/latency_pipeline.py"},
                "Freshness runs use different computation-code fingerprints; only an explicitly allowed driver change is supported.")
    return changes


def _completed(root):
    result = {}
    for marker in sorted((root / "days").glob("*/*/complete.json")):
        saved = read_json(marker)
        identity = saved["identity"]
        key = (identity["study"], identity["date"])
        require(marker.parent.name == key[1] and marker.parent.parent.name == key[0],
                "Checkpoint path differs from its saved study/date identity.")
        require(key not in result, "Duplicate completed study/date checkpoints.")
        result[key] = (marker.parent, saved)
    return result


def _verify_checkpoint(directory, saved, protocol, provenance):
    identity = saved["identity"]
    require(identity["protocol"] == protocol, "Checkpoint protocol differs from its run protocol.")
    outputs = saved["outputs"]
    require(outputs and "daily_latency.csv" in outputs, "Checkpoint has no authenticated daily latency output.")
    for filename, digest in outputs.items():
        require(Path(filename).name == filename, "Checkpoint output name must be a direct filename.")
        path = directory / filename
        require(path.is_file() and sha256_file(path) == digest,
                f"Checkpoint output hash differs: {path}.")
    own_sources = _source_keys(identity["sources"])
    provenance_sources = [source for source in provenance["sources"]
                          if source["study"] == identity["study"]
                          and source["request"]["start"][:10] == identity["date"]]
    require(own_sources and own_sources == _source_keys(provenance_sources, provenance=True),
            "Checkpoint source identities differ from saved run provenance.")
    computation = identity["computation_code"]
    require(computation and len({row["path"] for row in computation}) == len(computation),
            "Missing or duplicated computation-code identities.")
    known = {row["path"]: row["sha256"] for row in provenance["computation_code"]}
    require(all(known.get(row["path"]) == row["sha256"] for row in computation),
            "Checkpoint computation differs from its saved run provenance.")
    frame = pd.read_csv(directory / "daily_latency.csv")
    require(set(ROW_ID + ["n_events", "n_eligible", "n_matched", "median_us"]) <= set(frame),
            "Checkpoint daily latency table lacks required columns.")
    require(not frame.duplicated(ROW_ID).any(), "Duplicate checkpoint daily latency rows.")
    require(set(frame.study) == {identity["study"]} and set(frame.date) == {identity["date"]},
            "Checkpoint daily table contains a different study/date.")
    require(set(frame.clock) == set(protocol["clocks"])
            and set(frame.horizon_ms) == set(protocol["horizons_ms"]),
            "Checkpoint daily clock/horizon grid differs from protocol.")
    for field in ("n_events", "n_eligible", "n_matched"):
        values = frame[field].to_numpy(float)
        require(np.isfinite(values).all() and (values >= 0).all() and (values == np.floor(values)).all(),
                "Invalid checkpoint event counts.")
    require((frame.n_matched <= frame.n_eligible).all() and (frame.n_eligible <= frame.n_events).all(),
            "Checkpoint eligible/matched counts are inconsistent.")
    return frame.loc[frame.regime.eq("all")].copy(), own_sources


def compare_freshness(primary, reference, *, out_dir=None, allow_driver_change=False):
    """Verify and compare common completed dates, keeping changed cohorts visible."""
    primary, reference = Path(primary).resolve(), Path(reference).resolve()
    primary_protocol, reference_protocol = (read_json(root / "protocol.json") for root in (primary, reference))
    ignored = {"max_quote_age_ms", "limitations"}
    require({key: value for key, value in primary_protocol.items() if key not in ignored}
            == {key: value for key, value in reference_protocol.items() if key not in ignored},
            "Freshness runs differ in a protocol setting other than quote maximum age.")
    primary_age, reference_age = primary_protocol["max_quote_age_ms"], reference_protocol["max_quote_age_ms"]
    require(np.isfinite(primary_age) and np.isfinite(reference_age)
            and primary_age > 0 and reference_age > 0 and primary_age != reference_age,
            "Freshness sensitivity requires two different positive quote-age limits.")
    primary_provenance, reference_provenance = (read_json(root / "provenance.json") for root in (primary, reference))
    primary_days, reference_days = _completed(primary), _completed(reference)
    common = sorted(set(primary_days) & set(reference_days))
    require(common, "No common completed study/date checkpoints are available.")
    joined_frames, checkpoints, driver_changes = [], [], []
    for key in common:
        primary_dir, primary_saved = primary_days[key]
        reference_dir, reference_saved = reference_days[key]
        first, first_sources = _verify_checkpoint(primary_dir, primary_saved, primary_protocol, primary_provenance)
        second, second_sources = _verify_checkpoint(reference_dir, reference_saved, reference_protocol, reference_provenance)
        require(first_sources == second_sources, "Freshness runs use different raw requests, hashes or row counts.")
        changes = _compare_code(primary_saved["identity"]["computation_code"],
                                reference_saved["identity"]["computation_code"],
                                allow_driver_change=allow_driver_change)
        for change in changes:
            if change not in driver_changes:
                driver_changes.append(change)
        left_keys = set(first[ROW_ID].itertuples(index=False, name=None))
        right_keys = set(second[ROW_ID].itertuples(index=False, name=None))
        require(left_keys == right_keys, "Freshness runs have different daily task rows.")
        joined_frames.append(first.merge(second, on=ROW_ID, how="inner", validate="one_to_one",
                                         suffixes=("_primary", "_reference")))
        checkpoints.append({"study": key[0], "date": key[1], "phases": sorted(first.phase.unique()),
                            "source_requests": len(first_sources),
                            "raw_records": sum(row[2] for row in first_sources),
                            "primary_checkpoint_sha256": sha256_file(primary_dir / "complete.json"),
                            "reference_checkpoint_sha256": sha256_file(reference_dir / "complete.json")})
    joined = pd.concat(joined_frames, ignore_index=True)
    minimum = primary_protocol["minimum_matched_events_per_date"]
    records = []
    for key, group in joined.groupby(GROUP_ID, sort=True):
        row = dict(zip(GROUP_ID, key))
        row.update(primary_quote_age_ms=primary_age, reference_quote_age_ms=reference_age,
                   dates=int(group.date.nunique()), directed_pairs=int(group[["leader", "follower"]].drop_duplicates().shape[0]),
                   paired_daily_pair_rows=len(group))
        for name in ("primary", "reference"):
            for field in ("n_events", "n_eligible", "n_matched"):
                row[f"{name}_{field}"] = int(group[f"{field}_{name}"].sum())
            row[f"{name}_match_rate"] = (row[f"{name}_n_matched"] / row[f"{name}_n_eligible"]
                                         if row[f"{name}_n_eligible"] else None)
        row["eligible_change_percent"] = (100 * (row["primary_n_eligible"] - row["reference_n_eligible"])
                                           / row["reference_n_eligible"] if row["reference_n_eligible"] else None)
        row["match_rate_change_percentage_points"] = (100 * (row["primary_match_rate"] - row["reference_match_rate"])
                                                      if row["primary_match_rate"] is not None and row["reference_match_rate"] is not None else None)
        supported = ((group.n_matched_primary >= minimum) & (group.n_matched_reference >= minimum)
                     & np.isfinite(group.median_us_primary) & np.isfinite(group.median_us_reference))
        selected = group.loc[supported]
        delta = selected.median_us_primary.to_numpy(float) - selected.median_us_reference.to_numpy(float)
        row.update(supported_daily_pair_rows=len(selected), supported_dates=int(selected.date.nunique()),
                   supported_directed_pairs=int(selected[["leader", "follower"]].drop_duplicates().shape[0]),
                   median_primary_conditional_daily_pair_median_us=float(selected.median_us_primary.median()) if len(selected) else None,
                   median_reference_conditional_daily_pair_median_us=float(selected.median_us_reference.median()) if len(selected) else None,
                   median_paired_conditional_median_change_us=float(np.median(delta)) if len(delta) else None)
        positive = selected.median_us_reference.to_numpy(float) > 0
        relative = 100 * delta[positive] / selected.median_us_reference.to_numpy(float)[positive]
        row["median_paired_conditional_median_change_percent"] = float(np.median(relative)) if len(relative) else None
        records.append(row)
    output = Path(out_dir).resolve() if out_dir is not None else primary
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(records)
    table.to_csv(output / "freshness_sensitivity.csv", index=False)
    result = {
        "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": "Limited descriptive comparison on common completed source dates; primary minus reference.",
        "primary_quote_age_ms": primary_age, "reference_quote_age_ms": reference_age,
        "primary_root": str(primary), "reference_root": str(reference),
        "common_completed_study_dates": len(common), "dates": checkpoints,
        "primary_completed_dates_available": len(primary_days), "reference_completed_dates_available": len(reference_days),
        "joined_all_regime_daily_pair_rows": len(joined), "minimum_matches_in_both_rows": minimum,
        "checks": {"all_checkpoint_output_hashes": True, "checkpoint_and_top_level_protocols_match": True,
                   "only_quote_age_computation_setting_differs": True, "source_requests_hashes_rows_identical": True,
                   "matching_and_loader_fingerprints_identical": True,
                   "all_computation_fingerprints_identical": not driver_changes,
                   "daily_task_rows_identical": True},
        "explicitly_allowed_driver_changes": driver_changes,
        "driver_change_context": ("Primary driver adds configuration guards and report/summary ranking filters after the reference run was stopped. Matching/day-processing functions were reported unchanged by the run operator; the older whole driver source is not reconstructed by this script. Core matcher and quote loader fingerprints match exactly."
                                  if driver_changes else None),
        "limitations": [
            "Only common completed dates enter; this is not a second complete forty/fifty-date study.",
            "The included dates are reported explicitly, including inherited training/held_out labels; no untouched latency holdout is claimed.",
            "Changing quote age changes leader candidates and fully observed follow-up cohorts. These are matched task rows, not identical events.",
            "Match rates divide aggregate matched counts by eligible counts within each age convention; their event denominators differ.",
            "Median changes summarize rowwise differences of daily/pair conditional medians on rows with enough matches in both runs; events are not pooled.",
            "No p-values, confidence intervals, causal effect or general full-study robustness conclusion is computed.",
            "Raw DBNs are not rehashed here; both runs' exact request/content identities and authenticated output checkpoints are compared.",
            "If an orchestration/report driver change is explicitly allowed, its two fingerprints remain reported; identical whole-pipeline code is not asserted.",
        ],
        "aggregates": records,
        "script_sha256": sha256_file(Path(__file__)),
    }
    (output / "freshness_sensitivity.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=ROOT / "out" / "latency")
    parser.add_argument("--reference", type=Path, default=ROOT / "out" / "latency_age1000_partial")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--allow-driver-change", action="store_true",
                        help="Allow the explicitly documented orchestration/report-only driver change; matcher and loader must still match exactly.")
    args = parser.parse_args(argv)
    result = compare_freshness(args.primary, args.reference, out_dir=args.out_dir,
                               allow_driver_change=args.allow_driver_change)
    print(json.dumps({key: result[key] for key in ("status", "common_completed_study_dates", "primary_quote_age_ms", "reference_quote_age_ms", "joined_all_regime_daily_pair_rows")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
