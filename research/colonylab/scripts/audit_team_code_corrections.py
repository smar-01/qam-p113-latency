"""Independently audit the same-input teammate-code correction experiment.

The audit imports neither analysis variant nor its driver. It verifies saved
sources and code, exact comparison grids, numerical negative controls, fixed
bin normalization bounds, fixed-support cohorts and independently reconstructed
curve summaries. It does not claim to re-match every raw quote event.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MID_ID = ["day", "clock", "symbol", "venue"]
REACTION_ID = ["day", "clock", "symbol", "leader", "follower"]
EVENT_ID = ["stage", "day", "clock", "symbol", "leader", "venue", "offset_us"]
CC_ID = ["stage", "day", "clock", "symbol", "x", "y"]
DAY_FILES = ["midpoint_diagnostics.csv", "reaction_comparison.csv", "event_study_comparison.csv",
             "leadlag_comparison.csv", "fullcurves.csv", "regime_series.csv"]


def require(condition, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve(root: Path, value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def exact_grid(frame: pd.DataFrame, fields: list[str], expected: set[tuple], context: str) -> None:
    require(set(fields) <= set(frame), f"{context}: missing identity columns.")
    require(not frame.duplicated(fields).any(), f"{context}: duplicate identities.")
    observed = set(frame[fields].itertuples(index=False, name=None))
    require(observed == expected,
            f"{context}: missing {len(expected - observed)} or unexpected {len(observed - expected)} rows.")


def same(actual, expected, context: str, *, atol=1e-10) -> None:
    a, e = np.asarray(actual, float), np.asarray(expected, float)
    require(a.shape == e.shape and np.allclose(a, e, equal_nan=True, rtol=1e-10, atol=atol),
            f"{context}: numerical reconstruction disagrees.")


def integers(frame: pd.DataFrame, columns: list[str], context: str) -> None:
    require(set(columns) <= set(frame), f"{context}: missing counts.")
    values = frame[columns].to_numpy(float)
    require(np.isfinite(values).all() and (values >= 0).all() and (values == np.floor(values)).all(),
            f"{context}: counts must be nonnegative integers.")


def settings(protocol: dict, root: Path) -> dict:
    config = protocol.get("configuration") or protocol.get("config")
    if not isinstance(config, dict):
        source = protocol.get("source_config", protocol.get("config_path", "wide_network.toml"))
        config = tomllib.loads(resolve(root, source).read_text(encoding="utf-8"))
    days = protocol.get("days") or config["training_days"] + config["held_out_days"]
    datasets = protocol.get("datasets", config["datasets"])
    # The baseline package uses venue labels; acquisition uses dataset codes.
    mapping = {"XNAS.ITCH": "XNAS", "XNYS.PILLAR": "XNYS", "ARCX.PILLAR": "ARCX",
               "BATS.PITCH": "BATS", "EDGX.PITCH": "EDGX", "MEMX.MEMOIR": "MEMX",
               "BATY.PITCH": "BATY", "EDGA.PITCH": "EDGA"}
    venues = protocol.get("venues", [mapping.get(dataset, dataset) for dataset in datasets])
    symbols = protocol.get("symbols", config["symbols"])
    clocks = protocol.get("clocks", ["ts_recv", "ts_event"])
    horizon = protocol.get("horizon_ms", protocol.get("reaction_horizon_ms", 10))
    offsets = protocol.get("event_study_offsets_us", protocol.get("offsets_us"))
    require(offsets is not None, "Protocol lacks exact event-study offsets.")
    lag_bin = protocol.get("leadlag_bin_us", protocol.get("cc_bin_us", 10))
    max_lag = protocol.get("leadlag_max_lag_us", protocol.get("cc_max_lag_us", 2000))
    require(lag_bin > 0 and max_lag >= 0 and max_lag % lag_bin == 0,
            "Invalid fixed-bin lag protocol.")
    require(days == sorted(set(days)), "Protocol dates must be unique and chronological.")
    return {"config": config, "days": days, "datasets": datasets, "venues": venues,
            "symbols": symbols, "clocks": clocks, "horizon_ms": horizon, "offsets_us": offsets,
            "lag_bin_us": lag_bin, "max_lag_us": max_lag,
            "leader": protocol.get("event_study_leader", protocol.get("leader", "XNAS"))}


def _field(frame: pd.DataFrame, prefix: str, name: str) -> str:
    for column in (f"{prefix}_{name}", f"{prefix}_n_{name}" if not name.startswith("n_") else ""):
        if column in frame:
            return column
    raise ValueError(f"Reaction comparison lacks {prefix}_{name}.")


def audit_reaction(frame: pd.DataFrame, expected: set[tuple], horizon_ms: float) -> dict:
    exact_grid(frame, REACTION_ID, expected, "Reaction comparison")
    for name in ("n_events", "n_behind", "n_matched", "median_us"):
        old, fixed = _field(frame, "baseline", name), _field(frame, "numerical", name)
        if name.startswith("n_"):
            integers(frame, [old, fixed], "Reaction negative control")
        same(frame[old], frame[fixed], f"Unchanged bid/ask {name}")
    for prefix in ("baseline", "numerical"):
        events, behind, matched = (_field(frame, prefix, name) for name in ("n_events", "n_behind", "n_matched"))
        require((frame[matched] <= frame[behind]).all() and (frame[behind] <= frame[events]).all(),
                "Reaction counts are inconsistent.")
        medians = frame[_field(frame, prefix, "median_us")].to_numpy(float)
        have = frame[matched].to_numpy(float) > 0
        require(np.isfinite(medians[have]).all() and np.isnan(medians[~have]).all(),
                "Reaction medians do not correspond to matched support.")
        require((medians[have] > 0).all() and (medians[have] <= horizon_ms * 1000 + 1e-10).all(),
                "Reaction median lies outside its horizon.")
    require("exact_equal" in frame and frame.exact_equal.astype(str).str.lower().eq("true").all(),
            "The driver's complete reaction-event negative control did not pass.")
    return {"reaction_pair_cases": len(frame), "reaction_numerical_negative_control": True}


def audit_curves(curves: pd.DataFrame, summaries: pd.DataFrame, expected: set[tuple], lags: list[float]) -> dict:
    exact_grid(summaries, CC_ID, expected, "Lead-lag summaries")
    expected_curves = {(*key, lag) for key in expected for lag in lags}
    exact_grid(curves, [*CC_ID, "lag_us"], expected_curves, "Lead-lag curves")
    require("corr" in curves, "Lead-lag curves lack coefficients.")
    fixed = curves.loc[curves.stage.eq("numerical_only"), "corr"].to_numpy(float)
    require(not np.isinf(fixed).any() and (np.abs(fixed[np.isfinite(fixed)]) <= 1 + 1e-12).all(),
            "Corrected fixed-bin coefficient violates its unit bound.")
    indexed = summaries.set_index(CC_ID)
    tied_peaks = 0
    for key, group in curves.groupby(CC_ID, sort=False):
        row = indexed.loc[key]
        group = group.sort_values("lag_us")
        values, lag = group["corr"].to_numpy(float), group.lag_us.to_numpy(float)
        if np.isfinite(values).any():
            position = int(np.nanargmax(values))
            peak, value = lag[position], values[position]
            same([row.peak_corr], [value], "Lead-lag peak height")
            # CSV round trips can make two adjacent floating values identical.
            # The recorded peak must still be a true maximum within rounding
            # precision; do not force the first serialized maximum's lag.
            ties = np.isfinite(values) & np.isclose(values, value, rtol=1e-12, atol=1e-14)
            tied_peaks += int(ties.sum() > 1)
            require(np.isclose(lag[ties], row.peak_lag_us, rtol=0, atol=1e-10).any(),
                    "Lead-lag peak lag is not a maximizer of its saved curve.")
        else:
            peak = value = np.nan
            same([row.peak_lag_us, row.peak_corr], [peak, value], "Lead-lag peak")
        pos, neg = np.nansum(values[lag > 0] ** 2), np.nansum(values[lag < 0] ** 2)
        if neg > 0:
            same([row.llr], [pos / neg], "Lead-lag squared-energy ratio")
        else:
            require(pd.isna(row.llr) or np.isposinf(row.llr), "Undefined lead-lag ratio is neither missing nor infinity.")
    return {"leadlag_summary_rows": len(summaries), "leadlag_curve_rows": len(curves),
            "corrected_crossproducts_bounded": True,
            "curves_with_rounding_tied_peak_values": tied_peaks,
            "maximum_abs_corrected_crossproduct": float(np.nanmax(np.abs(fixed))) if np.isfinite(fixed).any() else None}


def audit_checkpoints(out: Path, protocol: dict, provenance: dict, cfg: dict) -> dict:
    code = [*provenance["original_code"], *provenance["corrected_code"], *provenance["runner_code"]]
    tables = {name: [] for name in DAY_FILES}
    quote_rows = 0
    for day in cfg["days"]:
        directory = out / "days" / day
        marker = read_json(directory / "complete.json")
        identity = marker["identity"]
        expected_sources = [entry["content_sha256"] for entry in provenance["sources"]
                            if entry["request"]["start"][:10] == day]
        require(identity == {"protocol": protocol, "code": code, "sources": expected_sources},
                f"Day checkpoint identity differs: {day}")
        before, after = marker["shared_quote_sha256_before"], marker["shared_quote_sha256_after"]
        require(before == after and len(before) == 64 and all(char in "0123456789abcdef" for char in before),
                f"A variant changed its shared quote input: {day}")
        raw_rows = sum(entry["rows"] for entry in provenance["sources"] if entry["request"]["start"][:10] == day)
        require(marker["raw_rows"] == raw_rows and 0 < marker["quote_rows"] <= raw_rows,
                f"Day checkpoint row counts differ: {day}")
        quote_rows += marker["quote_rows"]
        require(set(marker["files"]) == set(DAY_FILES), f"Day checkpoint does not cover every output: {day}")
        for name in DAY_FILES:
            path = directory / name
            require(path.is_file() and sha256(path) == marker["files"][name], f"Day output fingerprint changed: {day}/{name}")
            frame = pd.read_csv(path)
            require(frame.day.eq(day).all(), f"Day output contains another date: {day}/{name}")
            tables[name].append(frame)
    keys = {"midpoint_diagnostics.csv": MID_ID, "reaction_comparison.csv": REACTION_ID,
            "event_study_comparison.csv": EVENT_ID, "leadlag_comparison.csv": CC_ID,
            "fullcurves.csv": [*CC_ID, "lag_us"], "regime_series.csv": ["stage", "day", "clock", "symbol", "t"]}
    for name, pieces in tables.items():
        rebuilt = pd.concat(pieces, ignore_index=True)
        observed = pd.read_csv(out / name)
        require(set(rebuilt) == set(observed), f"Combined output columns differ: {name}")
        rebuilt = rebuilt.sort_values(keys[name], kind="stable").reset_index(drop=True)
        observed = observed.sort_values(keys[name], kind="stable").reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(observed[rebuilt.columns], rebuilt, check_dtype=False,
                                          check_exact=False, rtol=1e-10, atol=1e-10)
        except AssertionError as exc:
            raise ValueError(f"Combined output differs from hash-bound day results: {name}") from exc
    return {"day_checkpoints_verified": len(cfg["days"]), "shared_quote_inputs_unmutated": True,
            "shared_price_event_rows": quote_rows, "combined_tables_reconciled_to_day_files": len(DAY_FILES)}


def audit_summary(out: Path, midpoint: pd.DataFrame, reaction: pd.DataFrame,
                  event: pd.DataFrame, cc: pd.DataFrame, result: dict) -> dict:
    summary = read_json(out / "summary.json")
    require(summary["status"] == "completed", "Analysis summary is not complete.")
    values = {"source_requests": result["raw_source_requests"], "raw_records": result["raw_records_reconciled"],
              "midpoint_series_checked": len(midpoint), "float_midpoint_events": int(midpoint.float_events.sum()),
              "exact_midpoint_events": int(midpoint.exact_events.sum()),
              "float_only_phantom_events": int(midpoint.phantom_events.sum()),
              "reaction_tasks_compared": len(reaction), "reaction_outputs_identical": True}
    for field, expected in values.items():
        require(summary[field] == expected, f"Summary {field} differs from comparison tables.")
    paired = cc.loc[cc.stage.eq("original")].merge(cc.loc[cc.stage.eq("numerical_only")],
               on=["day", "clock", "symbol", "x", "y"], suffixes=("_original", "_fixed"), validate="one_to_one")
    change = paired.peak_lag_us_fixed.to_numpy() - paired.peak_lag_us_original.to_numpy()
    changed = (~np.isclose(paired.peak_lag_us_fixed, paired.peak_lag_us_original, equal_nan=True)).sum()
    metrics = {"leadlag_pair_tasks": len(paired), "leadlag_peak_lag_changed_tasks": int(changed),
               "leadlag_peak_lag_change_median_us": float(np.nanmedian(np.abs(change))),
               "leadlag_peak_lag_change_max_us": float(np.nanmax(np.abs(change))),
               "maximum_corrected_peak_crossproduct": float(cc.loc[cc.stage.eq("numerical_only"), "peak_corr"].max())}
    matched = event.loc[event.stage.eq("original")].merge(event.loc[event.stage.eq("numerical_only")],
              on=["day", "clock", "symbol", "leader", "venue", "offset_us"], suffixes=("_original", "_fixed"), validate="one_to_one")
    metrics["maximum_absolute_numerical_event_study_change"] = float((matched.share_fixed - matched.share_original).abs().max())
    for field, expected in metrics.items():
        same([summary[field]], [expected], f"Summary {field}")
    diagnostics = pd.DataFrame(summary["event_study_diagnostics"])
    exact_grid(diagnostics, ["stage", "symbol"], set(event[["stage", "symbol"]].itertuples(index=False, name=None)), "Summary event-study diagnostics")
    diagnostics = diagnostics.set_index(["stage", "symbol"])
    for key, group in event.groupby(["stage", "symbol"]):
        row = diagnostics.loc[key]
        require(int(row.rows) == len(group) and int(row.finite_rows) == int(group.share.notna().sum())
                and int(row.minimum_event_count) == int(group.n.min()) and int(row.maximum_event_count) == int(group.n.max()),
                "Summary event-study support differs from curves.")
        same([row.maximum_absolute_mean_share], [group.share.abs().max()], "Summary maximum response ratio")
    return {"summary_metrics_independently_reconstructed": True, **metrics}


def audit_quality_reuse(out: Path, root: Path, provenance: dict, cfg: dict,
                        reaction: pd.DataFrame, expected: set[tuple]) -> dict:
    receipt = read_json(out / "quality_comparison_provenance.json")
    previous = root / "out" / "latency"
    require(receipt["source_identity_equal"] is True and receipt["rows"] == len(reaction),
            "Quality-stage source or row identity is incomplete.")
    require(sha256(previous / "daily_latency.csv") == receipt["input_daily_sha256"]
            and sha256(previous / "result_audit.json") == receipt["input_audit_sha256"]
            and read_json(previous / "result_audit.json")["status"] == "passed",
            "Borrowed quality results differ from their saved audit identity.")
    prior_provenance = read_json(previous / "provenance.json")
    inventory = {(entry["request"]["dataset"], entry["request"]["start"]): entry
                 for entry in prior_provenance["sources"] if entry["study"] == "wide_network"}
    for source in provenance["sources"]:
        entry = inventory[(source["request"]["dataset"], source["request"]["start"])]
        require(entry["request"] == source["request"] and entry["content_sha256"] == source["content_sha256"],
                "Quality stage does not use the same exact raw sources.")
    frame = pd.read_csv(out / "quality_stage_comparison.csv")
    exact_grid(frame, REACTION_ID, expected, "Quality-stage comparison")
    rebuilt = pd.read_csv(previous / "daily_latency.csv")
    rebuilt = rebuilt.loc[rebuilt.study.eq("wide_network") & rebuilt.regime.eq("all") & rebuilt.horizon_ms.eq(cfg["horizon_ms"])].copy()
    rebuilt = rebuilt.rename(columns={"date": "day"})
    mapping = {"XNAS.ITCH": "XNAS", "XNYS.PILLAR": "XNYS", "ARCX.PILLAR": "ARCX", "BATS.PITCH": "BATS", "EDGX.PITCH": "EDGX"}
    for field in ("leader", "follower"):
        rebuilt[field] = rebuilt[field].map(mapping)
    rebuilt = rebuilt.sort_values(REACTION_ID).reset_index(drop=True)
    frame = frame.sort_values(REACTION_ID).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(frame[rebuilt.columns], rebuilt, check_dtype=False,
                                      check_exact=False, rtol=1e-10, atol=1e-10)
        original = reaction.sort_values(REACTION_ID).reset_index(drop=True)
        pd.testing.assert_frame_equal(frame[original.columns], original, check_dtype=False,
                                      check_exact=False, rtol=1e-10, atol=1e-10)
    except AssertionError as exc:
        raise ValueError("Quality-stage rows differ from the two separately audited input tables.") from exc
    return {"quality_stage_rows_verified": len(frame), "quality_stage_same_sources_verified": True,
            "quality_stage_comparison_sha256": sha256(out / "quality_stage_comparison.csv")}


def audit(out_dir: str | Path, root: str | Path = ROOT) -> dict:
    root, out = Path(root).resolve(), Path(out_dir).resolve()
    result = {"status": "failed", "created_utc": datetime.now(timezone.utc).isoformat(),
              "errors": [], "raw_sources_freshly_hashed": 0, "code_files_verified": 0,
              "scope": "Independent same-input correction grids, hashes and arithmetic; no raw re-matching by this final auditor."}
    try:
        protocol = read_json(out / "protocol.json")
        provenance = read_json(out / "provenance.json")
        require(provenance["protocol_sha256"] in {
            sha256(out / "protocol.json"), hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()},
            "Protocol fingerprint differs from run provenance.")
        cfg = settings(protocol, root)
        expected_sources = {(day, dataset) for day, dataset in itertools.product(cfg["days"], cfg["datasets"])}
        expected_requests = {}
        for day, dataset in expected_sources:
            start = pd.Timestamp(f"{day}T{cfg['config']['start_time_utc']}Z")
            expected_requests[(day, dataset)] = {
                "dataset": dataset, "schema": "mbp-1", "symbols": cfg["symbols"],
                "stype_in": "raw_symbol", "start": start.isoformat(),
                "end": (start + pd.Timedelta(minutes=cfg["config"]["minutes"])).isoformat(),
                "limit": cfg["config"]["record_limit_per_request"],
            }
        observed_sources, records = [], 0
        for source in provenance["sources"]:
            request = source["request"]
            observed_sources.append((request["start"][:10], request["dataset"]))
            require(request == expected_requests.get(observed_sources[-1]),
                    "Source request differs from the exact symbols/schema/window/limit grid.")
            require(type(source["rows"]) is int and 0 < source["rows"] < request["limit"],
                    "Source row count is empty or capped.")
            path = resolve(root, source["data_path"])
            require(path.is_file() and sha256(path) == source["content_sha256"], f"Raw source fingerprint changed: {path}")
            result["raw_sources_freshly_hashed"] += 1
            records += source["rows"]
        require(len(observed_sources) == len(set(observed_sources)) and set(observed_sources) == expected_sources,
                "Source grid is missing, duplicated or unexpected.")
        for key in ("original_code", "corrected_code", "runner_code"):
            require(provenance.get(key), f"Missing {key} fingerprints.")
            for source in provenance[key]:
                path = resolve(root, source["path"])
                require(path.is_file() and sha256(path) == source["sha256"], f"Code fingerprint changed: {path}")
                result["code_files_verified"] += 1
        result.update(raw_source_requests=len(observed_sources), raw_records_reconciled=records)
        base = list(itertools.product(cfg["days"], cfg["clocks"], cfg["symbols"]))
        midpoint = pd.read_csv(out / "midpoint_diagnostics.csv")
        exact_grid(midpoint, MID_ID, {(*key, venue) for key in base for venue in cfg["venues"]}, "Midpoint diagnostics")
        integers(midpoint, ["float_events", "exact_events", "phantom_events", "corrected_events"], "Midpoint diagnostics")
        require((midpoint.phantom_events <= midpoint.float_events).all(), "Phantom events exceed floating midpoint events.")
        require((midpoint.corrected_events == midpoint.exact_events).all(), "Corrected midpoint events differ from exact integer events.")
        if "min_abs_float_move" in midpoint:
            values = midpoint.min_abs_float_move.to_numpy(float)
            require((values[np.isfinite(values)] > 0).all(), "Nonzero floating move minimum is not positive.")
        result.update(midpoint_cases=len(midpoint), floating_midpoint_events=int(midpoint.float_events.sum()),
                      exact_midpoint_events=int(midpoint.exact_events.sum()),
                      floating_phantom_midpoint_events=int(midpoint.phantom_events.sum()),
                      corrected_midpoint_events=int(midpoint.corrected_events.sum()))
        directed = [(a, b) for a in cfg["venues"] for b in cfg["venues"] if a != b]
        reaction = pd.read_csv(out / "reaction_comparison.csv")
        result.update(audit_reaction(reaction, {(*key, a, b) for key in base for a, b in directed}, cfg["horizon_ms"]))
        event = pd.read_csv(out / "event_study_comparison.csv")
        stages = ["original", "numerical_only", "support_corrected"]
        expected_event = {(stage, *key, cfg["leader"], venue, offset) for stage in stages for key in base
                          for venue in cfg["venues"] for offset in cfg["offsets_us"]}
        exact_grid(event, EVENT_ID, expected_event, "Event-study curves")
        integers(event, ["n"], "Event-study support")
        require("share" in event and not np.isinf(event.share.to_numpy(float)).any(), "Event-study shares contain infinity.")
        require(np.isfinite(event.loc[event.n.gt(0), "share"]).all()
                and event.loc[event.n.eq(0), "share"].isna().all(), "Event-study means and support disagree.")
        leader_limits = midpoint.set_index(MID_ID)
        for row in event.itertuples(index=False):
            limits = leader_limits.loc[(row.day, row.clock, row.symbol, row.leader)]
            maximum = limits.float_events if row.stage == "original" else limits.corrected_events
            require(row.n <= maximum, "Event-study cohort exceeds available leader midpoint events.")
        fixed_support = event.loc[event.stage.eq("support_corrected")]
        require(fixed_support.groupby(["day", "clock", "symbol", "leader", "venue"]).n.nunique().eq(1).all(),
                "Complete-support event-study cohort changes across offsets.")
        counts = event.pivot(index=["day", "clock", "symbol", "leader", "venue", "offset_us"], columns="stage", values="n")
        require((counts.support_corrected <= counts.numerical_only).all(), "Complete-support cohort exceeds numerical-only cohort.")
        result["event_study_rows"] = len(event)
        result["max_abs_event_study_share_by_stage"] = {
            stage: float(group.share.abs().max()) if group.share.notna().any() else None
            for stage, group in event.groupby("stage", sort=True)}
        unordered = list(itertools.combinations(sorted(cfg["venues"]), 2))
        cc_stages = ["original", "integer_only", "numerical_only"]
        expected_cc = {(stage, *key, a, b) for stage in cc_stages for key in base for a, b in unordered}
        lags = np.arange(-cfg["max_lag_us"], cfg["max_lag_us"] + cfg["lag_bin_us"], cfg["lag_bin_us"]).astype(float).tolist()
        cc = pd.read_csv(out / "leadlag_comparison.csv", float_precision="round_trip")
        result.update(audit_curves(pd.read_csv(out / "fullcurves.csv", float_precision="round_trip"), cc, expected_cc, lags))
        if protocol.get("checkpoint_audit_required", True):
            result.update(audit_checkpoints(out, protocol, provenance, cfg))
        if protocol.get("summary_audit_required", True):
            result.update(audit_summary(out, midpoint, reaction, event, cc, result))
        if "quality_controlled" in protocol.get("stages", {}):
            result.update(audit_quality_reuse(out, root, provenance, cfg, reaction,
                          {(*key, a, b) for key in base for a, b in directed}))
        result["output_sha256"] = {name: sha256(out / name) for name in (
            "midpoint_diagnostics.csv", "reaction_comparison.csv", "event_study_comparison.csv",
            "leadlag_comparison.csv", "fullcurves.csv", "regime_series.csv", "quality_stage_comparison.csv")
            if (out / name).is_file()}
        result["status"] = "passed"
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    out.mkdir(parents=True, exist_ok=True)
    result["audit_script_sha256"] = sha256(Path(__file__))
    (out / "result_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "team_code_corrections")
    args = parser.parse_args()
    result = audit(args.out)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
