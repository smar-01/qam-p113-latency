"""Independent offline integrity and arithmetic audit of latency result tables.

Every source is freshly hashed and compared with the run's provenance and
available historical archive identity.  The analysis loader decoded every DBN;
this second audit does not repeat DBN parsing.  It independently checks the
exact acquisition/result grids, count conservation, conditional-mean and
restricted-wait arithmetic, sampled raw-event checks, and date-block inference.
It deliberately imports neither the latency engine nor its inference module.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tomllib

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK = ["clock", "symbol", "leader", "follower", "horizon_ms"]
DAILY_ID = ["study", "phase", "date", *TASK, "regime"]
INFERENCE_ID = ["study", "phase", *TASK, "contrast"]
COUNT_COLUMNS = [
    "n_events", "n_already", "n_no_state", "n_behind", "n_eligible", "n_matched", "n_censored",
    "n_quality_truncated", "n_right_truncated", "n_observed_matches", "known_matches_truncated",
    "sum_latency_ns", "restricted_wait_sum_ns",
]


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, path: str | Path) -> Path:
    result = Path(path)
    return result if result.is_absolute() else root / result


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _request_key(request: dict) -> str:
    return json.dumps(request, sort_keys=True)


def _acquisition_design(configuration: dict) -> dict:
    """Billing preferences do not change an exact historical data request."""
    return {key: value for key, value in configuration.items()
            if key not in {"billing_assumed_included", "max_cost_usd"}}


def _expected_requests(config: dict) -> list[dict]:
    """Reconstruct request identities without importing acquisition code."""
    require(config["schemas"] == ["mbp-1"], "Latency source configuration is not MBP-1.")
    result = []
    for day in config["training_days"] + config["held_out_days"]:
        start = pd.Timestamp(f"{day}T{config['start_time_utc']}Z")
        end = start + pd.Timedelta(minutes=config["minutes"])
        for dataset in config["datasets"]:
            result.append({"dataset": dataset, "schema": "mbp-1", "symbols": config["symbols"],
                           "stype_in": "raw_symbol", "start": start.isoformat(), "end": end.isoformat(),
                           "limit": config["record_limit_per_request"]})
    return result


def exact_grid(frame: pd.DataFrame, columns: list[str], expected: set[tuple], context: str) -> None:
    require(set(columns) <= set(frame), f"{context}: missing index columns.")
    require(not frame.duplicated(columns).any(), f"{context}: duplicate index rows.")
    observed = set(frame[columns].itertuples(index=False, name=None))
    require(observed == expected,
            f"{context}: missing {len(expected - observed)} or unexpected {len(observed - expected)} rows.")


def _same_array(actual, expected, context: str, *, atol=1e-8, rtol=1e-10) -> None:
    left, right = np.asarray(actual, float), np.asarray(expected, float)
    require(left.shape == right.shape and np.allclose(left, right, equal_nan=True, atol=atol, rtol=rtol),
            f"{context}: values disagree with independent reconstruction.")


def _protocol_settings(protocol: dict) -> dict:
    nested = protocol.get("inference", {})
    def setting(short, long, default):
        return nested.get(short, protocol.get(long, default))
    result = {
        "min_matches": setting("min_matches", "minimum_matched_events_per_date", 20),
        "replicates": setting("bootstrap_replicates", "bootstrap_replicates", 2000),
        "block": setting("block_sessions", "bootstrap_block_sessions", 5),
        "seed": setting("seed", "bootstrap_seed", 20261102),
        "primary_study": setting("primary_study", "primary_study", "fresh_confirmation"),
        "primary_phase": setting("primary_phase", "primary_phase", "held_out"),
        "min_dates": setting("min_inference_dates", "minimum_inference_dates", 20),
    }
    for field in ("min_matches", "replicates", "block", "min_dates"):
        require(type(result[field]) is int and result[field] > 0, f"Invalid protocol {field}.")
    require(type(result["seed"]) is int and result["seed"] >= 0, "Invalid protocol seed.")
    return result


def _independent_counts(n_dates: int, settings: dict) -> np.ndarray:
    """Independent loop implementation of the nonwrapping moving block draws."""
    length = min(settings["block"], n_dates)
    blocks = (n_dates + length - 1) // length
    starts = np.random.default_rng(settings["seed"]).integers(
        0, n_dates - length + 1, size=(settings["replicates"], blocks))
    weights = np.zeros((settings["replicates"], n_dates), dtype=np.int32)
    for replicate, row in enumerate(starts):
        position = 0
        for start in row:
            take = min(length, n_dates - position)
            weights[replicate, int(start):int(start) + take] += 1
            position += take
    return weights


def _independent_inference(daily: pd.DataFrame, configurations: dict, settings: dict) -> pd.DataFrame:
    """Rebuild equal-date task estimates and paired intervals from CSV only."""
    records = []
    for study, config in sorted(configurations.items()):
        for phase, date_key in (("training", "training_days"), ("held_out", "held_out_days")):
            dates = config[date_key]
            scope = daily.loc[daily.study.eq(study) & daily.phase.eq(phase)]
            primary = study == settings["primary_study"] and phase == settings["primary_phase"]
            weights = (_independent_counts(len(dates), settings)
                       if primary and len(dates) >= settings["min_dates"] else None)
            for task, group in scope.groupby(TASK, sort=True):
                pivot = group.pivot(index="date", columns="regime", values="median_us").reindex(dates)
                support = group.pivot(index="date", columns="regime", values="n_matched").reindex(dates)
                for regime in ("all", "calm", "volatile"):
                    pivot.loc[support[regime] < settings["min_matches"], regime] = np.nan
                for contrast in ("latency", "volatile_vs_calm"):
                    if contrast == "latency":
                        values = pivot["all"].to_numpy(float)
                        calm = np.full(len(dates), np.nan)
                        volatile = np.full(len(dates), np.nan)
                    else:
                        calm, volatile = pivot.calm.to_numpy(float), pivot.volatile.to_numpy(float)
                        paired = np.isfinite(calm) & np.isfinite(volatile)
                        calm, volatile = np.where(paired, calm, np.nan), np.where(paired, volatile, np.nan)
                        values = volatile - calm
                    finite = np.isfinite(values)
                    n = int(finite.sum())
                    estimate = float(values[finite].mean()) if n else np.nan
                    calm_mean = float(calm[finite].mean()) if contrast != "latency" and n else np.nan
                    volatile_mean = float(volatile[finite].mean()) if contrast != "latency" and n else np.nan
                    relative = 100 * estimate / calm_mean if calm_mean > 0 else np.nan
                    lower = upper = rel_lower = rel_upper = np.nan
                    if not primary:
                        status = "descriptive"
                    elif n < settings["min_dates"]:
                        status = "insufficient_supported_dates"
                    else:
                        denominator = weights[:, finite].sum(axis=1)
                        if (denominator == 0).any():
                            status = "undefined_resample"
                        else:
                            draws = weights[:, finite] @ values[finite] / denominator
                            lower, upper = np.quantile(draws, [.025, .975])
                            status = "ok"
                            if contrast != "latency":
                                calm_draws = weights[:, finite] @ calm[finite] / denominator
                                if (calm_draws > 0).all():
                                    rel_lower, rel_upper = np.quantile(100 * draws / calm_draws, [.025, .975])
                    records.append({
                        "study": study, "phase": phase, **dict(zip(TASK, task)), "contrast": contrast,
                        "n_dates": n, "n_calendar_dates": len(dates), "n_observed_calendar_dates": len(dates),
                        "n_missing_or_sparse_dates": len(dates) - n,
                        "estimate_us": estimate, "calm_us": calm_mean, "volatile_us": volatile_mean,
                        "relative_change_percent": relative, "ci95_low_us": lower, "ci95_high_us": upper,
                        "relative_ci95_low_percent": rel_lower, "relative_ci95_high_percent": rel_upper,
                        "interval_status": status,
                        "inference_role": "primary_post_hoc" if primary else "descriptive",
                        "bootstrap_replicates": settings["replicates"] if status == "ok" else 0,
                        "block_sessions": min(settings["block"], len(dates)), "seed": settings["seed"],
                    })
    return pd.DataFrame(records)


def _check_daily(daily: pd.DataFrame) -> dict:
    required = {*DAILY_ID, *COUNT_COLUMNS, "match_rate", "p10_us", "median_us", "p90_us", "mean_us",
                "restricted_mean_wait_us"}
    require(required <= set(daily), "Daily latency table lacks required counts/statistics.")
    values = daily[COUNT_COLUMNS].to_numpy(float)
    require(np.isfinite(values).all() and (values >= 0).all() and (values == np.floor(values)).all(),
            "Daily counts and latency sums must be finite nonnegative integers.")
    require((daily.n_events == daily.n_already + daily.n_no_state + daily.n_behind).all(),
            "Daily event conservation fails.")
    require((daily.n_behind == daily.n_eligible + daily.n_quality_truncated + daily.n_right_truncated).all(),
            "Daily behind-event conservation fails.")
    require((daily.n_eligible == daily.n_matched + daily.n_censored).all(),
            "Daily eligible-event conservation fails.")
    require((daily.n_observed_matches == daily.n_matched + daily.known_matches_truncated).all(),
            "Daily observed-match conservation fails.")
    require((daily.known_matches_truncated <= daily.n_quality_truncated + daily.n_right_truncated).all(),
            "Truncated known matches exceed truncated behind-events.")
    if "n_complete_behind" in daily:
        require((daily.n_complete_behind == daily.n_eligible).all(), "Complete-behind alias differs.")
    if "n_excluded" in daily:
        require((daily.n_excluded == daily.n_already + daily.n_no_state + daily.n_quality_truncated
                 + daily.n_right_truncated).all(), "Excluded-event count is inconsistent.")
    eligible = daily.n_eligible.to_numpy(float)
    matched = daily.n_matched.to_numpy(float)
    ratio = np.divide(matched, eligible, out=np.full(len(daily), np.nan), where=eligible > 0)
    _same_array(daily.match_rate, ratio, "Match rate")
    horizon_ns = daily.horizon_ms.to_numpy(float) * 1_000_000
    waiting = daily.sum_latency_ns.to_numpy(float) + daily.n_censored.to_numpy(float) * horizon_ns
    _same_array(daily.restricted_wait_sum_ns, waiting, "Restricted waiting-time sum", atol=.01)
    restricted = np.divide(waiting, eligible * 1000, out=np.full(len(daily), np.nan), where=eligible > 0)
    _same_array(daily.restricted_mean_wait_us, restricted, "Restricted mean waiting time")
    mean = np.divide(daily.sum_latency_ns.to_numpy(float), matched * 1000,
                     out=np.full(len(daily), np.nan), where=matched > 0)
    _same_array(daily.mean_us, mean, "Conditional mean latency")
    have_matches = matched > 0
    quantiles = daily[["p10_us", "median_us", "p90_us"]].to_numpy(float)
    require(np.isfinite(quantiles[have_matches]).all() and np.isnan(quantiles[~have_matches]).all(),
            "Matched quantiles must be finite exactly when there are matches.")
    require((quantiles[have_matches, 0] > 0).all() and
            (quantiles[have_matches, 0] <= quantiles[have_matches, 1] + 1e-9).all() and
            (quantiles[have_matches, 1] <= quantiles[have_matches, 2] + 1e-9).all() and
            (quantiles[have_matches, 2] <= daily.loc[have_matches, "horizon_ms"].to_numpy(float) * 1000 + 1e-8).all(),
            "Conditional quantiles are out of order or outside their matching horizon.")
    require((waiting >= 0).all() and (restricted[np.isfinite(restricted)] >= 0).all(),
            "Negative restricted waiting time.")
    # Calm/volatile are disjoint, but warm-up or unknown seconds need not be
    # either.  Their counts may be smaller than all; they cannot exceed it.
    group_id = [x for x in DAILY_ID if x != "regime"]
    all_counts = daily.loc[daily.regime.eq("all")].set_index(group_id)[COUNT_COLUMNS].sort_index()
    split = daily.loc[daily.regime.isin(["calm", "volatile"])].groupby(group_id)[COUNT_COLUMNS].sum().reindex(all_counts.index)
    require((split.to_numpy(float) <= all_counts.to_numpy(float) + .01).all(),
            "Calm/volatile counts overlap or exceed all-regime totals.")
    unknown_events = int((all_counts.n_events - split.n_events).sum())
    invariant = ["n_events", "n_already", "n_no_state", "n_behind"]
    base = [x for x in DAILY_ID if x != "horizon_ms"]
    require((daily.groupby(base)[invariant].nunique() <= 1).all().all(),
            "Leader-event counts change across horizons.")
    return {"daily_rows": len(daily), "unknown_regime_pair_events_across_horizons": unknown_events,
            "count_conservation": True, "conditional_quantiles_ordered": True,
            "conditional_mean_reconstructed": True, "restricted_wait_reconstructed": True}


def audit_latency(out_dir: str | Path, *, root: str | Path | None = None,
                  rehash: bool = True, strict_code: bool = True) -> dict:
    """Return a passed/failed audit; never mutate sources or call the network."""
    root = Path(root).resolve() if root is not None else ROOT
    out = _resolve(root, out_dir)
    result = {"status": "failed", "created_utc": datetime.now(timezone.utc).isoformat(),
              "errors": [], "raw_hash_verification_requested": bool(rehash),
              "raw_hashes_freshly_verified": False, "raw_sources_freshly_hashed": 0,
              "raw_dbn_reparsed_by_final_auditor": False,
              "raw_row_count_evidence": "Every source decoded by analysis loader and reconciled to completed acquisition receipt; final audit rehashes without DBN reparse.",
              "limitations": ["Independent event matching checks sample each directed pair/side; they do not re-match every raw event.",
                              "Date-block intervals are marginal exploratory estimates; they are not multiplicity-adjusted or causal effects.",
                              "Twenty sessions contain roughly four nonoverlapping five-session blocks; interval precision remains approximate."]}
    try:
        protocol = _read_json(out / "protocol.json")
        provenance = _read_json(out / "provenance.json")
        settings = _protocol_settings(protocol)
        horizons = protocol["horizons_ms"]
        clocks = protocol["clocks"]
        require(len(horizons) == len(set(horizons)) and all(float(x) > 0 for x in horizons),
                "Protocol horizons are invalid or duplicated.")
        require(len(clocks) == len(set(clocks)) and set(clocks) <= {"ts_recv", "ts_event"} and clocks,
                "Protocol clocks are invalid or duplicated.")
        studies = provenance["studies"]
        require(studies and len({entry["name"] for entry in studies}) == len(studies),
                "Provenance study list is empty or duplicated.")
        configurations = {}
        for study in studies:
            config = study["configuration"]
            config_path = _resolve(root, study["config_path"])
            current = tomllib.loads(config_path.read_text(encoding="utf-8"))
            require(current == config, f"Saved study configuration differs from current config: {study['name']}.")
            configurations[study["name"]] = config
        if strict_code:
            code = provenance.get("computation_code", [])
            require(code, "No computation-code fingerprints in provenance.")
            if isinstance(code, dict):
                code = [{"path": key, "sha256": value} for key, value in code.items()]
            for entry in code:
                require(sha256_file(_resolve(root, entry["path"])) == entry["sha256"],
                        f"Computation code differs from saved analysis: {entry['path']}.")
            result["computation_code_files_verified"] = len(code)
        historical = {}
        archive = root / "research-archive-manifest.json"
        if archive.exists():
            for entry in _read_json(archive).get("files", []):
                historical[str(_resolve(root, entry["path"]).resolve())] = entry["sha256"]
        sources = provenance["sources"]
        expected_source_keys, raw_count = set(), 0
        actual_source_keys = []
        receipt_cache, historical_count = {}, 0
        for study, config in configurations.items():
            for request in _expected_requests(config):
                expected_source_keys.add((study, _request_key(request)))
        for entry in sources:
            study, request = entry["study"], entry["request"]
            require(study in configurations, "Unknown source study.")
            actual_source_keys.append((study, _request_key(request)))
            path = _resolve(root, entry.get("data_path", entry.get("path", "")))
            require(path.is_file(), f"Raw source is missing: {path}.")
            require(type(entry["rows"]) is int and 0 < entry["rows"] < request["limit"],
                    "Raw source row count is invalid or capped.")
            raw_count += entry["rows"]
            if rehash:
                require(sha256_file(path) == entry["content_sha256"], f"Raw source content changed: {path}.")
                result["raw_sources_freshly_hashed"] += 1
            prior = historical.get(str(path.resolve()))
            if prior is not None:
                require(prior == entry["content_sha256"], f"Historical raw fingerprint disagrees: {path}.")
                historical_count += 1
            manifest_path = _resolve(root, entry["acquisition_manifest"])
            if manifest_path not in receipt_cache:
                acquisition = _read_json(manifest_path)
                require(acquisition.get("complete") is True, "Acquisition manifest is incomplete.")
                require(_acquisition_design(acquisition["configuration"]) == _acquisition_design(configurations[study]),
                        "Acquisition configuration differs from result provenance.")
                receipt_cache[manifest_path] = acquisition
            matches = [saved for saved in receipt_cache[manifest_path]["requests"] if saved["request"] == request]
            require(len(matches) == 1 and matches[0].get("complete") is True
                    and not matches[0].get("record_limit_reached") and matches[0]["rows"] == entry["rows"],
                    "Source is not reconciled to one completed uncapped acquisition receipt.")
            require(matches[0]["file_size_bytes"] == path.stat().st_size,
                    "Raw byte size differs from acquisition receipt.")
        require(len(actual_source_keys) == len(set(actual_source_keys))
                and set(actual_source_keys) == expected_source_keys,
                "Raw source grid is missing, duplicated or unexpected.")
        result["raw_hashes_freshly_verified"] = bool(rehash)
        result.update(raw_source_requests=len(sources), raw_records_reconciled=raw_count,
                      sources_with_historical_archive_fingerprint=historical_count)
        expected_daily, expected_quality, expected_audits = set(), set(), set()
        for study, config in configurations.items():
            pairs = [(a, b) for a in config["datasets"] for b in config["datasets"] if a != b]
            for phase, key in (("training", "training_days"), ("held_out", "held_out_days")):
                for date, clock, symbol, (leader, follower), horizon, regime in itertools.product(
                        config[key], clocks, config["symbols"], pairs, horizons, ("all", "calm", "volatile")):
                    expected_daily.add((study, phase, date, clock, symbol, leader, follower, horizon, regime))
                for date, clock, symbol, dataset in itertools.product(config[key], clocks, config["symbols"], config["datasets"]):
                    expected_quality.add((study, date, clock, symbol, dataset))
                for date, clock, symbol, (leader, follower), side in itertools.product(
                        config[key], clocks, config["symbols"], pairs, ("bid", "ask")):
                    expected_audits.add((study, date, clock, symbol, leader, follower, side))
        daily = pd.read_csv(out / "daily_latency.csv")
        exact_grid(daily, DAILY_ID, expected_daily, "Daily latency")
        result.update(_check_daily(daily))
        quality = pd.read_csv(out / "quote_quality.csv")
        exact_grid(quality, ["study", "date", "clock", "symbol", "dataset"], expected_quality, "Quote quality")
        require({"raw_records", "valid_state_records", "invalid_state_records", "state_records"} <= set(quality),
                "Quote quality table lacks raw/state counts.")
        for field in ("raw_records", "valid_state_records", "invalid_state_records", "state_records"):
            value = quality[field].to_numpy(float)
            require(np.isfinite(value).all() and (value >= 0).all() and (value == np.floor(value)).all(),
                    f"Invalid quote-quality count: {field}.")
        require((quality.state_records == quality.valid_state_records + quality.invalid_state_records).all(),
                "Quote state conservation fails.")
        raw_by_source = {(entry["study"], entry["request"]["start"][:10], entry["request"]["dataset"]): entry["rows"]
                         for entry in sources}
        for key, group in quality.groupby(["study", "date", "dataset", "clock"]):
            require(int(group.raw_records.sum()) == raw_by_source[key[:3]],
                    "Decoded symbol record counts differ from acquisition receipt.")
        result["quote_quality_rows"] = len(quality)
        audits = pd.read_csv(out / "day_audits.csv")
        audit_id = ["study", "date", "clock", "symbol", "leader", "follower", "side"]
        exact_grid(audits, audit_id, expected_audits, "Independent matching checks")
        require({"cases", "mismatches", "leader_events"} <= set(audits),
                "Independent matching check table lacks cases, mismatches or event counts.")
        for field in ("cases", "mismatches", "leader_events"):
            value = audits[field].to_numpy(float)
            require(np.isfinite(value).all() and (value >= 0).all() and (value == np.floor(value)).all(),
                    "Independent matching audit counts are invalid.")
        require((audits.mismatches == 0).all(), "Independent sampled event matching has mismatches.")
        require((audits.cases <= audits.leader_events).all()
                and ((audits.leader_events == 0) | (audits.cases > 0)).all(),
                "A nonempty pair/side lacks sampled event validation.")
        side_totals = audits.groupby(audit_id[:-1]).leader_events.sum().sort_index()
        all_daily = daily.loc[daily.regime.eq("all") & daily.horizon_ms.eq(horizons[0])].set_index(audit_id[:-1]).sort_index()
        _same_array(side_totals.to_numpy(), all_daily.n_events.to_numpy(), "Independent leader-event totals", atol=0, rtol=0)
        result.update(independent_matching_tasks=len(audits), independent_event_cases=int(audits.cases.sum()),
                      independent_matching_mismatches=0)
        stated = pd.read_csv(out / "inference.csv")
        reconstructed = _independent_inference(daily, configurations, settings)
        exact_grid(stated, INFERENCE_ID,
                   set(reconstructed[INFERENCE_ID].itertuples(index=False, name=None)), "Inference")
        stated = stated.set_index(INFERENCE_ID).sort_index()
        reconstructed = reconstructed.set_index(INFERENCE_ID).sort_index()
        numeric_fields = ["n_dates", "n_calendar_dates", "n_observed_calendar_dates", "n_missing_or_sparse_dates",
                          "estimate_us", "calm_us", "volatile_us", "relative_change_percent", "ci95_low_us",
                          "ci95_high_us", "relative_ci95_low_percent", "relative_ci95_high_percent",
                          "bootstrap_replicates", "block_sessions", "seed"]
        for field in numeric_fields:
            require(field in stated, f"Inference lacks {field}.")
            _same_array(stated[field], reconstructed[field], f"Inference {field}")
        require((stated.interval_status == reconstructed.interval_status).all(),
                "Inference support/interval statuses disagree.")
        require((stated.inference_role == reconstructed.inference_role).all(),
                "Inference primary/descriptive roles disagree.")
        result.update(inference_rows=len(stated), inference_estimates_and_intervals_reconstructed=True,
                      status="passed")
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("out/latency"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--no-rehash", action="store_true", help="Arithmetic-only mode; does not establish fresh raw integrity.")
    parser.add_argument("--allow-code-change", action="store_true", help="Do not verify analysis-code fingerprints.")
    args = parser.parse_args(argv)
    result = audit_latency(args.out_dir, root=args.root, rehash=not args.no_rehash,
                           strict_code=not args.allow_code_change)
    out = _resolve(args.root, args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
