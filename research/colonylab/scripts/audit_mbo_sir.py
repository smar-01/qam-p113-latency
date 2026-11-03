"""Independent final-artifact audit for the frozen full-MBO SIRS study.

No model is fitted or loaded from a fitted-model pickle. Probabilities are
rebuilt from JSON parameters with NumPy formulas, and session inference is
recomputed locally without importing the production prediction/statistics code.
Raw records are not decoded again: their independent full-grid audit is checked.
Audit consistency can pass for honestly reported nonconverged fits; scientific
eligibility and the overall support flag remain false in that case.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import time
import tomllib

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("independent_mbo_hazard", "sirs_network", "logistic_network", "past_shifted_sirs")
REFERENCES = ("independent_mbo_hazard", "logistic_network", "past_shifted_sirs")
SEMANTIC_FAILURES = (
    "duplicate_adds", "unmatched_cancels", "cancel_identity_mismatches", "oversize_cancels",
    "unmatched_modifies", "unknown_modify_invalidations", "modify_upserts", "modify_side_mismatches",
    "snapshot_without_clear", "invalid_snapshot_actions", "invalid_order_fields", "invalid_fill_fields",
    "invalid_trade_fields", "unknown_actions", "non_native_records", "unmatched_fills", "invalid_timestamps",
)
SCORE_FIELDS = ("rows", "positive_count", "log_loss_sum", "brier_sum", "numerical_guard_rows", "probability_clip_rows")


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def close(actual, expected, name, *, rtol=2e-8, atol=2e-9):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, rtol=rtol, atol=atol), f"Numeric mismatch: {name} ({actual!r} versus {expected!r}).")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_panel(path):
    # Only locally generated, provenance-checked study checkpoints are accepted.
    path = resolve(path)
    with path.open("rb") as stream:
        compressed = stream.read(2) == b"\x1f\x8b"
    with (gzip.open(path, "rb") if compressed else path.open("rb")) as stream:
        return pickle.load(stream)


def exact_grid(records, fields, expected, name):
    lookup = {}
    for record in records:
        key = tuple(record[field] for field in fields)
        require(key not in lookup, f"Duplicate {name}: {key}.")
        lookup[key] = record
    require(set(lookup) == set(expected), f"Incomplete or extra {name} grid.")
    return lookup


def request_grid(config):
    """Derive requests independently rather than calling acquisition helpers."""
    result = []
    def stamp(day, clock):
        return datetime.fromisoformat(day + "T" + clock).replace(tzinfo=timezone.utc)
    for day in config["training_days"] + config["held_out_days"]:
        end = stamp(day, config["session_end_utc"])
        validation = stamp(day, config["validation_start_utc"])
        for venue in config["datasets"]:
            for schema, start, stop, limit in (
                ("mbo", stamp(day, config["download_start_utc"]), end, config["record_limit_mbo"]),
                ("mbp-10", validation, validation + timedelta(minutes=config["validation_minutes"]), config["record_limit_aux"]),
            ):
                result.append({"dataset": venue, "schema": schema, "symbols": [config["symbol"]],
                               "stype_in": "raw_symbol", "start": start.isoformat(), "end": stop.isoformat(), "limit": limit})
        result.append({"dataset": config["market_dataset"], "schema": "bbo-1s", "symbols": [config["market_symbol"]],
                       "stype_in": "raw_symbol", "start": stamp(day, config["session_start_utc"]).isoformat(),
                       "end": end.isoformat(), "limit": config["record_limit_aux"]})
    return result


def validate_specification(config, protocol):
    venues = {"XNAS.ITCH", "XNYS.PILLAR", "BATS.PITCH", "ARCX.PILLAR", "EDGX.PITCH",
              "MEMX.MEMOIR", "BATY.PITCH", "EDGA.PITCH"}
    training = pd.bdate_range("2026-06-01", "2026-06-12").strftime("%Y-%m-%d").tolist()
    holdout = [day for day in pd.bdate_range("2026-06-15", "2026-07-14").strftime("%Y-%m-%d")
               if day not in {"2026-06-19", "2026-07-03"}]
    require(len(config["datasets"]) == 8 and set(config["datasets"]) == venues
            and config["training_days"] == training and config["held_out_days"] == holdout,
            "Venues or exact frozen training/test dates differ.")
    require(config["symbol"] == "AAPL" and config["market_symbol"] == "SPY"
            and config["market_dataset"] == "XNAS.ITCH" and config["anchor_step_ms"] == 100,
            "Frozen symbols or forecast grid differ.")
    for key, expected in (("download_start_utc", "00:00:00"), ("session_start_utc", "13:30:00"),
                          ("session_end_utc", "20:00:00"), ("validation_start_utc", "14:00:00"), ("validation_minutes", 5)):
        require(config[key] == expected, "Frozen request/session clock differs: " + key)
    require(protocol["forecast_horizon_ms"] == 10 and protocol["forecast_anchor_ms"] == 100
            and protocol["susceptible_infected_refractory"]["infectious_ms"] == 100
            and protocol["susceptible_infected_refractory"]["refractory_ms"] == 100,
            "SIRS state/horizon settings differ from the frozen specification.")


def finite_array(value, name, *, ndim=None):
    array = np.asarray(value, dtype=float)
    require(np.isfinite(array).all() and (ndim is None or array.ndim == ndim), f"Invalid {name} array.")
    return array


def parameter_vector(parameters, key, names):
    value = parameters[key]
    require(isinstance(value, dict) and set(value) == set(names), f"Parameter names differ for {key}.")
    return finite_array([value[name] for name in names], key, ndim=1)


def probability_from_json(panel, parameters, model):
    """Rebuild probabilities without sklearn or production model predictors."""
    require(model in MODELS, f"Unknown model: {model}.")
    require(type(parameters.get("converged")) is bool, f"Invalid convergence status: {model}.")
    x = finite_array(panel["x"], "controls", ndim=2)
    names = list(panel["names"])
    require(len(set(names)) == len(names) == x.shape[1], "Control names/order are invalid.")
    intercept = float(parameters["intercept"])
    require(np.isfinite(intercept), "Nonfinite model intercept.")
    if model == "logistic_network":
        keep = parameters["legacy_column_indices"]
        legacy_names = list(panel["legacy_names"])
        expected_keep = [i for i, name in enumerate(legacy_names)
                         if not (name == "own_infections_1000ms" and "own_infections_1s" in names)]
        require(keep == expected_keep, "Logistic frozen legacy indices differ from the declared deduplication rule.")
        legacy = finite_array(panel["legacy"], "legacy", ndim=2)
        require(legacy.shape == (len(x), len(legacy_names)), "Legacy matrix has incorrect dimensions.")
        design = np.column_stack([x, legacy[:, keep]])
        columns = names + [legacy_names[i] for i in keep]
        require(len(set(columns)) == len(columns), "Logistic features are duplicated.")
        if "feature_names" in parameters:
            require(parameters["feature_names"] == columns, "Logistic metadata feature order differs from the frozen panel order.")
        coefficients = parameter_vector(parameters, "coefficients", columns)
        mean = finite_array(parameters["control_mean"], "logistic mean", ndim=1)
        scale = finite_array(parameters["control_scale"], "logistic scale", ndim=1)
        require(mean.shape == scale.shape == coefficients.shape and np.all(scale > 0), "Invalid logistic scaling dimensions.")
        require(parameters["C"] == 1, "Logistic C differs from its frozen value.")
        eta = intercept + ((design - mean) / scale) @ coefficients
        result = np.empty(len(eta), dtype=float)
        nonnegative = eta >= 0
        result[nonnegative] = 1 / (1 + np.exp(-eta[nonnegative]))
        exponent = np.exp(eta[~nonnegative])
        result[~nonnegative] = exponent / (1 + exponent)
        return result, 0
    coefficients = parameter_vector(parameters, "control_coefficients", names)
    mean = parameter_vector(parameters, "control_mean", names)
    scale = parameter_vector(parameters, "control_scale", names)
    require(np.all(scale > 0), "Hazard scales must be positive.")
    close(parameters["dt_seconds"], .01, "frozen forecast duration", rtol=0, atol=0)
    close(parameters["ridge"], 1., "frozen hazard penalty", rtol=0, atol=0)
    require(parameters["log_baseline_rate_numerical_clip"] == [-60., 60.], "Unknown hazard numerical guard.")
    eta = intercept + ((x - mean) / scale) @ coefficients
    baseline = np.exp(np.minimum(np.maximum(eta, -60.), 60.))
    rates = parameters["beta_rates_per_second"]
    if model == "independent_mbo_hazard":
        require(rates == {}, "Independent hazard contains network rates.")
        contact = np.zeros(len(x))
    else:
        sources = list(panel["source_names"])
        require(set(rates) == set(sources), "Transmission source names differ from panel columns.")
        beta = finite_array([rates[source] for source in sources], "transmission rates", ndim=1)
        require(np.all(beta >= 0), "Negative transmission rate.")
        exposure_key = "shifted_infected" if model == "past_shifted_sirs" else "infected"
        exposure = finite_array(panel[exposure_key], exposure_key, ndim=2)
        require(exposure.shape == (len(x), len(sources)) and np.all((exposure == 0) | (exposure == 1)), "Invalid infectious source states.")
        contact = exposure @ beta
    result = -np.expm1(-.01 * (baseline + contact))
    return result, int(np.count_nonzero(np.abs(eta) >= 60.))


def validate_model_convergence(parameters, summary):
    """Audit reported optimizer outcomes independently of score arithmetic."""
    statuses, nonconverged = [], []
    for target in sorted(parameters):
        models = parameters[target]
        require(set(models) == set(MODELS), "Fitted model coverage differs.")
        for model in MODELS:
            metadata = models[model]
            status = metadata.get("converged")
            require(type(status) is bool, f"Invalid convergence status: {target}/{model}.")
            statuses.append(status)
            if status is False:
                nonconverged.append({"target": target, "model": model,
                                     "iterations": metadata.get("iterations"),
                                     "optimizer_message": metadata.get("optimizer_message")})
    require(len(parameters) == 8 and len(statuses) == 32, "Expected eight targets and 32 fitted convergence statuses.")
    all_converged = all(statuses)
    require(type(summary.get("all_models_converged")) is bool
            and summary["all_models_converged"] == all_converged,
            "Summary convergence aggregate differs from the 32 fitted statuses.")
    return {"all_models_converged": all_converged, "fitted_model_count": len(statuses),
            "nonconverged_models": nonconverged}


def validate_evidence_eligibility(summary, *, all_models_converged, guards, primary_statistical_support):
    """Keep arithmetic verification separate from scientific eligibility."""
    reasons = []
    if not all_models_converged:
        reasons.append("optimization_nonconvergence")
    if guards != 0:
        reasons.append("numerical_guards_activated")
    eligible = not reasons
    expected_support = bool(eligible and primary_statistical_support)
    require(type(summary.get("supports_incremental_prediction")) is bool
            and summary["supports_incremental_prediction"] == expected_support,
            "Overall evidence flag differs.")
    return {"scientifically_eligible": eligible, "scientific_ineligibility_reasons": reasons,
            "supports_incremental_prediction": expected_support}


def loss_sums(y, probability):
    target = finite_array(y, "outcome", ndim=1)
    probability = finite_array(probability, "probability", ndim=1)
    require(len(target) > 0 and target.shape == probability.shape and np.all((target == 0) | (target == 1)), "Invalid paired outcome cohort.")
    require(np.all((probability >= 0) & (probability <= 1)), "Probability is outside [0,1].")
    bounded = np.clip(probability, 1e-12, 1 - 1e-12)
    losses = np.where(target == 1, -np.log(bounded), -np.log1p(-bounded))
    return {"rows": len(target), "positive_count": int(target.sum()), "log_loss_sum": float(losses.sum()),
            "brier_sum": float(((probability - target) ** 2).sum()),
            "probability_clip_rows": int(np.count_nonzero(probability != bounded))}


def validate_score_cohort(recorded, times, y):
    times, y = np.asarray(times), np.asarray(y)
    require(times.ndim == y.ndim == 1 and len(times) == len(y) > 0
            and times.dtype.kind in "iu" and np.all(np.diff(times) > 0)
            and np.all((y == 0) | (y == 1)), "Invalid timestamp/label cohort.")
    expected = hashlib.sha256(times.tobytes() + y.tobytes()).hexdigest()
    require(recorded["cohort_sha256"] == expected and recorded["rows"] == len(y)
            and recorded["positive_count"] == int(y.sum()), "Stored model scored a different panel/cohort/label vector.")
    return expected


def bootstrap_intervals(totals, *, replicates=10_000, seed=20261027, block_length=5):
    """Independently draw paired consecutive session blocks without wrapping."""
    totals = finite_array(totals, "bootstrap totals", ndim=2)
    require(totals.shape[1] == 5 and len(totals) > 0 and np.all(totals[:, 4] > 0), "Invalid bootstrap date totals.")
    require(type(replicates) is int and replicates > 0 and type(block_length) is int and block_length > 0, "Invalid bootstrap controls.")
    n = len(totals)
    length = min(n, block_length)
    draws = np.random.default_rng(seed).integers(n - length + 1, size=(replicates, math.ceil(n / length)))
    outcomes = np.zeros((replicates, 5), dtype=float)
    # Iterating sessions instead of indexing a replicated three-dimensional
    # loss tensor provides a separate arithmetic path from production code.
    for position in range(n):
        block, offset = divmod(position, length)
        outcomes += totals[draws[:, block] + offset]
    result = {}
    for metric, column in (("log_loss", 0), ("brier", 2)):
        numerator = outcomes[:, column] - outcomes[:, column + 1]
        require(np.all(outcomes[:, column] > 0), "Undefined relative bootstrap gain.")
        result[metric] = {
            "absolute_gain_ci95": np.quantile(numerator / outcomes[:, 4], [.025, .975]).tolist(),
            "relative_gain_percent_ci95": np.quantile(100 * numerator / outcomes[:, column], [.025, .975]).tolist(),
        }
    return result


class Moments:
    """Streaming, numerically stable training-only first and second moments."""
    def __init__(self):
        self.n = 0
        self.mean = self.m2 = None
    def add(self, matrix):
        matrix = finite_array(matrix, "training matrix", ndim=2)
        require(len(matrix) > 0, "Zero-row training task.")
        mean = matrix.mean(axis=0)
        m2 = np.square(matrix - mean).sum(axis=0)
        if self.n == 0:
            self.n, self.mean, self.m2 = len(matrix), mean, m2
            return
        require(mean.shape == self.mean.shape, "Training feature dimensions changed.")
        total = self.n + len(matrix)
        difference = mean - self.mean
        self.m2 += m2 + difference ** 2 * (self.n * len(matrix) / total)
        self.mean += difference * (len(matrix) / total)
        self.n = total
    def scale(self, *, sklearn_constant_guard=False):
        variance = np.maximum(self.m2 / self.n, 0)
        scale = np.sqrt(variance)
        if sklearn_constant_guard:
            epsilon = np.finfo(float).eps
            constant = variance <= self.n * epsilon * variance + (self.n * self.mean * epsilon) ** 2
        else:
            constant = variance == 0
        scale[constant] = 1.
        return scale


def compare_summary_statistics(daily, summary, protocol):
    """Recompute every declared reference comparison and its paired intervals."""
    dates = summary["held_out_days"]
    lookup = exact_grid(daily.to_dict("records"), ("day", "model"),
                        {(day, model) for day in dates for model in MODELS}, "daily model/date")
    inference = protocol["inference"]
    require(inference["replicates"] == 10_000 and inference["block_sessions"] == 5
            and inference["seed"] == 20261026, "Inference settings differ from the frozen protocol.")
    report = {}
    for reference in REFERENCES:
        values = []
        for day in sorted(dates):
            a, b = lookup[day, reference], lookup[day, "sirs_network"]
            require(a["rows"] == b["rows"] and a["positive_count"] == b["positive_count"], "Unpaired daily model cohorts.")
            values.append([a["log_loss_sum"], b["log_loss_sum"], a["brier_sum"], b["brier_sum"], a["rows"]])
        values = finite_array(values, "paired daily totals", ndim=2)
        totals = values.sum(axis=0)
        claimed = summary["comparisons"][reference]
        require(claimed["status"] == "ok" and claimed["reference_model"] == reference
                and claimed["candidate_model"] == "sirs_network" and claimed["days"] == sorted(dates), "Comparison identity or coverage differs.")
        require(claimed["pooled"]["rows"] == int(totals[4]), "Pooled comparison denominator differs.")
        calculated = {"pooled": {"rows": int(totals[4])}}
        for metric, column in (("log_loss", 0), ("brier", 2)):
            difference = totals[column] - totals[column + 1]
            require(totals[column] > 0, "Relative pooled gain is undefined.")
            calculated["pooled"][metric] = {
                "reference": totals[column] / totals[4], "candidate": totals[column + 1] / totals[4],
                "absolute_gain": difference / totals[4], "relative_gain_percent": 100 * difference / totals[column],
            }
            for field, value in calculated["pooled"][metric].items():
                close(claimed["pooled"][metric][field], value, reference + ".pooled." + metric + "." + field)
            changes = values[:, column] - values[:, column + 1]
            for sign, count in (("positive", np.count_nonzero(changes > 0)),
                                ("negative", np.count_nonzero(changes < 0)), ("tied", np.count_nonzero(changes == 0))):
                require(claimed[f"{sign}_{metric}_days"] == int(count), f"Daily gain sign count differs: {reference}/{metric}.")
        for label, length, seed in (("paired_day_bootstrap", 1, inference["seed"]),
                                    ("five_session_block_bootstrap", 5, inference["seed"] + 1)):
            interval = bootstrap_intervals(values, replicates=inference["replicates"], seed=seed, block_length=length)
            for metric in ("log_loss", "brier"):
                for field, value in interval[metric].items():
                    close(claimed[label][metric][field], value, f"{reference}.{label}.{metric}.{field}")
            claimed_bootstrap = claimed[label]
            require(claimed_bootstrap["replicates"] == inference["replicates"]
                    and claimed_bootstrap["seed"] == seed and claimed_bootstrap["requested_block_sessions"] == length
                    and claimed_bootstrap["sessions_per_replicate"] == len(dates)
                    and claimed_bootstrap["wraps_date_range"] is False, "Bootstrap method metadata differs.")
            calculated[label] = interval
        lower = calculated["five_session_block_bootstrap"]["log_loss"]["absolute_gain_ci95"][0]
        supports = bool(len(dates) >= 20 and lower > 0 and calculated["pooled"]["log_loss"]["absolute_gain"] > 0)
        require(claimed["positive_block_interval"] == (lower > 0)
                and claimed["supports_positive_incremental_prediction"] == supports, "Evidence flag differs from the paired-block rule.")
        calculated["supports_positive_incremental_prediction"] = supports
        report[reference] = calculated
    return report


def _write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(12):
        try:
            pending.replace(path)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(.1 * 2 ** attempt, 1.))


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit_results(config_path=ROOT / "mbo_sir.toml", *, tables_only=False, progress=print):
    config_path = Path(config_path).resolve()
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    protocol = _read(config_path.with_name("mbo_sir_protocol.json"))
    out = resolve(config["out_dir"])
    report_path = out / "result_audit.json"
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "passed": False,
              "mode": "tables_only" if tables_only else "independent_bin_predictions",
              "full_probability_audit": not tables_only, "auditor_sha256": file_hash(__file__),
              "arithmetic_verified": False, "scientifically_eligible": None,
              "errors": [], "completed_stages": []}
    try:
        with threadpool_limits(limits=1):
            _audit_results(config, protocol, out, report, tables_only=tables_only, progress=progress)
        report["passed"] = True
        report["arithmetic_verified"] = not tables_only
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _write_report(report_path, report)
    return report


def _audit_results(config, protocol, out, report, *, tables_only, progress):
    days = config["training_days"] + config["held_out_days"]
    venues = config["datasets"]
    validate_specification(config, protocol)
    summary = _read(out / "summary.json")
    require(summary["complete"] is True and summary["config"] == config and summary["protocol"] == protocol,
            "Summary does not describe the frozen complete configuration/protocol.")
    require(summary["datasets"] == venues and summary["training_days"] == config["training_days"]
            and summary["held_out_days"] == config["held_out_days"], "Summary study dates or venues differ.")
    component_tests = _read(out / "current-component-test-summary.json")
    require(component_tests["all_current_components_verified"] is True
            and component_tests["errors"] == component_tests["failures"] == component_tests["skipped"] == 0,
            "Current component software tests are not all verified.")
    for path, expected in component_tests["final_source_and_test_sha256"].items():
        require(file_hash(resolve(path)) == expected, f"Component source/test changed since verification: {path}.")
    report["verified_component_tests"] = component_tests["passed"]
    provenance = _read(out / "provenance.json")
    require(provenance["config_sha256"] == digest(config) and provenance["protocol_sha256"] == digest(protocol), "Analysis configuration provenance differs.")
    for name, expected in provenance["source_code_sha256"].items():
        require(file_hash(ROOT / "src" / "colonylab" / name) == expected, "Analysis source provenance differs: " + name)
    require(file_hash(ROOT / "src" / "colonylab" / "sir_model.py") == provenance["sir_model_sha256"], "Hazard-model source provenance differs.")

    raw = _read(out / "data_manifest.json")
    raw_audit = _read(out / "raw_audit.json")
    requests = request_grid(config)
    require(len(requests) == 510 and raw["complete"] is True and raw_audit["complete"] is True
            and not raw_audit["failures"], "Independent full raw-grid audit did not pass.")
    require(raw["configuration_sha256"] == raw_audit["configuration_sha256"] == digest(config)
            and raw["request_grid_sha256"] == raw_audit["request_grid_sha256"] == digest({"requests": requests}),
            "Raw configuration/request-grid provenance differs.")
    require(raw_audit["validator_code_sha256"] == file_hash(ROOT / "src" / "colonylab" / "mbo_sir_data.py"),
            "Raw integrity audit was performed with a different validator source.")
    def raw_lookup(entries, label):
        lookup = {}
        for entry in entries:
            token = digest(entry["request"])
            require(token not in lookup and entry["complete"] is True and 0 < entry["rows"] < entry["request"]["limit"], "Invalid/duplicate " + label)
            lookup[token] = entry
        require(set(lookup) == {digest(request) for request in requests}, "Incomplete or extra " + label)
        return lookup
    raw_entries = raw_lookup(raw["requests"], "raw acquisition grid")
    audited_raw = raw_lookup(raw_audit["requests"], "independent raw audit grid")
    for token, entry in raw_entries.items():
        checked = audited_raw[token]
        require(entry["rows"] == checked["rows"] and entry["file_size_bytes"] == checked["file_size_bytes"]
                and not entry.get("record_limit_reached"), "Raw receipt disagrees with independent audit.")
        require(len(checked["sha256"]) == 64 and checked["request_sha256"] == token, "Raw audit is missing its content/request hash.")
        path = resolve(entry["data_path"])
        require(path.is_file() and path.stat().st_size == checked["file_size_bytes"], "Raw file is missing or its byte length changed after audit.")
    require(provenance["raw_requests_sha256"] == digest(raw["requests"]), "Raw manifest differs from analysis provenance.")
    report["raw_requests"] = 510
    report["raw_rows"] = sum(entry["rows"] for entry in raw_entries.values())
    require(summary["raw_requests"] == 510 and summary["raw_rows"] == report["raw_rows"], "Summary raw counts differ.")
    report["completed_stages"].append("exact_raw_grid_and_independent_integrity_audit")

    replays = _read(out / "replay_manifest.json")
    expected_tasks = {(day, venue) for day in days for venue in venues}
    require(replays["complete"] is True and provenance["replay_manifest_sha256"] == digest(replays), "Replay manifest is incomplete or differs from analysis provenance.")
    replay_lookup = exact_grid(replays["tasks"], ("day", "dataset"), expected_tasks, "replay task")
    request_lookup = {(request["start"][:10], request["dataset"], request["schema"]): request for request in requests}
    expected_replay_code = digest({name: file_hash(ROOT / "src" / "colonylab" / name)
                                   for name in ("mbo_book.py", "mbo_replay.py")})
    matched = 0
    for (day, venue), task in replay_lookup.items():
        validation, diagnostics = task["validation"], task["diagnostics"]
        require(validation["passed"] is True and validation["matched_keys"] > 0
                and validation["mismatch_keys"] == validation["missing_keys"] == 0, "Independent top-ten replay validation failed.")
        require(not any(diagnostics.get(key, 0) for key in SEMANTIC_FAILURES), "Whole-session order-semantic failure.")
        require(0 < diagnostics["valid_grid_rows"] <= diagnostics["grid_rows"], "Replay has zero or invalid observed book coverage.")
        require(task["provenance"]["code"] == expected_replay_code, "Replay was built with different book/adapter code.")
        for schema in ("mbo", "mbp-10"):
            source = audited_raw[digest(request_lookup[day, venue, schema])]
            require(task["provenance"]["sources"][schema] == source["sha256"], "Replay raw-source content hash differs.")
            require(task["provenance"]["requests"][schema] == request_lookup[day, venue, schema], "Replay request identity differs.")
            if schema == "mbo":
                require(diagnostics["records_replayed"] == source["rows"], "Replay omitted or added MBO records.")
        matched += validation["matched_keys"]
    report["replay_tasks"] = len(replay_lookup)
    report["mbo_rows_replayed"] = sum(task["diagnostics"]["records_replayed"] for task in replay_lookup.values())
    claimed_book = summary["book_validation"]
    require(claimed_book["tasks"] == 240 and claimed_book["all_passed"] is True
            and claimed_book["matched_keys"] == matched and claimed_book["mismatch_keys"] == claimed_book["missing_keys"]
            == claimed_book["full_session_semantic_errors"] == 0
            and summary["mbo_rows_replayed"] == report["mbo_rows_replayed"], "Summary reconstruction counts differ.")
    report["completed_stages"].append("exact_240_replays_depth_and_whole_session_semantics")
    book_table = pd.read_csv(out / "book_audit.csv", float_precision="round_trip")
    book_lookup = exact_grid(book_table.to_dict("records"), ("day", "dataset"), expected_tasks, "book audit table")
    for key, task in replay_lookup.items():
        row = book_lookup[key]
        require(str(row["passed"]).lower() == "true", "Published book audit table contains a failed task.")
        for field in ("matched_keys", "mismatch_keys", "missing_keys"):
            close(row[field], task["validation"][field], "published book audit " + field)
        close(row["records_replayed"], task["diagnostics"]["records_replayed"], "published replay record count")

    panels = _read(out / "panel_manifest.json")
    require(panels["complete"] is True and file_hash(out / "panel_manifest.json") == provenance["panel_manifest_sha256"], "Panel manifest provenance differs.")
    panel_lookup = exact_grid(panels["tasks"], ("day", "dataset"), expected_tasks, "panel task")
    require(panels["thresholds"] == summary["thresholds"] and set(panels["thresholds"]) == set(venues)
            and all(np.isfinite(value) and value >= .005 for value in panels["thresholds"].values()), "Frozen threshold maps differ.")
    expected_feature_code = digest({name: file_hash(ROOT / "src" / "colonylab" / name)
                                    for name in ("mbo_sir_features.py", "sir_model.py")})
    coverage = pd.read_csv(out / "coverage.csv", float_precision="round_trip")
    coverage_lookup = exact_grid(coverage.to_dict("records"), ("day", "dataset"), expected_tasks, "published coverage")
    task_scores = pd.read_csv(out / "task_daily_scores.csv", float_precision="round_trip")
    expected_scores = {(day, venue, model) for day in config["held_out_days"] for venue in venues for model in MODELS}
    score_lookup = exact_grid(task_scores.to_dict("records"), ("day", "dataset", "model"), expected_scores, "test task/model score")
    require(len(score_lookup) == 640, "There must be 160 held-out venue/date tasks times four models.")
    daily = pd.read_csv(out / "daily_scores.csv", float_precision="round_trip")
    daily_lookup = exact_grid(daily.to_dict("records"), ("day", "model"),
                              {(day, model) for day in config["held_out_days"] for model in MODELS}, "daily score")
    require(len(daily_lookup) == 80, "There must be twenty held-out dates times four models.")
    parameters = _read(out / "model_parameters.json")
    require(set(parameters) == set(venues) and all(set(value) == set(MODELS) for value in parameters.values()), "Fitted model coverage differs.")
    report.update(validate_model_convergence(parameters, summary))
    training = {venue: {"x": Moments(), "logistic": Moments(), "positives": 0} for venue in venues}
    recomputed = []
    max_error = {"log_loss_sum": 0., "brier_sum": 0.}
    heldout_training_guards = 0
    for number, (day, venue) in enumerate(sorted(panel_lookup), 1):
        panel = load_panel(panel_lookup[day, venue]["path"])
        require(panel["day"] == day and panel["dataset"] == venue and panel["names"] == protocol["controls"], "Panel identity/control specification differs.")
        require(set(panel["source_names"]) == set(venues) - {venue} and len(panel["source_names"]) == 7, "Panel infectious source columns differ.")
        times, y = np.asarray(panel["time_ns"]), np.asarray(panel["y"])
        x = finite_array(panel["x"], "panel x", ndim=2)
        require(times.dtype.kind in "iu" and times.ndim == y.ndim == 1 and len(times) == len(y) == len(x) > 0,
                "Zero, missing or dimensionally invalid panel cohort.")
        require(np.all(np.diff(times) > 0) and np.all((y == 0) | (y == 1)), "Panel timestamps or labels are invalid.")
        start = pd.Timestamp(day + "T" + config["session_start_utc"] + "Z").value
        end = pd.Timestamp(day + "T" + config["session_end_utc"] + "Z").value
        require(np.all((times - start) % 100_000_000 == 0) and int(times.min()) >= start + 30_000_000_000
                and int(times.max()) + 10_000_000 <= end, "Forecast anchors violate the frozen session/grid/horizon.")
        require(panel["coverage"]["eligible_bins"] == len(y) and panel["coverage"]["positive_bins"] == int(y.sum()), "Panel coverage counters differ.")
        require(coverage_lookup[day, venue]["eligible_bins"] == len(y)
                and coverage_lookup[day, venue]["positive_bins"] == int(y.sum()), "Published coverage differs from actual panel rows/labels.")
        require(panel["provenance"]["protocol"] == protocol
                and panel["provenance"]["replays"] == {target: replay_lookup[day, target]["provenance"] for target in venues},
                "Panel replay/protocol provenance differs.")
        market_request = request_lookup[day, config["market_dataset"], "bbo-1s"]
        require(panel["provenance"]["thresholds"] == panels["thresholds"]
                and panel["provenance"]["code"] == expected_feature_code
                and panel["provenance"]["market_sha256"] == audited_raw[digest(market_request)]["sha256"],
                "Panel thresholds, feature code or market source differs.")
        if day in config["training_days"]:
            training[venue]["x"].add(x)
            keep = parameters[venue]["logistic_network"]["legacy_column_indices"]
            training[venue]["logistic"].add(np.column_stack([x, panel["legacy"][:, keep]]))
            training[venue]["positives"] += int(y.sum())
            # Validate all parameter/source names without performing a full
            # training-bin rescore in either audit mode.
            tiny = {**panel, "x": x[:1], "legacy": panel["legacy"][:1],
                    "infected": panel["infected"][:1], "shifted_infected": panel["shifted_infected"][:1]}
            for model in MODELS:
                probability_from_json(tiny, parameters[venue][model], model)
            del tiny
        else:
            for model in MODELS:
                recorded = score_lookup[day, venue, model]
                validate_score_cohort(recorded, times, y)
                if tables_only:
                    calculated = {field: recorded[field] for field in SCORE_FIELDS}
                else:
                    probability, guards = probability_from_json(panel, parameters[venue][model], model)
                    calculated = dict(loss_sums(y, probability), numerical_guard_rows=guards)
                    for field in SCORE_FIELDS:
                        close(recorded[field], calculated[field], f"task {day}/{venue}/{model}/{field}")
                    for field in max_error:
                        max_error[field] = max(max_error[field], abs(recorded[field] - calculated[field]))
                recomputed.append(dict(day=day, dataset=venue, model=model, **calculated))
        del panel, x, y, times
        if number % 8 == 0:
            progress(f"Independent result audit: {number}/240 panel tasks checked.")
    for venue, moments in training.items():
        for model, metadata in parameters[venue].items():
            require(metadata["training_rows"] == moments["x"].n and metadata["training_infections"] == moments["positives"], "Fitted training cohort counts differ.")
            if model == "logistic_network":
                mean, scale = metadata["control_mean"], metadata["control_scale"]
                expected = moments["logistic"]
            else:
                mean = [metadata["control_mean"][name] for name in protocol["controls"]]
                scale = [metadata["control_scale"][name] for name in protocol["controls"]]
                expected = moments["x"]
                require(metadata["training_numerical_guard_rows"] == 0, "A training hazard numerical guard was activated.")
                heldout_training_guards += metadata["training_numerical_guard_rows"]
            close(mean, expected.mean, venue + ".training_only_mean")
            close(scale, expected.scale(sklearn_constant_guard=model == "logistic_network"), venue + ".training_only_scale")
    report["panel_tasks"] = len(panel_lookup)
    report["held_out_task_model_rows"] = len(score_lookup)
    report["max_independent_task_score_absolute_difference"] = max_error if not tables_only else None
    report["completed_stages"].append("all_panel_cohorts_training_only_scaling_and_json_models")
    if not tables_only:
        report["completed_stages"].append("independent_numpy_probability_and_bin_loss_reconstruction")

    recalculated = pd.DataFrame(recomputed)
    grouped = recalculated.groupby(["day", "model"], sort=True)[list(SCORE_FIELDS)].sum()
    for (day, model), values in grouped.iterrows():
        for field in SCORE_FIELDS:
            close(daily_lookup[day, model][field], values[field], f"daily aggregate {day}/{model}/{field}")
    totals = grouped.groupby("model").sum()
    metrics = exact_grid(summary["model_metrics"], ("model",), {(model,) for model in MODELS}, "summary model metric")
    for model, values in totals.iterrows():
        for field in ("rows", "positive_count", "log_loss_sum", "brier_sum"):
            close(metrics[model,][field], values[field], f"pooled {model}/{field}")
        close(metrics[model,]["log_loss"], values.log_loss_sum / values.rows, model + ".mean_log_loss")
        close(metrics[model,]["brier"], values.brier_sum / values.rows, model + ".mean_brier")
    guards = int(grouped.numerical_guard_rows.sum()) + heldout_training_guards
    require(summary["numerical_guard_rows"] == guards, "Summary numerical-guard count differs.")
    report["numerical_guard_rows"] = guards
    # Published session totals have already been checked against every
    # independently rebuilt bin score. Using their round-trip CSV values here
    # avoids changing exact daily tie classifications through benign addition
    # roundoff while independently redoing all inference arithmetic.
    report["comparisons"] = compare_summary_statistics(daily, summary, protocol)
    report.update(validate_evidence_eligibility(summary,
                  all_models_converged=report["all_models_converged"], guards=guards,
                  primary_statistical_support=report["comparisons"]["independent_mbo_hazard"]["supports_positive_incremental_prediction"]))
    edges = pd.read_csv(out / "edges.csv", float_precision="round_trip")
    edge_lookup = exact_grid(edges.to_dict("records"), ("source", "target"),
                             {(source, target) for source in venues for target in venues if source != target}, "directed edge")
    for (source, target), edge in edge_lookup.items():
        beta = parameters[target]["sirs_network"]["beta_rates_per_second"][source]
        require(beta >= 0, "Negative directed transmission rate.")
        close(edge["beta_per_second"], beta, "directed edge rate")
        close(edge["isolated_10ms_probability"], -np.expm1(-.01 * beta), "isolated edge probability")
    require(summary["positive_directed_edges"] == int(np.count_nonzero(edges.beta_per_second > 1e-8)), "Positive edge count differs.")
    report["completed_stages"].append("daily_and_pooled_scores_all_three_comparisons_and_paired_bootstraps")
    report["notes"] = ["Raw files were checked against the prior full-grid independent integrity audit without another record decode.",
                       "Model probabilities and inference were computed independently from JSON parameters and stored panel inputs.",
                       "Audit consistency does not establish optimizer convergence or scientific support; nonconverged fits are retained and reported.",
                       "A null optimizer message means that message was not stored in the fitted metadata."]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "mbo_sir.toml")
    parser.add_argument("--tables-only", action="store_true", help="Skip per-bin probability reconstruction; this is not the full final audit.")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        report = audit_results(args.config, tables_only=args.tables_only)
    print(json.dumps({key: value for key, value in report.items() if key != "comparisons"}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
