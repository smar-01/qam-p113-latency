"""Independently audit the completed fresh-date study using local DBN files.

This script makes no network requests and never reads API credentials except
through the existing error redactor. It reads one raw request at a time and
reconstructs conservation, calibration, paired totals and bootstrap intervals.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from colonylab.config import safe_error, write_json
from colonylab.network_wide_data import read_wide_config
from colonylab.volume_bars import build_volume_bars, normalize_trades


LOSS_COLUMNS = ["reference_log_loss_sum", "volume_log_loss_sum",
                "reference_brier_sum", "volume_brier_sum"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def same_number(actual, expected, context: str, *, atol: float = 1e-10) -> None:
    if expected is None:
        require(actual is None, f"{context}: expected an undefined value.")
        return
    require(actual is not None and np.isfinite(float(actual)) and
            np.isclose(float(actual), float(expected), rtol=1e-11, atol=atol),
            f"{context}: {actual!r} differs from reconstructed {expected!r}.")


def exact_grid(frame: pd.DataFrame, columns: list[str], expected: set, context: str) -> None:
    require(set(columns) <= set(frame.columns), f"{context}: required index columns are missing.")
    require(not frame.duplicated(columns).any(), f"{context}: duplicate index rows.")
    observed = set(frame[columns].itertuples(index=False, name=None))
    require(observed == expected, f"{context}: missing {len(expected - observed)} or extra {len(observed - expected)} index rows.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_code_fingerprints() -> dict:
    return {"audit_script": sha256_file(Path(__file__)),
            **{name: sha256_file(ROOT / "src" / "colonylab" / name)
               for name in ("volume_bars.py", "network.py")}}


def raw_request_facts(entry: dict, config: dict, thresholds: pd.DataFrame,
                      source: dict, cache_dir: Path, code: dict) -> tuple[dict, bool]:
    """Independently decode one request, or reuse content-bound audit facts.

    Every reuse still hashes the complete raw file. These are audit-owned small
    JSON facts; no engine pickle or prepared feature panel enters this cache.
    """
    import databento as db

    request = entry["request"]
    dataset, day = request["dataset"], request["start"][:10]
    require(entry.get("complete") and not entry.get("record_limit_reached"),
            f"Incomplete or capped request: {dataset} {day}.")
    start = pd.Timestamp(f"{day}T{config['start_time_utc']}Z")
    end = start + pd.Timedelta(minutes=config["minutes"])
    require(pd.Timestamp(request["start"]) == start and pd.Timestamp(request["end"]) == end and
            request["schema"] == "mbp-1" and request["symbols"] == config["symbols"] and
            request["stype_in"] == "raw_symbol" and request["limit"] == config["record_limit_per_request"],
            f"Raw request differs from the configured grid: {dataset} {day}.")
    path = Path(entry["data_path"])
    content_sha = sha256_file(path)
    require(source["content_sha256"] == content_sha and source["rows"] == entry["rows"] and
            source["request"] == request, f"Raw source changed after analysis: {dataset} {day}.")
    quanta = {symbol: int(thresholds.loc[(dataset, symbol), "threshold_shares"])
              for symbol in config["symbols"]}
    identity = {"raw_content_sha256": content_sha, "request": request,
                "expected_rows": entry["rows"], "thresholds": quanta, "code": code}
    cache_path = cache_dir / f"{day}_{dataset}.json"
    if cache_path.exists():
        try:
            saved = json.loads(cache_path.read_text(encoding="utf-8"))
            facts = saved["facts"]
            facts_sha = hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()
            if saved["identity"] == identity and saved["facts_sha256"] == facts_sha:
                return facts, True
        except (OSError, ValueError, KeyError, TypeError):
            pass  # A partial, malformed or outdated audit cache is decoded again.
    frame = db.DBNStore.from_file(path).to_df(price_type="float", pretty_ts=True, map_symbols=True)
    require(0 < len(frame) == entry["rows"] < request["limit"], f"Raw record count mismatch: {dataset} {day}.")
    require(isinstance(frame.index, pd.DatetimeIndex) and frame.index.name == "ts_recv" and
            frame.index.tz is not None and not frame.index.hasnans and
            (frame.index >= start).all() and (frame.index < end).all(), f"Invalid receive-clock coverage: {dataset} {day}.")
    require(set(frame.symbol.dropna().unique()) == set(config["symbols"]), f"Raw requested symbol mismatch: {dataset} {day}.")
    action = frame.action.map(lambda value: value.decode() if isinstance(value, bytes) else str(value))
    raw_t = action.eq("T").to_numpy()
    size = pd.to_numeric(frame["size"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    admissible_size = np.isfinite(size) & (size > 0) & (size == np.floor(size))
    raw_share_total = float(size[raw_t & admissible_size].sum())
    frame.attrs.update(requested_start=request["start"], requested_end=request["end"])
    trades, quality = normalize_trades(frame, dataset, day)
    require(int(raw_t.sum()) == quality["trade_records"], f"Raw T count mismatch: {dataset} {day}.")
    if quality["invalid_trade_records"] == 0:
        same_number(raw_share_total, quality["valid_volume_shares"], f"Independent raw T shares {dataset} {day}")
    symbols = {}
    for symbol in config["symbols"]:
        one = trades.loc[trades.symbol.eq(symbol)].copy()
        one.attrs.update(trades.attrs)
        bars = build_volume_bars(one, quanta[symbol])
        symbols[symbol] = {
            "valid_volume_shares": float(one.loc[one.valid, "size"].sum()),
            "valid_trade_records": int(one.valid.sum()), "invalid_trade_records": int((~one.valid).sum()),
            "completed_bars": len(bars), "threshold_shares": quanta[symbol],
            "completed_volume_shares": float(bars.volume_shares.sum()),
            "incomplete_volume_shares": float(bars.attrs.get("incomplete_volume_shares", 0.)),
            "barrier_discarded_volume_shares": float(bars.attrs.get("barrier_discarded_volume_shares", 0.)),
            "median_duration_ms": float(bars.duration_ms.median()) if len(bars) else None,
            "p95_duration_ms": float(bars.duration_ms.quantile(.95)) if len(bars) else None,
        }
    facts = {"raw_records": len(frame), "raw_action_t_records": int(raw_t.sum()),
             "raw_t_finite_positive_integer_size_shares": raw_share_total,
             "requested_seconds": (end - start).total_seconds(), "quality": quality, "symbols": symbols}
    facts_sha = hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".partial")
    write_json(temporary, {"identity": identity, "facts": facts, "facts_sha256": facts_sha})
    temporary.replace(cache_path)
    return facts, False


def prepare_raw_audit(config_path: Path, manifest_path: Path, out_dir: Path, *, progress=print) -> dict:
    """Prepare independent raw facts while the prediction engine is still running."""
    started = time.perf_counter()
    config = read_wide_config(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = json.loads((out_dir / "analysis_provenance.json").read_text(encoding="utf-8"))
    protocol_bytes = (config_path.parent / "fresh_confirmation_protocol.json").read_bytes()
    require((out_dir / "frozen_protocol.json").read_bytes() == protocol_bytes,
            "Saved analysis protocol differs from the frozen protocol source.")
    require(provenance["protocol_sha256"] == hashlib.sha256(protocol_bytes).hexdigest(), "Preparation protocol checksum mismatch.")
    require(manifest.get("complete") is True and manifest["configuration"] == config,
            "Raw preparation requires a complete manifest matching the configuration.")
    require(manifest["configuration_sha256"] == hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "Raw preparation manifest configuration checksum mismatch.")
    inventory = {str(Path(row["data_path"]).resolve()): row for row in provenance["sources"]}
    entries = manifest["requests"]
    expected = {(venue, day) for venue in config["datasets"] for day in config["training_days"] + config["held_out_days"]}
    observed = [(entry["request"]["dataset"], entry["request"]["start"][:10]) for entry in entries]
    require(len(observed) == len(set(observed)) and set(observed) == expected,
            "Raw preparation manifest does not contain the exact intended request grid.")
    thresholds = pd.read_csv(out_dir / "volume_thresholds.csv")
    exact_grid(thresholds, ["dataset", "symbol"],
               {(venue, symbol) for venue in config["datasets"] for symbol in config["symbols"]}, "Volume thresholds")
    require(thresholds.threshold_shares.ge(1).all() and
            thresholds.threshold_shares.eq(np.floor(thresholds.threshold_shares)).all(), "Invalid frozen volume thresholds.")
    thresholds = thresholds.set_index(["dataset", "symbol"])
    code, reused, rows = audit_code_fingerprints(), 0, 0
    for index, entry in enumerate(sorted(entries, key=lambda row: (row["request"]["start"], row["request"]["dataset"]))):
        source = inventory[str(Path(entry["data_path"]).resolve())]
        facts, cached = raw_request_facts(entry, config, thresholds, source,
                                         out_dir / "independent_raw_audit_cache", code)
        reused += int(cached)
        rows += facts["raw_records"]
        if (index + 1) % 20 == 0 or index + 1 == len(entries):
            progress(f"Prepared independent raw audit facts {index + 1}/{len(entries)} requests; {reused} reused.")
    result = {"status": "raw_facts_prepared", "raw_requests": len(entries), "raw_records": rows,
              "reused_requests": reused, "duration_seconds": round(time.perf_counter() - started, 3),
              "cache_directory": str((out_dir / "independent_raw_audit_cache").resolve()),
              "note": "Final audit is still required after model scoring; raw facts alone establish no prediction result."}
    write_json(out_dir / "raw_audit_preparation.json", result)
    return result


def independently_resample(daily: pd.DataFrame, protocol: dict) -> dict:
    """Recompute paired intervals directly, without importing the stats engine."""
    values = daily.sort_values("day")[LOSS_COLUMNS + ["rows"]].to_numpy(dtype=float)
    count = len(values)
    uncertainty = protocol["uncertainty"]
    results = {}
    for name, length, seed in (
        ("paired_day_bootstrap", 1, uncertainty["paired_day_seed"]),
        ("five_session_block_bootstrap", uncertainty["moving_block_sessions"], uncertainty["moving_block_seed"]),
    ):
        length = min(length, count)
        blocks = (count + length - 1) // length
        starts = np.random.default_rng(seed).integers(
            0, count - length + 1, size=(uncertainty["replicates"], blocks))
        indices = (starts[:, :, None] + np.arange(length)).reshape(len(starts), -1)[:, :count]
        sampled = values[indices].sum(axis=1)
        result = {}
        for metric, position in (("log_loss", 0), ("brier", 2)):
            delta = sampled[:, position] - sampled[:, position + 1]
            relative = np.divide(100 * delta, sampled[:, position],
                                 out=np.full(len(delta), np.nan), where=sampled[:, position] > 0)
            result[metric] = {
                "absolute_gain_ci95": np.quantile(delta / sampled[:, 4], [.025, .975]).tolist(),
                "relative_gain_percent_ci95": np.quantile(relative, [.025, .975]).tolist()
                if np.isfinite(relative).all() else None,
            }
        results[name] = result
    return results


def audit(config_path: Path, manifest_path: Path, out_dir: Path, *, progress=print) -> dict:
    started = time.perf_counter()
    config = read_wide_config(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads((out_dir / "confirmation_summary.json").read_text(encoding="utf-8"))
    protocol_path = config_path.parent / "fresh_confirmation_protocol.json"
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    require((out_dir / "frozen_protocol.json").read_bytes() == protocol_bytes,
            "Saved analysis protocol differs from the frozen protocol source.")
    require(summary["protocol_sha256"] == hashlib.sha256(protocol_bytes).hexdigest(),
            "Summary protocol checksum differs from the frozen protocol.")
    require(manifest.get("complete") is True, "Raw acquisition manifest is incomplete.")
    stored_config = manifest["configuration"]
    require(stored_config == config == summary["configuration"], "Configuration differs between source, manifest and summary.")
    require(manifest["configuration_sha256"] == hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "Manifest configuration checksum differs from its recorded configuration.")
    require(summary["source_clock"] == "ts_recv" and summary["label_horizon_ms"] == 10 and
            summary["anchor_step_ms"] == protocol["forecast_anchor_spacing_ms"], "Recorded clock, label horizon or anchor grid changed.")
    venues, symbols = config["datasets"], config["symbols"]
    train, test = config["training_days"], config["held_out_days"]
    days = train + test
    tasks = {(dataset, symbol) for dataset in venues for symbol in symbols}
    all_task_days = {(dataset, symbol, day) for dataset, symbol in tasks for day in days}
    test_task_days = {(dataset, symbol, day) for dataset, symbol in tasks for day in test}
    request_grid = {(dataset, day) for dataset in venues for day in days}
    require(len(train) == protocol["training_sessions"] and len(test) == protocol["held_out_sessions"]
            and len(venues) == protocol["venues"] and symbols == protocol["symbols"], "Protocol dimensions differ from the configuration.")
    require(not set(train) & set(test) and max(train) < min(test), "Training/held-out chronology overlaps.")
    entries = manifest["requests"]
    request_keys = [(entry["request"]["dataset"], entry["request"]["start"][:10]) for entry in entries]
    require(len(request_keys) == len(set(request_keys)) and set(request_keys) == request_grid,
            "Raw manifest does not contain exactly the intended venue/date grid.")
    require(summary["requests"] == len(entries), "Summary request count differs from the raw manifest.")
    provenance = json.loads((out_dir / "analysis_provenance.json").read_text(encoding="utf-8"))
    analysis_sha = provenance.pop("analysis_sha256")
    require(analysis_sha == summary["analysis_sha256"] == hashlib.sha256(
        json.dumps(provenance, sort_keys=True).encode()).hexdigest(), "Analysis provenance checksum is inconsistent.")
    for name, digest in provenance["code"].items():
        require(sha256_file(ROOT / "src" / "colonylab" / name) == digest,
                f"Analysis source changed after scoring: {name}.")
    inventory = {str(Path(row["data_path"]).resolve()): row for row in provenance["sources"]}
    require(len(inventory) == len(entries), "Provenance does not contain exactly one source per request.")

    quality = pd.read_csv(out_dir / "trade_quality.csv")
    candles = pd.read_csv(out_dir / "candle_statistics.csv")
    thresholds = pd.read_csv(out_dir / "volume_thresholds.csv")
    exact_grid(quality, ["dataset", "day"], request_grid, "Trade quality")
    exact_grid(candles, ["dataset", "symbol", "day"], all_task_days, "Candle statistics")
    exact_grid(thresholds, ["dataset", "symbol"], tasks, "Volume thresholds")
    quality = quality.set_index(["dataset", "day"])
    candles = candles.set_index(["dataset", "symbol", "day"])
    thresholds = thresholds.set_index(["dataset", "symbol"])
    raw_records = raw_trade_records = invalid_records = ambiguous_groups = 0
    raw_finite_positive_integer_trade_shares = valid_shares = 0.0
    maximum_conservation_error = 0.0
    training_totals = {task: {"seconds": 0., "volume": 0., "valid_trades": 0,
                              "invalid_trades": 0, "days": 0} for task in tasks}
    fields = ["raw_records", "trade_records", "fill_records_excluded", "valid_trade_records",
              "invalid_trade_records", "trade_records_without_last_flag", "ambiguous_price_timestamp_groups",
              "valid_volume_shares", "unknown_side_volume_shares"]
    code, reused_raw_requests = audit_code_fingerprints(), 0
    for index, entry in enumerate(sorted(entries, key=lambda row: (row["request"]["start"], row["request"]["dataset"]))):
        request = entry["request"]
        dataset, day = request["dataset"], request["start"][:10]
        source = inventory[str(Path(entry["data_path"]).resolve())]
        facts, cached = raw_request_facts(entry, config, thresholds, source,
                                         out_dir / "independent_raw_audit_cache", code)
        reused_raw_requests += int(cached)
        rebuilt_quality = facts["quality"]
        for field in fields:
            same_number(quality.loc[(dataset, day), field], rebuilt_quality[field], f"Trade quality {dataset} {day} {field}")
        raw_finite_positive_integer_trade_shares += facts["raw_t_finite_positive_integer_size_shares"]
        raw_records += facts["raw_records"]
        raw_trade_records += facts["raw_action_t_records"]
        invalid_records += rebuilt_quality["invalid_trade_records"]
        ambiguous_groups += rebuilt_quality["ambiguous_price_timestamp_groups"]
        valid_shares += rebuilt_quality["valid_volume_shares"]
        for symbol in symbols:
            actual = facts["symbols"][symbol]
            valid_volume = actual["valid_volume_shares"]
            saved = candles.loc[(dataset, symbol, day)]
            accounted = float(saved.completed_volume_shares + saved.incomplete_volume_shares + saved.barrier_discarded_volume_shares)
            maximum_conservation_error = max(maximum_conservation_error, abs(valid_volume - accounted))
            same_number(accounted, valid_volume, f"Candle conservation {dataset} {symbol} {day}", atol=0)
            for field in ("completed_bars", "threshold_shares", "completed_volume_shares",
                          "incomplete_volume_shares", "barrier_discarded_volume_shares"):
                same_number(saved[field], actual[field], f"Rebuilt candles {dataset} {symbol} {day} {field}")
            for field in ("median_duration_ms", "p95_duration_ms"):
                if actual["completed_bars"]:
                    same_number(saved[field], actual[field], f"Rebuilt candle duration {dataset} {symbol} {day} {field}")
                else:
                    require(pd.isna(saved[field]), f"Empty candles have a recorded duration: {dataset} {symbol} {day}.")
            require(saved["split"] == ("training" if day in train else "held_out"), f"Candle split mismatch: {dataset} {symbol} {day}.")
            if day in train:
                total = training_totals[(dataset, symbol)]
                total["seconds"] += facts["requested_seconds"]
                total["volume"] += valid_volume
                total["valid_trades"] += actual["valid_trade_records"]
                total["invalid_trades"] += actual["invalid_trade_records"]
                total["days"] += 1
        if (index + 1) % 20 == 0 or index + 1 == len(entries):
            progress(f"Audited {index + 1}/{len(entries)} raw requests and candle conservation; {reused_raw_requests} independent audit caches reused.")
    same_number(summary["source_rows"], raw_records, "Summary raw source rows", atol=0)
    for task, totals in training_totals.items():
        saved = thresholds.loc[task]
        expected_seconds = len(train) * config["minutes"] * 60
        require(totals["seconds"] == expected_seconds and totals["days"] == len(train), f"Incomplete training exposure: {task}.")
        expected_threshold = max(1, int(np.floor(totals["volume"] / expected_seconds * protocol["volume_target_training_seconds"] + .5)))
        expected = {"training_seconds": expected_seconds, "training_days": len(train),
                    "training_volume_shares": totals["volume"], "training_valid_trades": totals["valid_trades"],
                    "training_invalid_trades": totals["invalid_trades"],
                    "target_seconds": protocol["volume_target_training_seconds"], "threshold_shares": expected_threshold}
        for field, value in expected.items():
            same_number(saved[field], value, f"Training threshold {task} {field}", atol=0)

    task_daily = pd.read_csv(out_dir / "task_daily_scores.csv")
    daily = pd.read_csv(out_dir / "daily_scores.csv")
    coverage = pd.read_csv(out_dir / "coverage.csv")
    exact_grid(task_daily, ["dataset", "symbol", "day"], test_task_days, "Held-out task scores")
    exact_grid(daily, ["day"], {(day,) for day in test}, "Held-out daily scores")
    exact_grid(coverage, ["dataset", "symbol", "day"], all_task_days, "Forecast coverage")
    task_result_frame = pd.DataFrame(summary["task_results"])
    exact_grid(task_result_frame, ["dataset", "symbol"], tasks, "Summary model tasks")
    results = task_result_frame.set_index(["dataset", "symbol"])
    cov = coverage.set_index(["dataset", "symbol", "day"])
    require(((coverage.common_bins >= 0) & (coverage.common_bins <= coverage.quote_eligible_bins) &
             (coverage.quote_eligible_bins <= coverage.candidate_bins)).all(), "Coverage stages are not nested.")
    require(np.isfinite(task_daily[LOSS_COLUMNS + ["rows"]].to_numpy(dtype=float)).all(), "Task scores contain nonfinite values.")
    require((task_daily[LOSS_COLUMNS + ["rows"]] >= 0).all().all(), "Task scores contain negative values.")
    require(task_daily.rows.eq(np.floor(task_daily.rows)).all() and
            task_daily.positive_bins.between(0, task_daily.rows).all(), "Invalid task row or positive-bin counts.")
    require(task_daily.reference_brier_sum.le(task_daily.rows + 1e-10).all() and
            task_daily.volume_brier_sum.le(task_daily.rows + 1e-10).all(), "A task Brier total exceeds its paired observation count.")
    for row in task_daily.itertuples():
        expected_rows = cov.loc[(row.dataset, row.symbol, row.day), "common_bins"]
        status = results.loc[(row.dataset, row.symbol), "status"]
        same_number(row.rows, 0 if status == "insufficient_training_rows" else expected_rows,
                    f"Scored paired cohort {row.dataset} {row.symbol} {row.day}", atol=0)
    reconstructed_daily = task_daily.groupby("day")[LOSS_COLUMNS + ["rows", "positive_bins"]].sum().sort_index()
    actual_daily = daily.set_index("day").sort_index()
    for column in reconstructed_daily:
        require(np.allclose(actual_daily[column], reconstructed_daily[column], rtol=1e-12, atol=1e-9), f"Daily {column} does not equal the complete task sum.")
    for task, result in results.iterrows():
        train_counts = {day: int(cov.loc[(*task, day), "common_bins"]) for day in train}
        require(result.train_rows_by_day == train_counts and result.train_rows == sum(train_counts.values()), f"Training row counts mismatch: {task}.")
    zero_task_days = int(task_daily.rows.le(0).sum())
    actual_missing = set(task_daily.loc[task_daily.rows.le(0), ["dataset", "symbol", "day"]].itertuples(index=False, name=None))
    reported_missing = {(row["dataset"], row["symbol"], row["day"]) for row in summary["missing_task_days"]}
    require(actual_missing == reported_missing, "Summary omitted or invented zero-row task/dates.")
    pooled = reconstructed_daily.sum()
    bootstrap_checks = 0
    if (daily.rows > 0).all():
        same_number(summary["pooled"]["rows"], pooled["rows"], "Pooled paired rows", atol=0)
        for metric, first, second in (("log_loss", LOSS_COLUMNS[0], LOSS_COLUMNS[1]), ("brier", LOSS_COLUMNS[2], LOSS_COLUMNS[3])):
            delta = pooled[first] - pooled[second]
            expected = {"reference": pooled[first] / pooled["rows"], "volume": pooled[second] / pooled["rows"],
                        "absolute_gain": delta / pooled["rows"],
                        "relative_gain_percent": 100 * delta / pooled[first] if pooled[first] > 0 else None}
            for field, value in expected.items():
                same_number(summary["pooled"][metric][field], value, f"Pooled {metric} {field}")
            for word, count in (("positive", int((daily[first] > daily[second]).sum())),
                                ("negative", int((daily[first] < daily[second]).sum())),
                                ("tied", int((daily[first] == daily[second]).sum()))):
                same_number(summary[f"{word}_{metric}_days"], count, f"{word} {metric} days", atol=0)
        independently_sampled = independently_resample(daily, protocol)
        for name, metrics in independently_sampled.items():
            require(summary[name]["replicates"] == protocol["uncertainty"]["replicates"], f"Wrong bootstrap replicate count: {name}.")
            for metric, intervals in metrics.items():
                for field, expected in intervals.items():
                    actual = summary[name][metric][field]
                    if expected is None:
                        require(actual is None, f"{name} {metric} {field} should be undefined.")
                    else:
                        require(np.allclose(actual, expected, rtol=1e-9, atol=1e-12), f"Independent fixed-seed interval mismatch: {name} {metric} {field}.")
                    bootstrap_checks += 1
        interval = independently_sampled["five_session_block_bootstrap"]["log_loss"]["absolute_gain_ci95"]
        expected_support = bool(len(daily) >= 20 and interval[0] > 0 and
                                pooled[LOSS_COLUMNS[0]] > pooled[LOSS_COLUMNS[1]] and
                                not zero_task_days and not summary["nonconverged_models"])
        require(summary["supports_positive_incremental_prediction"] == expected_support,
                "Summary evidence flag differs from the frozen rule evaluated independently.")
    else:
        require(summary["status"] != "ok" and not summary["supports_positive_incremental_prediction"],
                "A zero-row intended held-out date was incorrectly called supported.")
    if zero_task_days or summary["nonconverged_models"]:
        require(not summary["supports_positive_incremental_prediction"], "Incomplete tasks or nonconverged models cannot support confirmation.")
    held = coverage.loc[coverage["split"].eq("held_out")]
    for field, expected in {"candidate_bins": int(held.candidate_bins.sum()),
                            "quote_eligible_bins": int(held.quote_eligible_bins.sum()), "paired_bins": int(task_daily.rows.sum())}.items():
        same_number(summary["coverage"][field], expected, f"Summary coverage {field}", atol=0)

    result = {
        "status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_sha256": analysis_sha, "protocol_sha256": summary["protocol_sha256"],
        "raw_requests_checked": len(entries), "raw_records_checked": raw_records,
        "reused_independent_raw_audit_requests": reused_raw_requests,
        "raw_action_t_records": raw_trade_records,
        "raw_t_finite_positive_integer_size_shares_before_flag_or_order_checks": raw_finite_positive_integer_trade_shares,
        "normalized_valid_trade_shares": valid_shares, "invalid_trade_records": invalid_records,
        "ambiguous_trade_timestamp_groups": ambiguous_groups,
        "raw_valid_volume_comparison": "All raw T shares directly equal normalized valid volume."
        if not invalid_records else "Unsafe or ambiguous prints occurred: raw T sizes are reported before validity exclusions; conservation uses independently renormalized valid volume.",
        "candle_task_days_checked": len(candles), "maximum_share_conservation_error": maximum_conservation_error,
        "training_thresholds_checked": len(thresholds), "training_seconds_per_task": len(train) * config["minutes"] * 60,
        "held_out_task_days_checked": len(task_daily), "zero_row_task_days": zero_task_days,
        "complete_held_out_dates": len(daily), "independent_bootstrap_intervals_checked": bootstrap_checks,
        "paired_held_out_rows": int(task_daily.rows.sum()), "summary_status": summary["status"],
        "duration_seconds": round(time.perf_counter() - started, 3),
        "checks": ["Raw source checksums, exact request grid and receive-clock bounds",
                   "Raw T counts, valid shares and per-request trade-quality agreement",
                   "Per-venue/stock/day completed + incomplete + barrier-discarded share conservation",
                   "Rebuilt completed candle counts and share totals",
                   "Training-only volume quanta reconstructed from every requested training second",
                   "All intended task/date records, paired cohort counts and aggregate loss/Brier sums",
                   "Independent fixed-seed whole-date and consecutive-session block intervals"],
    }
    write_json(out_dir / "audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "fresh_confirmation.toml")
    parser.add_argument("--manifest", type=Path, default=ROOT / "out/fresh_confirmation/data_manifest.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out/fresh_confirmation")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Prepare raw audit facts after training thresholds are frozen; no scoring outputs required.")
    args = parser.parse_args()
    try:
        operation = prepare_raw_audit if args.prepare_only else audit
        result = operation(args.config.resolve(), args.manifest.resolve(), args.out_dir.resolve(),
                           progress=lambda line: print(line, flush=True))
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(safe_error(exc), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
