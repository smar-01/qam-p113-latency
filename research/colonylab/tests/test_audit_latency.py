"""Tamper tests for the independently reconstructed latency result audit."""
from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from colonylab.latency_validation import infer_latency


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_latency.py"
SPEC = importlib.util.spec_from_file_location("audit_latency_independent", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


@pytest.fixture
def artifacts(tmp_path):
    out = tmp_path / "out" / "latency"
    out.mkdir(parents=True)
    dates = pd.bdate_range("2026-08-03", periods=22).strftime("%Y-%m-%d").tolist()
    config = {
        "name": "audit-fixture", "datasets": ["XNAS.ITCH", "BATS.PITCH"], "symbols": ["AAPL"],
        "schemas": ["mbp-1"], "training_days": dates[:2], "held_out_days": dates[2:],
        "start_time_utc": "14:00:00", "minutes": 30, "record_limit_per_request": 1000,
    }
    config_path = tmp_path / "study.toml"
    config_path.write_text("\n".join(f"{key} = {json.dumps(value)}" for key, value in config.items()), encoding="utf-8")
    protocol = {"horizons_ms": [1., 10., 100.], "clocks": ["ts_recv", "ts_event"],
                "minimum_matched_events_per_date": 20, "bootstrap_replicates": 100,
                "bootstrap_block_sessions": 5, "bootstrap_seed": 20261102,
                "primary_study": "fresh_confirmation", "primary_phase": "held_out", "minimum_inference_dates": 20}
    dump(out / "protocol.json", protocol)
    manifest_path = tmp_path / "acquisition.json"
    sources, receipts, archive = [], [], []
    for index, request in enumerate(AUDIT._expected_requests(config)):
        path = tmp_path / "raw" / f"source{index}.dbn.zst"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(f"mock source {index}".encode())
        receipt = {"request": request, "rows": 10, "file_size_bytes": path.stat().st_size,
                   "complete": True, "record_limit_reached": False}
        receipts.append(receipt)
        sources.append({**receipt, "study": "fresh_confirmation", "data_path": str(path),
                        "content_sha256": AUDIT.sha256_file(path), "acquisition_manifest": str(manifest_path)})
        archive.append({"path": str(path.relative_to(tmp_path)), "sha256": AUDIT.sha256_file(path)})
    dump(manifest_path, {"complete": True, "configuration": config, "requests": receipts})
    dump(tmp_path / "research-archive-manifest.json", {"files": archive})
    computation = tmp_path / "computation.py"
    computation.write_text("# fixture identity\n", encoding="utf-8")
    provenance = {"studies": [{"name": "fresh_confirmation", "configuration": config, "config_path": str(config_path)}],
                  "sources": sources, "computation_code": [{"path": str(computation), "sha256": AUDIT.sha256_file(computation)}]}
    dump(out / "provenance.json", provenance)
    daily, quality, checks = [], [], []
    pairs = [(leader, follower) for leader in config["datasets"] for follower in config["datasets"] if leader != follower]
    for phase, days in (("training", dates[:2]), ("held_out", dates[2:])):
        for day, clock, symbol, (leader, follower), horizon, regime in itertools.product(
                days, protocol["clocks"], config["symbols"], pairs, protocol["horizons_ms"], ("all", "calm", "volatile")):
            index = dates.index(day)
            median = {"all": 250., "calm": 200., "volatile": 300.}[regime] + index
            if clock == "ts_event":
                median *= .8
            multiplier = 2 if regime == "all" else 1
            matched, censored, eligible = 20 * multiplier, 10 * multiplier, 30 * multiplier
            latency_sum = int(round(matched * median * 1000))
            waiting_sum = latency_sum + censored * int(horizon * 1_000_000)
            daily.append({"study": "fresh_confirmation", "phase": phase, "date": day,
                          "clock": clock, "symbol": symbol, "leader": leader, "follower": follower,
                          "horizon_ms": horizon, "regime": regime, "n_events": 50 * multiplier,
                          "n_already": 10 * multiplier, "n_no_state": 5 * multiplier,
                          "n_behind": 35 * multiplier, "n_eligible": eligible, "n_matched": matched,
                          "n_censored": censored, "n_quality_truncated": 2 * multiplier,
                          "n_right_truncated": 3 * multiplier, "n_observed_matches": 23 * multiplier,
                          "known_matches_truncated": 3 * multiplier, "sum_latency_ns": latency_sum,
                          "restricted_wait_sum_ns": waiting_sum, "restricted_mean_wait_us": waiting_sum / eligible / 1000,
                          "match_rate": matched / eligible, "p10_us": median - 20, "median_us": median,
                          "p90_us": median + 20, "mean_us": median})
        for day, clock, symbol, dataset in itertools.product(days, protocol["clocks"], config["symbols"], config["datasets"]):
            quality.append({"study": "fresh_confirmation", "date": day, "clock": clock,
                            "symbol": symbol, "dataset": dataset, "raw_records": 10,
                            "valid_state_records": 7, "invalid_state_records": 1, "state_records": 8})
        for day, clock, symbol, (leader, follower), side in itertools.product(
                days, protocol["clocks"], config["symbols"], pairs, ("bid", "ask")):
            checks.append({"study": "fresh_confirmation", "date": day, "clock": clock,
                           "symbol": symbol, "leader": leader, "follower": follower, "side": side,
                           "leader_events": 50, "cases": 3, "mismatches": 0})
    daily = pd.DataFrame(daily)
    daily.to_csv(out / "daily_latency.csv", index=False)
    inferred = infer_latency(daily, bootstrap_replicates=100,
                             expected_dates={f"fresh_confirmation:{phase}": days
                                             for phase, days in (("training", dates[:2]), ("held_out", dates[2:]))})
    inferred.to_csv(out / "inference.csv", index=False)
    pd.DataFrame(quality).to_csv(out / "quote_quality.csv", index=False)
    pd.DataFrame(checks).to_csv(out / "day_audits.csv", index=False)
    return tmp_path, out


def run_audit(artifacts):
    root, out = artifacts
    return AUDIT.audit_latency(out, root=root)


def assert_failed(artifacts, phrase):
    result = run_audit(artifacts)
    assert result["status"] == "failed", result
    assert phrase in " ".join(result["errors"]), result


def test_complete_fixture_reconstructs_counts_inference_and_raw_identities(artifacts):
    result = run_audit(artifacts)
    assert result["status"] == "passed", result
    assert result["raw_source_requests"] == 44
    assert result["raw_records_reconciled"] == 440
    assert result["sources_with_historical_archive_fingerprint"] == 44
    assert result["daily_rows"] == 22 * 2 * 1 * 2 * 3 * 3
    assert result["independent_event_cases"] == 22 * 2 * 1 * 2 * 2 * 3
    assert result["inference_estimates_and_intervals_reconstructed"]
    assert result["raw_dbn_reparsed_by_final_auditor"] is False


def test_raw_byte_tampering_is_detected_by_fresh_hash(artifacts):
    root, out = artifacts
    source = json.loads((out / "provenance.json").read_text())["sources"][0]
    Path(source["data_path"]).write_bytes(b"tampered")
    assert_failed(artifacts, "Raw source content changed")
    result = run_audit(artifacts)
    assert result["raw_hash_verification_requested"]
    assert not result["raw_hashes_freshly_verified"]


def test_provenance_cannot_replace_the_historical_raw_identity(artifacts):
    _, out = artifacts
    provenance = json.loads((out / "provenance.json").read_text())
    source = provenance["sources"][0]
    path = Path(source["data_path"])
    # Change bytes while preserving byte count and supply the corresponding
    # replacement provenance hash. The older archive still catches this.
    path.write_bytes(b"X" * path.stat().st_size)
    source["content_sha256"] = AUDIT.sha256_file(path)
    dump(out / "provenance.json", provenance)
    assert_failed(artifacts, "Historical raw fingerprint")


def test_computation_code_mutation_is_not_silently_accepted(artifacts):
    root, _ = artifacts
    (root / "computation.py").write_text("# changed code\n", encoding="utf-8")
    assert_failed(artifacts, "Computation code differs")


@pytest.mark.parametrize("field,phrase", [("n_events", "event conservation"),
                                         ("n_behind", "event conservation"),
                                         ("restricted_wait_sum_ns", "Restricted waiting-time sum"),
                                         ("sum_latency_ns", "Restricted waiting-time sum"),
                                         ("mean_us", "Conditional mean latency"),
                                         ("match_rate", "Match rate")])
def test_independent_count_and_waiting_arithmetic_detects_tampering(artifacts, field, phrase):
    _, out = artifacts
    frame = pd.read_csv(out / "daily_latency.csv")
    frame.loc[0, field] += 1
    frame.to_csv(out / "daily_latency.csv", index=False)
    assert_failed(artifacts, phrase)


def test_missing_daily_grid_row_cannot_be_dropped_from_inference(artifacts):
    _, out = artifacts
    frame = pd.read_csv(out / "daily_latency.csv")
    frame.iloc[1:].to_csv(out / "daily_latency.csv", index=False)
    assert_failed(artifacts, "Daily latency: missing")


def test_zero_sampled_checks_for_nonempty_pair_is_rejected(artifacts):
    _, out = artifacts
    frame = pd.read_csv(out / "day_audits.csv")
    frame.loc[0, "cases"] = 0
    frame.to_csv(out / "day_audits.csv", index=False)
    assert_failed(artifacts, "lacks sampled event validation")


def test_sampled_matching_disagreement_fails_audit(artifacts):
    _, out = artifacts
    frame = pd.read_csv(out / "day_audits.csv")
    frame.loc[0, "mismatches"] = 1
    frame.to_csv(out / "day_audits.csv", index=False)
    assert_failed(artifacts, "has mismatches")


@pytest.mark.parametrize("field", ["estimate_us", "relative_change_percent", "ci95_low_us", "relative_ci95_high_percent", "seed"])
def test_independent_reconstruction_detects_inference_tampering(artifacts, field):
    _, out = artifacts
    frame = pd.read_csv(out / "inference.csv")
    frame[field] = frame[field].astype(float)
    index = frame.loc[frame[field].notna()].index[0]
    frame.loc[index, field] += .1
    frame.to_csv(out / "inference.csv", index=False)
    assert_failed(artifacts, f"Inference {field}")


def test_changed_raw_record_denominator_cannot_bypass_quality_reconciliation(artifacts):
    _, out = artifacts
    frame = pd.read_csv(out / "quote_quality.csv")
    frame.loc[0, "raw_records"] += 1
    frame.to_csv(out / "quote_quality.csv", index=False)
    assert_failed(artifacts, "Decoded symbol record counts")


def test_billing_preference_changes_do_not_change_the_historical_acquisition_design(artifacts):
    root, _ = artifacts
    path = root / "acquisition.json"
    saved = json.loads(path.read_text())
    saved["configuration"]["max_cost_usd"] = 5.
    saved["configuration"]["billing_assumed_included"] = False
    dump(path, saved)
    result = run_audit(artifacts)
    assert result["status"] == "passed", result
    assert result["raw_hashes_freshly_verified"]
    assert result["raw_sources_freshly_hashed"] == 44


def test_billing_exception_cannot_hide_a_changed_acquisition_window(artifacts):
    root, _ = artifacts
    path = root / "acquisition.json"
    saved = json.loads(path.read_text())
    saved["configuration"]["max_cost_usd"] = 5.
    saved["configuration"]["minutes"] += 1
    dump(path, saved)
    assert_failed(artifacts, "Acquisition configuration differs")
