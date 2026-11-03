"""Independent correction audit rejects changed inputs, cohorts and effects."""
from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SPEC = importlib.util.spec_from_file_location(
    "team_correction_audit", Path(__file__).resolve().parents[1] / "scripts" / "audit_team_code_corrections.py")
auditor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(auditor)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


@pytest.fixture
def result_set(tmp_path):
    out = tmp_path / "result"
    out.mkdir()
    config = {
        "training_days": ["2026-10-05"], "held_out_days": [],
        "datasets": ["XNAS.ITCH", "BATS.PITCH"], "symbols": ["AAPL"],
        "start_time_utc": "14:00:00", "minutes": 10, "record_limit_per_request": 100,
    }
    protocol = {"configuration": config, "clocks": ["ts_recv"], "horizon_ms": 10,
                "event_study_offsets_us": [-100, 0, 20], "leadlag_bin_us": 20,
                "leadlag_max_lag_us": 20, "event_study_leader": "XNAS",
                "checkpoint_audit_required": False, "summary_audit_required": False}
    write_json(out / "protocol.json", protocol)
    sources = []
    for dataset in config["datasets"]:
        raw = tmp_path / f"{dataset}.dbn.zst"
        raw.write_bytes(dataset.encode())
        request = {"dataset": dataset, "schema": "mbp-1", "symbols": ["AAPL"],
                   "stype_in": "raw_symbol", "start": "2026-10-05T14:00:00+00:00",
                   "end": "2026-10-05T14:10:00+00:00", "limit": 100}
        sources.append({"data_path": str(raw), "request": request, "rows": 2,
                        "content_sha256": auditor.sha256(raw)})
    codes = []
    for name in ("original.py", "corrected.py", "runner.py"):
        path = tmp_path / name
        path.write_text("source code\n")
        codes.append({"path": str(path), "sha256": auditor.sha256(path)})
    provenance = {"sources": sources, "original_code": codes[:1], "corrected_code": codes[1:2], "runner_code": codes[2:],
                  "protocol_sha256": auditor.sha256(out / "protocol.json")}
    write_json(out / "provenance.json", provenance)
    pd.DataFrame([{"day": "2026-10-05", "clock": "ts_recv", "symbol": "AAPL", "venue": venue,
                   "float_events": 2, "exact_events": 1, "phantom_events": 1,
                   "corrected_events": 1, "min_abs_float_move": 1e-14}
                  for venue in ("XNAS", "BATS")]).to_csv(out / "midpoint_diagnostics.csv", index=False)
    reactions = []
    for leader, follower in (("XNAS", "BATS"), ("BATS", "XNAS")):
        reactions.append({"day": "2026-10-05", "clock": "ts_recv", "symbol": "AAPL",
                          "leader": leader, "follower": follower, "exact_equal": True,
                          **{f"{prefix}_{key}": value for prefix in ("baseline", "numerical")
                             for key, value in {"n_events": 2, "n_behind": 2,
                                                "n_matched": 1, "median_us": 50}.items()}})
    pd.DataFrame(reactions).to_csv(out / "reaction_comparison.csv", index=False)
    events = []
    for stage, venue, offset in itertools.product(
            ("original", "numerical_only", "support_corrected"), ("XNAS", "BATS"), [-100, 0, 20]):
        events.append({"stage": stage, "day": "2026-10-05", "clock": "ts_recv", "symbol": "AAPL",
                       "leader": "XNAS", "venue": venue, "offset_us": offset,
                       "share": 2.0 if stage == "original" else 1.0, "n": 2 if stage == "original" else 1})
    pd.DataFrame(events).to_csv(out / "event_study_comparison.csv", index=False)
    curves, summaries = [], []
    for stage in ("original", "integer_only", "numerical_only"):
        values = [.05, .2, .1] if stage != "numerical_only" else [.1, .3, .2]
        key = {"stage": stage, "day": "2026-10-05", "clock": "ts_recv", "symbol": "AAPL",
               "x": "BATS", "y": "XNAS"}
        for lag, value in zip([-20., 0., 20.], values):
            curves.append({**key, "lag_us": lag, "corr": value})
        summaries.append({**key, "peak_lag_us": 0., "peak_corr": max(values), "llr": 4.})
    pd.DataFrame(curves).to_csv(out / "fullcurves.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "leadlag_comparison.csv", index=False)
    return out, tmp_path


def test_complete_same_input_artifacts_pass_and_bind_outputs(result_set):
    out, root = result_set
    report = auditor.audit(out, root=root)
    assert report["status"] == "passed", report["errors"]
    assert report["raw_sources_freshly_hashed"] == 2
    assert report["reaction_numerical_negative_control"]
    assert report["floating_phantom_midpoint_events"] == 2
    assert report["corrected_midpoint_events"] == 2
    assert report["event_study_rows"] == 18
    assert len(report["output_sha256"]) == 5


@pytest.mark.parametrize("kind", ["raw", "code", "protocol", "window"])
def test_changed_source_code_or_acquisition_identity_fails(result_set, kind):
    out, root = result_set
    provenance = auditor.read_json(out / "provenance.json")
    if kind == "raw":
        Path(provenance["sources"][0]["data_path"]).write_bytes(b"changed")
    elif kind == "code":
        Path(provenance["corrected_code"][0]["path"]).write_text("changed")
    elif kind == "protocol":
        protocol = auditor.read_json(out / "protocol.json")
        protocol["horizon_ms"] = 100
        write_json(out / "protocol.json", protocol)
    else:
        provenance["sources"][0]["request"]["start"] = "2026-10-05T14:01:00+00:00"
        write_json(out / "provenance.json", provenance)
    report = auditor.audit(out, root=root)
    assert report["status"] == "failed"
    assert report["errors"]


def test_numerical_fix_cannot_silently_change_bid_ask_measurement(result_set):
    out, root = result_set
    path = out / "reaction_comparison.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "numerical_median_us"] = 51
    frame.to_csv(path, index=False)
    report = auditor.audit(out, root=root)
    assert report["status"] == "failed"
    assert "Unchanged bid/ask median_us" in report["errors"][0]


def test_corrected_event_count_must_equal_exact_integer_events(result_set):
    out, root = result_set
    path = out / "midpoint_diagnostics.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "corrected_events"] += 1
    frame.to_csv(path, index=False)
    report = auditor.audit(out, root=root)
    assert report["status"] == "failed"
    assert "Corrected midpoint events" in report["errors"][0]


def test_corrected_crossproducts_must_be_bounded(result_set):
    out, root = result_set
    path = out / "fullcurves.csv"
    frame = pd.read_csv(path)
    frame.loc[(frame.stage == "numerical_only") & (frame.lag_us == 0), "corr"] = 1.1
    frame.to_csv(path, index=False)
    report = auditor.audit(out, root=root)
    assert report["status"] == "failed"
    assert "unit bound" in report["errors"][0]


def test_saved_curve_peak_is_reconstructed_independently(result_set):
    out, root = result_set
    path = out / "leadlag_comparison.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "peak_lag_us"] = 20
    frame.to_csv(path, index=False)
    report = auditor.audit(out, root=root)
    assert report["status"] == "failed"
    assert "Lead-lag peak" in report["errors"][0]


def test_serialized_equal_peak_values_allow_either_true_maximum(result_set):
    out, root = result_set
    path = out / "fullcurves.csv"
    curves = pd.read_csv(path)
    curves.loc[(curves.stage == "integer_only") & (curves.lag_us == 20), "corr"] = .2
    curves.to_csv(path, index=False)
    path = out / "leadlag_comparison.csv"
    summaries = pd.read_csv(path)
    summaries.loc[summaries.stage == "integer_only", ["peak_lag_us", "llr"]] = [20, 16]
    summaries.to_csv(path, index=False)
    report = auditor.audit(out, root=root)
    assert report["status"] == "passed", report["errors"]
    assert report["curves_with_rounding_tied_peak_values"] == 1


def test_support_corrected_cohort_is_fixed_across_offsets(result_set):
    out, root = result_set
    path = out / "event_study_comparison.csv"
    frame = pd.read_csv(path)
    index = frame.index[(frame.stage == "support_corrected") & (frame.offset_us == 20)][0]
    frame.loc[index, ["n", "share"]] = [0, np.nan]
    frame.to_csv(path, index=False)
    report = auditor.audit(out, root=root)
    assert report["status"] == "failed"
    assert "cohort changes" in report["errors"][0]


def test_event_study_ratios_are_allowed_outside_zero_one(result_set):
    out, root = result_set
    path = out / "event_study_comparison.csv"
    frame = pd.read_csv(path)
    frame.loc[frame.stage == "support_corrected", "share"] = -12
    frame.to_csv(path, index=False)
    report = auditor.audit(out, root=root)
    assert report["status"] == "passed", report["errors"]
    assert report["max_abs_event_study_share_by_stage"]["support_corrected"] == 12


def test_missing_curve_task_cannot_be_hidden_by_summary(result_set):
    out, root = result_set
    path = out / "fullcurves.csv"
    pd.read_csv(path).iloc[:-1].to_csv(path, index=False)
    report = auditor.audit(out, root=root)
    assert report["status"] == "failed"
    assert "Lead-lag curves" in report["errors"][0]
