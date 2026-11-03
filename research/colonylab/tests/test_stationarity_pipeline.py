"""Diagnostic registry completeness and isolation from the audited study."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from colonylab.event_midpoints import NativeMidpoints
from colonylab import stationarity_pipeline as pipeline


def test_selected_series_id_binds_both_time_and_exact_price():
    t, p = np.array([1, 2], np.int64), np.array([400_000_000_001, 400_000_000_003], np.int64)
    token = pipeline.series_identity(t, p)
    assert token == pipeline.series_identity(t.copy(), p.copy())
    assert token != pipeline.series_identity(t + 1, p)
    assert token != pipeline.series_identity(t, p + 1)


def test_missing_endpoint_cannot_round_another_native_timestamp(tmp_path):
    stamp = 1_780_938_000_123_456_789
    target = tmp_path / "runs.csv"
    pipeline.csv_rows(target, [{"series_id": "empty", "start_ns": None},
                               {"series_id": "present", "start_ns": stamp}])
    recovered = pd.read_csv(target, dtype={"start_ns": "Int64"})
    assert pd.isna(recovered.start_ns.iloc[0])
    assert recovered.start_ns.iloc[1] == stamp


def test_extraction_keeps_full_registry_and_shared_run_identity(monkeypatch):
    anchor = pd.Timestamp("2026-06-08T17:00:00Z").value
    first = anchor - 10_000_000_000
    times = np.arange(first - 1, anchor + 60_000_000_000, 1_000_000_000, dtype=np.int64)
    path = NativeMidpoints(times, np.arange(len(times), dtype=np.int64) + 400_000_000_000,
        np.ones(len(times), bool), np.zeros(len(times), np.int64), first, anchor + 60_000_000_000)
    monkeypatch.setattr(pipeline, "decode_group", lambda *args: ({("XNAS.ITCH", 250, "ts_recv"): path,
        ("XNAS.ITCH", 1000, "ts_recv"): path}, []))
    trials = [{"trial_id": name, "event_id": name, "event_type": "wwdc", "bundle_id": name,
        "role": "event", "anchor_utc": "2026-06-08T17:00:00Z"} for name in ("a", "b")]
    result = pipeline.extract_group({"group": {"group_id": "group"}, "entries": [],
        "source_protocol": {}, "trials": trials,
        "protocol": {"windows": [{"name": "post", "start_minute": 0, "end_minute": 1}],
                     "step_ns": 1_000_000_000, "longer_run_points": 600}})
    assert len(result["requests"]) == 4 and len(result["arrays"]) == 1
    assert len({row["request_id"] for row in result["requests"]}) == 4
    row = result["requests"][0]
    assert row["longest_points"] == 60 and row["event_class"] == "wwdc"
    assert not row["longer_run_600_points"]


def test_validation_refuses_changed_audited_artifact(tmp_path):
    (tmp_path / "result_audit.json").write_text(json.dumps({"passed": True, "status": "passed",
        "output_sha256": {"protocol.json": "wrong"}, "computation_code": []}))
    (tmp_path / "protocol.json").write_text("{}")
    with pytest.raises(ValueError, match="Audited source-study artifact changed"):
        pipeline.validate_source_study(tmp_path, tmp_path)


def test_validation_refuses_unpassed_source_audit(tmp_path):
    (tmp_path / "result_audit.json").write_text(json.dumps({"passed": False, "status": "failed"}))
    with pytest.raises(ValueError, match="passed source-study audit"):
        pipeline.validate_source_study(tmp_path, tmp_path)


def test_summary_deduplicates_shared_paths_and_is_strict_json():
    requests = [{"series_id": "same", "clock": "ts_recv", "age_ms": 250},
                {"series_id": "same", "clock": "ts_recv", "age_ms": 250}]
    tests = [{"series_id": "same", "transformation": "log_level", "regression": "c",
              "status": "skipped", "classification": "not_tested", "adf_pvalue_holm": None}]
    summary = pipeline.summarize(requests, tests)
    assert summary[0]["planned_windows"] == 2 and summary[0]["unique_series"] == 1
    assert summary[0]["skipped"] == 1 and summary[0]["tested"] == 0
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary
