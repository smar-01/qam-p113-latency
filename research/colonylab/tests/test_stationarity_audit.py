"""Independent selection, paired-test and multiplicity audit checks, offline."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from colonylab.stationarity import test_pairs as pipeline_pairs, adjust_adf_holm, sample_contiguous
from colonylab import stationarity_pipeline


spec = importlib.util.spec_from_file_location("stationarity_audit", Path(__file__).resolve().parents[1] / "scripts/audit_stationarity.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def prices(n=500, seed=778):
    noise = np.random.default_rng(seed).normal(size=n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = .6 * x[i-1] + noise[i]
    return 400_000_000_000 + np.rint(x * 1_000_000).astype(np.int64)


def test_selection_uses_strict_prior_and_earliest_equal_length_run():
    p = SimpleNamespace(times_ns=np.array([0, 20, 30, 50], np.int64), price2=np.array([100, 0, 110, 120], np.int64),
                        valid=np.array([True, False, True, True]), segments=np.array([0, 0, 1, 1], np.int64), start_ns=0, end_ns=70)
    values, fields = audit.independent_selection(p, 10, 70, 10)
    # Baselines at20 and30 exclude records at exactly that timestamp.
    assert values.tolist() == [[40, 50, 60], [110, 110, 120]]
    assert fields["valid_grid_points"] == 5 and fields["run_count"] == 2
    p.valid[-1] = False
    values, _ = audit.independent_selection(p, 10, 60, 10)
    assert values[0].tolist() == [10, 20]  # Two runs tie; select earliest.


def test_hidden_invalid_interval_splits_even_when_both_snapshots_and_segment_ids_match():
    p = SimpleNamespace(times_ns=np.array([0, 12, 13], np.int64), price2=np.array([100, 0, 100], np.int64),
                        valid=np.array([True, False, True]), segments=np.array([0, 0, 0], np.int64), start_ns=0, end_ns=40)
    values, fields = audit.independent_selection(p, 10, 40, 10)
    assert fields["valid_grid_points"] == 3 and fields["run_count"] == 2
    assert values[0].tolist() == [20, 30]


def test_independent_scan_agrees_with_sampler_on_randomized_quality_paths():
    rng = np.random.default_rng(418)
    for _ in range(100):
        times = np.unique(rng.integers(0, 1000, size=60)).astype(np.int64)
        valid = rng.random(len(times)) > .15
        path = SimpleNamespace(times_ns=times, price2=np.full(len(times), 100, np.int64), valid=valid,
             segments=np.cumsum(~valid).astype(np.int64), start_ns=0, end_ns=1000)
        values, fields = audit.independent_selection(path, 0, 1000, 10)
        expected = sample_contiguous(path, 0, 1000, 10)
        assert np.array_equal(values, np.array([expected["times_ns"], expected["price2"]]))
        for key in fields:
            assert fields[key] == expected[key]


def test_direct_statsmodels_pairs_and_table_boundaries_agree_with_persisted_contract():
    actual = audit.recompute_pairs(prices())
    expected = pipeline_pairs(prices())
    audit.same(actual, expected, "paired diagnostics")
    assert all(row["status"] == "tested" for row in actual)
    assert any(row["kpss_pvalue_relation"] in {">=", "<="} for row in actual)


@pytest.mark.parametrize("n,expected", [(0, ["too_short"]*3), (299, ["too_short"]*3), (300, ["", "", "too_short"])])
def test_minimum_applies_to_each_transformed_series(n, expected):
    rows = audit.recompute_pairs(prices(n))
    assert [row["skip_reason"] for row in rows] == expected
    assert rows[-1]["n_observations"] == max(0, n-1)


def test_constant_prices_are_not_classified_as_stationary_by_test_failure():
    rows = audit.recompute_pairs(np.full(500, 400_000_000_000, np.int64))
    assert all(row["status"] == "skipped" and row["skip_reason"] == "constant_series" and row["classification"] == "not_tested" for row in rows)


def test_global_holm_includes_finite_adf_from_a_pair_with_failed_kpss():
    rows = [{"adf_pvalue": p, "status": status} for p, status in [(.01, "tested"), (.03, "test_error"), (.04, "tested"), (None, "skipped")]]
    actual = audit.independent_holm(rows, .05)
    expected = adjust_adf_holm(rows)
    audit.same(actual, expected, "global Holm")
    assert [row["adf_pvalue_holm"] for row in actual] == [.03, .06, .06, None]


@pytest.mark.parametrize("bad", [np.array([100., 101.]), np.array([0, 1]), np.array([1, -1])])
def test_persisted_price_contract_rejects_non_native_or_nonpositive_values(bad):
    with pytest.raises(ValueError, match="positive native integers"):
        audit.recompute_pairs(bad)


def test_content_identity_keeps_native_int64_prices_and_time_order():
    values = np.array([[1, 2], [400_000_000_001, 400_000_000_003]], np.int64)
    assert audit.series_token(values) != audit.series_token(values[:, ::-1])
    changed = values.copy(); changed[1, 0] += 1
    assert audit.series_token(values) != audit.series_token(changed)


def test_source_study_destination_cannot_be_overwritten_even_with_failed_inputs(tmp_path, monkeypatch):
    old = {"passed": True, "status": "passed", "identity": "preserve"}
    (tmp_path / "result_audit.json").write_text(json.dumps(old))
    (tmp_path / "protocol.json").write_text(json.dumps({"source_study_dir": str(tmp_path)}))
    monkeypatch.setattr("sys.argv", ["audit_stationarity.py", "--out-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="separate from its source"):
        audit.main()
    assert json.loads((tmp_path / "result_audit.json").read_text()) == old


@pytest.fixture
def saved_tables(tmp_path):
    config = {"windows": [{"name": "pre30", "start_minute": -30, "end_minute": 0},
                           {"name": "post30", "start_minute": 0, "end_minute": 30}],
              "datasets": ["XNAS.ITCH"], "clocks": ["ts_recv"], "quote_age_ms": [5000],
              "step_ns": 1_000_000_000, "minimum_observations": 300, "alpha": .05, "longer_run_points": 600}
    trial = {"trial_id": "t", "event_id": "event", "event_type": "wwdc", "bundle_id": "bundle",
             "role": "event", "anchor_utc": "2024-06-10T17:00:00+00:00"}
    source_protocol = {"trials": [trial], "source_groups": [{"group_id": "g", "trial_ids": ["t"]}],
                       **{key: config[key] for key in ("datasets", "clocks", "quote_age_ms")}}
    anchor = pd.Timestamp(trial["anchor_utc"]).value
    requests, runs, arrays, tests = [], [], {}, []
    for n, window in enumerate(config["windows"]):
        start, end = anchor + window["start_minute"] * 60_000_000_000, anchor + window["end_minute"] * 60_000_000_000
        values = np.array([np.arange(start, start+301_000_000_000, 1_000_000_000, dtype=np.int64), prices(301, 77+n)], np.int64)
        sid = audit.series_token(values)
        arrays[sid] = values
        context = {"source_group_id": "g", "trial_id": "t", "event_id": "event", "event_class": "wwdc",
            "bundle_id": "bundle", "role": "event", "anchor_utc": trial["anchor_utc"], "window": window["name"],
            "venue": "XNAS.ITCH", "clock": "ts_recv", "age_ms": 5000, "start_ns": start, "end_ns": end}
        requests.append({"request_id": audit.signature(context), **context, "series_id": sid, "total_grid_points": 1800,
            "valid_grid_points": 301, "run_count": 1, "longest_points": 301, "run_start_ns": start,
            "run_end_ns": int(values[0, -1]), "fraction_window": 301/1800, "valid_fraction_window": 301/1800,
            "longer_run_600_points": False})
        runs.append({"series_id": sid, "n_price_points": 301, "start_ns": start, "end_ns": int(values[0, -1]), "duration_seconds": 300.})
        tests.extend({"series_id": sid, **row} for row in pipeline_pairs(values[1]))
    tests = adjust_adf_holm(tests)
    np.savez_compressed(tmp_path / "series.npz", **arrays)
    for name, rows in [("requests", requests), ("runs", runs), ("tests", tests)]:
        stationarity_pipeline.csv_rows(tmp_path / (name+".csv"), rows)
    counts = {"source_groups": 1, "diagnostic_windows": 2, "unique_series": 2, "unique_test_pairs": 6,
              "tested_pairs": 6, "skipped_pairs": 0}
    result = {"status": "completed", "protocol": config, "counts": counts,
              "summary": stationarity_pipeline.summarize(requests, tests)}
    (tmp_path / "results.json").write_text(json.dumps(result))
    return SimpleNamespace(out=tmp_path, protocol=config, source_protocol=source_protocol, arrays=arrays)


def test_all_saved_numeric_tables_and_summary_are_independently_reconstructed(saved_tables):
    s = saved_tables
    requests, counts, family = audit.audit_tables(s.out, s.protocol, s.source_protocol, progress=lambda *args, **kwargs: None)
    assert len(requests) == 2 and counts["tested_pairs"] == 6 and family == 6


@pytest.mark.parametrize("field", ["adf_stat", "adf_pvalue_holm", "classification"])
def test_numeric_or_classification_tamper_is_detected(saved_tables, field):
    s = saved_tables
    frame = audit.table(s.out / "tests.csv")
    frame.loc[0, field] = "unit_root_compatible" if field == "classification" else float(frame.loc[0, field]) + .02
    if field == "classification" and frame.loc[0, field] == audit.recompute_pairs(next(iter(s.arrays.values()))[1])[0]["classification"]:
        frame.loc[0, field] = "conflicting"
    stationarity_pipeline.csv_rows(s.out / "tests.csv", frame.to_dict("records"))
    with pytest.raises(ValueError, match=field):
        audit.audit_tables(s.out, s.protocol, s.source_protocol, progress=lambda *args, **kwargs: None)


def test_nullable_nanosecond_csv_reader_preserves_single_nanoseconds(tmp_path):
    path = tmp_path / "nanoseconds.csv"
    exact = 1_780_000_000_000_000_001
    path.write_text("start_ns,end_ns\n" + str(exact) + ",\n," + str(exact+1) + "\n")
    frame = audit.table(path)
    assert frame.start_ns.dtype == "Int64" and frame.loc[0, "start_ns"] == exact
    assert frame.loc[1, "end_ns"] == exact+1 and pd.isna(frame.loc[0, "end_ns"])
