"""Fixed-control, bundle/year and discovery-scope checks for multi-event stats."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from colonylab.multi_event_stats import summarize_multi_events


DIM = {"clock": "ts_recv", "age_ms": 250, "horizon_ms": 10, "metric": "latency_us"}


def add_event(protocol, rows, event_type, year, change, *, session="RTH", suffix="", bundle=None):
    trial_id = f"{event_type}_{year}{suffix}"
    event_day = pd.Timestamp(f"{year}-06-15")
    metadata = {"trial_id": trial_id, "bundle_id": bundle or trial_id, "event_id": trial_id,
                "event_type": event_type, "role": "event", "control_for": None,
                "year": year, "day": event_day.strftime("%Y-%m-%d"), "session": session}
    protocol["trials"].append(metadata)
    for window, value in (("pre30", 100), ("post30", 100 + change)):
        rows.append({**metadata, **DIM, "window": window, "value": value, "complete": True,
                     "reason": "", "quote_count": 10_000_000})
    for index in range(5):
        day = (event_day - pd.Timedelta(weeks=index + 2)).strftime("%Y-%m-%d")
        control = {**metadata, "trial_id": f"{trial_id}_c{index}", "role": "control",
                   "control_for": trial_id, "day": day}
        protocol["trials"].append(control)
        for window, value in (("pre30", 10), ("post30", 11 + index)):
            rows.append({**control, **DIM, "window": window, "value": value,
                         "complete": True, "reason": "", "quote_count": index + 1})
    return trial_id


def fixture():
    protocol = {"trials": [], "controls_per_event": 5, "discovery_event_ids": ["wwdc_2026"],
                "windows": [{"name": "pre30"}, {"name": "post30"}],
                "metric_dimensions": [DIM.copy()]}
    rows = []
    for year, wwdc, earnings in ((2022, 10, 4), (2023, 20, 8)):
        add_event(protocol, rows, "wwdc", year, wwdc)
        add_event(protocol, rows, "earnings_call", year, earnings)
    add_event(protocol, rows, "wwdc", 2026, 100)
    return pd.DataFrame(rows), protocol


def one(table, **identity):
    rows = [row for row in table if all(row.get(name) == value for name, value in identity.items())]
    assert len(rows) == 1, (identity, len(rows))
    return rows[0]


def test_exact_five_control_excess_is_event_change_minus_control_median():
    metrics, protocol = fixture()
    result = summarize_multi_events(metrics, protocol)
    event = one(result["event_excess"], trial_id="wwdc_2022")
    assert event["complete"] and event["status"] == "complete"
    assert event["event_pre_value"] == 100 and event["event_post_value"] == 110
    assert event["event_change"] == 10
    assert event["controls_valid"] == event["controls_expected"] == 5
    assert event["control_change_median"] == 3
    assert event["control_change_q25"] == 2 and event["control_change_q75"] == 4
    assert event["excess_change"] == 7
    assert len(result["event_changes"]) == 30  # Five event anchors plus their 25 controls.
    json.dumps(result, allow_nan=False)


def test_class_summary_uses_events_bundles_and_years_not_quote_counts():
    metrics, protocol = fixture()
    result = summarize_multi_events(metrics, protocol)
    summary = one(result["class_summary"], event_type="wwdc", analysis_scope="historical_primary")
    assert summary["n_events"] == summary["n_bundles"] == summary["n_years"] == 2
    assert summary["median_excess_change"] == 12  # Median of excesses 7 and 17.
    assert summary["excess_change_q25"] == 9.5 and summary["excess_change_q75"] == 14.5
    assert summary["leave_one_year_out_min"] == 7 and summary["leave_one_year_out_max"] == 17
    assert "p_value" not in summary and "confidence_interval" not in summary


def test_inspected_2026_wwdc_remains_separate_from_historical_primary():
    metrics, protocol = fixture()
    result = summarize_multi_events(metrics, protocol)
    historical = one(result["class_summary"], event_type="wwdc", analysis_scope="historical_primary")
    discovery = one(result["class_summary"], event_type="wwdc", analysis_scope="discovery")
    assert historical["years"] == [2022, 2023]
    assert discovery["years"] == [2026] and discovery["median_excess_change"] == 97
    assert discovery["leave_one_year_out_min"] is None


def test_missing_control_cannot_reduce_primary_cohort_to_four():
    metrics, protocol = fixture()
    metrics = metrics.loc[~metrics.trial_id.eq("wwdc_2022_c4")]
    result = summarize_multi_events(metrics, protocol)
    event = one(result["event_excess"], trial_id="wwdc_2022")
    assert not event["complete"] and event["controls_valid"] == 4
    assert event["status"] == "insufficient_control_support"
    assert event["event_change"] == 10  # The observed event still remains inspectable.
    assert event["control_change_median"] is event["excess_change"] is None
    assert event["excluded_controls"][0]["trial_id"] == "wwdc_2022_c4"
    assert "missing_pre_window" in event["excluded_controls"][0]["reason"]
    assert "missing_post_window" in event["excluded_controls"][0]["reason"]
    summary = one(result["class_summary"], event_type="wwdc", analysis_scope="historical_primary")
    assert summary["status"] == "incomplete_support"
    assert summary["n_events_expected"] == 2 and summary["n_events"] == 1
    json.dumps(result, allow_nan=False)


def test_quality_failure_is_retained_with_exact_control_and_window_reason():
    metrics, protocol = fixture()
    mask = metrics.trial_id.eq("wwdc_2022_c2") & metrics.window.eq("post30")
    metrics.loc[mask, ["complete", "value", "reason"]] = [False, np.nan, "insufficient_pair_matches:1"]
    result = summarize_multi_events(metrics, protocol)
    event = one(result["event_excess"], trial_id="wwdc_2022")
    assert event["controls_valid"] == 4 and event["excess_change"] is None
    assert event["excluded_controls"][0]["reason"] == "post:insufficient_pair_matches:1"


def test_missing_event_and_entire_planned_clock_are_explicit():
    metrics, protocol = fixture()
    metrics = metrics.loc[~metrics.trial_id.eq("wwdc_2022")]
    protocol["metric_dimensions"].append({**DIM, "clock": "ts_event"})
    result = summarize_multi_events(metrics, protocol)
    event = one(result["event_excess"], trial_id="wwdc_2022", clock="ts_recv")
    assert event["status"] == "event_insufficient_support"
    assert "missing_pre_window" in event["reason"]
    empty_clock = one(result["class_summary"], event_type="wwdc", analysis_scope="historical_primary", clock="ts_event")
    assert empty_clock["status"] == "insufficient_support"
    assert empty_clock["n_events_expected"] == 2 and empty_clock["n_events"] == 0
    assert empty_clock["median_excess_change"] is None


def test_within_year_class_comparison_pairs_years_before_summarizing():
    metrics, protocol = fixture()
    result = summarize_multi_events(metrics, protocol)
    comparison = one(result["class_comparisons"], analysis_scope="historical_primary")
    assert comparison["event_type_a"] == "earnings_call" and comparison["event_type_b"] == "wwdc"
    assert comparison["years"] == [2022, 2023] and comparison["n_paired_years"] == 2
    assert comparison["median_difference_a_minus_b"] == -9  # Annual differences -6 and -12.
    assert comparison["leave_one_year_out_min"] == -12 and comparison["leave_one_year_out_max"] == -6
    assert not comparison["session_confounded"]
    assert comparison["paired_year_details"][0]["n_unique_bundles"] == 2


def test_cross_session_comparison_is_explicitly_confounded():
    metrics, protocol = fixture()
    for row in protocol["trials"]:
        if row["event_type"] == "earnings_call":
            row["session"] = "afterhours"
    metrics.loc[metrics.event_type.eq("earnings_call"), "session"] = "afterhours"
    result = summarize_multi_events(metrics, protocol)
    comparison = one(result["class_comparisons"], analysis_scope="historical_primary")
    assert comparison["session_confounded"]
    assert "confounded" in comparison["interpretation"]
    assert comparison["session_a"] == "afterhours" and comparison["session_b"] == "RTH"


def test_partial_class_year_is_excluded_instead_of_changing_annual_bundle_cohort():
    metrics, protocol = fixture()
    rows = metrics.to_dict("records")
    trial_id = add_event(protocol, rows, "earnings_call", 2022, 50, suffix="_q2")
    metrics = pd.DataFrame(rows)
    mask = metrics.trial_id.eq(trial_id + "_c4") & metrics.window.eq("post30")
    metrics.loc[mask, ["complete", "value", "reason"]] = [False, np.nan, "low_coverage"]
    result = summarize_multi_events(metrics, protocol)
    comparison = one(result["class_comparisons"], analysis_scope="historical_primary")
    assert comparison["status"] == "descriptive_incomplete_paired_years"
    assert comparison["n_paired_years_expected"] == 2 and comparison["n_paired_years"] == 1
    assert comparison["years"] == [2023]
    assert comparison["median_difference_a_minus_b"] == -12
    assert comparison["excluded_years"] == [{"year": 2022,
                                              "unsupported_bundles_a": [trial_id], "unsupported_bundles_b": []}]


def test_multiple_same_bundle_anchors_do_not_gain_independent_weight():
    metrics, protocol = fixture()
    rows = metrics.to_dict("records")
    add_event(protocol, rows, "wwdc", 2022, 110, suffix="_second_anchor", bundle="wwdc_2022")
    result = summarize_multi_events(pd.DataFrame(rows), protocol)
    summary = one(result["class_summary"], event_type="wwdc", analysis_scope="historical_primary")
    assert summary["n_events"] == 3 and summary["n_bundles"] == summary["n_years"] == 2
    # Bundle2022 median=(7+107)/2=57, bundle2023=17; equal bundle median=37.
    assert summary["median_excess_change"] == 37


def test_one_bad_anchor_does_not_publish_partially_supported_bundle():
    metrics, protocol = fixture()
    rows = metrics.to_dict("records")
    trial_id = add_event(protocol, rows, "wwdc", 2022, 110, suffix="_second_anchor", bundle="wwdc_2022")
    metrics = pd.DataFrame(rows)
    mask = metrics.trial_id.eq(trial_id) & metrics.window.eq("post30")
    metrics.loc[mask, ["complete", "value", "reason"]] = [False, np.nan, "low_coverage"]
    result = summarize_multi_events(metrics, protocol)
    summary = one(result["class_summary"], event_type="wwdc", analysis_scope="historical_primary")
    assert summary["n_events"] == 2 and summary["n_events_expected"] == 3
    assert summary["n_bundles"] == 1 and summary["n_bundles_expected"] == 2
    assert summary["median_excess_change"] == 17


def test_freshness_policies_stay_separate_and_zero_baseline_has_no_infinity():
    metrics, protocol = fixture()
    slow = metrics.copy()
    slow["age_ms"] = 1000
    slow.loc[slow.role.eq("event") & slow.window.eq("post30"), "value"] += 10
    protocol["metric_dimensions"].append({**DIM, "age_ms": 1000})
    combined = pd.concat([metrics, slow], ignore_index=True)
    combined["metric"] = "shock_rate"
    combined.loc[combined.window.eq("pre30"), "value"] = 0
    for row in protocol["metric_dimensions"]:
        row["metric"] = "shock_rate"
    result = summarize_multi_events(combined, protocol)
    a = one(result["event_excess"], trial_id="wwdc_2022", age_ms=250)
    b = one(result["event_excess"], trial_id="wwdc_2022", age_ms=1000)
    assert b["excess_change"] - a["excess_change"] == 10
    assert a["event_relative_change_percent"] is None
    json.dumps(result, allow_nan=False)


def test_frozen_registry_duplicate_and_metadata_tampering_rejected():
    metrics, protocol = fixture()
    with pytest.raises(ValueError, match="Duplicate trial"):
        summarize_multi_events(pd.concat([metrics, metrics.iloc[:1]]), protocol)
    bad = metrics.copy()
    bad.loc[0, "session"] = "premarket"
    with pytest.raises(ValueError, match="metadata differs"):
        summarize_multi_events(bad, protocol)
    malformed = copy.deepcopy(protocol)
    malformed["trials"] = [row for row in malformed["trials"] if row["trial_id"] != "wwdc_2022_c4"]
    with pytest.raises(ValueError, match="exactly five"):
        summarize_multi_events(metrics, malformed)


def test_boolean_and_finite_value_validation_cannot_publish_nan():
    metrics, protocol = fixture()
    bad = metrics.copy()
    bad["complete"] = "False"
    with pytest.raises(ValueError, match="must be boolean"):
        summarize_multi_events(bad, protocol)
    metrics.loc[0, "value"] = np.nan
    with pytest.raises(ValueError, match="no finite value"):
        summarize_multi_events(metrics, protocol)


def test_csv_empty_reasons_remain_valid_and_discovery_applies_to_whole_bundle(tmp_path):
    metrics, protocol = fixture()
    rows = metrics.to_dict("records")
    add_event(protocol, rows, "wwdc", 2026, 200, suffix="_coannouncement", bundle="wwdc_2026")
    path = tmp_path / "metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    result = summarize_multi_events(pd.read_csv(path), protocol)
    historical = one(result["class_summary"], event_type="wwdc", analysis_scope="historical_primary")
    discovery = one(result["class_summary"], event_type="wwdc", analysis_scope="discovery")
    assert historical["n_events"] == 2 and 2026 not in historical["years"]
    assert discovery["n_events"] == 2 and discovery["n_bundles"] == 1
