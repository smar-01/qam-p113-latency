"""Outcome-independent controls, sessions and source-window design checks."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from colonylab.multi_event_design import (
    build_design,
    draw_controls,
    local_anchor,
    market_session,
    regular_sessions,
)


BASE_CONFIG = Path(__file__).resolve().parents[1] / "aapl_multi_event_protocol.json"


def config(**overrides):
    return {**json.loads(BASE_CONFIG.read_text(encoding="utf-8")), **overrides}


def event(event_id="wwdc_2024", anchor="2024-06-10T17:00:00Z", *, event_class="wwdc", bundle=None):
    return {"id": event_id, "bundle_id": bundle or event_id,
            "date": pd.Timestamp(anchor).tz_convert("America/New_York").strftime("%Y-%m-%d"),
            "event_time_utc": anchor, "event_class": event_class,
            "analysis_role": "primary_intraday_event", "timestamp_nature": "scheduled_start"}


def inputs(tmp_path, events, **overrides):
    calendar_path = tmp_path / "calendar.json"
    calendar_path.write_text(json.dumps({"events": events}), encoding="utf-8")
    settings = config(years=[2024], event_calendar="calendar.json", **overrides)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(settings), encoding="utf-8")
    return config_path, tmp_path / "out"


def test_exchange_holidays_and_conservative_early_close_blackouts():
    sessions = set(regular_sessions(2024))
    closed = ["2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
              "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25"]
    conservative = ["2024-07-02", "2024-07-03", "2024-11-29", "2024-12-23", "2024-12-24"]
    assert all(pd.Timestamp(day) not in sessions for day in closed + conservative)
    assert pd.Timestamp("2024-06-18") in sessions
    assert pd.Timestamp("2024-06-20") in sessions
    assert all(day.weekday() < 5 for day in sessions)


def test_actual_exchange_observance_differs_from_federal_holidays():
    # NYSE did not close the preceding Friday for Saturday New Year2022,
    # and Juneteenth first became an exchange holiday in2022.
    assert pd.Timestamp("2021-12-31") in regular_sessions(2021)
    assert pd.Timestamp("2021-06-18") in regular_sessions(2021)
    assert pd.Timestamp("2022-06-20") not in regular_sessions(2022)
    assert pd.Timestamp("2023-01-02") not in regular_sessions(2023)
    assert pd.Timestamp("2025-01-09") not in regular_sessions(2025)  # Carter national mourning.


@pytest.mark.parametrize("original,day,expected", [
    ("2024-03-07T22:00:00Z", "2024-03-14", "2024-03-14T21:00:00Z"),
    ("2024-10-31T21:00:00Z", "2024-11-07", "2024-11-07T22:00:00Z"),
])
def test_controls_preserve_new_york_clock_across_dst(original, day, expected):
    actual = local_anchor(day, original)
    assert actual == pd.Timestamp(expected)
    assert actual.tz_convert("America/New_York").hour == 17


@pytest.mark.parametrize("anchor,session", [
    ("2024-06-10T13:29:59Z", "premarket"),
    ("2024-06-10T13:30:00Z", "RTH"),
    ("2024-06-10T19:59:59Z", "RTH"),
    ("2024-06-10T20:00:00Z", "afterhours"),
])
def test_market_session_uses_local_open_and_close_boundaries(anchor, session):
    assert market_session(anchor) == session


def test_draw_is_reproducible_and_does_not_read_market_outcomes_or_event_name():
    source = event()
    calendar = {"events": [source]}
    chosen, audit = draw_controls(source, calendar, config())
    renamed = {**source, "id": "renamed_source_id", "observed_latency": 99_999, "return": -10}
    augmented = {"events": [renamed], "market_data": {"perfect_controls": ["2024-06-10"]}}
    again, second = draw_controls(renamed, augmented, config())
    assert chosen == again
    assert audit["randomized_order"] == second["randomized_order"]
    assert len(chosen) == len(set(chosen)) == 5
    assert audit["selection_uses_market_outcomes"] is False
    assert audit["replacements_after_quality_results"] is False
    assert audit["selected_dates"] == audit["randomized_order"][:5]
    changed, changed_audit = draw_controls(source, calendar, config(control_seed=20261103))
    assert changed_audit["eligible_dates"] == audit["eligible_dates"]
    assert changed_audit["randomized_order"] != audit["randomized_order"]


def test_draw_uses_same_year_weekday_and_preserves_anchor_session():
    source = event(anchor="2024-03-21T21:00:00Z", event_class="earnings_call")
    chosen, audit = draw_controls(source, {"events": [source]}, config())
    for value in audit["eligible_dates"]:
        day = pd.Timestamp(value)
        assert day.year == 2024 and day.weekday() == 3
        assert abs((day - pd.Timestamp(source["date"])).days) <= 56
        anchor = local_anchor(value, source["event_time_utc"])
        assert anchor.tz_convert("America/New_York").strftime("%H:%M") == "17:00"
        assert market_session(anchor) == "afterhours"
    assert set(chosen) <= set(audit["eligible_dates"])


def test_target_and_other_catalogued_announcement_blackouts_are_not_randomly_reintroduced():
    source = event()
    context = {"id": "context_release", "date": "2024-05-21", "event_time_utc": None}
    _, audit = draw_controls(source, {"events": [source, context]}, config())
    exclusions = {row["day"]: row["reason"] for row in audit["excluded_dates"]}
    assert exclusions["2024-06-03"] == exclusions["2024-06-10"] == exclusions["2024-06-17"] == "near_target_event"
    assert exclusions["2024-05-20"] == "near_catalogued_apple_event"
    assert "2024-05-20" not in audit["eligible_dates"]


def test_weekend_catalogued_news_blacks_out_adjacent_trading_day():
    source = event()
    context = {"id": "sunday_release", "date": "2024-05-19", "event_time_utc": None}
    _, audit = draw_controls(source, {"events": [source, context]}, config())
    assert "2024-05-20" not in audit["eligible_dates"]
    assert any(row["day"] == "2024-05-20" and row["reason"] == "near_catalogued_apple_event"
               for row in audit["excluded_dates"])


def test_control_history_bounds_apply_to_entire_source_interval_including_exact_end():
    source = event()
    settings = config(available_start_utc="2024-05-01T00:00:00Z", available_end_utc="2024-07-15T17:36:00Z")
    _, audit = draw_controls(source, {"events": [source]}, settings)
    assert "2024-07-15" in audit["eligible_dates"]
    exclusions = {row["day"]: row["reason"] for row in audit["excluded_dates"]}
    assert exclusions["2024-04-29"] == "before_shared_provider_history"
    assert exclusions["2024-07-22"] == "not_yet_available_at_design_freeze"
    for day in audit["eligible_dates"]:
        anchor = local_anchor(day, source["event_time_utc"])
        assert anchor + pd.Timedelta(minutes=-36) >= pd.Timestamp(settings["available_start_utc"])
        assert anchor + pd.Timedelta(minutes=36) <= pd.Timestamp(settings["available_end_utc"])


def test_fewer_than_five_controls_fails_instead_of_widening_or_reusing_dates():
    source = event()
    with pytest.raises(ValueError, match="Not enough outcome-independent controls"):
        draw_controls(source, {"events": [source]}, config(control_radius_days=14))


def test_release_and_call_share_controls_and_source_bytes_but_keep_distinct_anchor_trials(tmp_path):
    release = event("release", "2024-06-13T20:30:00Z", event_class="earnings_release", bundle="quarter_bundle")
    call = event("call", "2024-06-13T21:00:00Z", event_class="earnings_call", bundle="quarter_bundle")
    config_path, out = inputs(tmp_path, [release, call])
    protocol, requests = build_design(tmp_path, config_path, out)
    trials = protocol["trials"]
    assert len(trials) == 12 and len({row["trial_id"] for row in trials}) == 12
    by_parent = {parent: [row for row in trials if row["control_for"] == parent] for parent in ("release", "call")}
    assert [row["day"] for row in by_parent["release"]] == [row["day"] for row in by_parent["call"]]
    assert all(row["bundle_id"] == "quarter_bundle" for row in trials)
    assert len(protocol["source_groups"]) == 6
    assert len(requests) == 6 * 4
    groups = {row["group_id"]: row for row in protocol["source_groups"]}
    release_trial, call_trial = [next(row for row in trials if row["trial_id"] == name) for name in ("release", "call")]
    assert release_trial["source_group_id"] == call_trial["source_group_id"]
    merged = groups[release_trial["source_group_id"]]
    assert pd.Timestamp(merged["start"]) == pd.Timestamp("2024-06-13T19:54:00Z")
    assert pd.Timestamp(merged["end"]) == pd.Timestamp("2024-06-13T21:36:00Z")
    for trial in trials:
        group = groups[trial["source_group_id"]]
        anchor = pd.Timestamp(trial["anchor_utc"])
        assert trial["trial_id"] in group["trial_ids"]
        assert pd.Timestamp(group["start"]) <= anchor - pd.Timedelta(minutes=36)
        assert pd.Timestamp(group["end"]) >= anchor + pd.Timedelta(minutes=36)
    assert len({json.dumps(row, sort_keys=True) for row in requests}) == len(requests)
    assert json.loads((out / "protocol.json").read_text()) == protocol
    assert all(request["schema"] == "mbp-1" and request["symbols"] == ["AAPL"] for request in requests)


def test_calendar_order_does_not_change_control_draw_or_request_identities(tmp_path):
    a = event("release", "2024-06-13T20:30:00Z", event_class="earnings_release", bundle="quarter_bundle")
    b = event("call", "2024-06-13T21:00:00Z", event_class="earnings_call", bundle="quarter_bundle")
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_config, first_out = inputs(first_dir, [a, b])
    second_config, second_out = inputs(second_dir, [b, a])
    first, first_requests = build_design(first_dir, first_config, first_out)
    second, second_requests = build_design(second_dir, second_config, second_out)
    assert first["trials"] == second["trials"]
    assert first["source_groups"] == second["source_groups"]
    assert first_requests == second_requests


def test_context_date_only_events_do_not_gain_invented_anchors(tmp_path):
    primary = event()
    context = {**event("press_date_only", event_class="product_launch"), "event_time_utc": None,
               "analysis_role": "date_only_context"}
    config_path, out = inputs(tmp_path, [primary, context])
    protocol, _ = build_design(tmp_path, config_path, out)
    assert {row["event_id"] for row in protocol["trials"]} == {primary["id"]}


def test_event_source_interval_outside_available_history_fails_before_acquisition(tmp_path):
    config_path, out = inputs(tmp_path, [event()], available_end_utc="2024-06-10T17:35:59Z")
    with pytest.raises(ValueError, match="outside verified shared provider history"):
        build_design(tmp_path, config_path, out)
    assert not (out / "requests.json").exists()


def test_duplicate_primary_calendar_ids_and_date_anchor_disagreement_are_rejected(tmp_path):
    original = event()
    config_path, out = inputs(tmp_path, [original, copy.deepcopy(original)])
    with pytest.raises(ValueError, match="[Dd]uplicate|[Uu]nique"):
        build_design(tmp_path, config_path, out)
    mismatch = {**original, "date": "2024-06-11"}
    config_path, out = inputs(tmp_path, [mismatch])
    with pytest.raises(ValueError, match="date|[Aa]nchor"):
        build_design(tmp_path, config_path, out)
