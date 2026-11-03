"""Freeze sourced event anchors and reproducibly sampled exposure controls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, Holiday, USMartinLutherKingJr,
    USPresidentsDay, GoodFriday, USMemorialDay, USLaborDay, USThanksgivingDay,
    nearest_workday, sunday_to_monday)

from .aapl_event_trial import dump, file_sha, signature


class EquityHolidays(AbstractHolidayCalendar):
    rules = [Holiday("New Year", month=1, day=1, observance=sunday_to_monday),
             USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
             Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
             Holiday("Independence", month=7, day=4, observance=nearest_workday),
             USLaborDay, USThanksgivingDay,
             Holiday("Christmas", month=12, day=25, observance=nearest_workday),
             Holiday("Carter funeral", year=2025, month=1, day=9)]


def regular_sessions(year):
    """Conservative full-session control pool, with holiday-adjacent exclusions.

    Rules follow NYSE holidays. July 1-5, December 23-26 and Thanksgiving
    Friday are conservatively removed, including potential early closes.
    Provider per-day conditions are checked separately; no outcome is used.
    """
    holidays = set(EquityHolidays().holidays(f"{year}-01-01", f"{year}-12-31"))
    thanksgiving = USThanksgivingDay.dates(f"{year}-01-01", f"{year}-12-31")[0]
    return [day for day in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="B")
            if day not in holidays and day != thanksgiving + pd.Timedelta(days=1)
            and not (day.month == 7 and day.day <= 5)
            and not (day.month == 12 and 23 <= day.day <= 26)]


def local_anchor(day, original_anchor):
    original = pd.Timestamp(original_anchor).tz_convert("America/New_York")
    return pd.Timestamp(day + "T" + original.strftime("%H:%M:%S")).tz_localize(
        "America/New_York", ambiguous="raise", nonexistent="raise").tz_convert("UTC")


def market_session(anchor):
    local = pd.Timestamp(anchor).tz_convert("America/New_York")
    minute = local.hour * 60 + local.minute
    return "premarket" if minute < 570 else "RTH" if minute < 960 else "afterhours"


def draw_controls(event, calendar, config):
    anchor = pd.Timestamp(event["event_time_utc"])
    day = anchor.tz_convert("America/New_York").tz_localize(None).normalize()
    year = day.year
    sessions = regular_sessions(year)
    # Include actual trading weekdays around every catalogued corporate date;
    # the calendar is explicitly not an exhaustive news/macro blackout list.
    known = [pd.Timestamp(row["date"]) for row in calendar["events"] if int(row["date"][:4]) == year]
    eligible, excluded = [], []
    all_weekdays = list(pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="B"))
    weekday_index = {value: n for n, value in enumerate(all_weekdays)}
    known_bounds = []
    for known_day in known:
        position = int(np.searchsorted(all_weekdays, known_day))
        known_bounds.append((position, position) if known_day in weekday_index else (position - 1, position))
    for candidate in all_weekdays:
        if candidate.weekday() != day.weekday() or abs((candidate - day).days) > config["control_radius_days"]:
            continue
        reason = None
        if candidate not in sessions:
            reason = "holiday_or_conservative_short_session_blackout"
        elif abs(weekday_index[candidate] - weekday_index.get(day, -1000)) <= config["event_exclusion_weekdays"]:
            reason = "near_target_event"
        elif any(left - config["other_event_exclusion_weekdays"] <= weekday_index[candidate]
                 <= right + config["other_event_exclusion_weekdays"] for left, right in known_bounds):
            reason = "near_catalogued_apple_event"
        candidate_anchor = local_anchor(candidate.strftime("%Y-%m-%d"), anchor)
        if candidate_anchor + pd.Timedelta(minutes=config["download_start_minute"]) < pd.Timestamp(config.get("available_start_utc", "2018-05-01T00:00:00Z")):
            reason = "before_shared_provider_history"
        if candidate_anchor + pd.Timedelta(minutes=config["download_end_minute"]) > pd.Timestamp(config["available_end_utc"]):
            reason = "not_yet_available_at_design_freeze"
        if reason:
            excluded.append({"day": candidate.strftime("%Y-%m-%d"), "reason": reason})
        else:
            eligible.append(candidate.strftime("%Y-%m-%d"))
    if len(eligible) < config["controls_per_event"]:
        raise ValueError(f"Not enough outcome-independent controls for {event['id']}: {len(eligible)}")
    # Earnings release/call anchors in one bundle receive the same random
    # reference dates when their eligible pools coincide.
    seed_text = str(config["control_seed"]) + ":" + event["bundle_id"]
    seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "big")
    ordering = list(np.random.default_rng(seed).permutation(eligible))
    chosen = [str(value) for value in ordering[:config["controls_per_event"]]]
    return chosen, {"event_id": event["id"], "bundle_id": event["bundle_id"], "seed_sha256": hashlib.sha256(seed_text.encode()).hexdigest(),
                    "eligible_dates": eligible, "randomized_order": [str(value) for value in ordering],
                    "selected_dates": chosen, "excluded_dates": excluded,
                    "selection_uses_market_outcomes": False, "replacements_after_quality_results": False}


def build_design(root, config_path, out_dir, *, reuse_requests=None):
    root, out = Path(root).resolve(), Path(out_dir).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    calendar_path = root / config["event_calendar"]
    calendar = json.loads(calendar_path.read_text(encoding="utf-8"))
    events = [row for row in calendar["events"] if row.get("event_time_utc")
              and int(row["date"][:4]) in config["years"]
              and row["event_class"] in config["event_classes"]
              and row.get("analysis_role") == "primary_intraday_event"]
    if not events:
        raise ValueError("No verified primary event anchors in selected calendar")
    if len({row["id"] for row in events}) != len(events):
        raise ValueError("Duplicate primary calendar event IDs")
    if any(pd.Timestamp(row["event_time_utc"]).tz_convert("America/New_York").strftime("%Y-%m-%d") != row["date"] for row in events):
        raise ValueError("Calendar event date differs from its New York anchor date")
    if any(pd.Timestamp(row["event_time_utc"]) + pd.Timedelta(minutes=config["download_start_minute"]) < pd.Timestamp(config["available_start_utc"])
           or pd.Timestamp(row["event_time_utc"]) + pd.Timedelta(minutes=config["download_end_minute"]) > pd.Timestamp(config["available_end_utc"]) for row in events):
        raise ValueError("An event anchor is outside verified shared provider history")
    trials, draws = [], []
    for event in sorted(events, key=lambda row: (row["event_time_utc"], row["id"])):
        selected, draw = draw_controls(event, calendar, config)
        draws.append(draw)
        anchor = pd.Timestamp(event["event_time_utc"])
        common = {"event_id": event["id"], "bundle_id": event["bundle_id"], "event_type": event["event_class"],
                  "year": int(event["date"][:4]), "session": market_session(anchor),
                  "anchor_nature": event["timestamp_nature"]}
        trials.append({**common, "trial_id": event["id"], "role": "event", "control_for": None,
                       "day": event["date"], "anchor_utc": anchor.isoformat()})
        for n, day in enumerate(selected, 1):
            trials.append({**common, "trial_id": event["id"] + f"__control_{n}", "role": "control",
                           "control_for": event["id"], "day": day,
                           "anchor_utc": local_anchor(day, anchor).isoformat()})
    # Merge overlapping windows before pricing/retrieval. Separate anchors
    # retain separate trials even if their exact same source bytes are reused.
    intervals = []
    for trial in trials:
        anchor = pd.Timestamp(trial["anchor_utc"])
        intervals.append({"day": trial["day"], "start": anchor + pd.Timedelta(minutes=config["download_start_minute"]),
                          "end": anchor + pd.Timedelta(minutes=config["download_end_minute"]), "trial_ids": [trial["trial_id"]]})
    merged = []
    for row in sorted(intervals, key=lambda item: (item["start"], item["end"])):
        if merged and merged[-1]["day"] == row["day"] and row["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], row["end"])
            merged[-1]["trial_ids"].extend(row["trial_ids"])
        else:
            merged.append(row.copy())
    reuse_requests = list(reuse_requests or [])
    groups, requests = [], []
    for row in merged:
        identity = {"day": row["day"], "start": row["start"].isoformat(), "end": row["end"].isoformat()}
        group_id = row["day"] + "_" + signature(identity)[:12]
        group_requests = []
        for venue in config["datasets"]:
            prior = [request for request in reuse_requests if request["dataset"] == venue
                     and request["schema"] == "mbp-1" and request["symbols"] == ["AAPL"]
                     and request["stype_in"] == "raw_symbol" and request["limit"] == config["record_limit_per_request"]
                     and row["start"] <= pd.Timestamp(request["start"]) < pd.Timestamp(request["end"]) <= row["end"]]
            cursor = row["start"]
            for request in sorted(prior, key=lambda item: pd.Timestamp(item["start"])):
                begin, finish = pd.Timestamp(request["start"]), pd.Timestamp(request["end"])
                if begin < cursor:
                    raise ValueError("Reusable source intervals overlap or duplicate")
                if begin > cursor:
                    group_requests.append({"dataset": venue, "schema": "mbp-1", "symbols": ["AAPL"], "stype_in": "raw_symbol",
                                           "start": cursor.isoformat(), "end": begin.isoformat(), "limit": config["record_limit_per_request"]})
                group_requests.append(dict(request))
                cursor = finish
            if cursor < row["end"]:
                group_requests.append({"dataset": venue, "schema": "mbp-1", "symbols": ["AAPL"], "stype_in": "raw_symbol",
                                       "start": cursor.isoformat(), "end": row["end"].isoformat(), "limit": config["record_limit_per_request"]})
        groups.append({**identity, "group_id": group_id, "trial_ids": sorted(row["trial_ids"]),
                       "request_sha256": [signature(request) for request in group_requests]})
        requests.extend(group_requests)
        for trial in trials:
            if trial["trial_id"] in row["trial_ids"]:
                trial["source_group_id"] = group_id
    protocol = {**config, "trials": trials, "source_groups": groups, "requests": requests,
                "reuse_request_grid": reuse_requests,
                "calendar_sha256": file_sha(calendar_path), "base_config_sha256": file_sha(config_path),
                "discovery_event_ids": [row["id"] for row in events if row["event_class"] == "wwdc" and row["date"] == "2026-06-08"],
                "metric_dimensions": [{"clock": clock, "age_ms": age, "horizon_ms": h, "metric": metric}
                                      for clock in config["clocks"] for age in config["quote_age_ms"]
                                      for h in config["horizons_ms"] for metric in ("latency_us", "match_rate", "shock_rate")]}
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "protocol.json", protocol)
    dump(out / "control_draws.json", draws)
    dump(out / "requests.json", requests)
    dump(out / "event_calendar.json", calendar)
    pd.DataFrame(trials).to_csv(out / "trials.csv", index=False)
    return protocol, requests
