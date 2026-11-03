"""Event/date, fixed-pair and exposure-denominator checks for the AAPL trial."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from colonylab.event_trial_stats import summarize_trials


EVENT = "2026-06-08"
REFS = ["2026-06-01", "2026-06-15", "2026-06-22", "2026-06-29", "2026-07-06", "2026-07-13"]


def fixture():
    protocol = {
        "event_day": EVENT, "reference_days": REFS.copy(), "primary_reference_days": REFS.copy(),
        "primary_venues": ["A", "B"], "all_venues": ["A", "B"],
        "primary_horizon_ms": 10, "primary_shock_threshold_bp": 1,
        "minimum_matched_events": 20, "minimum_coverage_fraction": .9,
        "windows": [{"name": "pre30", "start_minute": -30, "end_minute": 0},
                    {"name": "post30", "start_minute": 0, "end_minute": 30}],
    }
    pairs, venues = [], []
    for day in sorted([EVENT, *REFS]):
        change = 10 if day == EVENT else REFS.index(day) + 1
        for window in ("pre30", "post30"):
            post = window == "post30"
            for leader, follower, median in (("A", "B", 10.), ("B", "A", 20.)):
                pairs.append({"day": day, "window": window, "horizon_ms": 10,
                              "leader": leader, "follower": follower, "n_events": 1_000_000,
                              "n_eligible": 100, "n_matched": 40, "n_censored": 60,
                              "n_quality_truncated": 0, "n_right_truncated": 0,
                              "median_us": median + change * post})
            for venue in ("A", "B"):
                raw = 180 + 30 * min(change, 6) * post
                venues.append({"day": day, "window": window, "venue": venue,
                               "shock_threshold_bp": 1., "observed_seconds": 1800.,
                               "midpoint_changes": 1000, "shocks_raw": raw,
                               "shocks_clustered": raw // 2, "shocks_per_minute": raw / 30,
                               "signed_return_bp": 1., "total_abs_return_bp": 10.,
                               "spread_median_bps": 2., "depth_median": 100., "coverage_fraction": 1.})
    return pd.DataFrame(pairs), pd.DataFrame(venues), protocol


def row(table, **identity):
    found = [r for r in table if all(r.get(key) == value for key, value in identity.items())]
    assert len(found) == 1, (identity, len(found))
    return found[0]


def primary_contrast(result, **extra):
    return row(result["contrasts"], kind="latency", universe="common5", horizon_ms=10,
               metric="mean_pair_median_us", contrast="primary30", reference_pool="primary_weekday", **extra)


def test_single_event_reference_unit_is_six_dates_not_millions_of_quotes():
    pairs, venues, protocol = fixture()
    result = summarize_trials(pairs, venues, protocol)
    estimate = primary_contrast(result)
    assert estimate["event_count"] == 1
    assert estimate["reference_n_expected"] == estimate["reference_n_valid"] == 6
    assert estimate["rank_population_size"] == 7
    assert estimate["event_pre_value"] == 15
    assert estimate["event_post_value"] == 25
    assert estimate["event_change"] == 10
    assert estimate["reference_change_median"] == 3.5
    assert estimate["event_minus_reference_median"] == 6.5
    assert estimate["reference_change_q25"] == 2.25
    assert estimate["reference_change_q75"] == 4.75
    assert estimate["descriptive_event_rank_low_ascending"] == 7
    assert estimate["descriptive_event_rank_high_ascending"] == 7
    assert "p_value" not in estimate and "ci95" not in estimate
    json.dumps(result, allow_nan=False)


def test_equal_pair_mean_is_not_event_count_weighted():
    pairs, venues, protocol = fixture()
    mask = pairs.day.eq(EVENT) & pairs.window.eq("pre30") & pairs.leader.eq("B")
    pairs.loc[mask, ["median_us", "n_eligible", "n_matched", "n_censored"]] = [30, 500, 400, 100]
    result = summarize_trials(pairs, venues, protocol)
    estimate = row(result["latency_daily"], universe="common5", day=EVENT, window="pre30", horizon_ms=10)
    assert estimate["mean_pair_median_us"] == 20
    assert estimate["eligible_pair_events"] == 600
    assert estimate["matched_pair_events"] == 440
    assert estimate["match_rate"] == pytest.approx(440 / 600)


def test_one_sparse_pair_suppresses_whole_fixed_pair_primary_metric():
    pairs, venues, protocol = fixture()
    mask = pairs.day.eq(EVENT) & pairs.window.eq("post30") & pairs.leader.eq("B")
    pairs.loc[mask, ["n_matched", "n_censored"]] = [19, 81]
    result = summarize_trials(pairs, venues, protocol)
    point = row(result["latency_daily"], universe="common5", day=EVENT, window="post30", horizon_ms=10)
    assert not point["complete"] and point["supported_pairs"] == 1
    assert point["mean_pair_median_us"] is None
    assert "insufficient_pair_matches:1" in point["reason"]
    assert primary_contrast(result)["status"] == "event_insufficient_coverage"


def test_missing_pair_is_not_averaged_out_of_the_universe():
    pairs, venues, protocol = fixture()
    mask = pairs.day.eq(EVENT) & pairs.window.eq("post30") & pairs.leader.eq("B")
    result = summarize_trials(pairs.loc[~mask], venues, protocol)
    point = row(result["latency_daily"], universe="common5", day=EVENT, window="post30", horizon_ms=10)
    assert point["expected_pairs"] == 2 and point["present_pairs"] == 1
    assert point["mean_pair_median_us"] is None
    assert "missing_pair_rows:1" in point["reason"]


def test_low_observation_coverage_cannot_enter_supported_latency():
    pairs, venues, protocol = fixture()
    mask = venues.day.eq(EVENT) & venues.window.eq("post30") & venues.venue.eq("A")
    venues.loc[mask, "observed_seconds"] = 1600
    venues.loc[mask, "coverage_fraction"] = 1600 / 1800
    venues.loc[mask, "shocks_per_minute"] = venues.loc[mask, "shocks_raw"] * 60 / 1600
    result = summarize_trials(pairs, venues, protocol)
    assert primary_contrast(result)["status"] == "event_insufficient_coverage"
    point = row(result["latency_daily"], universe="common5", day=EVENT, window="post30", horizon_ms=10)
    assert "insufficient_venue_coverage:A" in point["reason"]


def test_missing_primary_reference_stays_visible_and_suppresses_primary_rank():
    pairs, venues, protocol = fixture()
    mask = pairs.day.eq(REFS[-1]) & pairs.window.eq("post30") & pairs.leader.eq("B")
    result = summarize_trials(pairs.loc[~mask], venues, protocol)
    estimate = primary_contrast(result)
    assert estimate["reference_n_expected"] == 6 and estimate["reference_n_valid"] == 5
    assert estimate["status"] == "insufficient_reference_coverage"
    assert estimate["reference_change_median"] is None
    assert estimate["descriptive_event_rank_low_ascending"] is None
    assert estimate["excluded_reference_days"][0]["day"] == REFS[-1]
    secondary = row(result["contrasts"], kind="latency", universe="common5", horizon_ms=10,
                    metric="mean_pair_median_us", contrast="primary30", reference_pool="all_reference_days")
    assert secondary["status"] == "descriptive_incomplete_reference_coverage"
    assert secondary["reference_change_median"] == 3


def test_shock_exposure_pooling_and_equal_venue_weights_remain_distinct():
    pairs, venues, protocol = fixture()
    a = venues.day.eq(EVENT) & venues.window.eq("pre30") & venues.venue.eq("A")
    b = venues.day.eq(EVENT) & venues.window.eq("pre30") & venues.venue.eq("B")
    venues.loc[a, ["shocks_raw", "shocks_clustered", "shocks_per_minute"]] = [300, 150, 10]
    venues.loc[b, ["observed_seconds", "coverage_fraction", "shocks_raw", "shocks_clustered", "shocks_per_minute"]] = [1620, .9, 54, 27, 2]
    result = summarize_trials(pairs, venues, protocol)
    estimate = row(result["shock_daily"], universe="common5", day=EVENT, window="pre30", shock_threshold_bp=1)
    assert estimate["complete"]
    assert estimate["mean_venue_shocks_per_minute"] == 6
    assert estimate["pooled_shocks_per_venue_minute"] == pytest.approx(354 * 60 / 3420)
    assert estimate["observed_venue_seconds"] == 3420


def test_shock_sensitivities_do_not_enter_primary_threshold_counts():
    pairs, venues, protocol = fixture()
    sensitivity = venues.copy()
    sensitivity["shock_threshold_bp"] = .5
    sensitivity["shocks_raw"] *= 2
    sensitivity["shocks_clustered"] *= 2
    sensitivity["midpoint_changes"] *= 2
    sensitivity["shocks_per_minute"] *= 2
    result = summarize_trials(pairs, pd.concat([venues, sensitivity], ignore_index=True), protocol)
    primary = row(result["shock_daily"], universe="common5", day=EVENT, window="pre30", shock_threshold_bp=1)
    alternate = row(result["shock_daily"], universe="common5", day=EVENT, window="pre30", shock_threshold_bp=.5)
    assert primary["shocks_raw"] == 360
    assert alternate["shocks_raw"] == 720


def test_leadership_counts_conserve_and_shares_are_not_normalized_to_sum_one():
    pairs, venues, protocol = fixture()
    mask = pairs.day.eq(EVENT) & pairs.window.eq("pre30") & pairs.leader.eq("A")
    pairs.loc[mask, ["n_matched", "n_censored"]] = [80, 20]
    result = summarize_trials(pairs, venues, protocol)
    a = row(result["leadership_daily"], universe="common5", day=EVENT, window="pre30", horizon_ms=10, venue="A")
    b = row(result["leadership_daily"], universe="common5", day=EVENT, window="pre30", horizon_ms=10, venue="B")
    assert a["matched_outgoing"] == b["matched_incoming"] == 80
    assert b["matched_outgoing"] == a["matched_incoming"] == 40
    assert a["leadership_share"] == pytest.approx(2 / 3)
    assert b["leadership_share"] == pytest.approx(1 / 3)


def test_zero_shock_baseline_does_not_fabricate_infinite_percent_change():
    pairs, venues, protocol = fixture()
    mask = venues.day.eq(EVENT) & venues.window.eq("pre30")
    venues.loc[mask, ["shocks_raw", "shocks_clustered", "shocks_per_minute"]] = 0
    result = summarize_trials(pairs, venues, protocol)
    estimate = row(result["daily_changes"], kind="shock_intensity", universe="common5", shock_threshold_bp=1,
                   contrast="primary30", day=EVENT, metric="mean_venue_shocks_per_minute")
    assert estimate["complete"] and estimate["change"] == 12
    assert estimate["relative_change_percent"] is None
    json.dumps(result, allow_nan=False)


def test_pair_count_or_venue_rate_tampering_is_rejected():
    pairs, venues, protocol = fixture()
    bad = pairs.copy()
    bad.loc[0, "n_censored"] += 1
    with pytest.raises(ValueError, match="do not reconcile"):
        summarize_trials(bad, venues, protocol)
    bad_venue = venues.copy()
    bad_venue.loc[0, "shocks_per_minute"] += 1
    with pytest.raises(ValueError, match="normalized by observed time"):
        summarize_trials(pairs, bad_venue, protocol)


def test_duplicate_pair_or_event_in_reference_calendar_is_rejected():
    pairs, venues, protocol = fixture()
    with pytest.raises(ValueError, match="Duplicate pair"):
        summarize_trials(pd.concat([pairs, pairs.iloc[:1]]), venues, protocol)
    bad = copy.deepcopy(protocol)
    bad["reference_days"] = sorted([*REFS, EVENT])
    with pytest.raises(ValueError, match="disjoint"):
        summarize_trials(pairs, venues, bad)


def test_window_boundaries_determine_rates_not_raw_event_counts():
    pairs, venues, protocol = fixture()
    protocol["windows"][0]["start_minute"] = -5
    pre = venues.window.eq("pre30")
    venues.loc[pre, "observed_seconds"] = 300
    venues.loc[pre, "shocks_per_minute"] *= 6
    result = summarize_trials(pairs, venues, protocol)
    point = row(result["shock_daily"], universe="common5", day=EVENT, window="pre30", shock_threshold_bp=1)
    assert point["mean_venue_shocks_per_minute"] == 36
    assert point["shocks_raw"] == 360
    assert point["observed_venue_seconds"] == 600


def test_reference_ties_report_rank_interval_without_a_p_value():
    pairs, venues, protocol = fixture()
    post = pairs.window.eq("post30")
    for leader, pre in (("A", 10), ("B", 20)):
        pairs.loc[post & pairs.leader.eq(leader), "median_us"] = pre + 2
    result = summarize_trials(pairs, venues, protocol)
    estimate = primary_contrast(result)
    assert estimate["reference_changes_equal_event"] == 6
    assert estimate["descriptive_event_rank_low_ascending"] == 1
    assert estimate["descriptive_event_rank_high_ascending"] == 7
    assert estimate["event_minus_reference_median"] == 0
