"""Date-level descriptive contrasts for the saved-data AAPL event trial.

A corporate event, rather than a quote, is the observational unit.  This
module never turns the many within-event observations into an event-class
confidence interval.  The single event is compared with preselected same-hour
reference dates, keeping a fixed venue/pair universe and explicit exclusions.
"""
from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
import pandas as pd


PAIR_REQUIRED = {
    "day", "window", "horizon_ms", "leader", "follower", "n_events",
    "n_eligible", "n_matched", "n_censored", "n_quality_truncated",
    "n_right_truncated", "median_us",
}
VENUE_REQUIRED = {
    "day", "window", "venue", "shock_threshold_bp", "observed_seconds",
    "midpoint_changes", "shocks_raw", "shocks_clustered", "shocks_per_minute",
    "coverage_fraction",
}
DEFAULT_CONTRASTS = (
    ("primary30", "pre30", "post30"),
    ("immediate5", "pre5", "post5"),
    ("delayed25", "pre25", "post25"),
    ("anchor_minus5", "shift_minus5_pre", "shift_minus5_post"),
    ("anchor_plus5", "shift_plus5_pre", "shift_plus5_post"),
)


def _require(condition, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _number(value):
    return float(value) if value is not None and np.isfinite(value) else None


def _days(values: Sequence[str], name: str) -> list[str]:
    result = list(values)
    _require(result and result == sorted(set(result)), f"{name} must be unique chronological dates")
    for day in result:
        parsed = pd.Timestamp(day)
        _require(isinstance(day, str) and parsed.strftime("%Y-%m-%d") == day
                 and parsed.weekday() < 5, f"Invalid exchange weekday in {name}")
    return result


def _settings(protocol: dict) -> dict:
    event_day = protocol["event_day"]
    _days([event_day], "event_day")
    references = _days(protocol["reference_days"], "reference_days")
    primary_refs = _days(protocol["primary_reference_days"], "primary_reference_days")
    _require(event_day not in references and set(primary_refs) <= set(references),
             "Event day must be disjoint from all references; primary references must be a subset")
    primary = list(protocol["primary_venues"])
    all_venues = list(protocol["all_venues"])
    _require(len(primary) >= 2 and len(set(primary)) == len(primary)
             and len(set(all_venues)) == len(all_venues) and set(primary) <= set(all_venues),
             "Venue universes must be unique, with primary a subset of all venues")
    windows = {row["name"]: row for row in protocol["windows"]}
    _require(len(windows) == len(protocol["windows"]) and windows, "Window names must be unique")
    for window in windows.values():
        _require(np.isfinite(window["start_minute"]) and np.isfinite(window["end_minute"])
                 and window["start_minute"] < window["end_minute"], "Window bounds must increase")
    supplied = protocol.get("contrasts")
    contrasts = ([(row["name"], row["pre"], row["post"]) for row in supplied]
                 if supplied is not None else [row for row in DEFAULT_CONTRASTS
                                               if row[1] in windows and row[2] in windows])
    _require(contrasts and len({row[0] for row in contrasts}) == len(contrasts),
             "At least one uniquely named pre/post contrast is required")
    for _, pre, post in contrasts:
        _require(pre in windows and post in windows, "Contrast uses an unknown window")
    minimum = protocol.get("minimum_matched_events", 20)
    coverage = protocol.get("minimum_coverage_fraction", .9)
    horizon = protocol.get("primary_horizon_ms", 10)
    threshold = protocol.get("primary_shock_threshold_bp", 1)
    _require(type(minimum) is int and minimum > 0, "minimum_matched_events must be a positive integer")
    _require(np.isfinite(coverage) and 0 <= coverage <= 1, "Invalid minimum coverage fraction")
    _require(np.isfinite(horizon) and horizon > 0 and np.isfinite(threshold) and threshold > 0,
             "Primary horizon and shock threshold must be positive")
    return {"event_day": event_day, "references": references, "primary_refs": primary_refs,
            "days": sorted([event_day, *references]), "universes": {"common5": primary, "all8": all_venues},
            "windows": windows, "contrasts": contrasts, "minimum": minimum, "coverage": coverage,
            "horizon": float(horizon), "threshold": float(threshold)}


def _validate_pair(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    _require(PAIR_REQUIRED <= set(frame), "Pair summaries lack required fields")
    result = frame.copy()
    identity = ["day", "window", "horizon_ms", "leader", "follower"]
    _require(not result.duplicated(identity).any(), "Duplicate pair/window summaries")
    _require(set(result.day) <= set(cfg["days"]) and set(result.window) <= set(cfg["windows"]),
             "Pair summaries contain unplanned dates/windows")
    venues = set(cfg["universes"]["all8"])
    _require(set(result.leader) <= venues and set(result.follower) <= venues
             and not result.leader.eq(result.follower).any(), "Pair summaries contain invalid venues/self-pairs")
    counts = ["n_events", "n_eligible", "n_matched", "n_censored", "n_quality_truncated", "n_right_truncated"]
    values = result[counts].to_numpy(float)
    _require(np.isfinite(values).all() and (values >= 0).all() and (values == np.floor(values)).all(),
             "Pair counts must be finite nonnegative integers")
    _require((result.n_matched + result.n_censored == result.n_eligible).all()
             and (result.n_eligible <= result.n_events).all(), "Pair matched/censored counts do not reconcile")
    horizons = result.horizon_ms.to_numpy(float)
    medians = result.median_us.to_numpy(float)
    have = result.n_matched.to_numpy() > 0
    _require(np.isfinite(horizons).all() and (horizons > 0).all(), "Pair horizons must be positive")
    _require(np.isfinite(medians[have]).all() and np.isnan(medians[~have]).all()
             and (medians[have] > 0).all() and (medians[have] <= horizons[have] * 1000 + 1e-8).all(),
             "Pair medians do not agree with matched support/horizon")
    return result


def _validate_venue(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    _require(VENUE_REQUIRED <= set(frame), "Venue summaries lack required fields")
    result = frame.copy()
    _require(not result.duplicated(["day", "window", "venue", "shock_threshold_bp"]).any(),
             "Duplicate venue/window/shock-threshold summaries")
    _require(set(result.day) <= set(cfg["days"]) and set(result.window) <= set(cfg["windows"])
             and set(result.venue) <= set(cfg["universes"]["all8"]), "Venue summaries contain unplanned identities")
    count_columns = ["midpoint_changes", "shocks_raw", "shocks_clustered"]
    counts = result[count_columns].to_numpy(float)
    _require(np.isfinite(counts).all() and (counts >= 0).all() and (counts == np.floor(counts)).all()
             and (result.shocks_clustered <= result.shocks_raw).all()
             and (result.shocks_raw <= result.midpoint_changes).all(), "Venue shock counts do not reconcile")
    duration = result.window.map(lambda name: 60 * (cfg["windows"][name]["end_minute"]
                                                   - cfg["windows"][name]["start_minute"])).to_numpy(float)
    observed = result.observed_seconds.to_numpy(float)
    coverage = result.coverage_fraction.to_numpy(float)
    _require(np.isfinite(observed).all() and (observed >= 0).all()
             and (observed <= duration + 1e-6).all()
             and np.isfinite(coverage).all() and (coverage >= 0).all() and (coverage <= 1 + 1e-9).all()
             and np.allclose(coverage, observed / duration, rtol=1e-8, atol=1e-8),
             "Venue observation duration/coverage do not reconcile")
    threshold = result.shock_threshold_bp.to_numpy(float)
    _require(np.isfinite(threshold).all() and (threshold > 0).all(), "Shock thresholds must be positive")
    expected = np.divide(result.shocks_raw.to_numpy(float) * 60, observed,
                         out=np.full(len(result), np.nan), where=observed > 0)
    actual = result.shocks_per_minute.to_numpy(float)
    _require(np.allclose(actual, expected, equal_nan=True, atol=1e-8, rtol=1e-8),
             "Venue shock rate is not normalized by observed time")
    _require((result.loc[observed == 0, "shocks_raw"] == 0).all(), "Shocks cannot occur with zero observed time")
    return result


def _coverage(venue: pd.DataFrame, cfg: dict, day: str, window: str, venues: list[str]) -> tuple[bool, list[str], float]:
    selected = venue.loc[venue.day.eq(day) & venue.window.eq(window)
                         & venue.venue.isin(venues) & venue.shock_threshold_bp.eq(cfg["threshold"])]
    reasons = []
    missing = sorted(set(venues) - set(selected.venue))
    if missing:
        reasons.append("missing_venue_observation:" + ",".join(missing))
    low = selected.loc[selected.coverage_fraction.lt(cfg["coverage"]), "venue"].tolist()
    if low:
        reasons.append("insufficient_venue_coverage:" + ",".join(sorted(low)))
    minimum = float(selected.coverage_fraction.min()) if len(selected) else 0.
    return not reasons, reasons, minimum


def _latency_tables(pair: pd.DataFrame, venue: pd.DataFrame, cfg: dict) -> tuple[list[dict], list[dict]]:
    rows, leadership = [], []
    horizons = sorted(set(pair.horizon_ms.astype(float)) | {cfg["horizon"]})
    for universe, venues in cfg["universes"].items():
        expected = {(a, b) for a in venues for b in venues if a != b}
        for day in cfg["days"]:
            for window in cfg["windows"]:
                observable, coverage_reasons, minimum_coverage = _coverage(venue, cfg, day, window, venues)
                for horizon in horizons:
                    selected = pair.loc[pair.day.eq(day) & pair.window.eq(window) & pair.horizon_ms.eq(horizon)
                                        & pair.leader.isin(venues) & pair.follower.isin(venues)]
                    present = set(selected[["leader", "follower"]].itertuples(index=False, name=None))
                    enough = selected.n_matched.ge(cfg["minimum"]) & np.isfinite(selected.median_us)
                    reasons = list(coverage_reasons)
                    missing = expected - present
                    if missing:
                        reasons.append("missing_pair_rows:" + str(len(missing)))
                    if not enough.all():
                        reasons.append("insufficient_pair_matches:" + str(int((~enough).sum())))
                    complete = observable and not missing and bool(enough.all())
                    eligible, matched = int(selected.n_eligible.sum()), int(selected.n_matched.sum())
                    counts = {"pair_events": int(selected.n_events.sum()), "eligible_pair_events": eligible,
                              "matched_pair_events": matched, "censored_pair_events": int(selected.n_censored.sum()),
                              "quality_truncated_pair_events": int(selected.n_quality_truncated.sum()),
                              "right_truncated_pair_events": int(selected.n_right_truncated.sum())}
                    base = {"universe": universe, "day": day, "window": window, "horizon_ms": horizon,
                            "complete": complete, "reason": ";".join(reasons), "expected_pairs": len(expected),
                            "present_pairs": len(present), "supported_pairs": int(enough.sum()),
                            "minimum_venue_coverage": minimum_coverage, **counts}
                    rows.append({**base,
                        "mean_pair_median_us": float(selected.median_us.mean()) if complete else None,
                        "match_rate": matched / eligible if complete and eligible else None,
                        "available_pair_match_rate": matched / eligible if eligible else None,
                        "available_pair_mean_median_us": float(selected.loc[enough, "median_us"].mean()) if enough.any() else None,
                        "estimand": "Equal-pair mean of conditional pair medians on a fixed fully supported venue universe"})
                    if horizon == cfg["horizon"]:
                        outgoing = selected.groupby("leader").n_matched.sum()
                        incoming = selected.groupby("follower").n_matched.sum()
                        _require(int(outgoing.sum()) == matched == int(incoming.sum()), "Leadership match counts do not conserve")
                        for name in venues:
                            led, followed = int(outgoing.get(name, 0)), int(incoming.get(name, 0))
                            total = led + followed
                            leadership.append({**base, "venue": name,
                                "matched_outgoing": led, "matched_incoming": followed,
                                "leadership_share": led / total if complete and total else None,
                                "measure": "Outgoing matched counts / outgoing plus incoming; not Hasbrouck information share or causal leadership"})
    return rows, leadership


def _shock_tables(venue: pd.DataFrame, cfg: dict) -> list[dict]:
    rows = []
    thresholds = sorted(set(venue.shock_threshold_bp.astype(float)) | {cfg["threshold"]})
    for universe, venues in cfg["universes"].items():
        for day in cfg["days"]:
            for window in cfg["windows"]:
                for threshold in thresholds:
                    selected = venue.loc[venue.day.eq(day) & venue.window.eq(window)
                                         & venue.venue.isin(venues) & venue.shock_threshold_bp.eq(threshold)]
                    missing = sorted(set(venues) - set(selected.venue))
                    low = sorted(selected.loc[selected.coverage_fraction.lt(cfg["coverage"]), "venue"])
                    reasons = []
                    if missing:
                        reasons.append("missing_venue_rows:" + ",".join(missing))
                    if low:
                        reasons.append("insufficient_venue_coverage:" + ",".join(low))
                    complete = not reasons
                    observed = float(selected.observed_seconds.sum())
                    shocks = int(selected.shocks_raw.sum())
                    clustered = int(selected.shocks_clustered.sum())
                    clustered_rates = selected.shocks_clustered.to_numpy(float) * 60 / np.maximum(selected.observed_seconds.to_numpy(float), 1e-300)
                    row = {"universe": universe, "day": day, "window": window, "shock_threshold_bp": threshold,
                           "complete": complete, "reason": ";".join(reasons), "expected_venues": len(venues),
                           "present_venues": len(selected), "observed_venue_seconds": observed,
                           "midpoint_changes": int(selected.midpoint_changes.sum()), "shocks_raw": shocks,
                           "shocks_clustered": clustered,
                           "mean_venue_shocks_per_minute": float(selected.shocks_per_minute.mean()) if complete else None,
                           "pooled_shocks_per_venue_minute": shocks * 60 / observed if complete and observed else None,
                           "mean_venue_clustered_shocks_per_minute": float(clustered_rates.mean()) if complete else None,
                           "pooled_clustered_shocks_per_venue_minute": clustered * 60 / observed if complete and observed else None,
                           "minimum_venue_coverage": float(selected.coverage_fraction.min()) if len(selected) else 0.}
                    for source, target, operation in (
                        ("signed_return_bp", "median_venue_signed_return_bp", "median"),
                        ("total_abs_return_bp", "mean_venue_total_abs_return_bp", "mean"),
                        ("spread_median_bps", "median_venue_spread_bps", "median"),
                        ("depth_median", "median_venue_depth", "median")):
                        row[target] = (_number(getattr(selected[source], operation)())
                                       if complete and source in selected else None)
                    rows.append(row)
    return rows


def _changes(rows: list[dict], cfg: dict, dimensions: list[str], metrics: list[str], kind: str) -> list[dict]:
    frame = pd.DataFrame(rows)
    records = []
    for key, group in frame.groupby(dimensions, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        indexed = group.set_index(["day", "window"])
        for name, pre, post in cfg["contrasts"]:
            for day in cfg["days"]:
                left, right = indexed.loc[(day, pre)], indexed.loc[(day, post)]
                supported = bool(left.complete and right.complete)
                reason = ";".join(x for x in ("pre:" + left.reason if not left.complete else "",
                                                "post:" + right.reason if not right.complete else "") if x)
                for metric in metrics:
                    before, after = left[metric], right[metric]
                    valid = supported and before is not None and after is not None and np.isfinite(before) and np.isfinite(after)
                    difference = float(after - before) if valid else None
                    relative = 100 * difference / before if valid and before > 0 else None
                    records.append({"kind": kind, **dict(zip(dimensions, key)), "contrast": name,
                                    "day": day, "pre_window": pre, "post_window": post, "metric": metric,
                                    "complete": bool(valid), "reason": reason if supported is False else "" if valid else "undefined_metric",
                                    "pre_value": _number(before) if valid else None,
                                    "post_value": _number(after) if valid else None,
                                    "change": difference, "relative_change_percent": _number(relative)})
    return records


def _reference_summaries(changes: list[dict], cfg: dict) -> list[dict]:
    frame = pd.DataFrame(changes)
    nonkeys = {"day", "complete", "reason", "pre_value", "post_value", "change", "relative_change_percent",
               "pre_window", "post_window"}
    keys = [name for name in frame.columns if name not in nonkeys]
    records = []
    for key, group in frame.groupby(keys, sort=True, dropna=False):
        event = group.loc[group.day.eq(cfg["event_day"])].iloc[0]
        for pool, dates in (("primary_weekday", cfg["primary_refs"]), ("all_reference_days", cfg["references"])):
            reference = group.loc[group.day.isin(dates)].copy()
            valid = reference.complete.to_numpy(bool) & np.isfinite(reference.change.to_numpy(float))
            values = reference.loc[valid, "change"].to_numpy(float)
            excluded = reference.loc[~valid, ["day", "reason"]].to_dict("records")
            all_present = len(reference) == len(dates)
            complete_refs = all_present and len(values) == len(dates)
            event_valid = bool(event.complete and pd.notna(event.change))
            status = ("event_insufficient_coverage" if not event_valid else
                      "insufficient_reference_coverage" if pool == "primary_weekday" and not complete_refs else
                      "descriptive_incomplete_reference_coverage" if not complete_refs else "descriptive_complete")
            publish = event_valid and len(values) > 0 and (complete_refs or pool != "primary_weekday")
            event_change = float(event.change) if event_valid else None
            median = float(np.median(values)) if publish else None
            below = int(np.sum(values < event_change)) if publish else None
            equal = int(np.sum(values == event_change)) if publish else None
            relative = reference.loc[valid, "relative_change_percent"].to_numpy(float)
            relative = relative[np.isfinite(relative)]
            records.append({**dict(zip(keys, key)), "reference_pool": pool, "status": status,
                "event_day": cfg["event_day"], "event_count": 1, "reference_unit": "whole_date",
                "event_complete": event_valid, "event_reason": str(event.reason),
                "event_pre_value": _number(event.pre_value) if event_valid else None,
                "event_post_value": _number(event.post_value) if event_valid else None,
                "event_change": event_change, "event_relative_change_percent": _number(event.relative_change_percent) if event_valid else None,
                "reference_n_expected": len(dates), "reference_n_valid": len(values),
                "reference_coverage_complete": complete_refs, "excluded_reference_days": excluded,
                "reference_change_median": median,
                "reference_change_q25": float(np.quantile(values, .25)) if publish else None,
                "reference_change_q75": float(np.quantile(values, .75)) if publish else None,
                "reference_change_min": float(values.min()) if publish else None,
                "reference_change_max": float(values.max()) if publish else None,
                "event_minus_reference_median": event_change - median if publish else None,
                "reference_relative_change_median_percent": float(np.median(relative)) if publish and len(relative) else None,
                "reference_relative_change_n": len(relative),
                "reference_changes_below_event": below,
                "reference_changes_equal_event": equal,
                "reference_changes_above_event": len(values) - below - equal if publish else None,
                "descriptive_event_rank_low_ascending": 1 + below if publish else None,
                "descriptive_event_rank_high_ascending": 1 + below + equal if publish else None,
                "rank_population_size": len(values) + 1 if publish else None,
                "rank_interpretation": "Descriptive rank among event plus reference date changes; not a p-value or randomized causal test"})
    return records


def summarize_trials(pair_df: pd.DataFrame, venue_df: pd.DataFrame, protocol: dict) -> dict:
    """Build fixed-universe daily metrics and single-event reference contrasts.

    Returned tables are JSON-ready record lists. Primary reference statistics
    require every preselected weekday reference; incomplete secondary references
    remain explicitly labelled. No confidence intervals or p-values are made.
    """
    cfg = _settings(protocol)
    pairs = _validate_pair(pair_df, cfg)
    venues = _validate_venue(venue_df, cfg)
    latency, leadership = _latency_tables(pairs, venues, cfg)
    shocks = _shock_tables(venues, cfg)
    changes = _changes(latency, cfg, ["universe", "horizon_ms"],
                       ["mean_pair_median_us", "match_rate"], "latency")
    changes.extend(_changes(shocks, cfg, ["universe", "shock_threshold_bp"],
                            ["mean_venue_shocks_per_minute", "pooled_shocks_per_venue_minute",
                             "mean_venue_clustered_shocks_per_minute"], "shock_intensity"))
    changes.extend(_changes(leadership, cfg, ["universe", "horizon_ms", "venue"],
                            ["leadership_share"], "counting_leadership"))
    # Concatenating heterogeneous metric dimensions creates pandas NaNs. They
    # are metadata absence, not an invented zero-sized venue/horizon.
    contrasts = _reference_summaries(changes, cfg)
    for row in [*changes, *contrasts]:
        for key, value in list(row.items()):
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                row[key] = None
    return {"status": "descriptive_case_study", "event_day": cfg["event_day"], "event_count": 1,
            "primary_reference_days": cfg["primary_refs"], "reference_days": cfg["references"],
            "latency_daily": latency, "shock_daily": shocks, "leadership_daily": leadership,
            "daily_changes": changes, "contrasts": contrasts,
            "primary_definitions": {
                "latency": "Equal-pair mean of daily conditional medians, all intended directed pairs supported before and after",
                "shock_intensity": "Primary: mean_venue_clustered_shocks_per_minute, equal-venue mean rate after causal segment-local burst thinning at the protocol separation and common fixed basis-point threshold",
                "raw_shock_intensity": "Sensitivity: mean_venue_shocks_per_minute, equal-venue mean unthinned threshold-crossing rate per observed minute",
                "pooled_shock_intensity": "Sensitivity: raw shocks divided by summed observed venue-minutes; separate exposure weighting",
                "leadership": "Outgoing matched counts divided by outgoing plus incoming matched counts; not Hasbrouck information share",
                "rank": "Descriptive rank of one event-date change among complete reference-date changes"},
            "limitations": [
                "One corporate event is one event, regardless of the number of quotes or matched reactions.",
                "Reference dates are exposure controls, not guaranteed news-free or exchangeable randomized counterfactuals.",
                "No event-class confidence interval, p-value, causal effect or trading-profit inference is computed.",
                "Conditional matched delays require accompanying match rates and observation-exclusion counts.",
                "Different fixed venue universes and horizons can define different eligible event populations.",
                "Primary reference results require every planned weekday reference; secondary exclusions remain explicit.",
                "Scheduled public-event start is not a nanosecond first-information timestamp.",
            ]}
