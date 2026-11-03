"""Event- and year-level descriptive summaries for matched corporate events.

The frozen trial registry, rather than the observed quote table, determines
which controls and events must exist. Missing support never creates a smaller
successful-looking control cohort. Release/call anchors may share one bundle;
they do not become independent observations by producing more quote events.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


CONTRASTS = (
    ("primary30", "pre30", "post30"),
    ("immediate5", "pre5", "post5"),
    ("delayed25", "pre25", "post25"),
    ("anchor_minus5", "shift_minus5_pre", "shift_minus5_post"),
    ("anchor_plus5", "shift_plus5_pre", "shift_plus5_post"),
)
META = ("trial_id", "bundle_id", "event_id", "event_type", "role", "control_for",
        "year", "day", "session")
DIMS = ("clock", "age_ms", "horizon_ms", "metric")
REQUIRED = set(META) | set(DIMS) | {"window", "value", "complete", "reason"}


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _optional(value):
    return None if value is None or pd.isna(value) or value == "" else value


def _registry(protocol):
    controls = protocol.get("controls_per_event", 5)
    _require(type(controls) is int and controls == 5,
             "The primary design requires exactly five matched controls per event")
    rows = protocol.get("trials")
    _require(isinstance(rows, list) and rows, "An explicit frozen trials registry is required")
    registry = {}
    for source in rows:
        _require(set(META) - {"control_for"} <= set(source), "Trial registry lacks identity fields")
        row = {name: source.get(name) for name in META}
        row["control_for"] = _optional(row["control_for"])
        _require(isinstance(row["trial_id"], str) and row["trial_id"]
                 and row["trial_id"] not in registry, "Trial IDs must be unique nonempty strings")
        _require(row["role"] in ("event", "control"), "Trial role must be event or control")
        _require(row["session"] in ("RTH", "afterhours", "premarket"), "Unknown trading-session stratum")
        parsed = pd.Timestamp(row["day"])
        _require(parsed.strftime("%Y-%m-%d") == row["day"] and parsed.weekday() < 5
                 and int(row["year"]) == parsed.year, "Trial year/day must identify a valid exchange weekday")
        row["year"] = int(row["year"])
        _require(all(isinstance(row[name], str) and row[name]
                     for name in ("bundle_id", "event_id", "event_type")), "Event/bundle/type labels are required")
        _require((row["role"] == "event") == (row["control_for"] is None),
                 "Only control trials may refer to a parent event trial")
        registry[row["trial_id"]] = row
    events = [row for row in registry.values() if row["role"] == "event"]
    _require(events, "At least one event trial is required")
    for row in registry.values():
        if row["role"] == "control":
            parent = registry.get(row["control_for"])
            _require(parent is not None and parent["role"] == "event", "Control refers to an unknown event trial")
            _require(all(row[name] == parent[name] for name in ("year", "session", "event_type", "bundle_id", "event_id")),
                     "Controls must preserve the parent event's year/session/type/bundle identity")
            _require(row["day"] != parent["day"], "An event date cannot serve as its own control")
    for event in events:
        selected = [row for row in registry.values() if row["control_for"] == event["trial_id"]]
        _require(len(selected) == controls and len({row["day"] for row in selected}) == controls,
                 "Each event must have exactly five distinct frozen control dates")
    discovery = set(protocol.get("discovery_event_ids", []))
    _require(discovery <= {row["event_id"] for row in events}, "Unknown discovery event ID")
    discovery_bundles = {row["bundle_id"] for row in events if row["event_id"] in discovery}
    for row in registry.values():
        row["analysis_scope"] = "discovery" if row["bundle_id"] in discovery_bundles else "historical_primary"
    return registry, events


def _validate(metrics, registry):
    _require(REQUIRED <= set(metrics), "Metric table lacks required fields")
    frame = metrics.copy()
    # CSV round trips turn an empty textual support reason into a null.
    frame["reason"] = frame.reason.fillna("")
    _require(set(frame.trial_id) <= set(registry), "Metrics contain an unplanned trial")
    _require(not frame.duplicated(["trial_id", *DIMS, "window"]).any(), "Duplicate trial/window/metric summaries")
    for row in frame.itertuples(index=False):
        planned = registry[row.trial_id]
        for name in META:
            value = _optional(getattr(row, name)) if name == "control_for" else getattr(row, name)
            _require(value == planned[name], f"Metric metadata differs from frozen trial: {name}")
        _require(isinstance(row.complete, (bool, np.bool_)), "Metric completeness must be boolean")
        _require(isinstance(row.reason, str), "Every metric requires a textual support reason")
        _require(isinstance(row.clock, str) and row.clock and isinstance(row.metric, str) and row.metric,
                 "Metric and clock labels are required")
        _require(np.isfinite(row.age_ms) and row.age_ms > 0
                 and np.isfinite(row.horizon_ms) and row.horizon_ms > 0,
                 "Quote age and matching horizon must be positive")
        if row.complete:
            _require(pd.notna(row.value) and np.isfinite(row.value), "Complete metric has no finite value")
            if row.metric == "match_rate":
                _require(0 <= row.value <= 1, "Matching rate must lie in [0,1]")
            if row.metric in ("latency_us", "shock_rate"):
                _require(row.value >= 0, "Latency and shock intensity cannot be negative")
    return frame


def _contrasts(frame, protocol):
    supplied = protocol.get("contrasts")
    if supplied is not None:
        rows = [(row["name"], row["pre"], row["post"]) for row in supplied]
    elif protocol.get("windows"):
        names = {row["name"] for row in protocol["windows"]}
        rows = [row for row in CONTRASTS if row[1] in names and row[2] in names]
    else:
        names = set(frame.window)
        rows = [row for row in CONTRASTS if row[1] in names or row[2] in names]
    _require(rows and len({row[0] for row in rows}) == len(rows), "At least one unique contrast is required")
    _require(all(name and pre and post and pre != post for name, pre, post in rows), "Invalid pre/post contrast")
    allowed = {name for _, pre, post in rows for name in (pre, post)}
    if protocol.get("windows"):
        allowed |= {row["name"] for row in protocol["windows"]}
    _require(set(frame.window) <= allowed, "Metric table contains an unplanned window")
    return rows


def _dimensions(frame, protocol):
    supplied = protocol.get("metric_dimensions")
    if supplied is None:
        rows = frame[list(DIMS)].drop_duplicates().to_dict("records")
    else:
        _require(isinstance(supplied, list) and supplied, "metric_dimensions must be a nonempty list")
        rows = [{name: row[name] for name in DIMS} for row in supplied]
        _require(not pd.DataFrame(rows).duplicated(list(DIMS)).any(), "Duplicate planned metric dimensions")
        expected = {tuple(row[name] for name in DIMS) for row in rows}
        actual = {tuple(row) for row in frame[list(DIMS)].itertuples(index=False, name=None)}
        _require(actual <= expected, "Metric table contains an unplanned measurement dimension")
    _require(rows, "No measurement dimensions are planned or observed")
    for row in rows:
        _require(isinstance(row["clock"], str) and row["clock"]
                 and isinstance(row["metric"], str) and row["metric"]
                 and np.isfinite(row["age_ms"]) and row["age_ms"] > 0
                 and np.isfinite(row["horizon_ms"]) and row["horizon_ms"] > 0,
                 "Invalid planned measurement dimension")
    return rows


def _changes(frame, registry, dimensions, contrasts):
    lookup = {tuple(getattr(row, name) for name in ("trial_id", *DIMS, "window")): row
              for row in frame.itertuples(index=False)}
    output = []
    for planned in registry.values():
        for dims in dimensions:
            key = (planned["trial_id"], *(dims[name] for name in DIMS))
            for contrast, pre, post in contrasts:
                left, right = lookup.get((*key, pre)), lookup.get((*key, post))
                missing = []
                for label, row in (("pre", left), ("post", right)):
                    if row is None:
                        missing.append(f"missing_{label}_window")
                    elif not row.complete:
                        missing.append(f"{label}:{row.reason or 'insufficient_support'}")
                complete = not missing
                before, after = (float(left.value), float(right.value)) if complete else (None, None)
                change = after - before if complete else None
                output.append({**planned, **dims, "contrast": contrast, "pre_window": pre, "post_window": post,
                               "complete": complete, "reason": ";".join(missing), "pre_value": before,
                               "post_value": after, "change": change,
                               "relative_change_percent": 100 * change / before if complete and before > 0 else None})
    return output


def _excess(changes, registry):
    lookup = {(row["trial_id"], *(row[name] for name in DIMS), row["contrast"]): row for row in changes}
    records = []
    for event in changes:
        if event["role"] != "event":
            continue
        controls = [row for row in registry.values() if row["control_for"] == event["trial_id"]]
        key = (*(event[name] for name in DIMS), event["contrast"])
        values = [lookup[(row["trial_id"], *key)] for row in controls]
        valid = [row for row in values if row["complete"]]
        excluded = [{"trial_id": row["trial_id"], "day": row["day"], "reason": row["reason"]}
                    for row in values if not row["complete"]]
        complete = event["complete"] and len(valid) == 5
        control_changes = np.asarray([row["change"] for row in valid], dtype=float)
        median = float(np.median(control_changes)) if complete else None
        records.append({**{name: event[name] for name in (*META, "analysis_scope", *DIMS, "contrast", "pre_window", "post_window")},
                        "complete": complete,
                        "status": "complete" if complete else "event_insufficient_support" if not event["complete"] else "insufficient_control_support",
                        "reason": event["reason"] if not event["complete"] else "" if complete else "all_five_controls_required",
                        "event_complete": event["complete"], "event_pre_value": event["pre_value"],
                        "event_post_value": event["post_value"], "event_change": event["change"],
                        "event_relative_change_percent": event["relative_change_percent"],
                        "controls_expected": 5, "controls_valid": len(valid),
                        "control_trial_ids": [row["trial_id"] for row in controls], "excluded_controls": excluded,
                        "control_changes": [{"trial_id": row["trial_id"], "change": row["change"], "complete": row["complete"]} for row in values],
                        "control_change_median": median,
                        "control_change_q25": float(np.quantile(control_changes, .25)) if complete else None,
                        "control_change_q75": float(np.quantile(control_changes, .75)) if complete else None,
                        "control_change_min": float(control_changes.min()) if complete else None,
                        "control_change_max": float(control_changes.max()) if complete else None,
                        "excess_change": event["change"] - median if complete else None})
    return records


def _groups(rows, keys):
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row[name] for name in keys), []).append(row)
    return grouped


def _describe(values):
    values = np.asarray(values, dtype=float)
    return {"median": float(np.median(values)) if len(values) else None,
            "q25": float(np.quantile(values, .25)) if len(values) else None,
            "q75": float(np.quantile(values, .75)) if len(values) else None,
            "minimum": float(values.min()) if len(values) else None,
            "maximum": float(values.max()) if len(values) else None}


def _bundle_rows(excess):
    keys = ("analysis_scope", "event_type", "session", *DIMS, "contrast", "bundle_id", "year")
    bundles = []
    for key, anchors in _groups(excess, keys).items():
        complete = all(row["complete"] for row in anchors)
        bundles.append({**dict(zip(keys, key)), "complete": complete,
                        "anchor_trial_ids": [row["trial_id"] for row in anchors],
                        "excess_change": float(np.median([row["excess_change"] for row in anchors])) if complete else None})
    return bundles


def _class_summaries(excess, bundles):
    keys = ("analysis_scope", "event_type", "session", *DIMS, "contrast")
    anchor_groups = _groups(excess, keys)
    records = []
    for key, rows in _groups(bundles, keys).items():
        anchors = anchor_groups[key]
        valid = [row for row in rows if row["complete"]]
        years = sorted({row["year"] for row in valid})
        stats = _describe([row["excess_change"] for row in valid])
        omitted = []
        for year in years:
            remaining = [row["excess_change"] for row in valid if row["year"] != year]
            if remaining:
                omitted.append({"omitted_year": year, "median_excess_change": float(np.median(remaining)),
                                "n_bundles": len(remaining)})
        leave_values = [row["median_excess_change"] for row in omitted]
        records.append({**dict(zip(keys, key)), "status": "complete" if len(valid) == len(rows) else "incomplete_support" if valid else "insufficient_support",
                        "n_events_expected": len(anchors), "n_events": sum(row["complete"] for row in anchors),
                        "n_bundles_expected": len(rows), "n_bundles": len(valid),
                        "n_years_expected": len({row["year"] for row in rows}), "n_years": len(years), "years": years,
                        "median_excess_change": stats["median"], "excess_change_q25": stats["q25"],
                        "excess_change_q75": stats["q75"], "excess_change_min": stats["minimum"], "excess_change_max": stats["maximum"],
                        "leave_one_year_out_min": min(leave_values) if leave_values else None,
                        "leave_one_year_out_max": max(leave_values) if leave_values else None,
                        "leave_one_year_out": omitted,
                        "excluded_events": [{"trial_id": row["trial_id"], "status": row["status"], "reason": row["reason"]} for row in anchors if not row["complete"]],
                        "weighting": "Equal bundle weight; multiple same-class anchors within a bundle contribute their median only when all are supported"})
    return records


def _comparisons(bundles):
    keys = ("analysis_scope", *DIMS, "contrast")
    records = []
    for key, group in _groups(bundles, keys).items():
        classes = _groups(group, ("event_type", "session"))
        for (left_type, left_session), (right_type, right_session) in combinations(sorted(classes), 2):
            if left_type == right_type:
                continue
            left, right = classes[(left_type, left_session)], classes[(right_type, right_session)]
            left_years, right_years = {row["year"] for row in left}, {row["year"] for row in right}
            expected_years = sorted(left_years & right_years)
            paired, excluded = [], []
            for year in expected_years:
                a, b = [row for row in left if row["year"] == year], [row for row in right if row["year"] == year]
                if not all(row["complete"] for row in (*a, *b)):
                    excluded.append({"year": year,
                                     "unsupported_bundles_a": [row["bundle_id"] for row in a if not row["complete"]],
                                     "unsupported_bundles_b": [row["bundle_id"] for row in b if not row["complete"]]})
                    continue
                av, bv = float(np.median([row["excess_change"] for row in a])), float(np.median([row["excess_change"] for row in b]))
                aid, bid = {row["bundle_id"] for row in a}, {row["bundle_id"] for row in b}
                paired.append({"year": year, "class_a_median_excess": av, "class_b_median_excess": bv,
                               "difference_a_minus_b": av - bv, "n_bundles_a": len(a), "n_bundles_b": len(b),
                               "n_unique_bundles": len(aid | bid), "n_shared_bundles": len(aid & bid)})
            values = [row["difference_a_minus_b"] for row in paired]
            stats = _describe(values)
            leave = [{"omitted_year": row["year"], "median_difference_a_minus_b": float(np.median([other["difference_a_minus_b"] for other in paired if other["year"] != row["year"]]))}
                     for row in paired] if len(paired) > 1 else []
            leave_values = [row["median_difference_a_minus_b"] for row in leave]
            records.append({**dict(zip(keys, key)), "event_type_a": left_type, "session_a": left_session,
                            "event_type_b": right_type, "session_b": right_session,
                            "status": "descriptive_incomplete_paired_years" if paired and excluded else "descriptive_paired_years" if paired else "no_common_supported_years",
                            "session_confounded": left_session != right_session,
                            "n_paired_years_expected": len(expected_years), "n_paired_years": len(paired),
                            "years": [row["year"] for row in paired], "excluded_years": excluded,
                            "unpaired_years_a": sorted(left_years - right_years), "unpaired_years_b": sorted(right_years - left_years),
                            "median_difference_a_minus_b": stats["median"], "difference_q25": stats["q25"],
                            "difference_q75": stats["q75"], "difference_min": stats["minimum"], "difference_max": stats["maximum"],
                            "leave_one_year_out_min": min(leave_values) if leave_values else None,
                            "leave_one_year_out_max": max(leave_values) if leave_values else None,
                            "leave_one_year_out": leave, "paired_year_details": paired,
                            "interpretation": "Descriptive within-year difference of matched-control excess changes; session and event class are confounded" if left_session != right_session else "Descriptive within-year difference of matched-control excess changes; not a causal event-class effect"})
    return records


def summarize_multi_events(metrics: pd.DataFrame, protocol: dict) -> dict:
    """Return auditable trial changes, five-control excesses and class tables.

    ``protocol.trials`` is the frozen event/control registry; a control's
    ``control_for`` identifies its event trial_id. Optional
    ``metric_dimensions`` makes even an entirely absent measurement explicit.
    ``discovery_event_ids`` prevents already inspected events entering the
    historical primary class summary. No confidence intervals or p-values are
    calculated from this small number of calendar years.
    """
    registry, _ = _registry(protocol)
    frame = _validate(metrics, registry)
    changes = _changes(frame, registry, _dimensions(frame, protocol), _contrasts(frame, protocol))
    excess = _excess(changes, registry)
    bundles = _bundle_rows(excess)
    return _clean({"status": "descriptive_multi_event_study", "event_changes": changes,
                   "event_excess": excess, "class_summary": _class_summaries(excess, bundles),
                   "class_comparisons": _comparisons(bundles),
                   "limitations": [
                       "Each event anchor requires every one of its five frozen matched-control changes; missing support is not replaced or silently dropped.",
                       "Corporate-event bundles and calendar years, rather than quotes, define dependence and sample size.",
                       "Discovery events are kept separate from the historical primary summaries.",
                       "Class summaries with incomplete support describe supported bundles only, with planned and excluded counts retained.",
                       "A paired-year class comparison requires every planned bundle in both class/year strata; incomplete years remain explicitly excluded.",
                       "Cross-session event classes remain confounded with trading hours even after same-clock matched controls.",
                       "Scheduled call/keynote starts and filing acceptance times need not identify first public information.",
                       "Different clocks, freshness thresholds and horizons can define different observable event populations.",
                       "Leave-one-year-out ranges are sensitivity summaries, not confidence intervals; no p-values or causal effects are reported.",
                   ]})
