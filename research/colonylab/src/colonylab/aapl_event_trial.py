"""Saved-data, native-midpoint AAPL corporate-event case study.

The scheduled keynote is a macro-time anchor. Microsecond measurements use
native receive-time midpoint changes inside fixed windows, never the replay's
100-ms grid as if it were an event stream. Event/date is the inference unit.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .event_midpoints import load_day, match_events, summarize_events

DAY_FILES = ("pair_metrics.csv", "venue_metrics.csv", "price_profile.csv", "matching_audit.csv")
COMPUTATION_FILES = ("src/colonylab/event_midpoints.py", "src/colonylab/latency.py",
                     "src/colonylab/aapl_event_trial.py")


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def window_events(events, start_ns, end_ns):
    """Enforce each subwindow's exclusive end, including for early matches."""
    selected = events.loc[events.t.ge(start_ns) & events.t.lt(end_ns)].copy()
    selected.attrs.update(events.attrs)
    available = np.maximum(0, int(end_ns) - selected.t.to_numpy(np.int64) - 1)
    previous = selected.observed_ns.to_numpy(np.int64)
    shortened = available < previous
    selected["observed_ns"] = np.minimum(previous, available)
    selected.loc[shortened, "terminal_reason"] = "end"
    selected.loc[selected.latency_ns.gt(selected.observed_ns), "latency_ns"] = -1
    return selected


def observed_seconds(series, start_ns, end_ns):
    start_ns, end_ns = max(start_ns, series.start_ns), min(end_ns, series.end_ns)
    if start_ns >= end_ns:
        return 0.
    if not len(series.times_ns):
        return 0.
    first = np.maximum(series.times_ns, start_ns)
    following = np.r_[series.times_ns[1:], series.end_ns]
    last = np.minimum(following, end_ns)
    return float(np.maximum(0, last - first)[series.valid].sum() / 1e9)


def shock_arrays(series, threshold_bp, separation_ns):
    """Fixed economic-size threshold and causal, segment-local burst thinning."""
    p, t = series.price2, series.times_ns
    delta = np.diff(p)
    candidates = (series.valid[1:] & series.valid[:-1] & (series.segments[1:] == series.segments[:-1])
                  & (delta != 0) & (p[:-1] > 0))
    # Half-basis-point multiples permit exact integer comparisons on this study.
    twice_threshold = int(round(2 * threshold_bp))
    if twice_threshold / 2 != threshold_bp:
        raise ValueError("Shock thresholds must be half-basis-point multiples")
    qualifying = candidates & (np.abs(delta) * 20_000 >= twice_threshold * p[:-1])
    indices = np.flatnonzero(qualifying) + 1
    selected = []
    previous_time, previous_segment = -10**18, -1
    for index in indices:
        if series.segments[index] != previous_segment or int(t[index]) - previous_time >= separation_ns:
            selected.append(index)
            previous_time, previous_segment = int(t[index]), int(series.segments[index])
    return t[indices], t[np.asarray(selected, np.int64)]


def venue_metrics(series, windows, protocol):
    times, price = series.times_ns, series.price2
    changes = (series.valid[1:] & series.valid[:-1] & (series.segments[1:] == series.segments[:-1])
               & (price[1:] != price[:-1]))
    change_times = times[1:][changes]
    abs_bp = (10_000 * np.abs(np.diff(price)[changes]) / price[:-1][changes])
    shocks = {bp: shock_arrays(series, bp, int(protocol["shock_cluster_separation_ms"] * 1e6))
              for bp in protocol["shock_thresholds_bp"]}
    result = []
    for window in windows:
        start, end = window["start_ns"], window["end_ns"]
        seconds = observed_seconds(series, start, end)
        positions = (change_times >= start) & (change_times < end)
        at_start = int(np.searchsorted(times, start, side="left")) - 1
        at_end = int(np.searchsorted(times, end, side="left")) - 1
        continuous = (at_start >= 0 and at_end >= 0 and series.valid[at_start] and series.valid[at_end]
                      and series.segments[at_start] == series.segments[at_end]
                      and bool(series.valid[at_start:at_end + 1].all()))
        signed = 10_000 * np.log(price[at_end] / price[at_start]) if continuous else np.nan
        spread, depth, grid_count = np.nan, np.nan, 0
        grid = getattr(series, "grid", None)
        if grid is not None and len(grid):
            g = grid.loc[grid.time_ns.ge(start) & grid.time_ns.lt(end) & grid.valid]
            grid_count = len(g)
            if len(g):
                if {"bid", "ask"} <= set(g):
                    spread = float(np.median(20_000 * (g.ask.to_numpy(float) - g.bid.to_numpy(float)) /
                                             (g.ask.to_numpy(float) + g.bid.to_numpy(float))))
                if {"bid_size", "ask_size"} <= set(g):
                    depth = float((g.bid_size + g.ask_size).median())
        for bp, (raw, clustered) in shocks.items():
            nraw = int(np.sum((raw >= start) & (raw < end)))
            ncluster = int(np.sum((clustered >= start) & (clustered < end)))
            result.append({"window": window["name"], "shock_threshold_bp": bp,
                           "observed_seconds": seconds, "scheduled_seconds": (end - start) / 1e9,
                           "coverage_fraction": seconds / ((end - start) / 1e9),
                           "midpoint_changes": int(positions.sum()),
                           "midpoint_changes_per_minute": positions.sum() * 60 / seconds if seconds else np.nan,
                           "shocks_raw": nraw, "shocks_clustered": ncluster,
                           "shocks_per_minute": nraw * 60 / seconds if seconds else np.nan,
                           "clustered_shocks_per_minute": ncluster * 60 / seconds if seconds else np.nan,
                           "signed_return_bp": signed, "total_abs_return_bp": float(abs_bp[positions].sum()),
                           "spread_median_bps": spread, "depth_median": depth,
                           "grid_descriptive_rows": grid_count})
    return result


def audit_matches(leader, follower, events, horizon):
    """Independent bounded direct scans on every day/ordered-pair combination."""
    p, t = leader.price2, leader.times_ns
    candidate = (leader.valid[1:] & leader.valid[:-1] & (leader.segments[1:] == leader.segments[:-1])
                 & (p[1:] != p[:-1]) & (t[1:] >= leader.start_ns) & (t[1:] < leader.end_ns))
    if len(events) != int(candidate.sum()):
        raise ValueError("Independent native midpoint candidate count differs")
    selected = set(np.linspace(0, len(events) - 1, min(8, len(events))).astype(int)) if len(events) else set()
    for field in ("status", "terminal_reason"):
        for _, group in events.groupby(field, sort=False):
            selected.add(int(group.index[0]))
    for i in sorted(selected):
        event = events.iloc[i]
        now = int(event.t)
        li = int(np.searchsorted(t, now))
        direction = 1 if p[li] > p[li - 1] else -1
        target = int(p[li])
        f = int(np.searchsorted(follower.times_ns, now, side="right")) - 1
        has_state = f >= 0 and follower.start_ns <= now < follower.end_ns and follower.valid[f]
        stop = min(leader.end_ns, follower.end_ns)
        for stream, first in ((leader, li), (follower, f)):
            last = int(np.searchsorted(stream.times_ns, min(now + horizon + 1, stop), side="right"))
            for position in range(first + 1, last):
                if (not stream.valid[position] or (position > 0 and stream.segments[position] != stream.segments[position - 1])):
                    stop = min(stop, int(stream.times_ns[position]))
                    break
        available = min(horizon, max(0, stop - now - 1))
        if not len(follower.times_ns):
            reason = "end"
        else:
            reason = "horizon" if stop - now - 1 >= horizon else "quality" if stop < min(leader.end_ns, follower.end_ns) else "end"
        lag, status = -1, "no_state"
        if has_state:
            already = follower.price2[f] >= target if direction > 0 else follower.price2[f] <= target
            status = "already" if already else "behind"
            if not already:
                last = int(np.searchsorted(follower.times_ns, now + available, side="right"))
                for position in range(f + 1, last):
                    hit = follower.price2[position] >= target if direction > 0 else follower.price2[position] <= target
                    if hit:
                        lag = int(follower.times_ns[position]) - now
                        break
        if (event.status != status or event.latency_ns != lag
                or event.observed_ns != (available if has_state else 0) or event.terminal_reason != reason):
            raise ValueError(f"Independent midpoint first-passage mismatch at {now}")
    return {"leader_events": len(events), "sampled_cases": len(selected), "mismatches": 0}


def price_profile(series, primary_venues, anchor, protocol):
    grid = np.arange(anchor + protocol["outer_start_minute"] * 60_000_000_000,
                     anchor + protocol["outer_end_minute"] * 60_000_000_000,
                     protocol["price_path_grid_seconds"] * 1_000_000_000, dtype=np.int64)
    values = np.full((len(primary_venues), len(grid)), np.nan)
    for n, venue in enumerate(primary_venues):
        s = series[venue]
        if not len(s.times_ns):
            continue
        index = np.searchsorted(s.times_ns, grid, side="left") - 1
        safe = np.maximum(index, 0)
        good = (index >= 0) & s.valid[safe] & (grid >= s.start_ns) & (grid < s.end_ns)
        values[n, good] = s.price2[safe[good]] / 2e9
    supported = np.isfinite(values).sum(axis=0)
    middle = np.full(len(grid), np.nan)
    mask = supported >= (len(primary_venues) + 1) // 2
    middle[mask] = np.nanmedian(values[:, mask], axis=0)
    base_index = np.flatnonzero((grid == anchor) & np.isfinite(middle))
    baseline = middle[base_index[0]] if len(base_index) else np.nan
    return pd.DataFrame({"time_ns": grid, "minute_from_anchor": (grid - anchor) / 60e9,
                         "cross_venue_median_mid": middle, "n_venues": supported,
                         "anchor_supported": bool(len(base_index)),
                         "return_from_anchor_bp": 10_000 * np.log(middle / baseline)})


def process_day(task):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    started = time.perf_counter()
    root, directory, protocol = Path(task["root"]), Path(task["directory"]), task["protocol"]
    marker_path = directory / "complete.json"
    if marker_path.is_file():
        saved = json.loads(marker_path.read_text())
        if (saved["identity"] == task["identity"]
                and all((directory / name).is_file() and file_sha(directory / name) == saved["files"][name] for name in DAY_FILES)):
            # Reusing a derived day still rehashes native/cache inputs through load_day below.
            series, facts = load_day(root, task["day"], task["outer_start_ns"], task["outer_end_ns"], protocol["all_venues"])
            if signature(facts) == saved["source_facts_sha256"]:
                return {"day": task["day"], "reused": True}
    series, facts = load_day(root, task["day"], task["outer_start_ns"], task["outer_end_ns"], protocol["all_venues"])
    windows = [{**row, "start_ns": task["anchor_ns"] + row["start_minute"] * 60_000_000_000,
                "end_ns": task["anchor_ns"] + row["end_minute"] * 60_000_000_000} for row in protocol["windows"]]
    pairs, venues, audit = [], [], []
    for venue, states in series.items():
        venues.extend({"day": task["day"], "venue": venue, **row} for row in venue_metrics(states, windows, protocol))
    horizon = int(max(protocol["horizons_ms"]) * 1e6)
    for leader, a in series.items():
        for follower, b in series.items():
            if leader == follower:
                continue
            events = match_events(a, b, max_horizon_ns=horizon)
            audited = audit_matches(a, b, events, horizon)
            audit.append({"day": task["day"], "leader": leader, "follower": follower, **audited})
            for window in windows:
                selected = window_events(events, window["start_ns"], window["end_ns"])
                for h in protocol["horizons_ms"]:
                    row = summarize_events(selected, int(h * 1e6))
                    pairs.append({"day": task["day"], "window": window["name"], "leader": leader, "follower": follower, **row})
    frames = [pd.DataFrame(pairs), pd.DataFrame(venues),
              price_profile(series, protocol["primary_venues"], task["anchor_ns"], protocol).assign(day=task["day"]),
              pd.DataFrame(audit)]
    for filename, frame in zip(DAY_FILES, frames):
        frame.to_csv(directory / filename, index=False)
    dump(directory / "source_facts.json", facts)
    dump(marker_path, {"identity": task["identity"], "source_facts_sha256": signature(facts),
                       "files": {name: file_sha(directory / name) for name in DAY_FILES},
                       "elapsed_seconds": time.perf_counter() - started})
    return {"day": task["day"], "reused": False, "seconds": round(time.perf_counter() - started, 2),
            "matching_cases": sum(row["sampled_cases"] for row in audit),
            "native_window_rows": sum(len(states.times_ns) for states in series.values())}


def run(root, protocol_path, out_dir, *, workers=2, summarize_only=False, progress=print):
    from .event_trial_stats import summarize_trials

    root, out = Path(root).resolve(), Path(out_dir).resolve()
    protocol = json.loads(Path(protocol_path).read_text())
    if protocol["market_data_policy"] != "saved_data_only_no_downloads":
        raise ValueError("This event trial cannot download market data")
    calendar_path = root / protocol["event_calendar"]
    calendar = json.loads(calendar_path.read_text())
    anchor_event = next(row for row in calendar["events"] if row["id"] == protocol["event_id"])
    if (anchor_event["event_time_utc"] != protocol["event_day"] + "T" + protocol["anchor_time_utc"] + "Z"
            or anchor_event["coverage_status"] != "covered_native_midpoints"):
        raise ValueError("The selected anchor is not supported by the verified event calendar")
    code = [{"path": name, "sha256": file_sha(root / name)} for name in COMPUTATION_FILES]
    all_code = code + [{"path": "src/colonylab/event_trial_stats.py", "sha256": file_sha(root / "src/colonylab/event_trial_stats.py")}]
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "protocol.json", protocol)
    dump(out / "event_calendar.json", calendar)
    coverage = [{key: row.get(key) for key in ("id", "event_class", "label", "date", "event_time_utc", "timestamp_nature", "coverage_status", "analysis_role")}
                for row in calendar["events"]]
    pd.DataFrame(coverage).to_csv(out / "event_coverage.csv", index=False)
    dates = sorted(set([protocol["event_day"], *protocol["reference_days"]]))
    tasks = []
    for day in dates:
        anchor = pd.Timestamp(day + "T" + protocol["anchor_time_utc"] + "Z").value
        directory = out / "days" / day
        directory.mkdir(parents=True, exist_ok=True)
        tasks.append({"root": str(root), "directory": str(directory), "day": day, "protocol": protocol,
                      "anchor_ns": anchor, "outer_start_ns": anchor + protocol["outer_start_minute"] * 60_000_000_000,
                      "outer_end_ns": anchor + protocol["outer_end_minute"] * 60_000_000_000,
                      "identity": {"day": day, "protocol_sha256": signature(protocol), "code": code,
                                   "calendar_sha256": file_sha(calendar_path)}})
    if not summarize_only:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for n, future in enumerate(as_completed([executor.submit(process_day, task) for task in tasks]), 1):
                progress(f"Completed {n}/{len(tasks)} event/reference days: {future.result()}", flush=True)
    sources = []
    for task in tasks:
        directory = Path(task["directory"])
        marker = json.loads((directory / "complete.json").read_text())
        facts = json.loads((directory / "source_facts.json").read_text())
        if (marker["identity"] != task["identity"] or signature(facts) != marker["source_facts_sha256"]
                or any(file_sha(directory / name) != marker["files"][name] for name in DAY_FILES)):
            raise ValueError("Event day checkpoint differs from protocol, code, inputs or outputs")
        sources.extend(facts)
    for filename in DAY_FILES:
        pd.concat([pd.read_csv(Path(task["directory"]) / filename) for task in tasks], ignore_index=True).to_csv(out / filename, index=False)
    pairs, venues = pd.read_csv(out / "pair_metrics.csv"), pd.read_csv(out / "venue_metrics.csv")
    results = summarize_trials(pairs, venues, protocol)
    for key, value in results.items():
        if isinstance(value, list) and (not value or isinstance(value[0], dict)):
            pd.DataFrame(value).to_csv(out / (key + ".csv"), index=False)
    audit = pd.read_csv(out / "matching_audit.csv")
    summary = {"status": "completed", "measurement": protocol["measurement"], "event_count": 1,
               "reference_days": len(protocol["reference_days"]), "primary_reference_days": len(protocol["primary_reference_days"]),
               "observed_dates": len(dates), "source_cache_tasks": len(sources),
               "independent_matching_cases": int(audit.sampled_cases.sum()), "independent_matching_mismatches": int(audit.mismatches.sum()),
               "results": results, "limitations": protocol["limitations"]}
    dump(out / "summary.json", summary)
    dump(out / "provenance.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
         "protocol_sha256": file_sha(out / "protocol.json"), "calendar_sha256": file_sha(out / "event_calendar.json"),
         "computation_code": all_code, "sources": sources,
         "output_sha256": {filename: file_sha(out / filename) for filename in DAY_FILES}})
    return summary
