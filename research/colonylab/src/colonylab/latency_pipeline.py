"""Resumable offline replication of venue latency on existing native MBP-1 files.

All acquisition routes are deliberately absent. Saved request receipts, source
hashes, integer quote states and observation barriers bind each result to its
inputs. Raw bid/ask events are used rather than model-grid or midpoint caches.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

from .latency import QuoteSeries, reaction_events, summarize_events
from .latency_data import load_entry, load_study
from .latency_validation import infer_latency, render_latency_figures


DAY_FILES = ("daily_latency.csv", "daily_latency_by_side.csv", "quote_quality.csv",
             "realized_volatility.csv", "leadlag_summary.csv", "leadlag_curves.csv",
             "event_study.csv", "day_audits.csv", "latency_histograms.csv")
COMPUTATION_FILES = ("src/colonylab/latency.py", "src/colonylab/latency_data.py",
                     "src/colonylab/latency_pipeline.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signature(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _independent_case(leader: QuoteSeries, follower: QuoteSeries, side: str,
                      timestamp: int, horizon: int) -> dict:
    """Plain first-passage scan; does not use the threshold tree or core helpers."""
    t = int(timestamp)
    li = int(np.searchsorted(leader.times_ns, t))
    prices = getattr(leader, side)
    direction = 1 if int(prices[li]) > int(prices[li - 1]) else -1
    target = int(prices[li])
    before = int(np.searchsorted(follower.times_ns, t, side="right")) - 1
    if not len(follower.times_ns):
        return {"status": "no_state", "latency_ns": -1, "observed_ns": 0, "terminal_reason": "end"}
    has_state = (before >= 0 and follower.start_ns <= t < follower.end_ns and follower.valid[before])
    fp = getattr(follower, side)
    already = has_state and (int(fp[before]) >= target if direction > 0 else int(fp[before]) <= target)
    end = min(leader.end_ns, follower.end_ns)
    stop = end
    for stream, start in ((leader, li), (follower, before)):
        high = int(np.searchsorted(stream.times_ns, min(t + horizon + 1, end), side="right"))
        for index in range(start + 1, high):
            if not stream.valid[index] or (index > 0 and stream.segments[index] != stream.segments[index - 1]):
                stop = min(stop, int(stream.times_ns[index]))
                break
    observed = min(horizon, max(0, stop - t - 1))
    reason = "horizon" if stop - t - 1 >= horizon else "quality" if stop < end else "end"
    if not has_state:
        return {"status": "no_state", "latency_ns": -1, "observed_ns": 0, "terminal_reason": reason}
    lag = -1
    if not already:
        upper = int(np.searchsorted(follower.times_ns, t + observed, side="right"))
        for index in range(before + 1, upper):
            value = int(fp[index])
            if (value >= target if direction > 0 else value <= target):
                lag = int(follower.times_ns[index]) - t
                break
    return {"status": "already" if already else "behind", "latency_ns": lag,
            "observed_ns": observed, "terminal_reason": reason}


def audit_pair(leader: QuoteSeries, follower: QuoteSeries, side: str,
               events: pd.DataFrame, horizon: int) -> dict:
    """Audit candidate counts and sampled plain-loop matches for every pair/side."""
    price = getattr(leader, side)
    count = 0
    # Independent vector arithmetic for the entire candidate count, then a
    # bounded plain loop for representative first-passage outcomes.
    if len(price) > 1:
        count = int(np.sum(leader.valid[:-1] & leader.valid[1:]
                          & (leader.segments[:-1] == leader.segments[1:])
                          & (price[:-1] != price[1:])
                          & (leader.times_ns[1:] >= leader.start_ns)
                          & (leader.times_ns[1:] < leader.end_ns)))
    if count != len(events):
        raise ValueError("Independent leader-event count differs from latency engine")
    selected = set(np.linspace(0, len(events) - 1, min(8, len(events))).astype(int)) if len(events) else set()
    for column in ("status", "terminal_reason"):
        for _, group in events.groupby(column, sort=False):
            selected.add(int(group.index[0]))
    lag = events.latency_ns.to_numpy()
    for condition in (lag > 0, lag < 0):
        indices = np.flatnonzero(condition)
        if len(indices):
            selected.add(int(indices[0]))
    keys = ("status", "latency_ns", "observed_ns", "terminal_reason")
    for position in sorted(selected):
        row = events.iloc[position]
        expected = _independent_case(leader, follower, side, int(row.t), horizon)
        if any(row[key] != expected[key] for key in keys):
            raise ValueError(f"Independent first-passage mismatch at {int(row.t)}: {expected}")
    return {"cases": len(selected), "mismatches": 0, "leader_events": count}


def _day_identity(task: dict) -> dict:
    return {"study": task["study"], "date": task["date"], "protocol": task["protocol"],
            "computation_code": task["code"],
            "sources": [{"request": entry["request"], "sha256": entry["content_sha256"],
                         "rows": entry["rows"]} for entry in task["entries"]]}


def _day_complete(directory: Path, identity: dict) -> bool:
    marker = directory / "complete.json"
    if not marker.is_file():
        return False
    try:
        stored = json.loads(marker.read_text(encoding="utf-8"))
        return (stored["identity"] == identity
                and all((directory / name).is_file()
                        and sha256_file(directory / name) == digest
                        for name, digest in stored["outputs"].items())
                and set(stored["outputs"]) == set(DAY_FILES))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _process_day(task: dict) -> dict:
    """One day in bounded memory; only numeric/CSV checkpoints are persisted."""
    from .latency import binned_crosscorr, event_study, realized_volatility
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=1)
    started = time.perf_counter()
    protocol = task["protocol"]
    directory = Path(task["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    identity = _day_identity(task)
    if _day_complete(directory, identity):
        return {"study": task["study"], "date": task["date"], "reused": True}
    series, quality = {}, []
    for entry in task["entries"]:
        loaded, rows = load_entry(entry, clocks=protocol["clocks"], max_age_ms=protocol["max_quote_age_ms"])
        dataset = entry["request"]["dataset"]
        for (symbol, clock), quotes in loaded.items():
            series[(dataset, symbol, clock)] = quotes
        quality.extend(rows)
    common = {"study": task["study"], "date": task["date"], "phase": task["phase"]}
    summaries, by_side, vol_rows, correlations, curves, studies, audits, histograms = ([] for _ in range(8))
    horizons = [int(value * 1_000_000) for value in protocol["horizons_ms"]]
    max_horizon = max(horizons)
    bins = np.geomspace(.001, max(protocol["horizons_ms"]) * 1000, 65)
    offsets = np.asarray(protocol["event_study_offsets_us"], dtype=np.int64) * 1000
    venues = task["configuration"]["datasets"]
    for clock in protocol["clocks"]:
        for symbol in task["configuration"]["symbols"]:
            quotes = {venue: series[(venue, symbol, clock)] for venue in venues}
            rv = realized_volatility(quotes, step_ns=1_000_000_000,
                                     window=protocol["volatility_window_seconds"],
                                     min_venues=(len(venues) + 1) // 2)
            for row in rv.to_dict("records"):
                vol_rows.append({**common, "clock": clock, "symbol": symbol, **row})
            vt, vr = rv.t.to_numpy(np.int64), rv.regime.to_numpy()
            for leader in venues:
                for follower in venues:
                    if follower == leader:
                        continue
                    key = {**common, "clock": clock, "symbol": symbol, "leader": leader, "follower": follower}
                    pieces = []
                    for side in ("bid", "ask"):
                        events = reaction_events(quotes[leader], quotes[follower], side,
                                                 max_horizon_ns=max_horizon)
                        check = audit_pair(quotes[leader], quotes[follower], side, events, max_horizon)
                        audits.append({**key, "side": side, **check})
                        for horizon in horizons:
                            by_side.append({**key, "side": side, **summarize_events(events, horizon)})
                        events["side"] = side
                        pieces.append(events)
                    events = pd.concat(pieces, ignore_index=True)
                    events.attrs["max_horizon_ns"] = max_horizon
                    vi = np.searchsorted(vt, events.t.to_numpy(np.int64), side="right") - 1
                    regime = np.full(len(events), "unknown", dtype=object)
                    if len(vr):
                        present = vi >= 0
                        regime[present] = vr[vi[present]]
                    for label in ("all", "calm", "volatile"):
                        selection = events if label == "all" else events.loc[regime == label]
                        selection.attrs["max_horizon_ns"] = max_horizon
                        for horizon in horizons:
                            row = summarize_events(selection, horizon)
                            row["n_excluded"] = row["n_already"] + row["n_no_state"] + row["n_quality_truncated"] + row["n_right_truncated"]
                            summaries.append({**key, "regime": label, **row})
                    h = int(protocol["primary_horizon_ms"] * 1_000_000)
                    matched = ((events.status.to_numpy() == "behind") & (events.observed_ns.to_numpy() >= h)
                               & (events.latency_ns.to_numpy() > 0) & (events.latency_ns.to_numpy() <= h))
                    counts, _ = np.histogram(events.loc[matched, "latency_ns"].to_numpy() / 1000, bins=bins)
                    for low, high, count in zip(bins[:-1], bins[1:], counts):
                        histograms.append({**key, "horizon_ms": protocol["primary_horizon_ms"],
                                           "bin_left_us": low, "bin_right_us": high, "count": int(count)})
                    if leader < follower:
                        cc = binned_crosscorr(quotes[leader], quotes[follower],
                                              bin_ns=protocol["leadlag_bin_us"] * 1000,
                                              max_lag_ns=protocol["leadlag_max_lag_us"] * 1000)
                        finite = cc.loc[np.isfinite(cc["corr"])]
                        corr = cc["corr"].to_numpy()
                        lag = cc.lag_us.to_numpy()
                        positive, negative = np.nansum(corr[lag > 0] ** 2), np.nansum(corr[lag < 0] ** 2)
                        peak = finite.loc[finite["corr"].idxmax()] if len(finite) else None
                        correlations.append({**key, "peak_lag_us": float(peak.lag_us) if peak is not None else np.nan,
                                             "peak_corr": float(peak["corr"]) if peak is not None else np.nan,
                                             "llr": positive / negative if negative > 0 else np.nan,
                                             "positive_lag_means": "follower_after_leader", **cc.attrs})
                        if protocol["event_study_leader"] in (leader, follower):
                            curves.extend({**key, **row} for row in cc.to_dict("records"))
                    if leader == protocol["event_study_leader"]:
                        es = event_study(quotes[leader], quotes[follower], offsets)
                        studies.extend({**key, **row} for row in es.to_dict("records"))
    outputs = (summaries, by_side, [{**common, **row} for row in quality], vol_rows,
               correlations, curves, studies, audits, histograms)
    for filename, rows in zip(DAY_FILES, outputs):
        frame = pd.DataFrame(rows)
        if frame.empty:
            frame = pd.DataFrame(columns=list(common))
        frame.to_csv(directory / filename, index=False)
    _json(directory / "complete.json", {"identity": identity,
          "outputs": {name: sha256_file(directory / name) for name in DAY_FILES},
          "elapsed_seconds": time.perf_counter() - started})
    return {**common, "reused": False, "elapsed_seconds": round(time.perf_counter() - started, 2),
            "pair_side_checks": len(audits), "sampled_cases": sum(row["cases"] for row in audits)}


def _write_report(out_dir: Path, summary: dict, daily: pd.DataFrame, inference: pd.DataFrame) -> None:
    primary = inference.loc[(inference.study == "fresh_confirmation") & (inference.phase == "held_out")
                            & inference.horizon_ms.eq(10) & inference.contrast.eq("latency")
                            & inference.interval_status.eq("ok")]
    display = [column for column in ("clock", "symbol", "leader", "follower", "estimate_us", "n_dates") if column in primary]
    best = primary.sort_values("estimate_us").head(20)[display].round(3).to_html(index=False, escape=True)
    description = html.escape(summary["interpretation"])
    figures = "".join(f'<figure><img src="{name}" alt="{name}"><figcaption>{name}</figcaption></figure>'
                      for name in summary["figures"])
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>ColonyLab venue latency</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;max-width:1150px;margin:40px auto;padding:0 24px;color:#182436}}h1,h2{{line-height:1.2}}img{{max-width:100%}}figure{{margin:28px 0}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:6px;border-bottom:1px solid #ccd4df;text-align:right}}code{{background:#edf1f6;padding:2px 5px}}.note{{background:#edf2f7;padding:18px}}</style>
<h1>Native-event venue latency</h1><p>{description}</p>
<p class="note">Offline analysis of {summary['source_requests']:,} verified requests and {summary['raw_records']:,} records across {summary['dates']} dates. New exploratory analysis of previously studied data. The original study's held-out labels are inherited; these are not new untouched confirmation data.</p>
<h2>How to read the results</h2><p>Latency is time for a follower's bid or ask to reach a leader's new price. Tables separate already-at-price events, unknown prior state, fully observed nonmatches, feed-quality truncation and request-end truncation. Conditional matched medians require the entire chosen horizon observable in both feeds, even for early matches. Restricted mean waiting time also includes nonmatches at the horizon cap.</p>
<p>Symbols and study windows remain separate. Heatmaps summarize equal-date means of daily conditional medians, not a pooled-event median. Volatility contrasts pair calm and volatile periods on the same dates. Five-session date-block intervals are exploratory marginal intervals; sparse pairs and the short October study do not receive primary intervals.</p>
<h2>Figures</h2>{figures}<h2>Twenty smallest supported primary pair estimates</h2>{best}
<h2>Reproducibility</h2><p><a href="summary.json">Summary</a> · <a href="provenance.json">Source and code fingerprints</a> · <a href="protocol.json">Analysis protocol</a> · <a href="daily_latency.csv">Daily latency and censoring</a> · <a href="daily_latency_by_side.csv">Bid/ask sensitivity</a> · <a href="inference.csv">Date-level estimates and intervals</a> · <a href="quote_quality.csv">Quote quality</a> · <a href="leadlag_summary.csv">Lead-lag diagnostics</a> · <a href="event_study.csv">Event studies</a> · <a href="day_audits.csv">Independent sampled first-passage checks</a></p>
<p>Receive timestamps include feed transport; exchange timestamps compare different clocks. Price matching is not a causal transmission estimate. Only saved 10/30-minute MBP-1 windows are analyzed; IEX is absent. Feed flags do not guarantee unreported upstream completeness. The MBO model grid is not used as a substitute for event-level quotes.</p>
<p>Measurement ideas adapted independently from <a href="https://github.com/smar-01/qam-p113-latency/tree/62d6dcd8af9ecd7136096dbcce3eb40371987422">smar-01/qam-p113-latency</a>, with ColonyLab integrity checks, observation barriers and date-level inference.</p></html>'''
    (out_dir / "index.html").write_text(body, encoding="utf-8")


def run_latency(root: Path, protocol_path: Path, out_dir: Path, *, workers: int = 2,
                summarize_only: bool = False, progress=print) -> dict:
    root, out_dir = Path(root).resolve(), Path(out_dir).resolve()
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if protocol.get("offline_only") is not True:
        raise ValueError("The latency analysis requires offline_only=true")
    if protocol.get("volatility_grid_ms") != 1000 or protocol.get("volatility_quantile") != .75:
        raise ValueError("This protocol supports a one-second volatility grid and the 75th-percentile regime cut")
    horizons = protocol.get("horizons_ms", [])
    if (not horizons or horizons != sorted(set(horizons))
            or any(not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0 for value in horizons)
            or protocol.get("primary_horizon_ms") not in horizons):
        raise ValueError("Use unique increasing positive horizons including the primary horizon")
    if not 1 <= workers <= 4:
        raise ValueError("workers must be between one and four")
    code = [{"path": path, "sha256": sha256_file(root / path)} for path in COMPUTATION_FILES]
    all_code = code + [{"path": "src/colonylab/latency_validation.py", "sha256": sha256_file(root / "src/colonylab/latency_validation.py")}]
    out_dir.mkdir(parents=True, exist_ok=True)
    _json(out_dir / "protocol.json", protocol)
    sources, studies, tasks = [], [], []
    expected_dates = {}
    for config_name in protocol["study_configs"]:
        config_path = root / config_name
        progress(f"Verifying saved source bytes and request grid: {config_path.stem}", flush=True)
        config, entries = load_study(config_path, root)
        name = config_path.stem
        acquisition_manifest = str((root / config["out_dir"] / "data_manifest.json").resolve())
        studies.append({"name": name, "configuration": config, "config_path": str(config_path.resolve())})
        for entry in entries:
            sources.append({"study": name, "data_path": str(Path(entry["data_path"]).resolve()),
                            "content_sha256": entry["content_sha256"], "request": entry["request"],
                            "rows": int(entry["rows"]), "acquisition_manifest": acquisition_manifest,
                            "source_hash_verified_against": entry.get("source_hash_verified_against")})
        for phase, date_key in (("training", "training_days"), ("held_out", "held_out_days")):
            expected_dates[f"{name}:{phase}"] = config[date_key]
            for date in config[date_key]:
                tasks.append({"study": name, "date": date, "phase": phase, "configuration": config,
                              "entries": [entry for entry in entries if entry["request"]["start"][:10] == date],
                              "protocol": protocol, "code": code,
                              "directory": str(out_dir / "days" / name / date)})
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "offline_only": True,
                  "protocol_sha256": _signature(protocol), "studies": studies, "sources": sources,
                  "computation_code": all_code, "python": sys.version, "platform": platform.platform(),
                  "raw_row_count_validation": "Every newly prepared DBN is parsed and compared with its complete receipt; resumed day checkpoints are content-bound to the same raw hashes and code."}
    _json(out_dir / "provenance.json", provenance)
    missing = [task for task in tasks if not _day_complete(Path(task["directory"]), _day_identity(task))]
    if summarize_only and missing:
        raise ValueError(f"{len(missing)} day checkpoints are absent or differ from current inputs/code")
    if missing:
        if workers == 1:
            for index, task in enumerate(missing, 1):
                result = _process_day(task)
                progress(f"Completed {index}/{len(missing)}: {result}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                pending = {executor.submit(_process_day, task): task for task in missing}
                for index, future in enumerate(as_completed(pending), 1):
                    result = future.result()
                    progress(f"Completed {index}/{len(missing)}: {result}", flush=True)
    for filename in DAY_FILES:
        frames = [pd.read_csv(Path(task["directory"]) / filename) for task in tasks]
        pd.concat(frames, ignore_index=True).to_csv(out_dir / filename, index=False)
    daily = pd.read_csv(out_dir / "daily_latency.csv")
    inference = infer_latency(daily, min_matches=protocol["minimum_matched_events_per_date"],
                              bootstrap_replicates=protocol["bootstrap_replicates"],
                              block_sessions=protocol["bootstrap_block_sessions"], seed=protocol["bootstrap_seed"],
                              primary_study=protocol["primary_study"], primary_phase=protocol["primary_phase"],
                              min_inference_dates=protocol["minimum_inference_dates"], expected_dates=expected_dates)
    inference.to_csv(out_dir / "inference.csv", index=False)
    figures = render_latency_figures(daily, inference, out_dir)
    audits = pd.read_csv(out_dir / "day_audits.csv")
    primary = daily.loc[daily.study.eq(protocol["primary_study"]) & daily.phase.eq(protocol["primary_phase"])
                        & daily.horizon_ms.eq(protocol["primary_horizon_ms"]) & daily.regime.eq("all")]
    aggregates = []
    for clock, frame in primary.groupby("clock"):
        eligible = int(frame.n_eligible.sum())
        matched = int(frame.n_matched.sum())
        estimates = inference.loc[inference.study.eq(protocol["primary_study"]) & inference.phase.eq(protocol["primary_phase"])
                                 & inference.horizon_ms.eq(protocol["primary_horizon_ms"]) & inference.clock.eq(clock)
                                 & inference.contrast.eq("latency") & inference.interval_status.eq("ok")]
        supported = estimates.estimate_us.dropna()
        aggregates.append({"clock": clock, "eligible_pair_events": eligible, "matched_pair_events": matched,
                           "match_rate": matched / eligible if eligible else None,
                           "median_across_supported_pair_symbol_equal_date_means_us": float(supported.median()) if len(supported) else None,
                           "supported_pair_symbol_tasks": len(supported)})
    summary = {"status": "completed", "analysis_status": protocol["analysis_status"],
               "source_requests": len(sources), "raw_records": sum(row["rows"] for row in sources),
               "dates": len(tasks), "studies": [{"name": study["name"], "dates": len(study["configuration"]["training_days"]) + len(study["configuration"]["held_out_days"]),
                                                   "venues": len(study["configuration"]["datasets"]), "symbols": study["configuration"]["symbols"],
                                                   "minutes_per_date": study["configuration"]["minutes"]} for study in studies],
               "primary_10ms": aggregates, "independent_matching_checks": int(audits.cases.sum()),
               "independent_matching_mismatches": int(audits.mismatches.sum()),
               "figures": [Path(path).name for path in figures],
               "interpretation": "Conditional first-passage quote matching, not causal propagation or pure exchange latency. Compare match rates, quality exclusions, clock sensitivity and horizon sensitivity alongside matched medians.",
               "limitations": protocol["limitations"]}
    _json(out_dir / "summary.json", summary)
    _write_report(out_dir, summary, daily, inference)
    return summary
