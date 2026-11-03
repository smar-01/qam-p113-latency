"""Independently rebuild selected native quote states and complete pair results.

This audit decodes actual DBN sources.  Its pandas normalization and direct
first-passage scans do not call the production quality normalizer, segment
tree, reaction matcher or summary functions.  The production loader is used
only as the object being checked.  All requests remain offline.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def independently_normalize(frame: pd.DataFrame, request: dict, symbol: str, clock: str,
                            max_age_ms: float) -> SimpleNamespace:
    """Separate dataframe reconstruction, retaining size-only refresh states."""
    start, request_end = pd.Timestamp(request["start"]).value, pd.Timestamp(request["end"]).value
    raw = frame.loc[frame.symbol.eq(symbol)].reset_index(drop=True).copy()
    values = raw[clock].to_numpy()
    require(values.dtype.kind in "ui", "Raw native timestamps must be integers.")
    clock_ok = (values >= 0) & (values <= np.iinfo(np.int64).max)
    raw["time"] = np.where(clock_ok, values, 0).astype(np.int64)
    bits = raw["flags"].astype(np.uint64)
    bad = (bits & (32 | 8 | 4)) != 0
    completed = (bits & 128) != 0
    raw["include"] = completed | bad | ~clock_ok
    raw["good"] = (completed & ~bad & clock_ok & raw.bid_px_00.gt(0) &
                   raw.ask_px_00.gt(raw.bid_px_00) & raw.ask_px_00.ne(np.iinfo(np.int64).max) &
                   raw.bid_sz_00.gt(0) & raw.ask_sz_00.gt(0))
    for missing in np.flatnonzero(~clock_ok):
        for neighbor in (missing - 1, missing + 1):
            if 0 <= neighbor < len(raw):
                raw.loc[neighbor, ["good", "include"]] = [False, True]
    observed = raw.loc[raw.include & clock_ok].copy()
    end = min(request_end, int(observed.time.max())) if clock == "ts_event" and len(observed) else request_end
    end = max(start, end)
    # Nullable integer shift preserves nanoseconds at contemporary epochs.
    previous_high = observed.time.astype("Int64").cummax().shift()
    reversals = observed.time.lt(previous_high).fillna(False)
    intervals = list(zip(observed.loc[reversals, "time"].tolist(), previous_high.loc[reversals].tolist()))
    observed = observed.loc[observed.time.ge(start) & observed.time.lt(end)].copy()
    for low, high in intervals:
        observed.loc[observed.time.between(int(low), int(high)), "good"] = False
    observed = observed.sort_values("time", kind="stable")
    state_columns = ["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]
    grouped = observed.groupby("time", sort=True)
    states = grouped[state_columns].first()
    if len(states):
        conflicts = grouped[state_columns].nunique(dropna=False).gt(1).any(axis=1)
        states["valid"] = grouped.good.all() & ~conflicts
    else:
        states["valid"] = pd.Series(dtype=bool)
    states = states.reset_index()
    age_ns = int(max_age_ms * 1_000_000)
    next_time = pd.Series(np.r_[states.time.to_numpy(np.int64)[1:], end], index=states.index, dtype=np.int64)
    deadline = states.time.astype(np.int64) + age_ns
    expires = states.valid & deadline.lt(next_time) & deadline.lt(end)
    if expires.any():
        inserts = pd.DataFrame({"time": deadline.loc[expires].to_numpy(np.int64), "valid": False})
        for column in state_columns:
            inserts[column] = 0
        states = pd.concat([states, inserts], ignore_index=True).sort_values("time", kind="stable")
    states["valid"] = states.valid.astype(bool)
    states.loc[~states.valid, ["bid_px_00", "ask_px_00"]] = 0
    states["segment"] = (~states.valid).cumsum()
    return SimpleNamespace(times_ns=states.time.to_numpy(np.int64),
                           bid=states.bid_px_00.to_numpy(np.int64), ask=states.ask_px_00.to_numpy(np.int64),
                           valid=states.valid.to_numpy(bool), segments=states.segment.to_numpy(np.int64),
                           start_ns=int(start), end_ns=int(end))


def assert_state_equivalence(expected, actual, context: str) -> int:
    """Compare all state boundaries, even when one representation is compressed."""
    require(expected.start_ns == actual.start_ns and expected.end_ns == actual.end_ns,
            f"Raw-state observation bounds differ: {context}")
    points = np.unique(np.r_[expected.times_ns, actual.times_ns, expected.start_ns, expected.end_ns - 1])
    left = np.searchsorted(expected.times_ns, points, side="right") - 1
    right = np.searchsorted(actual.times_ns, points, side="right") - 1
    require(np.array_equal(left >= 0, right >= 0), f"Raw-state initial support differs: {context}")
    present = (left >= 0) & (right >= 0)
    for field in ("bid", "ask", "valid", "segments"):
        require(np.array_equal(getattr(expected, field)[left[present]], getattr(actual, field)[right[present]]),
                f"Independent raw-state {field} reconstruction differs: {context}")
    return len(points)


def independently_match(leader, follower, side: str, horizon_ns: int) -> dict[str, np.ndarray]:
    """Direct event-by-event threshold scans, with no production matcher calls."""
    price = getattr(leader, side)
    candidates = (leader.valid[:-1] & leader.valid[1:] &
                  (leader.segments[:-1] == leader.segments[1:]) & (price[:-1] != price[1:]) &
                  (leader.times_ns[1:] >= leader.start_ns) & (leader.times_ns[1:] < leader.end_ns))
    positions = np.flatnonzero(candidates) + 1
    times = leader.times_ns[positions]
    target = price[positions]
    signs = np.where(price[positions] > price[positions - 1], 1, -1)
    state = np.full(len(times), "no_state", dtype=object)
    lag = np.full(len(times), -1, dtype=np.int64)
    observed = np.zeros(len(times), dtype=np.int64)
    reason = np.full(len(times), "end", dtype=object)
    if not len(follower.times_ns):
        return {"status": state, "latency_ns": lag, "observed_ns": observed, "terminal_reason": reason}
    predecessor = np.searchsorted(follower.times_ns, times, side="right") - 1
    supported = ((predecessor >= 0) & (times >= follower.start_ns) & (times < follower.end_ns))
    supported &= follower.valid[np.maximum(predecessor, 0)]
    fp = getattr(follower, side)
    equal_or_ahead = np.where(signs > 0, fp[np.maximum(predecessor, 0)] >= target,
                              fp[np.maximum(predecessor, 0)] <= target)
    state[supported & equal_or_ahead] = "already"
    state[supported & ~equal_or_ahead] = "behind"
    end = min(leader.end_ns, follower.end_ns)
    stop = np.full(len(times), end, dtype=np.int64)
    for stream in (leader, follower):
        barrier = ~stream.valid
        if len(barrier) > 1:
            barrier = barrier | np.r_[False, np.diff(stream.segments) != 0]
        boundaries = np.r_[stream.times_ns[barrier], stream.end_ns]
        # A boundary exactly at the event already enters predecessor validity;
        # the first strictly later discontinuity terminates follow-up.
        bi = np.searchsorted(boundaries, times, side="right")
        stop = np.minimum(stop, boundaries[np.minimum(bi, len(boundaries) - 1)])
    available = np.maximum(0, stop - times - 1)
    observed[supported] = np.minimum(available[supported], horizon_ns)
    reason[:] = np.where(stop < end, "quality", "end")
    reason[available >= horizon_ns] = "horizon"
    for index in np.flatnonzero(state == "behind"):
        first = int(predecessor[index]) + 1
        last = int(np.searchsorted(follower.times_ns, times[index] + observed[index], side="right"))
        future_prices = fp[first:last]
        hit = np.flatnonzero(future_prices >= target[index] if signs[index] > 0 else future_prices <= target[index])
        if len(hit):
            lag[index] = follower.times_ns[first + hit[0]] - times[index]
    return {"status": state, "latency_ns": lag, "observed_ns": observed, "terminal_reason": reason}


def independently_summarize(events: dict[str, np.ndarray], horizon_ns: int) -> dict:
    state, lag = events["status"], events["latency_ns"]
    behind = state == "behind"
    complete = behind & (events["observed_ns"] >= horizon_ns)
    known_match = behind & (lag > 0) & (lag <= horizon_ns)
    match = complete & known_match
    partial = behind & ~complete
    reason = events["terminal_reason"]
    n, m = int(complete.sum()), int(match.sum())
    values = lag[match] / 1000
    result = {
        "n_events": len(state), "n_already": int((state == "already").sum()),
        "n_no_state": int((state == "no_state").sum()), "n_behind": int(behind.sum()),
        "n_complete_behind": n, "n_eligible": n, "n_matched": m,
        "n_observed_matches": int(known_match.sum()), "known_matches_truncated": int((known_match & partial).sum()),
        "n_censored": n - m, "n_quality_truncated": int((partial & (reason == "quality")).sum()),
        "n_right_truncated": int((partial & (reason == "end")).sum()),
        "n_quality_censored": int((partial & ~known_match & (reason == "quality")).sum()),
        "n_end_censored": int((partial & ~known_match & (reason == "end")).sum()),
        "sum_latency_ns": int(lag[match].sum()),
        "restricted_wait_sum_ns": int(lag[match].sum()) + (n - m) * int(horizon_ns),
        "match_rate": m / n if n else np.nan,
        "mean_us": float(values.mean()) if m else np.nan,
        "p10_us": float(np.quantile(values, .1)) if m else np.nan,
        "median_us": float(np.quantile(values, .5)) if m else np.nan,
        "p90_us": float(np.quantile(values, .9)) if m else np.nan,
    }
    result["restricted_mean_wait_us"] = result["restricted_wait_sum_ns"] / n / 1000 if n else np.nan
    return result


def compare_summary(expected: dict, actual: pd.Series, context: str) -> None:
    for field, value in expected.items():
        require(field in actual, f"Missing summary field {field}: {context}")
        observed = actual[field]
        if isinstance(value, int):
            require(int(observed) == value, f"Independent raw {field} differs: {context}: {observed} != {value}")
        elif np.isnan(value):
            require(pd.isna(observed), f"Expected undefined raw summary {field}: {context}")
        else:
            require(np.isclose(float(observed), value, rtol=1e-11, atol=1e-8),
                    f"Independent raw {field} differs: {context}: {observed} != {value}")


FACT_KEYS = ["study", "date", "symbol", "clock", "leader", "follower", "horizon_ms", "side"]
FACT_COUNTS = {
    "n_events", "n_already", "n_no_state", "n_behind", "n_complete_behind", "n_eligible", "n_matched",
    "n_observed_matches", "known_matches_truncated", "n_censored", "n_quality_truncated", "n_right_truncated",
    "n_quality_censored", "n_end_censored", "sum_latency_ns", "restricted_wait_sum_ns",
}
FACT_METRICS = [*sorted(FACT_COUNTS), "match_rate", "mean_us", "p10_us", "median_us", "p90_us", "restricted_mean_wait_us"]
PREPARED_CSV = "raw_audit_prepared_summaries.csv"
PREPARED_JSON = "raw_audit_prepared.json"


def selected_date_grid(provenance: dict) -> dict[str, list[str]]:
    return {study["name"]: sorted({study["configuration"]["training_days"][0],
                                   study["configuration"]["training_days"][-1],
                                   study["configuration"]["held_out_days"][0],
                                   study["configuration"]["held_out_days"][-1]})
            for study in provenance["studies"]}


def prepared_identity(out_dir: Path, root: Path, protocol: dict, provenance: dict) -> dict:
    """Fresh byte identities; never uses file modification times as evidence."""
    dates = selected_date_grid(provenance)
    recorded_code = {row["path"]: row["sha256"] for row in provenance["computation_code"]}
    code = {}
    for relative in ("src/colonylab/latency.py", "src/colonylab/latency_data.py"):
        current = sha256(root / relative)
        require(recorded_code.get(relative) == current,
                f"Current computation code differs from analysis provenance: {relative}")
        code[relative] = current
    code["scripts/audit_latency_raw.py"] = sha256(Path(__file__))
    sources = []
    for row in sorted(provenance["sources"], key=lambda x: (x["study"], x["request"]["start"], x["request"]["dataset"])):
        if row["request"]["start"][:10] not in dates[row["study"]]:
            continue
        digest = sha256(Path(row["data_path"]))
        require(digest == row["content_sha256"], f"Raw source changed after analysis: {row['data_path']}")
        sources.append({"study": row["study"], "request": row["request"], "rows": row["rows"],
                        "data_path": row["data_path"], "content_sha256": digest})
    require(len(sources) == sum(len(dates[study["name"]]) * len(study["configuration"]["datasets"])
                               for study in provenance["studies"]), "Prepared raw audit source grid is incomplete.")
    return {"version": 1, "protocol_sha256": sha256(out_dir / "protocol.json"), "protocol": protocol,
            "studies": [{"name": study["name"], "configuration": study["configuration"]}
                        for study in provenance["studies"]],
            "selected_dates": dates, "code": code, "sources": sources}


def write_prepared(out_dir: Path, identity: dict, facts: pd.DataFrame, statistics: dict) -> dict:
    """Persist audit-owned numeric facts, bound to their exact byte identities."""
    csv_path = out_dir / PREPARED_CSV
    temporary_csv = csv_path.with_suffix(".csv.partial")
    facts.to_csv(temporary_csv, index=False)
    temporary_csv.replace(csv_path)
    result = {"status": "raw_facts_prepared", "created_utc": datetime.now(timezone.utc).isoformat(),
              "identity": identity, "facts_file": PREPARED_CSV, "facts_sha256": sha256(csv_path),
              "fact_rows": len(facts), "statistics": statistics,
              "note": "Independent raw-state and complete first-passage facts are ready; final result comparison remains required."}
    path = out_dir / PREPARED_JSON
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
    return result


def read_prepared(out_dir: Path, current_identity: dict) -> tuple[dict, pd.DataFrame]:
    saved = json.loads((out_dir / PREPARED_JSON).read_text(encoding="utf-8"))
    require(saved.get("status") == "raw_facts_prepared" and saved.get("identity") == current_identity,
            "Prepared raw audit identity differs from current sources, code or protocol; rebuild it.")
    require(saved.get("facts_file") == PREPARED_CSV and sha256(out_dir / PREPARED_CSV) == saved.get("facts_sha256"),
            "Prepared raw audit CSV hash differs from its saved identity.")
    facts = pd.read_csv(out_dir / PREPARED_CSV)
    require(len(facts) == saved["fact_rows"], "Prepared raw audit row count differs from its saved identity.")
    return saved, facts


def verify_prepared(out_dir: Path, *, root: Path = ROOT, progress=print) -> dict:
    """Rehash source bytes and compare prepared raw facts without DBN reparse."""
    started = time.perf_counter()
    out_dir, root = Path(out_dir).resolve(), Path(root).resolve()
    protocol = json.loads((out_dir / "protocol.json").read_text(encoding="utf-8"))
    provenance = json.loads((out_dir / "provenance.json").read_text(encoding="utf-8"))
    identity = prepared_identity(out_dir, root, protocol, provenance)
    saved, facts = read_prepared(out_dir, identity)
    require(set(FACT_KEYS + FACT_METRICS) == set(facts), "Prepared raw audit columns differ from the audit schema.")
    require(not facts.duplicated(FACT_KEYS).any(), "Prepared raw audit contains duplicate summary keys.")
    expected_grid = {
        (study["name"], day, symbol, clock, leader, follower, horizon, side)
        for study in provenance["studies"] for day in identity["selected_dates"][study["name"]]
        for symbol in study["configuration"]["symbols"] for clock in protocol["clocks"]
        for leader, follower in (("XNAS.ITCH", "BATS.PITCH"), ("BATS.PITCH", "XNAS.ITCH"))
        for horizon in protocol["horizons_ms"] for side in ("pooled", "bid", "ask")
    }
    require(set(facts[FACT_KEYS].itertuples(index=False, name=None)) == expected_grid,
            "Prepared raw audit summary grid has missing or unexpected cases.")
    daily = pd.read_csv(out_dir / "daily_latency.csv")
    daily = daily.loc[daily.regime.eq("all")].copy()
    keys = FACT_KEYS[:-1]
    by_side = pd.read_csv(out_dir / "daily_latency_by_side.csv")
    require(not daily.duplicated(keys).any() and not by_side.duplicated(FACT_KEYS).any(),
            "Saved analysis has duplicate raw audit target rows.")
    daily, by_side = daily.set_index(keys), by_side.set_index(FACT_KEYS)
    checks = []
    for row in facts.to_dict("records"):
        key = tuple(row[field] for field in FACT_KEYS)
        target, target_key = (daily, key[:-1]) if row["side"] == "pooled" else (by_side, key)
        require(target_key in target.index, f"Saved analysis lacks prepared raw audit case: {key}")
        expected = {field: int(row[field]) if field in FACT_COUNTS else float(row[field]) for field in FACT_METRICS}
        compare_summary(expected, target.loc[target_key], str(key))
        if row["side"] == "pooled":
            checks.append({**{field: row[field] for field in keys}, "independent_events": row["n_events"],
                           "independent_matches": row["n_matched"], "mismatches": 0})
    pd.DataFrame(checks).to_csv(out_dir / "raw_audit_cases.csv", index=False)
    result = {
        "status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(), "offline_only": True,
        **saved["statistics"], "selected_dates": identity["selected_dates"],
        "full_pair_horizon_summaries_checked": len(checks), "full_bid_ask_summaries_checked": len(facts) - len(checks),
        "mismatches": 0, "sources_freshly_rehashed_during_verification": len(identity["sources"]),
        "prepared_facts_file": PREPARED_CSV, "prepared_facts_sha256": saved["facts_sha256"],
        "audit_script_sha256": sha256(Path(__file__)), "protocol_sha256": sha256(out_dir / "protocol.json"),
        "summary_sha256": sha256(out_dir / "summary.json"),
        "verification_duration_seconds": round(time.perf_counter() - started, 3),
        "scope": "Independent selected-date raw normalization and complete Nasdaq/BZX first-passage results; broader all-source/hash/arithmetic/date-bootstrap audit is separate.",
    }
    (out_dir / "raw_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    progress(f"Prepared independent raw audit matches all {len(facts)} saved pooled/bid/ask summaries.", flush=True)
    return result


def audit(out_dir: Path, *, root: Path = ROOT, progress=print, prepare_only: bool = False) -> dict:
    import databento as db
    from colonylab.latency_data import load_entry

    started = time.perf_counter()
    out_dir, root = Path(out_dir).resolve(), Path(root).resolve()
    protocol = json.loads((out_dir / "protocol.json").read_text(encoding="utf-8"))
    provenance = json.loads((out_dir / "provenance.json").read_text(encoding="utf-8"))
    identity = prepared_identity(out_dir, root, protocol, provenance)
    sources = {(row["study"], row["request"]["start"][:10], row["request"]["dataset"]): row
               for row in provenance["sources"]}
    facts, source_rows, state_points, quote_series = [], 0, 0, 0
    selected_dates = {}
    for study in provenance["studies"]:
        name, config = study["name"], study["configuration"]
        dates = sorted({config["training_days"][0], config["training_days"][-1],
                        config["held_out_days"][0], config["held_out_days"][-1]})
        selected_dates[name] = dates
        manifest = json.loads((root / config["out_dir"] / "data_manifest.json").read_text(encoding="utf-8"))
        entries = {(row["request"]["start"][:10], row["request"]["dataset"]): row for row in manifest["requests"]}
        for day in dates:
            independent = {}
            for dataset in config["datasets"]:
                source = sources[(name, day, dataset)]
                entry = dict(entries[(day, dataset)], data_path=source["data_path"],
                             content_sha256=source["content_sha256"])
                request = source["request"]
                path = Path(source["data_path"])
                require(sha256(path) == source["content_sha256"], f"Raw source changed after analysis: {path}")
                store = db.DBNStore.from_file(path)
                metadata = store.metadata
                require(str(metadata.schema) == "mbp-1" and str(metadata.dataset) == dataset and
                        int(metadata.start) == pd.Timestamp(request["start"]).value and
                        int(metadata.end) == pd.Timestamp(request["end"]).value and
                        set(metadata.symbols) == set(config["symbols"]), f"Raw metadata differs: {path}")
                frame = store.to_df(price_type="fixed", pretty_ts=False, map_symbols=True).reset_index()
                require(len(frame) == entry["rows"] and 0 < len(frame) < request["limit"], f"Raw count mismatch: {path}")
                require(set(frame.symbol.unique()) == set(config["symbols"]), f"Raw symbol mismatch: {path}")
                canonical, _ = load_entry(entry, clocks=protocol["clocks"], max_age_ms=protocol["max_quote_age_ms"])
                source_rows += len(frame)
                for symbol in config["symbols"]:
                    for clock in protocol["clocks"]:
                        expected = independently_normalize(frame, request, symbol, clock, protocol["max_quote_age_ms"])
                        state_points += assert_state_equivalence(expected, canonical[(symbol, clock)],
                                                                 f"{name}/{day}/{dataset}/{symbol}/{clock}")
                        quote_series += 1
                        independent[(dataset, symbol, clock)] = expected
            pairs = [("XNAS.ITCH", "BATS.PITCH"), ("BATS.PITCH", "XNAS.ITCH")]
            pairs = [(a, b) for a, b in pairs if a in config["datasets"] and b in config["datasets"]]
            require(len(pairs) == 2, "Raw audit requires the saved Nasdaq/BZX overlap.")
            for symbol in config["symbols"]:
                for clock in protocol["clocks"]:
                    for leader, follower in pairs:
                        records = {}
                        for side in ("bid", "ask"):
                            records[side] = independently_match(independent[(leader, symbol, clock)],
                                                                 independent[(follower, symbol, clock)], side,
                                                                 int(max(protocol["horizons_ms"]) * 1_000_000))
                        pooled = {key: np.r_[records["bid"][key], records["ask"][key]] for key in records["bid"]}
                        for horizon in protocol["horizons_ms"]:
                            for side, events in (("pooled", pooled), ("bid", records["bid"]), ("ask", records["ask"])):
                                facts.append({"study": name, "date": day, "symbol": symbol, "clock": clock,
                                              "leader": leader, "follower": follower, "horizon_ms": horizon, "side": side,
                                              **independently_summarize(events, int(horizon * 1_000_000))})
            progress(f"Independent raw-state and full first-passage facts prepared: {name} {day}", flush=True)
    statistics = {
        "raw_requests_checked": sum(len(study["configuration"]["datasets"]) *
                                    len(selected_dates[study["name"]]) for study in provenance["studies"]),
        "raw_records_checked": source_rows, "quote_series_independently_rebuilt": quote_series,
        "state_boundary_points_compared": state_points,
        "preparation_duration_seconds": round(time.perf_counter() - started, 3),
    }
    prepared = write_prepared(out_dir, identity, pd.DataFrame(facts), statistics)
    return prepared if prepare_only else verify_prepared(out_dir, root=root, progress=progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "latency")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="Independently rebuild raw facts before analysis outputs finish")
    mode.add_argument("--verify-prepared", action="store_true", help="Rehash sources and compare prepared raw facts without re-decoding DBNs")
    args = parser.parse_args()
    result = verify_prepared(args.out) if args.verify_prepared else audit(args.out, prepare_only=args.prepare_only)
    if args.prepare_only:
        result = {key: value for key, value in result.items() if key != "identity"}
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
