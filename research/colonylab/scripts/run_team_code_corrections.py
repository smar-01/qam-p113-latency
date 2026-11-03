"""Controlled offline original/fixed team-code experiment on identical saved quotes.

The teammate checkout is read only. An immutable source snapshot and a corrected
local copy are compared on the ten-date, five-venue saved MBP-1 study. Numerical
changes, observation-support changes and ColonyLab quality policies are separate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import difflib
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from colonylab.latency_data import load_study

ORIGINAL = ROOT / "comparisons" / "qam_p113_baseline"
CORRECTED = ROOT / "comparisons" / "qam_p113_corrected"
VENUE_CODES = {"XNAS.ITCH": "XNAS", "XNYS.PILLAR": "XNYS", "ARCX.PILLAR": "ARCX",
               "BATS.PITCH": "BATS", "EDGX.PITCH": "EDGX"}
DAY_FILES = ["midpoint_diagnostics.csv", "reaction_comparison.csv", "event_study_comparison.csv",
             "leadlag_comparison.csv", "fullcurves.csv", "regime_series.csv"]
USED_FILES = ["p113/load.py", "p113/leadlag.py", "p113/latency.py", "p113/regimes.py", "p113/venues.py", "p113/__init__.py"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".partial")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def module(name, path, package=False):
    spec = importlib.util.spec_from_file_location(name, path,
             submodule_search_locations=[str(Path(path).parent)] if package else None)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def engines(snapshot):
    original = {name: module("_original_team_" + name, Path(snapshot) / "p113" / (name + ".py"))
                for name in ("load", "latency", "leadlag", "regimes")}
    module("_corrected_team", CORRECTED / "p113" / "__init__.py", package=True)
    corrected = {name: module("_corrected_team." + name, CORRECTED / "p113" / (name + ".py"))
                 for name in ("load", "latency", "leadlag", "regimes")}
    return original, corrected


def reaction_summary(events):
    matched = events.loc[events.status.eq("matched"), "latency_ns"] / 1000
    behind = int((~events.status.eq("already")).sum())
    return {"n_events": len(events), "n_behind": behind, "n_matched": len(matched),
            "median_us": float(matched.median()) if len(matched) else np.nan,
            "match_rate": len(matched) / behind if behind else np.nan}


def midpoint_facts(q, clock, fixed):
    ordered = q.sort_values(clock, kind="stable")
    bid, ask = ordered.bid_raw.to_numpy(np.int64), ordered.ask_raw.to_numpy(np.int64)
    valid = (bid != np.iinfo(np.int64).max) & (ask != np.iinfo(np.int64).max)
    twice = np.where(valid, bid, 0) + np.where(valid, ask, 0)
    differences = np.diff(((ordered.bid + ordered.ask) / 2).to_numpy(float))
    moves = np.isfinite(differences) & (differences != 0)
    adjacent = valid[:-1] & valid[1:]
    exact = adjacent & (np.diff(twice) != 0)
    phantom = adjacent & moves & (np.diff(twice) == 0)
    _, corrected = fixed.mid_changes_integer(q, clock)
    if int(exact.sum()) != len(corrected):
        raise ValueError("Independent integer midpoint-event count does not match corrected code")
    return {"float_events": int(moves.sum()), "exact_events": int(exact.sum()),
            "phantom_events": int(phantom.sum()), "corrected_events": len(corrected),
            "min_abs_float_move": float(np.min(np.abs(differences[moves]))) if moves.any() else np.nan}


def peak(curve):
    finite = curve.loc[np.isfinite(curve["corr"])]
    if not len(finite):
        return {"peak_lag_us": np.nan, "peak_corr": np.nan, "llr": np.nan}
    row = finite.loc[finite["corr"].idxmax()]
    positive = float(np.square(finite.loc[finite.lag_us.gt(0), "corr"]).sum())
    negative = float(np.square(finite.loc[finite.lag_us.lt(0), "corr"]).sum())
    return {"peak_lag_us": float(row.lag_us), "peak_corr": float(row["corr"]),
            "llr": positive / negative if negative > 0 else np.nan}


def run_day(task):
    import databento as db
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    start_time = time.perf_counter()
    directory = Path(task["directory"])
    identity = task["identity"]
    marker = directory / "complete.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["identity"] == identity and all(sha(directory / name) == saved["files"][name] for name in DAY_FILES):
            return {"day": task["day"], "reused": True}
    old, fixed = engines(task["snapshot"])
    raw_parts = []
    for entry in task["entries"]:
        if sha(entry["data_path"]) != entry["content_sha256"]:
            raise ValueError("Raw data changed after source verification")
        store = db.DBNStore.from_file(entry["data_path"])
        raw = fixed["load"].from_store(store, quality=False)
        if len(raw) != entry["rows"]:
            raise ValueError("Decoded source count differs from exact acquisition receipt")
        raw_parts.append(raw)
    raw = pd.concat(raw_parts, ignore_index=True)
    request = task["entries"][0]["request"]
    raw.attrs.update(requested_start=request["start"], requested_end=request["end"],
                     requested_start_ns=pd.Timestamp(request["start"]).value,
                     requested_end_ns=pd.Timestamp(request["end"]).value, price_changes_only=False)
    # Exactly the original starter's price-event selection is shared by all
    # numerical variants, including its receive-order selection before event-clock sorting.
    quotes = old["load"].bbo_changes(raw)
    quotes.attrs.update(raw.attrs, price_changes_only=True)
    # Metadata never silently introduces flags/tie/freshness policies into the
    # numerical experiment. Those belong to the separate qualified comparison.
    quotes = quotes.drop(columns=[name for name in ("valid", "segment") if name in quotes])
    quotes.attrs.update(raw.attrs, price_changes_only=True)
    shared_columns = ["ts_recv", "ts_event", "venue", "symbol", "bid", "ask", "bid_raw", "ask_raw"]
    quote_hash = hashlib.sha256(pd.util.hash_pandas_object(quotes[shared_columns], index=False).to_numpy().tobytes()).hexdigest()
    diagnostics, reactions, event_rows, peaks, curves, regimes = ([] for _ in range(6))
    protocol = task["protocol"]
    offsets = np.asarray(protocol["event_study_offsets_us"], np.int64) * 1000
    horizon = int(protocol["horizon_ms"] * 1_000_000)
    original_rv = old["regimes"].label(old["regimes"].realized_vol(quotes), q=.75)
    for clock in protocol["clocks"]:
        for row in original_rv.to_dict("records"):
            regimes.append({"day": task["day"], "clock": clock, "rv_clock": "ts_recv", "stage": "original", **row})
        corrected_rv = fixed["regimes"].label(fixed["regimes"].realized_vol(raw, clock=clock), q=.75)
        for row in corrected_rv.to_dict("records"):
            regimes.append({"day": task["day"], "clock": clock, "rv_clock": clock, "stage": "quality_corrected", **row})
        for symbol, one in quotes.groupby("symbol", sort=True):
            one.attrs.update(quotes.attrs)
            grouped = {venue: group for venue, group in one.groupby("venue", sort=True)}
            for venue, group in grouped.items():
                facts = midpoint_facts(group, clock, fixed["leadlag"])
                diagnostics.append({"day": task["day"], "clock": clock, "symbol": symbol, "venue": venue, **facts})
            for leader, leader_quotes in grouped.items():
                for follower, follower_quotes in grouped.items():
                    if leader == follower:
                        continue
                    baseline, numerical = [], []
                    for side in ("bid", "ask"):
                        a = old["latency"].reaction_latency(leader_quotes, follower_quotes, side, clock, horizon)
                        b = fixed["latency"].reaction_latency(leader_quotes, follower_quotes, side, clock, horizon)
                        pd.testing.assert_frame_equal(a, b, check_exact=True)
                        baseline.append(a)
                        numerical.append(b)
                    a, b = reaction_summary(pd.concat(baseline)), reaction_summary(pd.concat(numerical))
                    reactions.append({"day": task["day"], "clock": clock, "symbol": symbol,
                                      "leader": leader, "follower": follower, "exact_equal": True,
                                      **{"baseline_" + key: value for key, value in a.items()},
                                      **{"numerical_" + key: value for key, value in b.items()}})
            venues = sorted(grouped)
            original_changes = {venue: old["leadlag"].mid_changes(grouped[venue], clock) for venue in venues}
            integer_changes = {venue: fixed["leadlag"].mid_changes(grouped[venue], clock) for venue in venues}
            for ix, x in enumerate(venues):
                for y in venues[ix + 1:]:
                    for stage in ("original", "integer_only", "numerical_only"):
                        changes = original_changes if stage == "original" else integer_changes
                        method = fixed["leadlag"].crosscorr if stage == "numerical_only" else old["leadlag"].crosscorr
                        curve = method(*changes[x], *changes[y], dt_ns=protocol["leadlag_bin_us"] * 1000,
                                       max_lag_ns=protocol["leadlag_max_lag_us"] * 1000)
                        key = {"day": task["day"], "clock": clock, "symbol": symbol, "x": x, "y": y, "stage": stage}
                        peaks.append({**key, **peak(curve)})
                        curves.extend({**key, **row} for row in curve.to_dict("records"))
            for stage in ("original", "numerical_only", "support_corrected"):
                method = old["leadlag"].event_study if stage == "original" else fixed["leadlag"].event_study
                kwargs = {} if stage == "original" else {"support": "legacy" if stage == "numerical_only" else "complete"}
                table = method(quotes, "XNAS", symbol, offsets, clock=clock, **kwargs)
                event_rows.extend({"day": task["day"], "clock": clock, "symbol": symbol,
                                   "leader": "XNAS", "stage": stage, **row} for row in table.to_dict("records"))
    after = hashlib.sha256(pd.util.hash_pandas_object(quotes[shared_columns], index=False).to_numpy().tobytes()).hexdigest()
    if after != quote_hash:
        raise ValueError("An analysis variant mutated the shared input quotes")
    tables = [diagnostics, reactions, event_rows, peaks, curves, regimes]
    for name, rows in zip(DAY_FILES, tables):
        pd.DataFrame(rows).to_csv(directory / name, index=False)
    dump(marker, {"identity": identity, "files": {name: sha(directory / name) for name in DAY_FILES},
                  "shared_quote_sha256_before": quote_hash, "shared_quote_sha256_after": after,
                  "quote_rows": len(quotes), "raw_rows": len(raw)})
    return {"day": task["day"], "reused": False, "seconds": round(time.perf_counter() - start_time, 2),
            "raw_rows": len(raw), "quote_rows": len(quotes), "phantom_moves": sum(row["phantom_events"] for row in diagnostics)}


def add_quality_comparison(root, out, provenance):
    previous = root / "out" / "latency"
    audit = json.loads((previous / "result_audit.json").read_text())
    if audit["status"] != "passed":
        raise ValueError("The prior quality-controlled source/result audit must pass")
    saved = json.loads((previous / "provenance.json").read_text())
    inventory = {(row["request"]["dataset"], row["request"]["start"]): row for row in saved["sources"] if row["study"] == "wide_network"}
    for source in provenance["sources"]:
        match = inventory[(source["request"]["dataset"], source["request"]["start"])]
        if (source["request"] != match["request"] or source["content_sha256"] != match["content_sha256"]):
            raise ValueError("Quality comparison does not use the same exact source requests")
    for code in saved["computation_code"]:
        if sha(root / code["path"]) != code["sha256"]:
            raise ValueError("Previously audited quality code has changed")
    quality = pd.read_csv(previous / "daily_latency.csv")
    quality = quality.loc[quality.study.eq("wide_network") & quality.regime.eq("all") & quality.horizon_ms.eq(10)].copy()
    quality = quality.rename(columns={"date": "day"})
    quality["leader"] = quality.leader.map(VENUE_CODES)
    quality["follower"] = quality.follower.map(VENUE_CODES)
    original = pd.read_csv(out / "reaction_comparison.csv")
    compared = original.merge(quality, on=["day", "clock", "symbol", "leader", "follower"], validate="one_to_one")
    if len(compared) != len(original):
        raise ValueError("Quality comparison is missing original task rows")
    compared.to_csv(out / "quality_stage_comparison.csv", index=False)
    return {"kind": "Previously audited ColonyLab complete-horizon quality results on the exact same sources",
            "rows": len(compared), "input_daily_sha256": sha(previous / "daily_latency.csv"),
            "input_audit_sha256": sha(previous / "result_audit.json"), "source_identity_equal": True}


def summarize(out):
    midpoint = pd.read_csv(out / "midpoint_diagnostics.csv")
    reaction = pd.read_csv(out / "reaction_comparison.csv")
    event = pd.read_csv(out / "event_study_comparison.csv")
    cc = pd.read_csv(out / "leadlag_comparison.csv")
    ratios = []
    for (stage, symbol), group in event.groupby(["stage", "symbol"]):
        finite = group.share.dropna()
        ratios.append({"stage": stage, "symbol": symbol,
                       "maximum_absolute_mean_share": float(finite.abs().max()) if len(finite) else None,
                       "rows": len(group), "finite_rows": len(finite), "minimum_event_count": int(group.n.min()),
                       "maximum_event_count": int(group.n.max())})
    comparison = cc.loc[cc.stage.eq("original")].merge(cc.loc[cc.stage.eq("numerical_only")],
        on=["day", "clock", "symbol", "x", "y"], suffixes=("_original", "_fixed"), validate="one_to_one")
    changes = comparison.peak_lag_us_fixed - comparison.peak_lag_us_original
    max_delta = event.loc[event.stage.eq("original")].merge(event.loc[event.stage.eq("numerical_only")],
        on=["day", "clock", "symbol", "leader", "venue", "offset_us"], suffixes=("_original", "_fixed"), validate="one_to_one")
    delta = max_delta.share_fixed - max_delta.share_original
    result = {"status": "completed", "source_requests": 50, "raw_records": None,
              "midpoint_series_checked": len(midpoint), "float_midpoint_events": int(midpoint.float_events.sum()),
              "exact_midpoint_events": int(midpoint.exact_events.sum()), "float_only_phantom_events": int(midpoint.phantom_events.sum()),
              "reaction_tasks_compared": len(reaction), "reaction_outputs_identical": bool(reaction.exact_equal.all()),
              "event_study_diagnostics": ratios,
              "maximum_absolute_numerical_event_study_change": float(delta.abs().max()) if delta.notna().any() else None,
              "leadlag_pair_tasks": len(comparison), "leadlag_peak_lag_changed_tasks": int(changes.ne(0).sum()),
              "leadlag_peak_lag_change_median_us": float(changes.abs().median()),
              "leadlag_peak_lag_change_max_us": float(changes.abs().max()),
              "maximum_corrected_peak_crossproduct": float(cc.loc[cc.stage.eq("numerical_only"), "peak_corr"].max()),
              "interpretation": "Exact same quotes isolate midpoint arithmetic and fixed-bin cross-product changes; full-support and quality policy changes are reported separately. Missing original full-day raw files prevent rerunning the published three-day results or IEX.",
              "no_trading_or_causal_claim": True}
    provenance = json.loads((out / "provenance.json").read_text())
    result["source_requests"] = len(provenance["sources"])
    result["raw_records"] = sum(row["rows"] for row in provenance["sources"])
    dump(out / "summary.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "team_code_corrections")
    parser.add_argument("--workers", type=int, choices=[1, 2, 3, 4], default=2)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out / "original_source"
    for name in [*USED_FILES, "analyze.py", "pull.py"]:
        destination = snapshot / name
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ORIGINAL / name, destination)
    config, entries = load_study(ROOT / "wide_network.toml", ROOT)
    offsets = np.r_[-100, 0, np.unique(np.logspace(0, 4, 25).astype(int))].astype(int).tolist()
    protocol = {"version": 1, "configuration": config, "source_config": "wide_network.toml",
                "clocks": ["ts_recv", "ts_event"], "horizon_ms": 10,
                "venues": sorted(VENUE_CODES.values()), "event_study_offsets_us": offsets,
                "leadlag_bin_us": 10, "leadlag_max_lag_us": 2000,
                "stages": {"original": "Unmodified source on legacy selected price events",
                           "integer_only": "Exact midpoint increments, original rounded event-pair cross-products",
                           "numerical_only": "Exact midpoint arithmetic; consistent fixed-bin normalized cross-products; legacy event-study support",
                           "support_corrected": "Exact event ratios; fixed observable event-study cohort",
                           "quality_controlled": "Previously audited 250-ms, flags/ties/barriers/complete-horizon ColonyLab results, same raw sources"},
                "offline_only": True, "original_full_day_raw_available": False,
                "scope_note": "Ten saved10-minute dates, three symbols, five venues; not the original three full-day/IEX runs. Exploratory paired code comparison, not fresh economic confirmation."}
    dump(out / "protocol.json", protocol)
    original_code = [{"path": str((snapshot / name).resolve()), "sha256": sha(snapshot / name)} for name in USED_FILES]
    corrected_code = [{"path": str((CORRECTED / name).resolve()), "sha256": sha(CORRECTED / name)} for name in USED_FILES]
    code = [*original_code, *corrected_code, {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))}]
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": sha(out / "protocol.json"),
                  "original_repository": "https://github.com/smar-01/qam-p113-latency", "original_commit": "62d6dcd8af9ecd7136096dbcce3eb40371987422",
                  "sources": [{key: entry[key] for key in ("data_path", "request", "rows", "content_sha256")} for entry in entries],
                  "original_code": original_code, "corrected_code": corrected_code,
                  "runner_code": [{"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))}]}
    dump(out / "provenance.json", provenance)
    tasks = []
    for day in config["training_days"] + config["held_out_days"]:
        directory = out / "days" / day
        directory.mkdir(parents=True, exist_ok=True)
        selected = [entry for entry in entries if entry["request"]["start"][:10] == day]
        identity = {"protocol": protocol, "code": code, "sources": [entry["content_sha256"] for entry in selected]}
        tasks.append({"day": day, "entries": selected, "protocol": protocol, "snapshot": str(snapshot),
                      "directory": str(directory), "identity": identity})
    if not args.summarize_only:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for future in as_completed([executor.submit(run_day, task) for task in tasks]):
                print(json.dumps(future.result()), flush=True)
    for task in tasks:
        directory = Path(task["directory"])
        marker = json.loads((directory / "complete.json").read_text())
        if (marker["identity"] != task["identity"]
                or marker["shared_quote_sha256_before"] != marker["shared_quote_sha256_after"]
                or any(sha(directory / name) != marker["files"][name] for name in DAY_FILES)):
            raise ValueError("A day checkpoint is incomplete or differs from the exact code/source/input identity")
    for name in DAY_FILES:
        pd.concat([pd.read_csv(Path(task["directory"]) / name) for task in tasks], ignore_index=True).to_csv(out / name, index=False)
    quality = add_quality_comparison(ROOT, out, provenance)
    dump(out / "quality_comparison_provenance.json", quality)
    result = summarize(out)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
