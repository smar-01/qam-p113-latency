"""Checkpointed full-session displayed-book reconstruction and SIRS experiment.

Raw streams, reconstruction, feature panels and fitted models have separate
content-addressed provenance. Replaying a received file never requires another
network request, and changing reporting code does not invalidate book caches.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import time
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .mbo_sir_data import build_requests, cache_directory, fetch_study, read_config
from .mbo_sir_stats import summarize_sir_comparison
from .sir_model import fit_sirs, fit_sirs_models


MODELS = ("independent_mbo_hazard", "sirs_network", "logistic_network", "past_shifted_sirs")
SEMANTIC_FAILURES = ("duplicate_adds", "unmatched_cancels", "cancel_identity_mismatches", "oversize_cancels",
                     "unmatched_modifies", "unknown_modify_invalidations", "modify_upserts", "modify_side_mismatches",
                     "snapshot_without_clear", "invalid_snapshot_actions", "invalid_order_fields", "invalid_fill_fields",
                     "invalid_trade_fields", "unknown_actions", "non_native_records", "unmatched_fills",
                     "invalid_timestamps")
HERE = Path(__file__).resolve().parent


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _file_hash(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=1024)
def _raw_hash(path, byte_length, modified_ns):
    return _file_hash(path)


def _code_hash(*names) -> str:
    return _digest({name: _file_hash(HERE / name) for name in names})


def _replace(pending: Path, destination: Path) -> None:
    for attempt in range(12):
        try:
            pending.replace(destination)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(.1 * 2 ** attempt, 1.0))


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    _replace(tmp, path)


def _dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=1) as stream:
        pickle.dump(value, stream, protocol=5)
    _replace(tmp, path)


def _load(path: Path):
    # These are locally created research checkpoints, never external pickles.
    with path.open("rb") as stream:
        compressed = stream.read(2) == b"\x1f\x8b"
    with (gzip.open(path, "rb") if compressed else path.open("rb")) as stream:
        return pickle.load(stream)


def _log(message) -> None:
    print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True)


def _receipt(config, request):
    folder = cache_directory(Path(config["data_dir"]), request)
    path = folder / "fetch.json"
    if not path.exists():
        return None
    entry = json.loads(path.read_text(encoding="utf-8"))
    if not entry.get("complete"):
        return None
    if entry.get("request") != request or entry.get("record_limit_reached"):
        raise ValueError("Raw request receipt mismatch or record cap reached.")
    raw = folder / "sample.dbn.zst"
    if not raw.is_file() or raw.stat().st_size != entry["file_size_bytes"]:
        raise ValueError("Completed raw stream is missing or its byte length changed.")
    marker = folder / "stream_complete.json"
    if not marker.exists() or not 0 < entry["rows"] < request["limit"]:
        raise ValueError("Raw stream lacks complete, uncapped termination provenance.")
    if json.loads(marker.read_text(encoding="utf-8")).get("request") != request:
        raise ValueError("Successful stream marker differs from the exact request.")
    stat = raw.stat()
    return dict(entry, data_path=str(raw.resolve()), sha256=_raw_hash(str(raw.resolve()), stat.st_size, stat.st_mtime_ns))


def _replay_job(job):
    from .mbo_replay import replay_file
    started = time.monotonic()
    result = replay_file(job["mbo_path"], job["validation_path"], day=job["day"], anchor_step_ms=job["anchor_step_ms"])
    if result["diagnostics"]["records_replayed"] != job["mbo_rows"]:
        raise ValueError("Replay count differs from the independently parsed acquisition receipt.")
    result["provenance"] = job["provenance"]
    result["diagnostics"]["replay_elapsed_seconds"] = time.monotonic() - started
    _dump(Path(job["path"]), result)
    return {"day": job["day"], "dataset": job["dataset"], "path": job["path"],
            "validation": result["validation"], "diagnostics": result["diagnostics"],
            "provenance": job["provenance"]}


def _assert_replay_clean(result, out):
    if not result["validation"]["passed"]:
        raise ValueError(f"Independent top-ten reconstruction audit failed for {(result['day'], result['dataset'])}; preserved replay for diagnosis.")
    counts = {name: result["diagnostics"].get(name, 0) for name in SEMANTIC_FAILURES
              if result["diagnostics"].get(name, 0)}
    if counts:
        _json(Path(out) / "early_semantic_failure.json", {"day": result["day"], "dataset": result["dataset"], "counts": counts, "path": result["path"]})
        raise ValueError(f"Full-session order semantics failed for {(result['day'], result['dataset'])}: {counts}; replay preserved.")


def replay_study(config, *, wait_for_downloads=False, workers=8):
    """Replay independent venue/day files; optionally overlap acquisition.

    Waiting mode checks the durable exact-request receipts. A failed completed
    download manifest stops the run instead of omitting the missing task.
    """
    requests = build_requests(config)
    pairs = {}
    for request in requests:
        if request["schema"] in {"mbo", "mbp-10"}:
            pairs.setdefault((request["start"][:10], request["dataset"]), {})[request["schema"]] = request
    code = _code_hash("mbo_book.py", "mbo_replay.py")
    out = Path(config["out_dir"])
    cache = out / "replay_cache"
    cache.mkdir(parents=True, exist_ok=True)
    pending, running, results = dict(pairs), {}, {}
    last_status = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while pending or running:
            for key, pair in list(pending.items()):
                if len(running) >= workers:
                    break
                mbo, mbp = _receipt(config, pair["mbo"]), _receipt(config, pair["mbp-10"])
                if mbo is None or mbp is None:
                    continue
                sources = {"mbo": mbo["sha256"], "mbp-10": mbp["sha256"]}
                provenance = {"sources": sources, "code": code, "anchor_step_ms": config["anchor_step_ms"],
                              "requests": pair, "cache_version": 1}
                path = cache / f"{key[0]}_{key[1]}_{_digest(provenance)[:20]}.pkl.gz"
                if path.exists():
                    value = _load(path)
                    if value.get("provenance") != provenance:
                        raise ValueError("Reconstruction cache provenance differs.")
                    results[key] = dict(day=key[0], dataset=key[1], path=str(path), provenance=provenance,
                                        validation=value["validation"], diagnostics=value["diagnostics"])
                    del value
                    del pending[key]
                    _assert_replay_clean(results[key], out)
                    _log(f"Reused replay {key}: {len(results)}/{len(pairs)}")
                    continue
                job = dict(day=key[0], dataset=key[1], mbo_path=mbo["data_path"], validation_path=mbp["data_path"],
                           mbo_rows=mbo["rows"], anchor_step_ms=config["anchor_step_ms"], path=str(path), provenance=provenance)
                running[pool.submit(_replay_job, job)] = key
                del pending[key]
            if running:
                completed, _ = wait(running, timeout=2, return_when=FIRST_COMPLETED)
                for future in completed:
                    key = running.pop(future)
                    result = future.result()
                    results[key] = result
                    val, diag = result["validation"], result["diagnostics"]
                    _log(f"Replayed {key}: {diag['records_replayed']:,} records; valid grid {diag['valid_grid_rows']:,}; "
                         f"depth audit={val['passed']} ({val['mismatch_keys']:,} mismatches, {val['missing_keys']:,} missing); "
                         f"{len(results)}/{len(pairs)}")
                    _json(out / "replay_manifest.json", {"complete": len(results) == len(pairs), "tasks": list(results.values())})
                    # Reconstruction correctness is a gate, never a filter.
                    _assert_replay_clean(result, out)
            elif pending:
                if not wait_for_downloads:
                    raise ValueError("Exact full-session raw request grid is incomplete.")
                path = out / "data_manifest.json"
                if path.exists():
                    manifest = json.loads(path.read_text(encoding="utf-8"))
                    if manifest.get("finished_utc") and not manifest.get("complete"):
                        raise ValueError("Acquisition finished with missing requests; replay stopped without dropping tasks.")
                time.sleep(2)
            if time.monotonic() - last_status >= 30:
                _log(f"Replay progress: {len(results)}/{len(pairs)} complete, {len(running)} active, {len(pending)} awaiting data or worker.")
                last_status = time.monotonic()
    result = {"complete": True, "tasks": [results[key] for key in pairs]}
    _json(out / "replay_manifest.json", result)
    return result


def _build_panels(config, protocol, replays):
    from .mbo_sir_features import build_day_panels, fit_thresholds
    out = Path(config["out_dir"])
    tasks = {(row["day"], row["dataset"]): row for row in replays["tasks"]}
    def training():
        for day in config["training_days"]:
            for dataset in config["datasets"]:
                yield dataset, day, _load(Path(tasks[day, dataset]["path"]))
    thresholds = fit_thresholds(training(), training_days=config["training_days"], expected_datasets=config["datasets"])
    thresholds.to_csv(out / "shock_thresholds.csv", index=False)
    threshold_map = thresholds.set_index("dataset").threshold.to_dict()
    features_code = _code_hash("mbo_sir_features.py", "sir_model.py")
    market_requests = {r["start"][:10]: r for r in build_requests(config) if r["schema"] == "bbo-1s"}
    panel_rows, coverage = [], []
    for day in config["training_days"] + config["held_out_days"]:
        market = _receipt(config, market_requests[day])
        if market is None:
            raise ValueError(f"Missing market-control request: {day}")
        provenance = {"code": features_code, "thresholds": threshold_map, "protocol": protocol,
                      "market_sha256": market["sha256"],
                      "replays": {d: tasks[day, d]["provenance"] for d in config["datasets"]}}
        signature = _digest(provenance)[:20]
        paths = {d: out / "panel_cache" / f"{day}_{d}_{signature}.pkl.gz" for d in config["datasets"]}
        if not all(path.exists() for path in paths.values()):
            books = {d: _load(Path(tasks[day, d]["path"])) for d in config["datasets"]}
            panels = build_day_panels(books, market["data_path"], threshold_map, protocol)
            if set(panels) != set(config["datasets"]):
                raise ValueError("Feature builder omitted an intended venue.")
            for dataset, panel in panels.items():
                panel["provenance"] = provenance
                _dump(paths[dataset], panel)
            del panels, books
            gc.collect()
        for dataset, path in paths.items():
            panel = _load(path)
            if panel.get("provenance") != provenance:
                raise ValueError("Feature cache provenance differs.")
            panel_rows.append(dict(day=day, dataset=dataset, path=str(path), signature=signature))
            coverage.append(dict(day=day, dataset=dataset, **panel["coverage"]))
            del panel
        _log(f"Built common four-model panel {day}: {len(panel_rows)}/{len(tasks)} tasks")
    pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)
    _json(out / "panel_manifest.json", {"complete": True, "tasks": panel_rows, "thresholds": threshold_map})
    return panel_rows, threshold_map


def _training(panels, dataset, days):
    parts = [_load(Path(row["path"])) for row in panels if row["dataset"] == dataset and row["day"] in days]
    selected = [row for row in panels if row["dataset"] == dataset and row["day"] in days]
    if (len(parts) != len(days) or {row["day"] for row in selected} != set(days)
            or len({row["day"] for row in selected}) != len(selected)
            or any(not len(part["y"]) for part in parts)):
        raise ValueError("Each intended venue/training day must have eligible observations.")
    first = parts[0]
    for part in parts[1:]:
        for key in ("names", "legacy_names", "source_names"):
            if list(part[key]) != list(first[key]):
                raise ValueError("Feature column order differs across dates.")
    result = {key: np.concatenate([part[key] for part in parts], axis=0)
              for key in ("x", "infected", "shifted_infected", "legacy", "y")}
    result.update({key: first[key] for key in ("names", "legacy_names", "source_names")})
    return result


def _fit_target(train, protocol):
    x, y = train["x"], train["y"]
    kwargs = dict(dt_seconds=protocol["forecast_horizon_ms"] / 1000,
                  ridge=protocol["optimization"]["hazard_ridge"], control_names=train["names"], max_iter=1000)
    fits = fit_sirs_models(x, train["infected"], y, source_names=train["source_names"], **kwargs)
    shifted = fit_sirs(x, train["shifted_infected"], y, source_names=train["source_names"], **kwargs)
    # The shared controls already contain own one-second infection history.
    # Include that semantic feature once so L2 does not change its penalty.
    keep = [i for i, name in enumerate(train["legacy_names"])
            if not (name == "own_infections_1000ms" and "own_infections_1s" in train["names"])]
    legacy_x = np.column_stack([x, train["legacy"][:, keep]])
    scaler = StandardScaler()
    legacy_x = scaler.fit_transform(legacy_x)
    logistic = LogisticRegression(C=1, solver="lbfgs", max_iter=1000, tol=1e-8)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always", ConvergenceWarning)
        logistic.fit(legacy_x, y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    models = dict(independent_mbo_hazard=fits["exogenous"], sirs_network=fits["network"],
                  past_shifted_sirs=shifted, logistic_network=(scaler, logistic))
    metadata = {name: model.to_dict() for name, model in models.items() if name != "logistic_network"}
    metadata["logistic_network"] = {
        "converged": converged, "iterations": list(map(int, logistic.n_iter_)), "C": 1,
        "training_rows": len(y), "training_infections": int(y.sum()), "intercept": float(logistic.intercept_[0]),
        "coefficients": dict(zip(list(train["names"]) + [train["legacy_names"][i] for i in keep], map(float, logistic.coef_[0]))),
        "feature_names": list(train["names"]) + [train["legacy_names"][i] for i in keep],
        "legacy_column_indices": keep,
        "control_mean": list(map(float, scaler.mean_)), "control_scale": list(map(float, scaler.scale_)),
    }
    return models, metadata


def _score(y, p):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if (p.shape != y.shape or not len(y) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            or not np.isfinite(y).all() or not np.all((y == 0) | (y == 1))):
        raise ValueError("Cannot score missing, nonfinite or invalid paired predictions.")
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    return dict(rows=len(y), positive_count=int(y.sum()), log_loss_sum=float(-(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)).sum()),
                brier_sum=float(np.square(p - y).sum()), probability_clip_rows=int(np.count_nonzero(clipped != p)))


def _fit_and_score(config, protocol, panels):
    out = Path(config["out_dir"])
    fits_dir = out / "fit_cache"
    fits_dir.mkdir(parents=True, exist_ok=True)
    code = _code_hash("sir_model.py", "mbo_sir_pipeline.py")
    score_rows, edge_rows, all_metadata = [], [], {}
    for dataset in config["datasets"]:
        signature = _digest({"code": code, "protocol": protocol,
                             "training_panels": [p for p in panels if p["dataset"] == dataset and p["day"] in config["training_days"]]})
        path = fits_dir / f"{dataset}_{signature[:20]}.pkl.gz"
        if path.exists():
            cached = _load(path)
            if cached.get("signature") != signature:
                raise ValueError("Fitted model cache provenance differs.")
            models, metadata = cached["models"], cached["metadata"]
            _log(f"Reused four training fits {dataset}")
        else:
            _log(f"Fitting four models {dataset}")
            train = _training(panels, dataset, config["training_days"])
            models, metadata = _fit_target(train, protocol)
            del train
            gc.collect()
            _dump(path, dict(models=models, metadata=metadata, signature=signature))
        all_metadata[dataset] = metadata
        _json(out / "model_parameters.json", all_metadata)
        _log(f"Fitted {dataset}: " + ", ".join(f"{name} convergence={value['converged']}" for name, value in metadata.items()))
        network = models["sirs_network"]
        for source, beta in zip(network.source_names_, network.beta_rates_):
            edge_rows.append(dict(source=source, target=dataset, beta_per_second=float(beta),
                                  isolated_10ms_probability=float(-np.expm1(-.01 * beta))))
        for row in panels:
            if row["dataset"] != dataset or row["day"] not in config["held_out_days"]:
                continue
            panel = _load(Path(row["path"]))
            x, y = panel["x"], panel["y"]
            cohort_hash = hashlib.sha256(panel["time_ns"].tobytes() + y.tobytes()).hexdigest()
            for name, model in models.items():
                with threadpool_limits(limits=1):
                    if name == "logistic_network":
                        scaler, logistic = model
                        keep = metadata[name]["legacy_column_indices"]
                        p = logistic.predict_proba(scaler.transform(np.column_stack([x, panel["legacy"][:, keep]])))[:, 1]
                        guarded = 0
                    else:
                        sources = panel["shifted_infected"] if name == "past_shifted_sirs" else panel["infected"]
                        p = model.predict_proba(x, sources if name != "independent_mbo_hazard" else None)
                        eta = model.intercept_ + ((x - model.control_mean_) / model.control_scale_) @ model.control_coefficients_
                        guarded = int(np.count_nonzero(np.abs(eta) >= 60))
                score_rows.append(dict(day=row["day"], dataset=dataset, model=name, cohort_sha256=cohort_hash,
                                       numerical_guard_rows=guarded, **_score(y, p)))
            del panel, x, y
        pd.DataFrame(score_rows).to_csv(out / "task_daily_scores.csv", index=False)
        pd.DataFrame(edge_rows).to_csv(out / "edges.csv", index=False)
    return pd.DataFrame(score_rows), pd.DataFrame(edge_rows), all_metadata


def analyze_study(config, protocol, replays):
    out = Path(config["out_dir"])
    if not replays.get("complete") or any(not t["validation"]["passed"] for t in replays["tasks"]):
        raise ValueError("Every intended venue/day must pass independent depth validation before fitting.")
    intended = {(day, dataset) for day in config["training_days"] + config["held_out_days"] for dataset in config["datasets"]}
    observed = [(t["day"], t["dataset"]) for t in replays["tasks"]]
    if set(observed) != intended or len(observed) != len(intended):
        raise ValueError("Reconstruction must contain the exact unique intended venue/day grid.")
    audit = pd.DataFrame([dict(day=t["day"], dataset=t["dataset"], **t["validation"],
                               records_replayed=t["diagnostics"]["records_replayed"],
                               valid_grid_rows=t["diagnostics"]["valid_grid_rows"],
                               initialization_source=t["diagnostics"]["initialization_source"],
                               **{"diag_" + key: value for key, value in t["diagnostics"].items()
                                  if isinstance(value, (int, float, bool))})
                          for t in replays["tasks"]])
    audit.to_csv(out / "book_audit.csv", index=False)
    semantic_errors = [dict(day=t["day"], dataset=t["dataset"],
                            counts={name: t["diagnostics"].get(name, 0) for name in SEMANTIC_FAILURES
                                    if t["diagnostics"].get(name, 0)})
                       for t in replays["tasks"]
                       if any(t["diagnostics"].get(name, 0) for name in SEMANTIC_FAILURES)]
    if semantic_errors:
        _json(out / "semantic_failures.json", {"passed": False, "tasks": semantic_errors})
        raise ValueError("Full-session order semantics/clock audit failed; book_audit.csv and semantic_failures.json preserve details. Fix before fitting.")
    panels, thresholds = _build_panels(config, protocol, replays)
    task_daily, edges, metadata = _fit_and_score(config, protocol, panels)
    expected = {(day, dataset, model) for day in config["held_out_days"] for dataset in config["datasets"] for model in MODELS}
    actual = set(zip(task_daily.day, task_daily.dataset, task_daily.model))
    if actual != expected or task_daily.duplicated(["day", "dataset", "model"]).any():
        raise ValueError("Held-out task/date/model grid is incomplete or duplicated.")
    if task_daily.groupby(["day", "dataset"]).cohort_sha256.nunique().max() != 1:
        raise ValueError("Models were scored on different held-out timestamps or labels.")
    daily = task_daily.groupby(["day", "model"], as_index=False)[["rows", "positive_count", "log_loss_sum", "brier_sum", "numerical_guard_rows", "probability_clip_rows"]].sum()
    daily.to_csv(out / "daily_scores.csv", index=False)
    comparisons = {reference: summarize_sir_comparison(daily, reference_model=reference, candidate_model="sirs_network",
                    expected_days=config["held_out_days"], bootstrap_replicates=protocol["inference"]["replicates"],
                    seed=protocol["inference"]["seed"], block_length=protocol["inference"]["block_sessions"])
                   for reference in ("independent_mbo_hazard", "logistic_network", "past_shifted_sirs")}
    totals = daily.groupby("model")[["rows", "positive_count", "log_loss_sum", "brier_sum"]].sum()
    totals["log_loss"] = totals.log_loss_sum / totals.rows
    totals["brier"] = totals.brier_sum / totals.rows
    converged = all(m["converged"] for value in metadata.values() for m in value.values())
    guards = int(task_daily.numerical_guard_rows.sum()) + sum(m.get("training_numerical_guard_rows", 0) for value in metadata.values() for m in value.values())
    manifest = json.loads((out / "data_manifest.json").read_text(encoding="utf-8"))
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "name": config["name"], "complete": True,
        "datasets": config["datasets"], "symbol": config["symbol"], "training_days": config["training_days"],
        "held_out_days": config["held_out_days"], "raw_requests": len(manifest["requests"]),
        "raw_rows": sum(r["rows"] for r in manifest["requests"]), "mbo_rows_replayed": int(audit.records_replayed.sum()),
        "book_validation": {"tasks": len(audit), "all_passed": bool(audit.passed.all()),
                            "matched_keys": int(audit.matched_keys.sum()), "mismatch_keys": int(audit.mismatch_keys.sum()), "missing_keys": int(audit.missing_keys.sum()),
                            "full_session_semantic_errors": 0,
                            "quality": [{"dataset": dataset, "candidate_grid_rows": sum(t["diagnostics"]["grid_rows"] for t in replays["tasks"] if t["dataset"] == dataset),
                                         "valid_grid_rows": sum(t["diagnostics"]["valid_grid_rows"] for t in replays["tasks"] if t["dataset"] == dataset),
                                         "native_bad_book_flags": sum(t["diagnostics"].get("maybe_bad_book_flags", 0) for t in replays["tasks"] if t["dataset"] == dataset),
                                         "native_bad_receive_flags": sum(t["diagnostics"].get("bad_receive_timestamp_flags", 0) for t in replays["tasks"] if t["dataset"] == dataset),
                                         "receive_time_reversals": sum(t["diagnostics"].get("receive_time_reversals", 0) for t in replays["tasks"] if t["dataset"] == dataset)}
                                        for dataset in config["datasets"]]},
        "all_models_converged": converged, "numerical_guard_rows": guards,
        "model_metrics": totals.reset_index().to_dict(orient="records"), "comparisons": comparisons,
        "supports_incremental_prediction": bool(converged and guards == 0 and comparisons["independent_mbo_hazard"].get("supports_positive_incremental_prediction")),
        "thresholds": thresholds, "positive_directed_edges": int((edges.beta_per_second > 1e-8).sum()),
        "protocol": protocol, "config": config,
        "comparability": "Previous MBP experiment used different dates, stocks, exposure duration, shock labels and eligible cohort; compare models within this experiment only.",
        "limitations": protocol["limitations"],
    }
    _json(out / "summary.json", summary)
    _json(out / "provenance.json", {"protocol_sha256": _digest(protocol), "config_sha256": _digest(config),
          "source_code_sha256": {p.name: _file_hash(p) for p in HERE.glob("mbo*.py")},
          "sir_model_sha256": _file_hash(HERE / "sir_model.py"),
          "raw_requests_sha256": _digest(manifest["requests"]), "replay_manifest_sha256": _digest(replays),
          "panel_manifest_sha256": _file_hash(out / "panel_manifest.json")})
    _log(json.dumps({k: summary[k] for k in ("complete", "mbo_rows_replayed", "book_validation", "all_models_converged", "supports_incremental_prediction")}))
    return summary


def run_mbo_sir(config_path, *, fetch=True, analyze=True):
    config_path = Path(config_path)
    config = read_config(config_path)
    protocol = json.loads(config_path.with_name("mbo_sir_protocol.json").read_text(encoding="utf-8"))
    if fetch:
        fetch_study(config, config_path=config_path, progress=_log)
    if not analyze:
        return
    manifest_path = Path(config["out_dir"]) / "data_manifest.json"
    if not manifest_path.exists() or not json.loads(manifest_path.read_text(encoding="utf-8")).get("complete"):
        raise ValueError("The exact acquisition grid is incomplete. Use --replay-available to overlap book reconstruction with the running download.")
    replays = replay_study(config)
    from .mbo_sir_data import audit_study
    audit_study(config, progress=_log, workers=4)
    return analyze_study(config, protocol, replays)
