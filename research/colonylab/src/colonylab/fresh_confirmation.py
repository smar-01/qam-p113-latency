"""Streaming, frozen confirmation of volume features added to venue networks.

Raw requests are validated and normalized one at a time. Only one date's quote
states and one target's training matrix are materialized at once. Checkpoints
belong to a fingerprint of the exact source files, code and frozen protocol.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .config import write_json
from .network import MAX_AGE_MS, _jump_candidates, extract_shocks, normalize_quotes
from .network_models import build_prediction_rows
from .network_wide_data import (_signature, _study_design, read_wide_config,
                                requests_for, validate_config)
from .volume_bars import build_volume_bars, normalize_trades, sample_completed_bars


FEATURES = ("return_bps", "range_bps", "log_volume_multiple", "log_trade_count",
            "log_duration_ms", "signed_volume_fraction", "unknown_volume_fraction",
            "vwap_vs_close_bps", "log_age_ms")
ENGINE_VERSION = 1


def attach_volume_features(
    rows: pd.DataFrame, volume_bars: dict, thresholds: dict, *, venues: list[str],
    max_age_ms: float = 10_000,
) -> tuple[pd.DataFrame, list[str], dict]:
    """Add nine strictly completed volume states per venue on one paired mask."""
    if not np.isfinite(max_age_ms) or max_age_ms <= 0:
        raise ValueError("Candle age must be positive and finite.")
    if not venues or len(venues) != len(set(venues)):
        raise ValueError("Candle venues must be nonempty and unique.")
    names = [f"volume_{venue}_{feature}" for venue in venues for feature in FEATURES]
    if rows.empty:
        return rows.copy(), names, {"quote_eligible_bins": 0, "common_bins": 0,
                                    "candle_excluded_bins": 0, "by_venue": []}
    if any(rows[column].nunique() != 1 for column in ("dataset", "symbol", "day")):
        raise ValueError("Attach volume candles to one target venue/stock/day at a time.")
    symbol, day = str(rows.symbol.iloc[0]), str(rows.day.iloc[0])
    anchors = rows.time_ns.to_numpy(dtype=np.int64)
    common, additions, coverage = np.ones(len(rows), dtype=bool), {}, []
    for venue in venues:
        threshold = float(thresholds[(venue, symbol)])
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("Every volume quantum must be finite and positive.")
        sample = sample_completed_bars(volume_bars.get((venue, symbol, day), pd.DataFrame()),
                                       anchors, max_age_ns=int(max_age_ms * 1e6),
                                       include_equal_close=False)
        values = (
            sample.return_bps.to_numpy(dtype=float), sample.range_bps.to_numpy(dtype=float),
            np.log1p(np.expm1(sample.log_volume.to_numpy(dtype=float)) / threshold),
            sample.log_trade_count.to_numpy(dtype=float), sample.log_duration_ms.to_numpy(dtype=float),
            sample.signed_volume_fraction.to_numpy(dtype=float),
            1 - sample.known_side_fraction.to_numpy(dtype=float),
            sample.vwap_close_bps.to_numpy(dtype=float),
            np.log1p(sample.age_ms.to_numpy(dtype=float)),
        )
        usable = sample.valid.to_numpy(dtype=bool)
        for feature, value in zip(FEATURES, values):
            additions[f"volume_{venue}_{feature}"] = value
            usable &= np.isfinite(value)
        common &= usable
        coverage.append({"predictor_dataset": venue, "available_bins": int(usable.sum()),
                         "excluded_bins": int((~usable).sum())})
    result = pd.concat([rows.reset_index(drop=True), pd.DataFrame(additions)], axis=1)
    result = result.loc[common].reset_index(drop=True)
    return result, names, {"quote_eligible_bins": len(rows), "common_bins": int(common.sum()),
                           "candle_excluded_bins": int((~common).sum()), "by_venue": coverage}


def fit_training_calibration(
    prepared: Iterable[tuple[str, str, dict]], training_days: list[str], symbols: list[str],
    *, target_seconds: float = 1.,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Accumulate only training jumps and full requested trade exposure."""
    if (not training_days or len(set(training_days)) != len(training_days)
            or training_days != sorted(training_days)):
        raise ValueError("Training dates must be nonempty, unique and ordered.")
    if not np.isfinite(target_seconds) or target_seconds <= 0:
        raise ValueError("target_seconds must be positive and finite.")
    train = set(training_days)
    seen, jumps, totals = set(), defaultdict(list), {}
    for dataset, day, payload in prepared:
        # Do not inspect a held-out payload, even for its shape or attributes.
        if day not in train:
            continue
        if (dataset, day) in seen:
            raise ValueError("Duplicate training venue/day exposure.")
        seen.add((dataset, day))
        quotes, trades = payload["quotes"], payload["trades"]
        if "requested_start" not in trades.attrs or "requested_end" not in trades.attrs:
            raise ValueError("Training exposure requires explicit requested start/end attributes.")
        seconds = (pd.Timestamp(trades.attrs["requested_end"]) -
                   pd.Timestamp(trades.attrs["requested_start"])).total_seconds()
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError("Requested training exposure must be positive.")
        for symbol in symbols:
            key = dataset, symbol
            q = quotes.loc[quotes.symbol.eq(symbol)]
            _, delta = _jump_candidates(q, MAX_AGE_MS)
            jumps[key].append(np.abs(delta))
            t = trades.loc[trades.symbol.eq(symbol)]
            row = totals.setdefault(key, {"training_seconds": 0., "training_volume_shares": 0.,
                                         "training_valid_trades": 0, "training_invalid_trades": 0,
                                         "training_days": 0})
            row["training_seconds"] += seconds
            row["training_volume_shares"] += float(t.loc[t.valid, "size"].sum())
            row["training_valid_trades"] += int(t.valid.sum())
            row["training_invalid_trades"] += int((~t.valid).sum())
            row["training_days"] += 1
    if not seen:
        raise ValueError("No training venue/day frames supplied.")
    venues = {dataset for dataset, _ in seen}
    if seen != {(venue, day) for venue in venues for day in training_days}:
        raise ValueError("Every observed training venue requires every intended training date.")
    shock_rows, volume_rows = [], []
    for (dataset, symbol), row in sorted(totals.items()):
        delta = np.concatenate(jumps[(dataset, symbol)])
        threshold = float(max(.005, np.quantile(delta, .75))) if len(delta) else .005
        shock_rows.append({"dataset": dataset, "symbol": symbol, "threshold": threshold,
                           "threshold_usd": threshold, "training_nonzero_jumps": len(delta),
                           "training_days": row["training_days"]})
        volume_rows.append({"dataset": dataset, "symbol": symbol, **row,
                            "target_seconds": float(target_seconds),
                            "threshold_shares": max(1, int(np.floor(
                                row["training_volume_shares"] / row["training_seconds"] * target_seconds + .5)))})
    return pd.DataFrame(shock_rows), pd.DataFrame(volume_rows)


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _read(path: Path):
    # Only trusted caches created by this local pipeline are read as pickle.
    with path.open("rb") as handle:
        return pickle.load(handle)


def _manifest_entries(manifest_path: Path, config: dict) -> tuple[dict, list[dict]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError("Confirmation requires a complete acquisition manifest.")
    stored = validate_config(manifest["configuration"])
    if manifest.get("configuration_sha256") != _signature(stored):
        raise ValueError("Manifest configuration checksum is inconsistent.")
    if _study_design(stored) != _study_design(config):
        raise ValueError("Confirmation configuration differs from acquired study.")
    expected = {json.dumps(request, sort_keys=True) for request in requests_for(config)}
    entries = manifest["requests"]
    actual = [json.dumps(entry["request"], sort_keys=True) for entry in entries]
    if len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Manifest must contain exactly the intended venue/date grid.")
    if any(not entry.get("complete") or entry.get("record_limit_reached") for entry in entries):
        raise ValueError("Incomplete or capped requests cannot enter confirmation.")
    return manifest, sorted(entries, key=lambda item: (item["request"]["start"], item["request"]["dataset"]))


def _prepare_request(entry: dict) -> dict:
    import databento as db

    request = entry["request"]
    frame = db.DBNStore.from_file(entry["data_path"]).to_df(
        price_type="float", pretty_ts=True, map_symbols=True)
    if not 0 < len(frame) < request["limit"] or len(frame) != entry["rows"]:
        raise ValueError("Raw DBN is empty, capped or inconsistent with its completed receipt.")
    if (not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None
            or frame.index.hasnans or frame.index.name != "ts_recv"):
        raise ValueError("Confirmation requires valid timezone-aware receive timestamps.")
    if ((frame.index < pd.Timestamp(request["start"])).any()
            or (frame.index >= pd.Timestamp(request["end"])).any()):
        raise ValueError("DBN timestamps fall outside its requested half-open interval.")
    if "symbol" not in frame or set(frame.symbol.dropna().unique()) != set(request["symbols"]):
        raise ValueError("DBN must contain exactly every requested symbol.")
    frame.attrs.update(requested_start=request["start"], requested_end=request["end"])
    dataset, day = request["dataset"], request["start"][:10]
    quotes, quote_quality = normalize_quotes(frame, dataset, day)
    trades, trade_quality = normalize_trades(frame, dataset, day)
    quotes = quotes[["time_ns", "symbol", "mid", "valid", "segment", "dataset", "day"]]
    trades = trades[["time_ns", "price", "size", "sign", "valid", "symbol", "segment", "dataset", "day"]]
    return {"quotes": quotes, "trades": trades, "quote_quality": quote_quality,
            "trade_quality": trade_quality}


def _reference_names(dataset: str, symbol: str, venues: list[str], symbols: list[str]) -> list[str]:
    controls = (["SPY"] if symbol != "SPY" and "SPY" in symbols else
                [s for s in ("AAPL", "MSFT") if s in symbols] if symbol == "SPY" else
                [s for s in sorted(symbols) if s != symbol])
    return (["elapsed_seconds"] + [f"own_shocks_{w}ms" for w in (10, 100, 1000)] +
            [f"market_{s}_shocks_{w}ms" for s in controls for w in (10, 100, 1000)] +
            [f"other_{v}_shocks_{w}ms" for v in sorted(venues) if v != dataset for w in (10, 100, 1000)])


def _panel_path(cache: Path, dataset: str, symbol: str, day: str) -> Path:
    return cache / "panels" / f"{dataset}_{symbol}_{day}.pkl"


def _loss_sums(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    p = np.clip(probability, 1e-12, 1 - 1e-12)
    return (float((-y * np.log(p) - (1 - y) * np.log1p(-p)).sum()),
            float(np.square(y - p).sum()))


def _fit_task(cache: Path, dataset: str, symbol: str, config: dict) -> dict:
    """Fit both models on identical training arrays; test dates stream in order."""
    train = [_read(_panel_path(cache, dataset, symbol, day)) for day in config["training_days"]]
    names = train[0]["feature_names"]
    reference_count = train[0]["reference_feature_count"]
    if any(panel["feature_names"] != names or panel["reference_feature_count"] != reference_count for panel in train):
        raise ValueError("Training feature order changed between dates.")
    x = np.concatenate([panel["x"] for panel in train], axis=0)
    y = np.concatenate([panel["y"] for panel in train])
    train_rows_by_day = {day: len(panel["y"]) for day, panel in zip(config["training_days"], train)}
    del train
    models, coefficients, convergence = {}, [], {}
    if not len(y):
        status = "insufficient_training_rows"
    elif len(np.unique(y)) < 2:
        status = "one_training_class_constant_fallback"
    else:
        status = "ok"
    with threadpool_limits(limits=1):
        for name, count in (("reference", reference_count), ("volume", len(names))):
            if status != "ok":
                models[name] = float(y.mean()) if len(y) else None
                convergence[name] = {"converged": status != "insufficient_training_rows", "iterations": 0}
                continue
            scaler = StandardScaler()
            scaled = scaler.fit_transform(x[:, :count])
            model = LogisticRegression(C=1., solver="lbfgs", max_iter=1000,
                                       class_weight=None, random_state=0)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(scaled, y)
            convergence[name] = {"converged": not any(issubclass(w.category, ConvergenceWarning) for w in caught),
                                 "iterations": int(model.n_iter_[0])}
            models[name] = scaler, model, count
            coefficients.extend({"dataset": dataset, "symbol": symbol, "model": name,
                                 "feature": feature, "coefficient_standardized": float(model.coef_[0, i]),
                                 "training_mean": float(scaler.mean_[i]), "training_scale": float(scaler.scale_[i]),
                                 "intercept": float(model.intercept_[0])}
                                for i, feature in enumerate(names[:count]))
            del scaled
        del x
        train_count, train_positives = len(y), int(y.sum())
        del y
        daily = []
        for day in config["held_out_days"]:
            panel = _read(_panel_path(cache, dataset, symbol, day))
            if panel["feature_names"] != names:
                raise ValueError("Held-out feature order differs from frozen training order.")
            y_test, x_test = panel["y"], panel["x"]
            row = {"dataset": dataset, "symbol": symbol, "day": day,
                   "rows": len(y_test) if status != "insufficient_training_rows" else 0,
                   "positive_bins": int(y_test.sum()) if status != "insufficient_training_rows" else 0}
            for name in ("reference", "volume"):
                fitted = models[name]
                if fitted is None or not len(y_test):
                    sums = 0., 0.
                else:
                    if isinstance(fitted, tuple):
                        scaler, model, count = fitted
                        probability = model.predict_proba(scaler.transform(x_test[:, :count]))[:, 1]
                    else:
                        probability = np.full(len(y_test), fitted)
                    sums = _loss_sums(y_test, probability)
                row[f"{name}_log_loss_sum"], row[f"{name}_brier_sum"] = sums
            daily.append(row)
    return {"dataset": dataset, "symbol": symbol, "status": status, "train_rows": train_count,
            "train_positive_bins": train_positives, "train_rows_by_day": train_rows_by_day,
            "reference_feature_count": reference_count, "volume_feature_count": len(names),
            "convergence": convergence, "daily": daily, "coefficients": coefficients}


def run_fresh_confirmation(
    manifest_path: Path, config_path: Path, out_dir: Path, *, anchor_step_ms: int = 100,
    progress: Callable[[str], None] = print,
) -> dict:
    """Run the exact fresh confirmation with resumable source-bound checkpoints."""
    from .confirmation_stats import summarize_confirmation

    manifest_path, config_path, out_dir = Path(manifest_path), Path(config_path), Path(out_dir)
    config = read_wide_config(config_path)
    protocol_path = config_path.parent / "fresh_confirmation_protocol.json"
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol["forecast_anchor_spacing_ms"] != anchor_step_ms or protocol["prediction_horizon_ms"] != 10:
        raise ValueError("Requested anchor grid differs from the frozen protocol.")
    if (protocol["training_sessions"] != len(config["training_days"])
            or protocol["held_out_sessions"] != len(config["held_out_days"])
            or protocol["venues"] != len(config["datasets"])
            or protocol["symbols"] != config["symbols"]
            or protocol["requested_minutes_per_session"] != config["minutes"]):
        raise ValueError("Frozen protocol dimensions differ from acquired configuration.")
    manifest, entries = _manifest_entries(manifest_path, config)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frozen_protocol.json").write_bytes(protocol_bytes)
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    inventory_path = out_dir / "source_inventory.json"
    old_inventory = json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.exists() else {}
    sources = []
    for entry in entries:
        path = Path(entry["data_path"])
        stat = path.stat()
        identity = {"data_path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        previous = old_inventory.get(str(path.resolve()), {})
        content_sha = previous.get("content_sha256") if all(previous.get(k) == v for k, v in identity.items()) else None
        if content_sha is None:
            content_sha = _file_sha(path)
        sources.append({**identity, "content_sha256": content_sha, "request": entry["request"], "rows": entry["rows"]})
    write_json(inventory_path, {source["data_path"]: source for source in sources})
    source_files = ("fresh_confirmation.py", "network.py", "network_models.py", "volume_bars.py", "confirmation_stats.py")
    signature = {"engine_version": ENGINE_VERSION, "configuration": _study_design(config),
                 "protocol_sha256": protocol_sha, "sources": sources,
                 "code": {name: _file_sha(Path(__file__).parent / name) for name in source_files}}
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    write_json(out_dir / "analysis_provenance.json", {**signature, "analysis_sha256": digest})
    cache = out_dir / "cache" / digest[:20]
    cache.mkdir(parents=True, exist_ok=True)
    prepared_paths, quote_quality, trade_quality = {}, [], []
    train_days, test_days = config["training_days"], config["held_out_days"]
    venues, symbols = sorted(config["datasets"]), config["symbols"]

    def prepare_entries(selected: set[str]) -> None:
        for entry in entries:
            request = entry["request"]
            dataset, day = request["dataset"], request["start"][:10]
            if day not in selected:
                continue
            path = cache / "prepared" / f"{day}_{dataset}.pkl"
            if not path.exists():
                payload = _prepare_request(entry)
                _dump(path, payload)
            else:
                payload = _read(path)
            prepared_paths[(dataset, day)] = path
            quote_quality.append(payload["quote_quality"])
            trade_quality.append(payload["trade_quality"])
            progress(f"Prepared {day} {dataset}: {len(payload['quotes']):,} quote states; {len(payload['trades']):,} trade prints.")
            del payload

    prepare_entries(set(train_days))
    shock_table, volume_table = fit_training_calibration(
        ((dataset, day, _read(prepared_paths[(dataset, day)])) for day in train_days for dataset in venues),
        train_days, symbols, target_seconds=protocol["volume_target_training_seconds"])
    shock_table.to_csv(out_dir / "shock_thresholds.csv", index=False)
    volume_table.to_csv(out_dir / "volume_thresholds.csv", index=False)
    expected_tasks = {(venue, symbol) for venue in venues for symbol in symbols}
    if set(zip(shock_table.dataset, shock_table.symbol)) != expected_tasks:
        raise ValueError("Training calibration omitted an intended venue/stock task.")
    thresholds = {(r.dataset, r.symbol): int(r.threshold_shares) for r in volume_table.itertuples()}
    shocks = {(r.dataset, r.symbol): float(r.threshold) for r in shock_table.itertuples()}
    progress("Training-only shock and volume thresholds frozen; preparing held-out requests.")
    prepare_entries(set(test_days))
    pd.DataFrame(quote_quality).to_csv(out_dir / "quote_quality.csv", index=False)
    pd.DataFrame(trade_quality).to_csv(out_dir / "trade_quality.csv", index=False)
    coverage, feed_coverage, candle_statistics, shock_statistics = [], [], [], []
    for day in train_days + test_days:
        day_checkpoint = cache / "day_metadata" / f"{day}.pkl"
        if day_checkpoint.exists() and all(_panel_path(cache, v, s, day).exists() for v, s in expected_tasks):
            meta = _read(day_checkpoint)
            coverage.extend(meta["coverage"]); feed_coverage.extend(meta["feed_coverage"])
            candle_statistics.extend(meta["candle_statistics"]); shock_statistics.extend(meta["shock_statistics"])
            progress(f"Reused complete forecast panels for {day}.")
            continue
        quote_groups, event_groups, volume_groups = {}, {}, {}
        day_coverage, day_feed, day_candles, day_shocks = [], [], [], []
        for dataset in venues:
            payload = _read(prepared_paths[(dataset, day)])
            for symbol in symbols:
                key = dataset, symbol, day
                quotes = payload["quotes"].loc[payload["quotes"].symbol.eq(symbol)].reset_index(drop=True)
                quote_groups[key] = quotes
                events = extract_shocks(quotes, shocks[(dataset, symbol)])
                event_groups[key] = events.time_ns.to_numpy(dtype=np.int64)
                day_shocks.append({"dataset": dataset, "symbol": symbol, "day": day, "shocks": len(events),
                                   "split": "training" if day in train_days else "held_out"})
                trades = payload["trades"].loc[payload["trades"].symbol.eq(symbol)].reset_index(drop=True)
                trades.attrs.update(payload["trades"].attrs)
                bars = build_volume_bars(trades, thresholds[(dataset, symbol)])
                volume_groups[key] = bars
                day_candles.append({"dataset": dataset, "symbol": symbol, "day": day,
                                    "split": "training" if day in train_days else "held_out",
                                    "completed_bars": len(bars), "threshold_shares": thresholds[(dataset, symbol)],
                                    "completed_volume_shares": float(bars.volume_shares.sum()),
                                    "median_duration_ms": float(bars.duration_ms.median()) if len(bars) else None,
                                    "p95_duration_ms": float(bars.duration_ms.quantile(.95)) if len(bars) else None,
                                    "incomplete_volume_shares": bars.attrs.get("incomplete_volume_shares", 0.),
                                    "barrier_discarded_volume_shares": bars.attrs.get("barrier_discarded_volume_shares", 0.)})
            del payload
        for dataset in venues:
            for symbol in symbols:
                reference = _reference_names(dataset, symbol, venues, symbols)
                volume_names = [f"volume_{v}_{feature}" for v in venues for feature in FEATURES]
                # Require exactly the original model's same-stock and proxy
                # feeds; an unrelated stock cannot censor this target.
                controls = (["SPY"] if symbol != "SPY" and "SPY" in symbols else
                            [s for s in ("AAPL", "MSFT") if s in symbols] if symbol == "SPY" else
                            [s for s in symbols if s != symbol])
                required_symbols = set([symbol, *controls])
                if any(group.empty for key, group in quote_groups.items() if key[1] in required_symbols):
                    rows, attr, candle = pd.DataFrame(), {}, {"quote_eligible_bins": 0, "common_bins": 0, "by_venue": []}
                else:
                    rows, baseline, additional = build_prediction_rows(
                        quote_groups, event_groups, dataset, symbol, day, anchor_step_ms=anchor_step_ms)
                    attr = dict(rows.attrs)
                    if len(rows) and baseline + additional != reference:
                        raise ValueError("Reference feature order differs from existing network model.")
                    rows, volume_names, candle = attach_volume_features(
                        rows, volume_groups, thresholds, venues=venues,
                        max_age_ms=protocol["volume_state_maximum_age_ms"])
                feature_names = reference + volume_names
                x = rows[feature_names].to_numpy(dtype=np.float64) if len(rows) else np.empty((0, len(feature_names)), dtype=np.float64)
                y = rows.y.to_numpy(dtype=np.int8) if len(rows) else np.empty(0, dtype=np.int8)
                panel = {"feature_names": feature_names, "reference_feature_count": len(reference),
                         "x": x, "y": y, "time_ns": rows.time_ns.to_numpy(dtype=np.int64) if len(rows) else np.empty(0, dtype=np.int64)}
                _dump(_panel_path(cache, dataset, symbol, day), panel)
                day_coverage.append({"dataset": dataset, "symbol": symbol, "day": day,
                                     "split": "training" if day in train_days else "held_out",
                                     "candidate_bins": attr.get("candidate_bins", 0),
                                     "quote_eligible_bins": candle["quote_eligible_bins"],
                                     "common_bins": len(rows), "positive_bins": int(y.sum())})
                day_feed.extend({"dataset": dataset, "symbol": symbol, "day": day, "kind": "quote", **r}
                                for r in attr.get("feed_coverage", []))
                day_feed.extend({"dataset": dataset, "symbol": symbol, "day": day, "kind": "volume", **r}
                                for r in candle.get("by_venue", []))
                del panel, x, y, rows
        meta = {"coverage": day_coverage, "feed_coverage": day_feed,
                "candle_statistics": day_candles, "shock_statistics": day_shocks}
        _dump(day_checkpoint, meta)
        coverage.extend(day_coverage); feed_coverage.extend(day_feed)
        candle_statistics.extend(day_candles); shock_statistics.extend(day_shocks)
        progress(f"Forecast panels {day}: {sum(r['common_bins'] for r in day_coverage):,} paired rows across {len(expected_tasks)} targets.")
        del quote_groups, event_groups, volume_groups
        gc.collect()
    pd.DataFrame(coverage).to_csv(out_dir / "coverage.csv", index=False)
    pd.DataFrame(feed_coverage).to_csv(out_dir / "feed_coverage.csv", index=False)
    pd.DataFrame(candle_statistics).to_csv(out_dir / "candle_statistics.csv", index=False)
    pd.DataFrame(shock_statistics).to_csv(out_dir / "shock_statistics.csv", index=False)
    results, daily_rows, coefficient_rows = [], [], []
    for dataset in venues:
        for symbol in symbols:
            checkpoint = cache / "models" / f"{dataset}_{symbol}.pkl"
            if checkpoint.exists():
                result = _read(checkpoint)
            else:
                progress(f"Fitting frozen reference and volume models: {dataset} {symbol}.")
                result = _fit_task(cache, dataset, symbol, config)
                _dump(checkpoint, result)
            daily_rows.extend(result["daily"]); coefficient_rows.extend(result["coefficients"])
            results.append({k: v for k, v in result.items() if k not in ("daily", "coefficients")})
            progress(f"Scored {dataset} {symbol}: {sum(r['rows'] for r in result['daily']):,} held-out rows; status={result['status']}.")
            gc.collect()
    daily_tasks = pd.DataFrame(daily_rows)
    daily_tasks.to_csv(out_dir / "task_daily_scores.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(out_dir / "coefficients.csv", index=False)
    daily = daily_tasks.groupby("day", as_index=False).sum(numeric_only=True)
    daily.to_csv(out_dir / "daily_scores.csv", index=False)
    summary = summarize_confirmation(daily, expected_days=test_days)
    missing_task_days = daily_tasks.loc[daily_tasks.rows.le(0), ["dataset", "symbol", "day"]].to_dict("records")
    nonconverged = [{"dataset": r["dataset"], "symbol": r["symbol"], "model": name}
                    for r in results for name, meta in r["convergence"].items() if not meta["converged"]]
    if missing_task_days or nonconverged:
        summary["status"] = "insufficient_coverage_or_model_convergence"
        summary["supports_positive_incremental_prediction"] = False
    summary.update({"created_utc": datetime.now(timezone.utc).isoformat(),
                    "analysis_sha256": digest, "protocol_sha256": protocol_sha,
                    "configuration": config, "source_rows": sum(entry["rows"] for entry in entries),
                    "requests": len(entries), "anchor_step_ms": anchor_step_ms, "label_horizon_ms": 10,
                    "source_clock": "ts_recv", "task_results": results, "missing_task_days": missing_task_days,
                    "nonconverged_models": nonconverged,
                    "coverage": {"candidate_bins": int(sum(r["candidate_bins"] for r in coverage if r["split"] == "held_out")),
                                 "quote_eligible_bins": int(sum(r["quote_eligible_bins"] for r in coverage if r["split"] == "held_out")),
                                 "paired_bins": int(daily.rows.sum())},
                    "cache_directory": str(cache.resolve()),
                    "interpretation": protocol["interpretation"]})
    write_json(out_dir / "confirmation_summary.json", summary)
    progress(f"Confirmation complete: {int(daily.rows.sum()):,} paired held-out rows; {summary.get('status', 'ok')}.")
    return summary
