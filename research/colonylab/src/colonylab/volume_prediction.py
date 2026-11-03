"""Matched comparison of completed time and volume candles on one target clock.

The candles change the predictors, while the original 10-ms shock labels and
quote-coverage rules stay fixed. Test dates have been used in earlier research,
so this is an exploratory paired comparison, not a new independent validation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .network import chronological_split, prepare_quotes
from .network_models import _scores, build_prediction_rows
from .volume_bars import (build_time_bars, build_volume_bars, fit_volume_thresholds,
                          normalize_trades, sample_completed_bars)


MODEL_NAMES = ("reference", "time", "volume")


def attach_candle_features(
    rows: pd.DataFrame, time_bars: dict, volume_bars: dict, thresholds: dict,
    *, venues: list[str], max_age_ms: float = 10_000,
) -> tuple[pd.DataFrame, list[str], list[str], dict]:
    """Attach the same nine candle quantities from each venue and one common mask.

    Time candles contain [start,end) prints and are available at end<=t.
    Volume candles include the closing print, requiring close<t. Thus every
    underlying trade used by either family precedes the forecast anchor.
    """
    if not np.isfinite(max_age_ms) or max_age_ms <= 0:
        raise ValueError("Candle age must be positive and finite.")
    if rows.empty:
        return rows.copy(), [], [], {"quote_eligible_bins": 0, "common_bins": 0}
    if any(rows[column].nunique() != 1 for column in ("dataset", "symbol", "day")):
        raise ValueError("Attach candles to one target venue/stock/day at a time.")
    if not venues or len(set(venues)) != len(venues):
        raise ValueError("Candle venues must be nonempty and unique.")
    symbol, day = str(rows.symbol.iloc[0]), str(rows.day.iloc[0])
    anchors = rows.time_ns.to_numpy(dtype=np.int64)
    additions, feature_names = {}, {name: [] for name in ("time", "volume")}
    common = np.ones(len(rows), dtype=bool)
    coverage = []
    for family, groups in (("time", time_bars), ("volume", volume_bars)):
        for venue in venues:
            threshold = float(thresholds[(venue, symbol)])
            if not np.isfinite(threshold) or threshold <= 0:
                raise ValueError("Every volume quantum must be finite and positive.")
            bars = groups.get((venue, symbol, day), pd.DataFrame())
            sample = sample_completed_bars(
                bars, anchors, max_age_ns=int(max_age_ms * 1e6),
                include_equal_close=family == "time")
            values = {
                "return_bps": sample.return_bps.to_numpy(dtype=float),
                "range_bps": sample.range_bps.to_numpy(dtype=float),
                "log_volume_multiple": np.log1p(np.expm1(sample.log_volume.to_numpy(dtype=float)) / threshold),
                "log_trade_count": sample.log_trade_count.to_numpy(dtype=float),
                "log_duration_ms": sample.log_duration_ms.to_numpy(dtype=float),
                "signed_volume_fraction": sample.signed_volume_fraction.to_numpy(dtype=float),
                "unknown_volume_fraction": 1 - sample.known_side_fraction.to_numpy(dtype=float),
                "vwap_vs_close_bps": sample.vwap_close_bps.to_numpy(dtype=float),
                "log_age_ms": np.log1p(sample.age_ms.to_numpy(dtype=float)),
            }
            usable = sample.valid.to_numpy(dtype=bool)
            for feature, value in values.items():
                name = f"{family}_{venue}_{feature}"
                additions[name] = value
                feature_names[family].append(name)
                usable &= np.isfinite(value)
            common &= usable
            coverage.append({"family": family, "predictor_dataset": venue,
                             "available_bins": int(usable.sum()), "excluded_bins": int((~usable).sum())})
    result = pd.concat([rows.reset_index(drop=True), pd.DataFrame(additions)], axis=1).loc[common].reset_index(drop=True)
    return result, feature_names["time"], feature_names["volume"], {
        "quote_eligible_bins": len(rows), "common_bins": int(common.sum()),
        "candle_excluded_bins": int((~common).sum()), "by_family_venue": coverage,
    }


def _deltas(scores: dict) -> dict:
    return {
        "delta_reference_vs_time_log_loss": scores["reference"]["log_loss"] - scores["time"]["log_loss"],
        "delta_reference_vs_volume_log_loss": scores["reference"]["log_loss"] - scores["volume"]["log_loss"],
        "delta_time_vs_volume_log_loss": scores["time"]["log_loss"] - scores["volume"]["log_loss"],
        "delta_time_vs_volume_brier": scores["time"]["brier"] - scores["volume"]["brier"],
    }


def fit_candle_models(
    rows: pd.DataFrame, baseline_features: list[str], time_features: list[str],
    volume_features: list[str], training_days: list[str], held_out_days: list[str],
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Fit three fixed models on identical training rows and score identical dates."""
    if len(set(training_days)) != len(training_days):
        raise ValueError("Training dates must be unique.")
    train, held = chronological_split(sorted(set(training_days) | set(held_out_days)), held_out_days)
    if set(train) != set(training_days):
        raise ValueError("Training and held-out dates overlap.")
    if not set(rows.day) <= set(train + held):
        raise ValueError("Unexpected dates entered the candle comparison.")
    training, test = rows.loc[rows.day.isin(train)], rows.loc[rows.day.isin(held)]
    if training.empty or test.empty:
        return {"status": "insufficient_rows", "train_rows": len(training), "test_rows": len(test)}, pd.DataFrame(), pd.DataFrame()
    y_train, y_test = training.y.to_numpy(dtype=int), test.y.to_numpy(dtype=int)
    predictions = test[["dataset", "symbol", "day", "time_ns", "y"]].copy()
    scores, coefficients = {}, []
    one_class = len(np.unique(y_train)) < 2
    with threadpool_limits(limits=1):
        for name, features in (("reference", baseline_features),
                               ("time", baseline_features + time_features),
                               ("volume", baseline_features + volume_features)):
            if one_class:
                probabilities = np.full(len(test), float(y_train.mean()))
            else:
                scaler = StandardScaler()
                x_train = scaler.fit_transform(training[features].to_numpy(dtype=float))
                model = LogisticRegression(C=1., solver="lbfgs", max_iter=1000,
                                           class_weight=None, random_state=0)
                model.fit(x_train, y_train)
                probabilities = model.predict_proba(scaler.transform(test[features].to_numpy(dtype=float)))[:, 1]
                coefficients.extend({"model": name, "feature": feature,
                                     "coefficient_standardized": float(model.coef_[0, i]),
                                     "training_mean": float(scaler.mean_[i]),
                                     "training_scale": float(scaler.scale_[i]),
                                     "intercept": float(model.intercept_[0])}
                                    for i, feature in enumerate(features))
            predictions[f"{name}_probability"] = probabilities
            scores[name] = _scores(y_test, probabilities)
    result = {"status": "one_class_constant_fallback" if one_class else "ok",
              "train_rows": len(training), "test_rows": len(test),
              "train_positive_bins": int(y_train.sum()), "test_positive_bins": int(y_test.sum()),
              "reference_feature_count": len(baseline_features),
              "time_feature_count": len(baseline_features) + len(time_features),
              "volume_feature_count": len(baseline_features) + len(volume_features),
              **scores, **_deltas(scores)}
    return result, predictions, pd.DataFrame(coefficients)


def _new_chunks() -> dict:
    return {name: [] for name in ("y", *MODEL_NAMES)}


def _pool(chunks: dict) -> dict:
    if not chunks["y"]:
        return {"test_rows": 0, "status": "no_eligible_bins"}
    y = np.concatenate(chunks["y"])
    scores = {name: _scores(y, np.concatenate(chunks[name])) for name in MODEL_NAMES}
    return {"test_rows": len(y), "test_positive_bins": int(y.sum()), **scores, **_deltas(scores)}


def run_volume_comparison(
    frames: dict, shock_path: Path, outdir: Path, *, training_days: list[str],
    held_out_days: list[str], target_seconds: float = 1., max_age_ms: float = 10_000,
    progress: Callable[[str], None] = print,
) -> dict:
    """Build genuine trade candles and compare them on a fixed common forecast set."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    observed = sorted({key[2] for key in frames})
    checked_train, checked_test = chronological_split(observed, held_out_days)
    if checked_train != training_days or checked_test != held_out_days:
        raise ValueError("Candle comparison must use the exact explicit chronological split.")
    trades_by_frame, quote_groups, quality = {}, {}, []
    for (dataset, schema, day), frame in sorted(frames.items()):
        if schema != "mbp-1":
            continue
        trades, diagnostics = normalize_trades(frame, dataset, day)
        trades_by_frame[(dataset, schema, day)] = trades
        quality.append(diagnostics)
        prepared = prepare_quotes(frame, dataset, day)
        for symbol, group in prepared.groupby("symbol", sort=True):
            quote_groups[(dataset, str(symbol), day)] = group.reset_index(drop=True)
        progress(f"Prepared trade prints and quote states: {day} {dataset}; {len(trades):,} trades.")
    threshold_table = fit_volume_thresholds(trades_by_frame, training_days, target_seconds=target_seconds)
    threshold_table.to_csv(outdir / "volume_thresholds.csv", index=False)
    thresholds = {(r.dataset, r.symbol): int(r.threshold_shares) for r in threshold_table.itertuples()}
    time_groups, volume_groups, statistics, candle_frames = {}, {}, [], {"time": [], "volume": []}
    for (dataset, _, day), trades in sorted(trades_by_frame.items()):
        for symbol, group in trades.groupby("symbol", sort=True):
            group = group.reset_index(drop=True)
            group.attrs.update(trades.attrs)
            key = dataset, str(symbol), day
            time_bars = build_time_bars(group, seconds=target_seconds)
            volume_bars = build_volume_bars(group, thresholds[(dataset, str(symbol))])
            time_groups[key], volume_groups[key] = time_bars, volume_bars
            for name, bars in (("time", time_bars), ("volume", volume_bars)):
                exported = bars.copy(deep=False)
                exported.attrs = {}
                candle_frames[name].append(exported)
                statistics.append({"dataset": dataset, "symbol": str(symbol), "day": day,
                    "split": "training" if day in training_days else "held_out", "family": name,
                    "completed_bars": len(bars), "threshold_shares": thresholds[(dataset, str(symbol))],
                    "median_duration_ms": float(bars.duration_ms.median()) if len(bars) else None,
                    "p95_duration_ms": float(bars.duration_ms.quantile(.95)) if len(bars) else None,
                    "median_volume_shares": float(bars.volume_shares.median()) if len(bars) else None,
                    "median_overrun_shares": float(bars.overrun_shares.median()) if name == "volume" and len(bars) else None,
                    "completed_volume_shares": float(bars.volume_shares.sum()),
                    "trailing_incomplete_volume_shares": bars.attrs.get("incomplete_volume_shares"),
                    "barrier_discarded_volume_shares": bars.attrs.get("barrier_discarded_volume_shares")})
    for name, bars in candle_frames.items():
        pd.concat(bars, ignore_index=True).to_csv(outdir / f"{name}_candles.csv", index=False)
    del candle_frames, trades_by_frame
    pd.DataFrame(quality).to_csv(outdir / "trade_quality.csv", index=False)
    pd.DataFrame(statistics).to_csv(outdir / "candle_statistics.csv", index=False)
    progress("Completed time and volume candles built; fitting matched forecast comparisons.")
    shocks = pd.read_csv(shock_path)
    if set(shocks.day) != set(observed) or "split" not in shocks:
        raise ValueError("Frozen shock events must cover exactly the configured dates and split.")
    expected_split = np.where(shocks.day.isin(held_out_days), "held_out", "training")
    if not np.array_equal(shocks.split.to_numpy(), expected_split):
        raise ValueError("Shock calibration split differs from candle comparison split.")
    event_groups = {tuple(map(str, key)): np.sort(g.time_ns.to_numpy(dtype=np.int64))
                    for key, g in shocks.groupby(["dataset", "symbol", "day"], sort=True)}
    pooled, daily = _new_chunks(), {day: _new_chunks() for day in held_out_days}
    tasks, coverage, scores_out, coefficients_out, task_days = [], [], [], [], []
    for dataset, symbol in sorted({key[:2] for key in quote_groups}):
        pieces, specs = [], None
        for day in observed:
            rows, base, network = build_prediction_rows(quote_groups, event_groups, dataset, symbol, day)
            venues = sorted(key[0] for key in quote_groups if key[1:] == (symbol, day))
            common, time_features, volume_features, diagnostics = attach_candle_features(
                rows, time_groups, volume_groups, thresholds, venues=venues, max_age_ms=max_age_ms)
            coverage.append({"dataset": dataset, "symbol": symbol, "day": day, **diagnostics})
            if common.empty:
                continue
            specification = base + network, time_features, volume_features
            if specs is not None and specs != specification:
                raise ValueError("Candle or venue feature coverage changes across dates.")
            specs = specification
            pieces.append(common)
        identity = {"dataset": dataset, "symbol": symbol}
        if not pieces:
            tasks.append({**identity, "status": "no_common_rows"})
            continue
        panel = pd.concat(pieces, ignore_index=True)
        del pieces
        result, predictions, coefficients = fit_candle_models(
            panel, *specs, training_days, held_out_days)
        del panel
        tasks.append({**identity, **result})
        if not coefficients.empty:
            coefficients_out.append(coefficients.assign(**identity))
        if predictions.empty:
            continue
        for name in MODEL_NAMES:
            scores_out.append({**identity, "model": name, "train_rows": result["train_rows"],
                               "test_rows": result["test_rows"], **result[name]})
        for day, group in predictions.groupby("day", sort=True):
            chunks = _new_chunks()
            chunks["y"].append(group.y.to_numpy(dtype=np.int8, copy=True))
            for name in MODEL_NAMES:
                chunks[name].append(group[f"{name}_probability"].to_numpy(dtype=float, copy=True))
            task_days.append({**identity, "day": day, **_pool(chunks)})
            for name, arrays in chunks.items():
                pooled[name].extend(arrays)
                daily[day][name].extend(arrays)
        progress(f"Scored {dataset}/{symbol}: {result['test_rows']:,} common held-out bins.")
    pd.DataFrame(scores_out).to_csv(outdir / "candle_model_scores.csv", index=False)
    pd.DataFrame([{k: v for k, v in row.items() if k != "by_family_venue"} for row in coverage]).to_csv(
        outdir / "candle_coverage.csv", index=False)
    pd.DataFrame([{**{k: row[k] for k in ("dataset", "symbol", "day")}, **item}
                  for row in coverage for item in row.get("by_family_venue", [])]).to_csv(
        outdir / "candle_feed_coverage.csv", index=False)
    pd.concat(coefficients_out, ignore_index=True).to_csv(outdir / "candle_coefficients.csv", index=False) if coefficients_out else None
    daily_scores = [{"day": day, **_pool(daily[day])} for day in held_out_days]
    pd.DataFrame([{**{k: r[k] for k in ("dataset", "symbol", "day", "test_rows") if k in r},
                   "model": name, **r[name]} for r in [*daily_scores, *task_days]
                  for name in MODEL_NAMES if name in r]).to_csv(outdir / "candle_daily_scores.csv", index=False)
    result = {
        "experiment": "matched completed volume-candle versus time-candle predictor comparison",
        "training_days": training_days, "held_out_days": held_out_days,
        "volume_target_seconds": target_seconds, "time_candle_seconds": target_seconds,
        "max_candle_age_ms": max_age_ms, "forecast_bin_ms": 10,
        "target": "Unchanged next-10-ms frozen-threshold quote-shock label from the wider study.",
        "reference": "Original network shock-history predictors, refitted on the common candle-observable rows.",
        "candle_features": "Nine matched completed-candle features on every same-stock venue: return, range, volume/quantum, trade count, duration, signed fraction, unknown fraction, VWAP/close difference and age.",
        "volume_rule": "Per venue/stock shares = rounded training trade volume / full requested training seconds, times target_seconds; whole timestamp groups may overshoot; incomplete tails excluded.",
        "availability": "Time candle [start,end) is available at end<=t; volume candle includes its closing print and requires close<t. All predictor trades strictly precede t.",
        "matching": "All three models use identical eligible rows, labels and training-only scalers; fixed L2 C=1, no tuning.",
        "delta_sign": "Positive time-minus-volume loss favors volume candles.",
        "pooled": _pool(pooled), "per_day": daily_scores, "tasks": tasks, "task_days": task_days,
        "limitations": ["Exploratory reuse of five previously inspected test dates; fresh-date confirmation is still needed.",
                        "The forecasting clock stays at 10 ms; this tests candle representations, not a change to volume-event prediction targets.",
                        "Bars from different venues are asynchronous; quote and candle coverage select active observable periods.",
                        "Whole-print bars have volume overshoot and can close immediately on large trades; no synthetic split prints are created.",
                        "Dependent bins and market factors prevent independent-bin significance or causal impact claims.",
                        "Participant identities, maker opportunity and RPI reports are not used as predictors."],
        "outputs": {name: str((outdir / name).resolve()) for name in (
            "volume_thresholds.csv", "time_candles.csv", "volume_candles.csv", "candle_statistics.csv",
            "trade_quality.csv", "candle_model_scores.csv", "candle_coverage.csv", "candle_feed_coverage.csv",
            "candle_coefficients.csv", "candle_daily_scores.csv")},
    }
    (outdir / "volume_comparison_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result
