"""Chronological prediction check for incremental cross-venue information.

Targets are quote-shock arrivals in the next 10 ms. Every predictor precedes
the bin boundary. Models are frozen before the held-out dates; scores describe
predictive association and do not identify causal venue effects.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .network import chronological_split, prepare_quotes


BIN_MS = 10
LAG_WINDOWS_MS = (10, 100, 1000)
HISTORY_MS = 1000


def lag_counts(events: np.ndarray, anchors: np.ndarray, window_ms: int) -> np.ndarray:
    """Count timestamps in [t-window, t), never the current timestamp."""
    events = np.sort(np.asarray(events, dtype=np.int64))
    return (np.searchsorted(events, anchors, side="left") -
            np.searchsorted(events, anchors - window_ms * 1_000_000, side="left"))


def future_labels(events: np.ndarray, anchors: np.ndarray, horizon_ms: int = BIN_MS) -> np.ndarray:
    """An event at t belongs to this outcome; one at t+h belongs to the next."""
    events = np.sort(np.asarray(events, dtype=np.int64))
    return (np.searchsorted(events, anchors + horizon_ms * 1_000_000, side="left") >
            np.searchsorted(events, anchors, side="left")).astype(np.int8)


def quote_coverage(quotes: pd.DataFrame, anchors: np.ndarray, *, future_ms: int = 0) -> np.ndarray:
    """Require a complete fresh past second, plus an observable outcome if set."""
    if quotes.empty:
        return np.zeros(len(anchors), dtype=bool)
    qt = quotes.time_ns.to_numpy(dtype=np.int64)
    valid = quotes.valid.to_numpy(dtype=bool) & np.isfinite(quotes.mid.to_numpy(dtype=float))
    segment = quotes.segment.to_numpy(dtype=np.int64)
    starts = anchors - HISTORY_MS * 1_000_000
    ends = anchors + future_ms * 1_000_000
    before = np.searchsorted(qt, anchors, side="left") - 1
    at_start = np.searchsorted(qt, starts, side="right") - 1
    # Outcome bins are half-open; a barrier exactly at the next bin belongs there.
    right = np.searchsorted(qt, ends, side="left")
    endpoint = right - 1
    left = np.searchsorted(qt, starts, side="left")
    b, a, e = np.maximum(before, 0), np.maximum(at_start, 0), np.maximum(endpoint, 0)
    good = (before >= 0) & (at_start >= 0) & (endpoint >= 0) & (ends <= qt[-1])
    good &= valid[b] & valid[a] & valid[e]
    good &= anchors - qt[b] <= HISTORY_MS * 1_000_000
    good &= ends - qt[e] <= HISTORY_MS * 1_000_000
    good &= segment[a] == segment[e]
    invalid_prefix = np.r_[0, np.cumsum(~valid)]
    good &= invalid_prefix[right] - invalid_prefix[left] == 0
    return good


def _event_times(events: dict, key: tuple[str, str, str]) -> np.ndarray:
    return np.asarray(events.get(key, np.array([], dtype=np.int64)), dtype=np.int64)


def build_prediction_rows(
    quotes: dict[tuple[str, str, str], pd.DataFrame],
    events: dict[tuple[str, str, str], np.ndarray],
    dataset: str, symbol: str, day: str,
    *, anchor_step_ms: int = BIN_MS,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build a task/day panel from normalized quotes and frozen shock events.

    Keys use (dataset, symbol, day). The same eligible rows feed both models.
    Market controls use SPY across observed venues for stock targets; SPY uses
    AAPL and MSFT where available. Synthetic symbols use the other symbols.
    """
    if isinstance(anchor_step_ms, bool) or not isinstance(anchor_step_ms, (int, np.integer)) or anchor_step_ms <= 0:
        raise ValueError("Forecast anchor spacing must be a positive integer number of milliseconds.")
    target_key = dataset, symbol, day
    if target_key not in quotes:
        return pd.DataFrame(), [], []
    symbols = sorted({key[1] for key in quotes if key[2] == day})
    if symbol != "SPY" and "SPY" in symbols:
        controls = ["SPY"]
    elif symbol == "SPY":
        controls = [item for item in ("AAPL", "MSFT") if item in symbols]
    else:
        controls = [item for item in symbols if item != symbol]
    venues = sorted({key[0] for key in quotes if key[1:] == (symbol, day)})
    required = [(venue, symbol, day) for venue in venues]
    control_keys = [key for key in quotes if key[2] == day and key[1] in controls]
    required.extend(control_keys)
    if len(venues) < 2 or not required:
        return pd.DataFrame(), [], []
    start = max(int(quotes[key].time_ns.min()) for key in required)
    end = min(int(quotes[key].time_ns.max()) for key in required)
    step = int(anchor_step_ms) * 1_000_000
    first = ((start + HISTORY_MS * 1_000_000 + step - 1) // step) * step
    # Thinning forecast anchors never changes the 10-ms outcome horizon.
    anchors = np.arange(first, end - BIN_MS * 1_000_000 + 1, step, dtype=np.int64)
    if not len(anchors):
        return pd.DataFrame(), [], []
    eligible = np.ones(len(anchors), dtype=bool)
    feed_coverage = []
    for key in required:
        future_ms = BIN_MS if key == target_key else 0
        feed_good = quote_coverage(quotes[key], anchors, future_ms=future_ms)
        eligible &= feed_good
        feed_coverage.append({"predictor_dataset": key[0], "predictor_symbol": key[1],
                              "future_ms": future_ms, "candidate_bins": len(anchors),
                              "eligible_bins": int(feed_good.sum()),
                              "excluded_bins": int((~feed_good).sum())})
    rows = pd.DataFrame({
        "time_ns": anchors, "dataset": dataset, "symbol": symbol, "day": day,
        "y": future_labels(_event_times(events, target_key), anchors),
        # Fixed elapsed seconds are comparable between dates without fitting a
        # test-window normalization or using its labels.
        "elapsed_seconds": (anchors - start) / 1e9,
    })
    baseline = ["elapsed_seconds"]
    for window in LAG_WINDOWS_MS:
        name = f"own_shocks_{window}ms"
        rows[name] = lag_counts(_event_times(events, target_key), anchors, window)
        baseline.append(name)
    for control in controls:
        times = np.sort(np.concatenate([_event_times(events, key) for key in control_keys if key[1] == control]))
        for window in LAG_WINDOWS_MS:
            name = f"market_{control}_shocks_{window}ms"
            rows[name] = lag_counts(times, anchors, window)
            baseline.append(name)
    additional = []
    for venue in venues:
        if venue == dataset:
            continue
        for window in LAG_WINDOWS_MS:
            name = f"other_{venue}_shocks_{window}ms"
            rows[name] = lag_counts(_event_times(events, (venue, symbol, day)), anchors, window)
            additional.append(name)
    result = rows.loc[eligible].reset_index(drop=True)
    result.attrs.update({"candidate_bins": len(anchors), "eligible_bins": int(eligible.sum()),
                         "excluded_bins": int((~eligible).sum()), "coverage_feed_count": len(required),
                         "market_controls": ",".join(controls), "common_start_ns": start, "common_end_ns": end,
                         "feed_coverage": feed_coverage})
    return result, baseline, additional


def _scores(y: np.ndarray, probabilities: np.ndarray) -> dict:
    p = np.clip(probabilities, 1e-12, 1 - 1e-12)
    both = len(np.unique(y)) == 2
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "average_precision": float(average_precision_score(y, p)) if np.any(y == 1) else None,
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
    }


def fit_prediction_models(
    rows: pd.DataFrame, baseline_features: list[str], additional_features: list[str],
    training_days: list[str], held_out_day: str | list[str],
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Fit both scalers and L2 logistic models on earlier dates only."""
    held_out_days = [held_out_day] if isinstance(held_out_day, str) else list(held_out_day)
    observed = sorted(set(training_days) | set(held_out_days))
    checked_training, held_out_days = chronological_split(observed, held_out_days)
    if set(checked_training) != set(training_days):
        raise ValueError("Training and held-out dates must be disjoint.")
    training = rows.loc[rows.day.isin(training_days)]
    held = rows.loc[rows.day.isin(held_out_days)]
    if training.empty or held.empty:
        return {"status": "insufficient_eligible_train_or_test_rows", "train_rows": len(training), "test_rows": len(held)}, pd.DataFrame(), pd.DataFrame()
    y_train, y_test = training.y.to_numpy(dtype=int), held.y.to_numpy(dtype=int)
    rate = float(np.mean(y_train))
    predictions = held[["dataset", "symbol", "day", "time_ns", "y"]].copy()
    predictions["constant_probability"] = rate
    scores = {"constant": _scores(y_test, np.full(len(held), rate))}
    coefficients = []
    status = "ok" if len(np.unique(y_train)) == 2 else "one_training_class_constant_fallback"
    with threadpool_limits(limits=1):
        for name, features in (("baseline", baseline_features), ("network", baseline_features + additional_features)):
            if status != "ok":
                probability = np.full(len(held), rate)
            else:
                scaler = StandardScaler()
                x = scaler.fit_transform(training[features].to_numpy(dtype=float))
                model = LogisticRegression(C=1.0, class_weight=None, solver="lbfgs", max_iter=1000, random_state=0)
                model.fit(x, y_train)
                probability = model.predict_proba(scaler.transform(held[features].to_numpy(dtype=float)))[:, 1]
                for i, feature in enumerate(features):
                    coefficients.append({"model": name, "feature": feature,
                        "coefficient_standardized": float(model.coef_[0, i]),
                        "training_mean": float(scaler.mean_[i]), "training_scale": float(scaler.scale_[i]),
                        "intercept": float(model.intercept_[0])})
            predictions[f"{name}_probability"] = probability
            scores[name] = _scores(y_test, probability)
    result = {
        "status": status, "train_rows": len(training), "train_positive_bins": int(y_train.sum()),
        "test_rows": len(held), "test_positive_bins": int(y_test.sum()), "train_positive_rate": rate,
        "test_positive_rate": float(y_test.mean()), "baseline_feature_count": len(baseline_features),
        "network_feature_count": len(baseline_features) + len(additional_features), **scores,
        "delta_log_loss": scores["baseline"]["log_loss"] - scores["network"]["log_loss"],
        "delta_brier": scores["baseline"]["brier"] - scores["network"]["brier"],
    }
    return result, predictions, pd.DataFrame(coefficients)


def _new_probability_chunks() -> dict[str, list[np.ndarray]]:
    return {name: [] for name in ("y", "constant", "baseline", "network")}


def _chunk_scores(chunks: dict[str, list[np.ndarray]]) -> dict:
    """Pool compact numeric arrays without retaining stock/venue/bin tables."""
    if not chunks["y"]:
        return {"status": "no_eligible_rows", "test_rows": 0, "test_positive_bins": 0}
    y = np.concatenate(chunks["y"])
    scores = {model: _scores(y, np.concatenate(chunks[model]))
              for model in ("constant", "baseline", "network")}
    return {"test_rows": len(y), "test_positive_bins": int(y.sum()), **scores,
            "delta_log_loss": scores["baseline"]["log_loss"] - scores["network"]["log_loss"],
            "delta_brier": scores["baseline"]["brier"] - scores["network"]["brier"]}


def _metric_rows(records: list[dict]) -> list[dict]:
    """Flatten daily nested metrics into one auditable row per model."""
    result = []
    for record in records:
        identity = {key: record[key] for key in ("dataset", "symbol", "day", "test_rows", "test_positive_bins")
                    if key in record}
        for model in ("constant", "baseline", "network"):
            if model in record:
                result.append({**identity, "model": model, **record[model]})
    return result


def run_prediction_test(
    frames: dict[tuple[str, str, str], pd.DataFrame], shock_path: Path, outdir: Path,
    *, held_out_days: list[str] | None = None, save_predictions: bool = True,
) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    quote_groups = {}
    clocks = set()
    for (dataset, schema, day), frame in sorted(frames.items()):
        if schema != "mbp-1":
            continue
        # Same audited atomic quote states and barriers used by the event study.
        prepared = prepare_quotes(frame, dataset, day)
        clocks.add("ts_recv" if "ts_recv" in frame or frame.index.name == "ts_recv" else "ts_event")
        for symbol, group in prepared.groupby("symbol", sort=True):
            quote_groups[(dataset, str(symbol), day)] = group.reset_index(drop=True)
    if len(clocks) != 1:
        raise ValueError("Prediction tests require one consistent analysis clock.")
    days = sorted({key[2] for key in quote_groups})
    training_days, held_out_days = chronological_split(days, held_out_days)
    held_out_day = held_out_days[0] if len(held_out_days) == 1 else None
    shocks = pd.read_csv(shock_path)
    required_columns = {"dataset", "symbol", "day", "time_ns"}
    if not required_columns.issubset(shocks.columns):
        raise ValueError("shock_events.csv is missing event provenance columns.")
    if "split" in shocks:
        expected_split = np.where(shocks["day"].isin(held_out_days), "held_out", "training")
        if not np.array_equal(shocks["split"].to_numpy(), expected_split):
            raise ValueError("Shock-event split differs from the requested training/held-out dates; regenerate thresholds with the same split.")
    event_groups = {tuple(map(str, key)): np.sort(group.time_ns.to_numpy(dtype=np.int64))
                    for key, group in shocks.groupby(["dataset", "symbol", "day"], sort=True)}
    metrics, coefficient_frames, summaries, coverage, task_day_scores = [], [], [], [], []
    feed_coverage = []
    pooled_chunks = _new_probability_chunks()
    day_chunks = {day: _new_probability_chunks() for day in held_out_days}
    prediction_path = outdir / "prediction_test_bins.csv"
    # A compact rerun must not leave a stale prediction table from a prior run.
    prediction_path.unlink(missing_ok=True)
    prediction_header_written = False
    for dataset, symbol in sorted({key[:2] for key in quote_groups}):
        task_rows, feature_specification = [], None
        for day in days:
            rows, baseline, additional = build_prediction_rows(quote_groups, event_groups, dataset, symbol, day)
            coverage.append({"dataset": dataset, "symbol": symbol, "day": day,
                             **{name: value for name, value in rows.attrs.items() if name != "feed_coverage"}})
            feed_coverage.extend({"target_dataset": dataset, "target_symbol": symbol, "day": day, **item}
                                 for item in rows.attrs.get("feed_coverage", []))
            if rows.empty:
                continue
            specification = (baseline, additional)
            if feature_specification is not None and specification != feature_specification:
                raise ValueError("Venue/market-control feature coverage differs between dates.")
            feature_specification = specification
            task_rows.append(rows)
        identity = {"dataset": dataset, "symbol": symbol}
        if not task_rows:
            summaries.append({**identity, "status": "no_eligible_rows"})
            continue
        panel = pd.concat(task_rows, ignore_index=True)
        del task_rows
        result, predictions, coefficients = fit_prediction_models(panel, *feature_specification, training_days, held_out_days)
        del panel
        summaries.append({**identity, **result})
        if predictions.empty:
            continue
        if save_predictions:
            predictions.to_csv(prediction_path, mode="a", header=not prediction_header_written, index=False)
            prediction_header_written = True
        for day, group in predictions.groupby("day", sort=True):
            chunks = _new_probability_chunks()
            chunks["y"].append(group.y.to_numpy(dtype=np.int8, copy=True))
            for model in ("constant", "baseline", "network"):
                chunks[model].append(group[f"{model}_probability"].to_numpy(dtype=float, copy=True))
            task_day_scores.append({**identity, "day": day, **_chunk_scores(chunks)})
            # The task/day and pooled collections share each compact array.
            for name, arrays in chunks.items():
                day_chunks[day][name].extend(arrays)
                pooled_chunks[name].extend(arrays)
        if not coefficients.empty:
            coefficient_frames.append(coefficients.assign(**identity))
        for model in ("constant", "baseline", "network"):
            metrics.append({**identity, "model": model, "train_rows": result["train_rows"],
                "train_positive_bins": result["train_positive_bins"], "test_rows": result["test_rows"],
                "test_positive_bins": result["test_positive_bins"], **result[model]})
    pd.DataFrame(metrics).to_csv(outdir / "prediction_scores.csv", index=False)
    pd.DataFrame(coverage).to_csv(outdir / "prediction_coverage.csv", index=False)
    pd.DataFrame(feed_coverage).to_csv(outdir / "prediction_feed_coverage.csv", index=False)
    if save_predictions and not prediction_header_written:
        pd.DataFrame().to_csv(prediction_path, index=False)
    pd.concat(coefficient_frames, ignore_index=True).to_csv(outdir / "prediction_coefficients.csv", index=False) if coefficient_frames else pd.DataFrame().to_csv(outdir / "prediction_coefficients.csv", index=False)
    per_day_scores = [{"day": day, **_chunk_scores(day_chunks[day])} for day in held_out_days]
    pooled_result = _chunk_scores(pooled_chunks)
    pooled = {key: value for key, value in pooled_result.items()
              if key not in ("test_rows", "test_positive_bins", "status")}
    pd.DataFrame(_metric_rows(per_day_scores)).to_csv(outdir / "prediction_scores_by_day.csv", index=False)
    pd.DataFrame(_metric_rows(task_day_scores)).to_csv(outdir / "prediction_task_scores_by_day.csv", index=False)
    output_names = ["prediction_scores.csv", "prediction_coverage.csv", "prediction_feed_coverage.csv", "prediction_coefficients.csv",
                    "prediction_scores_by_day.csv", "prediction_task_scores_by_day.csv"]
    if save_predictions:
        output_names.append("prediction_test_bins.csv")
    summary = {
        "experiment": "incremental cross-venue prediction of a quote shock in the next 10 ms",
        "training_days": training_days, "held_out_day": held_out_day, "held_out_days": held_out_days,
        "analysis_clock": next(iter(clocks)),
        "bin_ms": BIN_MS, "lag_windows_ms": list(LAG_WINDOWS_MS), "history_ms": HISTORY_MS,
        "target": "Any frozen-threshold, de-duplicated midpoint shock in [t,t+10ms).",
        "baseline": "Own past shock counts, elapsed seconds, past SPY shock counts across observed venues; SPY targets use past AAPL/MSFT activity.",
        "network": "Identical baseline plus strictly past same-stock shock counts on the other observed venues.",
        "mbo_predictors_used": False,
        "model": "StandardScaler fitted on training rows only, then L2 logistic regression C=1, no class weighting or hyperparameter tuning.",
        "eligibility": "Common observed window, full prior second, fresh prior quotes, no invalid or cross-segment historical states on predictor feeds and no target outcome-bin barriers. Both models score the same rows.",
        "feed_coverage_note": "Per-feed exclusions use the same candidate grid and are not additive because excluded intervals overlap; the joint eligibility rule is unchanged.",
        "improvement_sign": "Positive delta log loss or Brier = baseline loss minus network loss; positive favors network.",
        "tasks": summaries, "coverage_by_day": coverage,
        "pooled_test_bins": pooled_result["test_rows"], "pooled_scores": pooled,
        "per_day_scores": per_day_scores, "task_day_scores": task_day_scores,
        "predictions_saved": save_predictions,
        "prediction_storage": "Task predictions are streamed to CSV; pooled scoring retains only labels and probabilities as numeric arrays."
                              if save_predictions else "Per-bin CSV disabled; pooled scoring retains only compact numeric labels and probabilities.",
        "limitations": [
            "A limited collection of held-out intraday windows does not establish stable prediction across market regimes.",
            "Bins and stock/venue tasks are dependent; pooled scores are descriptive with no independent-bin confidence intervals or p-values.",
            "Past market-proxy activity is a limited common-shock control; news, transport delays, omitted venues and correlated order flow remain confounders.",
            "This estimates predictive association, not causal influence, a maker's identity, a trading strategy or fitted SIR/Hawkes contagion parameters.",
            "De-duplicated target shocks have a minimum separation; target-own histories let the baseline learn part of that construction.",
            "Monthly venue weights and public maker/RPI fields are not predictors because publication-aware historical alignment has not yet been established.",
        ],
        "outputs": {name: str((outdir / name).resolve()) for name in output_names},
    }
    (outdir / "prediction_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary
