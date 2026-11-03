"""Past-only execution features and held-out descriptive response horizons.

The units being clustered are execution contexts, not investors or parent orders.
Timestamp joins deliberately avoid any assumed ordering among tied records.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


FEATURE_COLUMNS = [
    "log_trade_count", "log_total_volume", "log_mean_size", "size_cv",
    "signed_volume_imbalance", "known_direction_fraction", "sign_persistence",
    "log_mean_gap_ms", "gap_cv", "past_return_bps", "past_volatility_bps",
    "log_last_trade_age_ms", "past_spread_bps", "past_depth_imbalance",
]


def _text(series: pd.Series) -> pd.Series:
    return series.map(lambda x: x.decode() if isinstance(x, bytes) else str(x))


def _number(frame: pd.DataFrame, name: str, default: float = np.nan) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _normalise(frame: pd.DataFrame, dataset: str, schema: str) -> pd.DataFrame:
    if schema not in {"mbp-1", "trades"}:
        raise ValueError("This MVP supports mbp-1 and trades, not order-lifecycle MBO inference.")
    if frame.empty:
        raise ValueError("The input data contain no records.")
    data = frame.copy()
    if "ts_event" in data:
        timestamps = pd.to_datetime(data["ts_event"], utc=True, errors="coerce")
    elif isinstance(data.index, pd.DatetimeIndex) and data.index.name == "ts_event":
        timestamps = pd.Series(pd.to_datetime(data.index, utc=True), index=data.index)
    else:
        raise ValueError("A ts_event column (or named DatetimeIndex) is required; ts_recv is not a substitute.")
    data["ts_event"] = np.asarray(timestamps)
    data["ts_event"] = pd.to_datetime(data["ts_event"], utc=True).astype("datetime64[ns, UTC]")
    if "ts_recv" in data:
        receive_times = pd.to_datetime(data["ts_recv"], utc=True, errors="coerce")
        data["ts_analysis"] = np.asarray(receive_times)
        data["analysis_clock"] = "ts_recv"
    elif isinstance(data.index, pd.DatetimeIndex) and data.index.name == "ts_recv":
        data["ts_analysis"] = np.asarray(pd.to_datetime(data.index, utc=True))
        data["analysis_clock"] = "ts_recv"
    else:
        data["ts_analysis"] = data["ts_event"]
        data["analysis_clock"] = "ts_event"
    data["ts_analysis"] = pd.to_datetime(data["ts_analysis"], utc=True).astype("datetime64[ns, UTC]")
    if data["ts_analysis"].isna().any():
        raise ValueError("The selected analysis clock has missing timestamps; supply a complete timestamp field.")
    data = data.loc[data["ts_event"].notna()].copy()
    if data.empty:
        raise ValueError("No usable event timestamps were found.")
    if "symbol" in data:
        data["symbol"] = _text(data["symbol"])
    elif "instrument_id" in data:
        data["symbol"] = _text(data["instrument_id"])
    else:
        data["symbol"] = "UNKNOWN"
    if dataset.upper() == "EQUS.MINI":
        data["response_feed"] = "EQUS.MINI_AGGREGATE"
    elif "publisher_id" in data:
        data["response_feed"] = _text(data["publisher_id"])
    else:
        data["response_feed"] = dataset
    data["session_date"] = data["ts_analysis"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    data["_time_ns"] = data["ts_analysis"].astype("int64")
    data["_price"] = _number(data, "price")
    data["_size"] = _number(data, "size", 0)
    # DBNStore.to_df(pretty_ts=True, pretty_px=True) supplies price-unit floats.
    # Sentinel/implausibly large values must not become prices or midpoints.
    data.loc[(data["_price"] <= 0) | (data["_price"] >= 1e8), "_price"] = np.nan
    if "side" in data:
        sides = _text(data["side"]).str.upper()
        data["_sign"] = sides.map({"B": 1, "A": -1}).fillna(0).astype(int)
    else:
        data["_sign"] = 0
    if schema == "mbp-1" and "action" not in data:
        raise ValueError("mbp-1 requires action to distinguish trades from quote records.")
    is_trade = _text(data["action"]).str.upper().eq("T") if "action" in data else np.ones(len(data), dtype=bool)
    data["_is_trade"] = is_trade & data["_price"].notna() & (data["_size"] > 0)
    data["_quote_mode"] = schema == "mbp-1"
    if schema == "mbp-1":
        if not {"bid_px_00", "ask_px_00"}.issubset(data.columns):
            raise ValueError("mbp-1 analysis requires bid_px_00 and ask_px_00.")
        bid, ask = _number(data, "bid_px_00"), _number(data, "ask_px_00")
        valid = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask >= bid) & (ask < 1e8)
        mid = (bid + ask) / 2
        data["_response_price"] = np.where(valid, mid, np.nan)
        data["_spread_bps"] = np.where(valid, (ask - bid) / mid * 1e4, np.nan)
        bsize, asize = _number(data, "bid_sz_00"), _number(data, "ask_sz_00")
        total = bsize + asize
        depth = np.divide(bsize - asize, total, out=np.zeros(len(data)), where=np.isfinite(total) & (total > 0))
        data["_depth_imbalance"] = np.where(valid, depth, np.nan)
    else:
        data["_response_price"] = np.where(data["_is_trade"], data["_price"], np.nan)
        data["_spread_bps"] = 0.0
        data["_depth_imbalance"] = 0.0
    return data.sort_values("_time_ns", kind="stable").reset_index(drop=True)


def _prefix(values: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(values, dtype=float)]


def _sums(prefix: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return prefix[right] - prefix[left]


def _price_series(group: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep invalid/ambiguous quote states as barriers to carrying old prices."""
    quote_mode = bool(group["_quote_mode"].iloc[0])
    included = np.ones(len(group), dtype=bool) if quote_mode else group["_response_price"].notna()
    prices = group.loc[included, ["_time_ns", "_response_price", "_spread_bps", "_depth_imbalance"]].copy()
    if prices.empty:
        return prices, 0
    # Equal midpoint with different spreads is also an ambiguous quote state.
    distinct = prices.groupby("_time_ns", sort=False)[["_response_price", "_spread_bps", "_depth_imbalance"]].nunique()
    ambiguous = distinct.gt(1).any(axis=1)
    invalid_times = prices.loc[prices["_response_price"].isna(), "_time_ns"]
    invalid_at_time = distinct.index.isin(invalid_times)
    barrier_times = distinct.index[ambiguous | invalid_at_time]
    prices = prices.drop_duplicates("_time_ns", keep="first")
    prices.loc[prices["_time_ns"].isin(barrier_times), ["_response_price", "_spread_bps", "_depth_imbalance"]] = np.nan
    return prices, int(ambiguous.sum())


def _features(group: pd.DataFrame, window_ns: int, max_age_ns: int) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    trades = group.loc[group["_is_trade"]].copy()
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame(), 0
    times = trades["_time_ns"].to_numpy(dtype=np.int64)
    size = trades["_size"].to_numpy(dtype=float)
    price = trades["_price"].to_numpy(dtype=float)
    sign = trades["_sign"].to_numpy(dtype=int, copy=True)
    quote, ambiguous = _price_series(group)
    quote_times = quote["_time_ns"].to_numpy(dtype=np.int64)
    baseline_index = np.searchsorted(quote_times, times, side="left") - 1
    baseline_ok = baseline_index >= 0
    safe_index = np.maximum(baseline_index, 0)
    baseline = np.full(len(trades), np.nan)
    baseline_age = np.full(len(trades), np.nan)
    spread = np.zeros(len(trades))
    depth = np.zeros(len(trades))
    if len(quote):
        baseline[baseline_ok] = quote["_response_price"].to_numpy()[safe_index[baseline_ok]]
        baseline_age[baseline_ok] = times[baseline_ok] - quote_times[safe_index[baseline_ok]]
        spread[baseline_ok] = quote["_spread_bps"].to_numpy()[safe_index[baseline_ok]]
        depth[baseline_ok] = quote["_depth_imbalance"].to_numpy()[safe_index[baseline_ok]]
    baseline_ok &= (baseline_age <= max_age_ns) & np.isfinite(baseline)
    direction_source = np.where(sign != 0, "feed_B_or_A", "unknown_not_inferred").astype(object)
    # A quote-rule inference remains an estimate, distinct from a supplied side.
    # Transaction-only runs deliberately do not use trade-price comparisons here.
    if bool(group["_quote_mode"].iloc[0]):
        at_midpoint = np.isclose(price, baseline, rtol=1e-12, atol=1e-10)
        infer = (sign == 0) & baseline_ok & ~at_midpoint
        sign[infer] = np.sign(price[infer] - baseline[infer]).astype(int)
        direction_source[infer] = "estimated_prior_midquote"
    trades["_sign"] = sign
    left = np.searchsorted(times, times - window_ns, side="left")
    right = np.searchsorted(times, times, side="left")
    count = right - left
    safe_count = np.maximum(count, 1)
    volume = _sums(_prefix(size), left, right)
    mean_size = volume / safe_count
    size_var = np.maximum(0, _sums(_prefix(size**2), left, right) / safe_count - mean_size**2)
    known = (sign != 0).astype(float)
    known_volume = _sums(_prefix(size * known), left, right)
    signed_volume = _sums(_prefix(size * sign), left, right)
    # Differences are used only when both observations are within the past window.
    inner_left = np.minimum(left + 1, right)
    gaps_ms = np.r_[0.0, np.diff(times) / 1e6]
    pair_count = np.maximum(count - 1, 1)
    mean_gap = _sums(_prefix(gaps_ms), inner_left, right) / pair_count
    gap_var = np.maximum(0, _sums(_prefix(gaps_ms**2), inner_left, right) / pair_count - mean_gap**2)
    untied = ~trades["_time_ns"].duplicated(keep=False).to_numpy()
    unambiguous_pair = untied[1:] & untied[:-1]
    pair_sign = np.r_[0.0, sign[1:] * sign[:-1] * unambiguous_pair]
    pair_known = np.r_[0.0, known[1:] * known[:-1] * unambiguous_pair]
    persistence = _sums(_prefix(pair_sign), inner_left, right) / np.maximum(1, _sums(_prefix(pair_known), inner_left, right))
    # The average price of an entire prior tied timestamp is order invariant.
    # The current tied timestamp is still excluded from every past-window query.
    context_price = trades.groupby("_time_ns", sort=False)["_price"].transform("mean").to_numpy()
    returns = np.r_[0.0, np.diff(np.log(context_price)) * 1e4]
    return_mean = _sums(_prefix(returns), inner_left, right) / pair_count
    return_var = np.maximum(0, _sums(_prefix(returns**2), inner_left, right) / pair_count - return_mean**2)
    last = np.maximum(right - 1, 0)
    first = np.minimum(left, len(trades) - 1)
    past_return = np.where(count >= 2, np.log(context_price[last] / context_price[first]) * 1e4, 0)
    last_age_ms = np.where(count > 0, (times - times[last]) / 1e6, window_ns / 1e6)
    values = np.column_stack([
        np.log1p(count), np.log1p(volume), np.log1p(mean_size), np.sqrt(size_var) / np.maximum(mean_size, 1),
        signed_volume / np.maximum(known_volume, 1), _sums(_prefix(known), left, right) / safe_count,
        persistence, np.log1p(mean_gap), np.sqrt(gap_var) / np.maximum(mean_gap, 1e-6),
        past_return, np.sqrt(return_var), np.log1p(last_age_ms), spread, depth,
    ])
    result = trades[["ts_event", "ts_analysis", "symbol", "response_feed", "session_date", "_time_ns", "_sign", "_size", "_price"]].reset_index(drop=True)
    for i, name in enumerate(FEATURE_COLUMNS):
        result[name] = values[:, i]
    result["past_trade_count"] = count
    result["baseline_price"] = baseline
    result["baseline_age_ms"] = baseline_age / 1e6
    result["baseline_valid"] = baseline_ok
    result["features_valid"] = (count >= 5) & (times - int(group["_time_ns"].min()) >= window_ns) & baseline_ok & np.isfinite(values).all(axis=1)
    result["direction_source"] = direction_source
    result["tick_rule_sign_diagnostic_only"] = np.where(right > 0, np.sign(price - context_price[last]), 0).astype(int)
    result["group_end_ns"] = int(group["_time_ns"].max())
    return result, quote, ambiguous


def _finite_json(value):
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_analysis(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    dataset: str,
    schema: str,
    window_ms: int = 1000,
    horizons_ms: list[int] | None = None,
    n_clusters: int = 3,
) -> dict:
    """Fit behavioral contexts in chronological training data and describe holdout.

    All prices must be floating-point price units, as produced by DBNStore.to_df
    with pretty_px=True. Horizons are as-of (at or before), never next-tick joins.
    """
    horizons = sorted(set([1, 10, 100, 1000, 10000, 60000] if horizons_ms is None else horizons_ms))
    if window_ms <= 0 or not horizons or any(h <= 0 for h in horizons):
        raise ValueError("Window and response horizons must be positive milliseconds.")
    if not 1 <= n_clusters <= 12:
        raise ValueError("n_clusters must be between 1 and 12.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic = bool(frame.attrs.get("synthetic", False)) or ("synthetic" in frame and bool(frame["synthetic"].fillna(False).all()))
    data = _normalise(frame, dataset, schema)
    max_age_ms = 1000
    groups: dict[tuple[str, str, str], tuple[pd.DataFrame, int]] = {}
    event_frames = []
    ambiguous_total = 0
    for key, group in data.groupby(["symbol", "response_feed", "session_date"], sort=False):
        events, prices, ambiguous = _features(group, int(window_ms * 1e6), int(max_age_ms * 1e6))
        if not events.empty:
            event_frames.append(events)
            groups[key] = (prices, int(group["_time_ns"].max()))
        ambiguous_total += ambiguous
    if not event_frames:
        raise ValueError("No positive-size executed trades with valid prices were found.")
    events = pd.concat(event_frames, ignore_index=True).sort_values("_time_ns", kind="stable").reset_index(drop=True)
    events.insert(0, "event_id", np.arange(len(events)))
    eligible = events["features_valid"].to_numpy(dtype=bool)
    unique_times = np.unique(events.loc[eligible, "_time_ns"].to_numpy(dtype=np.int64))
    if len(unique_times) < 12:
        raise ValueError("Too few usable timestamps: obtain a longer sample with at least five past-window trades and fresh prices.")
    cutoff = int(unique_times[min(len(unique_times) - 2, int(len(unique_times) * 0.7))])
    embargo_ns = int(max(horizons) * 1e6)
    times = events["_time_ns"].to_numpy(dtype=np.int64)
    train = eligible & (times < cutoff - embargo_ns)
    test = eligible & (times >= cutoff)
    if train.sum() < 6 or test.sum() < 3:
        raise ValueError("Insufficient chronological train/test data after the maximum-horizon embargo; use a longer sample or shorter horizons.")
    x = events[FEATURE_COLUMNS].to_numpy(dtype=float)
    scaler = StandardScaler().fit(x[train])
    distinct_training = len(np.unique(np.round(scaler.transform(x[train]), 8), axis=0))
    fitted_clusters = min(n_clusters, max(1, int(train.sum()) // 5), distinct_training)
    notes = []
    if fitted_clusters < n_clusters:
        notes.append(f"Reduced clusters from {n_clusters} to {fitted_clusters} for sample support.")
    model = GaussianMixture(n_components=fitted_clusters, covariance_type="diag", init_params="k-means++", reg_covar=1e-4, random_state=4317, n_init=3, max_iter=300)
    # The MVP has few features. Large BLAS thread pools can make these small fits
    # dramatically slower, especially while parallel validation is running.
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        model.fit(scaler.transform(x[train]))
    notes.extend(str(item.message) for item in caught)
    if not model.converged_:
        notes.append("GMM did not converge; membership probabilities require review.")
    events["split"] = "ineligible"
    events.loc[eligible, "split"] = "embargo"
    events.loc[train, "split"] = "train"
    events.loc[test, "split"] = "test"
    probabilities = np.full((len(events), fitted_clusters), np.nan)
    with threadpool_limits(limits=1):
        probabilities[eligible] = model.predict_proba(scaler.transform(x[eligible]))
    events["colony"] = -1
    events.loc[eligible, "colony"] = probabilities[eligible].argmax(axis=1)
    for colony in range(fitted_clusters):
        events[f"p_colony_{colony}"] = probabilities[:, colony]
    # A membership label never sees the current trade or its future response.
    # The initiating direction is used only after the model is frozen.
    signed_test = test & (events["_sign"].to_numpy() != 0)
    response_rows, coverage_rows, event_response_frames = [], [], []
    for horizon in horizons:
        target = times + int(horizon * 1e6)
        future = np.full(len(events), np.nan)
        future_age = np.full(len(events), np.nan)
        observed = np.zeros(len(events), dtype=bool)
        for key, indices in events.groupby(["symbol", "response_feed", "session_date"], sort=False).groups.items():
            indices = np.asarray(indices, dtype=int)
            prices, group_end = groups[key]
            price_times = prices["_time_ns"].to_numpy(dtype=np.int64)
            if not len(price_times):
                continue
            location = np.searchsorted(price_times, target[indices], side="right") - 1
            available = (location >= 0) & (target[indices] <= group_end)
            selected = indices[available]
            future[selected] = prices["_response_price"].to_numpy()[location[available]]
            future_age[selected] = (target[selected] - price_times[location[available]]) / 1e6
            observed[selected] = np.isfinite(future[selected])
        fresh = observed & (future_age <= max_age_ms)
        valid = signed_test & fresh & events["baseline_valid"].to_numpy(dtype=bool)
        response = events["_sign"].to_numpy() * (future - events["baseline_price"].to_numpy()) / events["baseline_price"].to_numpy() * 1e4
        coverage_rows.append({
            "horizon_ms": horizon, "test_events": int(test.sum()),
            "known_direction_test_events": int(signed_test.sum()),
            "unknown_direction_test_events": int((test & ~signed_test).sum()),
            "missing_horizon_test_events": int((signed_test & ~observed).sum()),
            "stale_horizon_test_events": int((signed_test & observed & ~fresh).sum()),
            "usable_test_events": int(valid.sum()),
            "coverage_of_known_direction": float(valid.sum() / signed_test.sum()) if signed_test.sum() else np.nan,
        })
        for colony in range(fitted_clusters):
            weight = probabilities[valid, colony]
            values = response[valid]
            total_weight = weight.sum()
            hard = valid & (events["colony"].to_numpy() == colony)
            response_rows.append({
                "colony": colony, "horizon_ms": horizon,
                "weighted_mean_signed_response_bps": float(np.average(values, weights=weight)) if total_weight > 0 else np.nan,
                "weighted_negative_response_fraction": float(np.average(values < 0, weights=weight)) if total_weight > 0 else np.nan,
                "probability_mass": float(total_weight),
                "effective_event_count": float(total_weight**2 / np.square(weight).sum()) if total_weight > 0 else 0.0,
                "hard_assignment_event_count": int(hard.sum()),
                "hard_assignment_mean_signed_response_bps": float(response[hard].mean()) if hard.any() else np.nan,
            })
        detail = events.loc[test, ["event_id", "symbol", "response_feed", "ts_event", "ts_analysis", "colony"]].copy()
        detail["horizon_ms"] = horizon
        detail["horizon_observed"] = observed[test]
        detail["horizon_price_age_ms"] = future_age[test]
        detail["response_valid"] = valid[test]
        detail["signed_response_bps"] = np.where(valid[test], response[test], np.nan)
        detail["response_price"] = future[test]
        event_response_frames.append(detail)
    response_curve = pd.DataFrame(response_rows)
    coverage = pd.DataFrame(coverage_rows)
    probability_columns = [f"p_colony_{colony}" for colony in range(fitted_clusters)]
    membership_columns = ["event_id", "ts_event", "ts_analysis", "symbol", "response_feed", "session_date", "split", "features_valid", "colony", "_sign", "direction_source", "tick_rule_sign_diagnostic_only", "baseline_price", "baseline_age_ms", "baseline_valid"] + probability_columns
    events[membership_columns].rename(columns={"_sign": "aggressor_sign"}).to_csv(output_dir / "event_memberships.csv", index=False)
    events[["event_id", "ts_event", "ts_analysis", "split", "features_valid", "past_trade_count"] + FEATURE_COLUMNS].to_csv(output_dir / "event_features.csv", index=False)
    response_curve.to_csv(output_dir / "response_curve.csv", index=False)
    coverage.to_csv(output_dir / "coverage.csv", index=False)
    pd.concat(event_response_frames, ignore_index=True).to_csv(output_dir / "event_responses.csv", index=False)
    profiles = []
    for colony in range(fitted_clusters):
        weight = probabilities[train, colony]
        item = {"colony": colony, "train_probability_mass": float(weight.sum()), "train_hard_event_count": int((events.loc[train, "colony"] == colony).sum())}
        item.update(dict(zip(FEATURE_COLUMNS, np.average(x[train], axis=0, weights=weight))))
        profiles.append(item)
    pd.DataFrame(profiles).to_csv(output_dir / "colony_profiles.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(9, 5.4))
    for colony, group in response_curve.groupby("colony"):
        axis.plot(group["horizon_ms"], group["weighted_mean_signed_response_bps"], marker="o", label=f"Colony {colony}")
    axis.set_xscale("log")
    axis.axhline(0, color="#6b7280", linewidth=0.8)
    axis.set(xlabel="Elapsed horizon (milliseconds, logarithmic scale)", ylabel="Probability-weighted signed response (bps)", title=("SYNTHETIC DEMONSTRATION — " if synthetic else "Held-out ") + ("midquote response" if schema == "mbp-1" else "transaction-price response"))
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "response_curve.png", dpi=160)
    plt.close(fig)
    limitations = [
        "Colonies describe prior execution contexts; they do not identify traders, funds, parent orders, or investment styles.",
        "This is descriptive association, not causal impact, profitable trading evidence, or a measurement of exchange/infrastructure latency.",
        "Feed B/A sides are preserved; missing/unknown sides use a strictly preceding fresh midpoint comparison when available, and remain unknown at midpoint ties. This estimated sign is not a verified aggressor side. Tick diagnostics never determine signed responses.",
        "Chronological train/test split and an outcome-horizon embargo prevent direct response leakage; no tuning or independent multi-day replication has been performed.",
        "Response events overlap and are dependent; effective event counts describe probability concentration, not independent sample sizes. No inferential confidence intervals are claimed.",
        "Raw response differences can reflect liquidity, volatility, time of day, symbol composition and common shocks; this MVP does not control for those confounders.",
        "Prices use the last unambiguous observation at or before the horizon, with a strictly pre-event baseline; sparse updates limit horizon resolution. Staleness excludes rows and can introduce selection effects.",
        "An invalid, crossed or conflicting quote timestamp blocks prior quotes until a subsequent valid quote; unknown horizon states are counted as missing. The price staleness cap is fixed at 1,000ms independently of the feature window.",
        "Features and prices are grouped by symbol, publisher/feed and New York calendar date; the MVP does not classify exchange sessions or handle corporate actions and halts.",
        "Training features/scaling/model use only past-window observations, with equal timestamps excluded from that window. GMM probabilities are fitted mixture responsibilities, not calibrated identity probabilities.",
        "Within an earlier tied trade timestamp, context prices use the arithmetic mean and sign-persistence transitions touching ties are excluded; no submission ordering is inferred.",
        "The current study measures within-feed response; multi-feed synchronization and lead/lag identification require a separate design.",
        "The receive clock is preferred when present so later-received records cannot enter earlier features; otherwise event timestamps are used and late arrival/reordering remains a limitation. No clock measures pure exchange latency.",
        "Feed-specific quality flags and trading conditions are not yet audited or filtered; live-data results require this review before substantive conclusions.",
    ]
    if dataset.upper() == "EQUS.MINI":
        limitations.append("EQUS.MINI is treated as one anonymized aggregate publisher; publisher_id is not an originating exchange and no venue comparison is possible.")
    if schema == "trades":
        limitations.append("Transaction-price responses include bid/ask bounce and are not quote reactions.")
    if synthetic:
        limitations.insert(0, "ALL INPUT DATA AND RESULTS IN THIS RUN ARE SYNTHETIC; this validates software operation only.")
    summary = {
        "dataset": dataset, "schema": schema, "synthetic": synthetic,
        "input_records": len(data), "executed_trade_events": len(events),
        "analysis_clock": str(data["analysis_clock"].iloc[0]),
        "eligible_events": int(eligible.sum()), "train_events": int(train.sum()), "test_events": int(test.sum()),
        "events_without_valid_fresh_baseline": int((~events["baseline_valid"]).sum()),
        "events_with_fewer_than_five_past_window_trades": int((events["past_trade_count"] < 5).sum()),
        "embargo_events": int((events["split"] == "embargo").sum()),
        "unknown_direction_test_events": int((test & (events["_sign"].to_numpy() == 0)).sum()),
        "train_test_cutoff_utc": pd.Timestamp(cutoff, tz="UTC").isoformat(),
        "train_last_event_utc": events.loc[train, "ts_analysis"].max().isoformat(),
        "test_first_event_utc": events.loc[test, "ts_analysis"].min().isoformat(),
        "test_direction_sources": events.loc[test, "direction_source"].value_counts().to_dict(),
        "embargo_ms": max(horizons), "window_ms": window_ms,
        "horizons_ms": horizons, "max_price_staleness_ms": max_age_ms,
        "requested_clusters": n_clusters, "fitted_clusters": fitted_clusters,
        "gmm_converged": bool(model.converged_), "ambiguous_price_timestamps_excluded": ambiguous_total,
        "invalid_quote_records": int(data["_response_price"].isna().sum()) if schema == "mbp-1" else 0,
        "price_measure": "midquote" if schema == "mbp-1" else "transaction_price",
        "feature_columns": FEATURE_COLUMNS, "notes": notes, "limitations": limitations,
        "coverage": coverage.to_dict(orient="records"),
        "outputs": ["event_memberships.csv", "event_features.csv", "response_curve.csv", "coverage.csv", "event_responses.csv", "colony_profiles.csv", "response_curve.png", "run_summary.json"],
    }
    summary = _finite_json(summary)
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary
