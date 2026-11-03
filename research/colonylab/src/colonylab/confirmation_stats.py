"""Paired date-level uncertainty for the frozen fresh-date candle comparison.

Every input row contains all scored stocks and venues for one held-out session.
Bootstrap indices resample the entire row, keeping the two models, all loss
metrics and the observation denominator together. No bin independence is used.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "day", "reference_log_loss_sum", "volume_log_loss_sum",
    "reference_brier_sum", "volume_brier_sum", "rows",
)
LOSS_COLUMNS = REQUIRED_COLUMNS[1:-1]
MINIMUM_CONFIRMATION_DAYS = 20


def _interval(values: np.ndarray) -> list[float] | None:
    # A zero reference loss leaves a relative gain undefined. Do not silently
    # discard those replicates and condition the reported interval on success.
    if not np.isfinite(values).all():
        return None
    return np.quantile(values, [.025, .975]).astype(float).tolist()


def _relative(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(100 * numerator, denominator,
                     out=np.full(np.shape(numerator), np.nan, dtype=float),
                     where=denominator > 0)


def _bootstrap(totals: np.ndarray, *, replicates: int, seed: int,
               block_length: int) -> dict:
    """Resample consecutive session positions without wrapping the date range."""
    count = len(totals)
    length = min(block_length, count)
    rng = np.random.default_rng(seed)
    blocks = math.ceil(count / length)
    starts = rng.integers(0, count - length + 1, size=(replicates, blocks))
    indices = (starts[..., None] + np.arange(length)).reshape(replicates, -1)[:, :count]
    # Columns: reference LL, volume LL, reference Brier, volume Brier, rows.
    sampled = totals[indices].sum(axis=1)
    result = {
        "method": "paired_session_bootstrap" if block_length == 1 else "paired_moving_session_block_bootstrap",
        "replicates": replicates, "seed": seed,
        "requested_block_sessions": block_length,
        "effective_block_sessions": length, "sessions_per_replicate": count,
        "blocks_per_replicate": blocks, "wraps_date_range": False,
        "confidence_level": .95,
    }
    for metric, position in (("log_loss", 0), ("brier", 2)):
        gain = sampled[:, position] - sampled[:, position + 1]
        result[metric] = {
            "absolute_gain_ci95": _interval(gain / sampled[:, 4]),
            "relative_gain_percent_ci95": _interval(_relative(gain, sampled[:, position])),
        }
    return result


def summarize_confirmation(
    daily: pd.DataFrame, *, expected_days: Sequence[str] | None = None,
    bootstrap_replicates: int = 10_000, seed: int = 20261026,
    block_length: int = 5,
) -> dict:
    """Summarize paired daily loss totals without treating bins as independent.

    An invalid or zero-row date invalidates inference for the supplied study;
    nothing is dropped. ``expected_days`` is required to establish that the
    intended holdout is complete. Fewer than twenty sessions are descriptive.
    The evidence flag requires a positive lower bound in the predeclared
    five-session block interval and at least twenty complete held-out sessions.
    """
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be a positive integer.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if type(block_length) is not int or block_length < 1:
        raise ValueError("block_length must be a positive integer.")
    result = {
        "status": "insufficient_coverage", "primary_metric": "pooled_log_loss",
        "delta_sign": "Positive reference-minus-volume loss means volume candles improve the network.",
        "resampling_unit": "One complete date containing all scored stocks and venues together.",
        "minimum_confirmation_days": MINIMUM_CONFIRMATION_DAYS,
        "observed_day_count": len(daily),
        "expected_day_count": len(expected_days) if expected_days is not None else None,
        "coverage_errors": [], "supports_positive_incremental_prediction": False,
        "limitations": [
            "Dependent bins are not independent experiments; no independent-bin p-values are computed.",
            "Twenty dates contain only about four non-overlapping five-session blocks; intervals remain approximate.",
            "The moving-block interval uses overlapping consecutive sessions without wrapping; finite-sample edge weighting and regime changes can affect it.",
            "Results are conditional on the frozen model, the observed coverage cohort and sampled intraday window.",
            "Predictive loss improvement establishes neither causal impact nor trading profitability.",
        ],
    }
    errors = result["coverage_errors"]
    missing = sorted(set(REQUIRED_COLUMNS) - set(daily.columns))
    if missing:
        errors.append("Missing required columns: " + ", ".join(missing))
        return result
    if daily.empty:
        errors.append("No held-out dates supplied.")
        return result
    days = daily.day.astype(str).tolist()
    valid_days = []
    for day in days:
        try:
            parsed = pd.Timestamp(day)
            valid_days.append(parsed.strftime("%Y-%m-%d") == day and parsed.weekday() < 5)
        except (ValueError, TypeError, OverflowError):
            valid_days.append(False)
    if not all(valid_days):
        errors.append("Every held-out day must be an explicit valid ISO exchange weekday.")
    if len(set(days)) != len(days):
        errors.append("Duplicate held-out dates; aggregate all stocks and venues into one row per date first.")
    if expected_days is not None:
        expected = list(expected_days)
        if len(set(expected)) != len(expected):
            errors.append("Expected held-out dates contain duplicates.")
        missing_days, extra_days = sorted(set(expected) - set(days)), sorted(set(days) - set(expected))
        if missing_days:
            errors.append("Missing intended held-out dates: " + ", ".join(missing_days))
        if extra_days:
            errors.append("Unexpected held-out dates: " + ", ".join(extra_days))
    try:
        values = daily[list(LOSS_COLUMNS) + ["rows"]].to_numpy(dtype=float)
    except (ValueError, TypeError):
        errors.append("Daily loss totals and rows must be numeric.")
        return result
    if not np.isfinite(values).all():
        errors.append("Daily loss totals and rows must all be finite.")
    if np.any(values[:, :4] < 0):
        errors.append("Daily log-loss and Brier totals must be nonnegative.")
    rows = values[:, 4]
    if np.any(rows <= 0) or np.any(rows != np.floor(rows)) or daily.rows.map(lambda value: isinstance(value, (bool, np.bool_))).any():
        errors.append("Every intended date requires a positive integer number of paired rows.")
    if np.any(values[:, 2:4] > rows[:, None] + 1e-10):
        errors.append("Brier totals cannot exceed the number of paired rows.")
    if errors:
        return result
    order = np.argsort(days)
    ordered_days = [days[i] for i in order]
    values = values[order]
    summed = values.sum(axis=0)
    result["days"] = ordered_days
    result["pooled"] = {"rows": int(summed[4])}
    for metric, position in (("log_loss", 0), ("brier", 2)):
        gain = summed[position] - summed[position + 1]
        result["pooled"][metric] = {
            "reference": float(summed[position] / summed[4]),
            "volume": float(summed[position + 1] / summed[4]),
            "absolute_gain": float(gain / summed[4]),
            "relative_gain_percent": float(100 * gain / summed[position]) if summed[position] > 0 else None,
        }
        difference = values[:, position] - values[:, position + 1]
        result[f"positive_{metric}_days"] = int(np.sum(difference > 0))
        result[f"negative_{metric}_days"] = int(np.sum(difference < 0))
        result[f"tied_{metric}_days"] = int(np.sum(difference == 0))
    result["paired_day_bootstrap"] = _bootstrap(values, replicates=bootstrap_replicates, seed=seed, block_length=1)
    result["five_session_block_bootstrap"] = _bootstrap(values, replicates=bootstrap_replicates, seed=seed + 1, block_length=block_length)
    complete = expected_days is not None and len(values) >= MINIMUM_CONFIRMATION_DAYS
    result["status"] = "ok" if complete else "insufficient_coverage"
    if expected_days is None:
        errors.append("Intended held-out dates were not supplied; completeness cannot be established.")
    if len(values) < MINIMUM_CONFIRMATION_DAYS:
        errors.append(f"Only {len(values)} paired dates; at least {MINIMUM_CONFIRMATION_DAYS} were predeclared for confirmation.")
    interval = result["five_session_block_bootstrap"]["log_loss"]["absolute_gain_ci95"]
    result["positive_block_interval"] = interval is not None and interval[0] > 0
    result["supports_positive_incremental_prediction"] = bool(
        complete and block_length == 5 and result["positive_block_interval"]
        and result["pooled"]["log_loss"]["absolute_gain"] > 0)
    result["evidence_rule"] = "At least twenty complete intended dates, positive pooled gain and a positive lower 95% bound for paired five-session-block log-loss gain; no practical-profit threshold is asserted."
    return result
