"""Isolated regular-grid stationarity diagnostics; no latency transformations.

Only one contiguous observed run is tested. The tests describe the selected
price path, not irregular/censored matching delays or an event's causal effect.
"""
from __future__ import annotations

import copy
import math
import warnings

import numpy as np


INT64_MAX = np.iinfo(np.int64).max
SPECS = (("log_level", "c"), ("log_level", "ct"), ("log_return", "c"))


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _integer_vector(values, name):
    raw = np.asarray(values)
    _require(raw.ndim == 1 and raw.dtype.kind in "iu", f"{name} must be an integer vector")
    _require(not (raw.dtype.kind == "u" and np.any(raw > INT64_MAX)), f"{name} exceeds signed integer range")
    return raw.astype(np.int64, copy=False)


def sample_contiguous(path, start_ns, end_ns, step_ns=1_000_000_000):
    """Select the earliest longest supported run on a strict-prior time grid.

    Grid points are ``start_ns + k*step_ns < end_ns``. A state timestamped
    exactly at a grid point has not entered that point's baseline. Known
    quality boundaries split runs even when they occur between grid points,
    recover before the next point, or reuse a segment identifier.

    ``fraction_window`` is longest-run points / all requested grid points;
    ``valid_fraction_window`` separately includes every valid sampled point.
    Run endpoints identify the first and last included grid timestamps.
    """
    _require(all(isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))
                 for x in (start_ns, end_ns, step_ns)), "Grid bounds and step must be integer nanoseconds")
    start_ns, end_ns, step_ns = int(start_ns), int(end_ns), int(step_ns)
    _require(0 <= start_ns < end_ns <= INT64_MAX and step_ns > 0, "Invalid half-open grid interval or step")
    times = _integer_vector(path.times_ns, "times_ns")
    prices = _integer_vector(path.price2, "price2")
    segments = _integer_vector(path.segments, "segments")
    valid = np.asarray(path.valid, dtype=bool)
    _require(valid.ndim == 1 and len(times) == len(prices) == len(segments) == len(valid), "Native path lengths differ")
    _require((times >= 0).all() and (np.diff(times) > 0).all(), "Native timestamps must strictly increase")
    _require((segments >= 0).all() and (prices[valid] > 0).all(), "Invalid native price or segment")
    grid = np.arange(start_ns, end_ns, step_ns, dtype=np.int64)
    result = {"times_ns": np.array([], np.int64), "price2": np.array([], np.int64),
              "segments": np.array([], np.int64), "total_grid_points": len(grid),
              "valid_grid_points": 0, "run_count": 0, "longest_points": 0,
              "run_start_ns": None, "run_end_ns": None, "fraction_window": 0., "valid_fraction_window": 0.}
    if not len(times):
        return result
    positions = np.searchsorted(times, grid, side="left") - 1
    safe = np.maximum(positions, 0)
    good = ((positions >= 0) & valid[safe] & (grid >= int(path.start_ns)) & (grid < int(path.end_ns)))
    selected = np.flatnonzero(good)
    result["valid_grid_points"] = int(good.sum())
    result["valid_fraction_window"] = float(good.mean())
    if not len(selected):
        return result
    barriers = times[(~valid) | np.r_[False, segments[1:] != segments[:-1]]]
    barrier_count = np.searchsorted(barriers, grid, side="left")
    break_before = np.r_[True, (np.diff(selected) != 1)
                         | (segments[safe[selected[1:]]] != segments[safe[selected[:-1]]])
                         | (barrier_count[selected[1:]] != barrier_count[selected[:-1]])]
    if hasattr(path, "next_barrier") and len(selected) > 1:
        # Strict-prior sampling includes a boundary exactly at the previous
        # grid timestamp only at the next grid point, hence querying t-1ns.
        following = np.asarray(path.next_barrier(grid[selected[:-1]] - 1))
        _require(following.shape == (len(selected) - 1,), "next_barrier returned an invalid shape")
        break_before[1:] |= following < grid[selected[1:]]
    starts = np.flatnonzero(break_before)
    stops = np.r_[starts[1:], len(selected)]
    longest = int(np.argmax(stops - starts))
    chosen = selected[starts[longest]:stops[longest]]
    result.update(times_ns=grid[chosen].copy(), price2=prices[safe[chosen]].copy(),
                  segments=segments[safe[chosen]].copy(), run_count=len(starts), longest_points=len(chosen),
                  run_start_ns=int(grid[chosen[0]]), run_end_ns=int(grid[chosen[-1]]),
                  fraction_window=float(len(chosen) / len(grid)))
    return result


def _row(transformation, regression, n, minimum, alpha):
    return {"transformation": transformation, "regression": regression, "n_observations": int(n),
            "min_observations": minimum, "alpha": alpha, "status": "skipped", "skip_reason": "",
            "adf_stat": None, "adf_pvalue": None, "adf_maxlag": None, "adf_lags": None,
            "adf_nobs": None, "adf_critical": None, "kpss_stat": None, "kpss_pvalue": None,
            "kpss_pvalue_relation": None, "kpss_lags": None, "kpss_critical": None,
            "adf_reject_nominal": None, "kpss_reject_nominal": None,
            "warnings": [], "classification": "not_tested"}


def _warning_rows(captured, test):
    return [{"test": test, "category": item.category.__name__, "message": str(item.message)} for item in captured]


def _kpss_relation(pvalue, statistic, critical, captured):
    messages = " ".join(str(item.message).lower() for item in captured)
    if "smaller" in messages or pvalue <= .01 and statistic > critical["1%"]:
        return "<="
    if "greater" in messages or "larger" in messages or pvalue >= .1 and statistic < critical["10%"]:
        return ">="
    return "="


def _decision(pvalue, relation, alpha):
    if relation == "<=":
        return True if pvalue <= alpha else None
    if relation == ">=":
        return False if pvalue > alpha else None
    return bool(pvalue <= alpha)


def test_pairs(price2, *, min_observations=300, alpha=.05):
    """Run the three prespecified ADF/KPSS pairs on one contiguous price run.

    ADF uses BIC with the capped Schwert maximum lag; KPSS uses its automatic
    Newey-West bandwidth. Logs are centered without detrending. KPSS table
    bounds retain their inequality and warning; nominal joint classifications
    are descriptive and never select a transformation automatically.
    """
    _require(type(min_observations) is int and min_observations >= 8, "min_observations must be an integer >=8")
    _require(np.isfinite(alpha) and 0 < alpha < 1, "alpha must lie strictly between zero and one")
    alpha = float(alpha)
    raw = np.asarray(price2)
    n = len(raw) if raw.ndim == 1 else 0
    rows = [_row(kind, trend, n if kind == "log_level" else max(0, n - 1), min_observations, alpha)
            for kind, trend in SPECS]
    if raw.ndim == 1 and n == 0:
        for row in rows:
            row["skip_reason"] = "too_short"
        return rows
    invalid = ("price2_not_one_dimensional" if raw.ndim != 1 else
               "nonfinite_price2" if raw.dtype.kind in "fiu" and not np.isfinite(raw).all() else
               "noninteger_price2" if raw.dtype.kind not in "iu" else
               "price2_out_of_range" if raw.dtype.kind == "u" and np.any(raw > INT64_MAX) else
               "nonpositive_price2" if np.any(raw <= 0) else None)
    if invalid:
        for row in rows:
            row["skip_reason"] = invalid
        return rows
    raw = raw.astype(np.int64, copy=False)
    if not n:
        for row in rows:
            row["skip_reason"] = "too_short"
        return rows
    # Integer subtraction retains small native price moves before logarithms;
    # use the direct log ratio only for very large downward relative moves.
    base = int(raw[0])
    relative = (raw - base).astype(float) / float(base)
    logged = np.empty(n, float)
    near = relative > -.5
    logged[near] = np.log1p(relative[near])
    logged[~near] = np.log(raw[~near].astype(float)) - math.log(base)
    from statsmodels.tsa.stattools import adfuller, kpss
    for row in rows:
        x = logged.copy() if row["transformation"] == "log_level" else np.diff(logged)
        if len(x) < min_observations:
            row["skip_reason"] = "too_short"
            continue
        if not np.isfinite(x).all():
            row["skip_reason"] = "nonfinite_transformation"
            continue
        span = float(np.ptp(x))
        if span == 0:
            row["skip_reason"] = "constant_series"
            continue
        if span <= 64 * np.finfo(float).eps:
            row["skip_reason"] = "numerically_degenerate_series"
            continue
        x -= x.mean()
        maximum = min(int(math.ceil(12 * (len(x) / 100) ** .25)), len(x) // 4)
        row["adf_maxlag"] = maximum
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                a = adfuller(x, maxlag=maximum, regression=row["regression"], autolag="BIC")
            row["warnings"].extend(_warning_rows(captured, "adf"))
            _require(np.isfinite(a[0]) and np.isfinite(a[1]), "ADF returned a nonfinite statistic")
            row.update(adf_stat=float(a[0]), adf_pvalue=float(a[1]), adf_lags=int(a[2]), adf_nobs=int(a[3]),
                       adf_critical={key: float(value) for key, value in a[4].items()})
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                k = kpss(x, regression=row["regression"], nlags="auto")
            row["warnings"].extend(_warning_rows(captured, "kpss"))
            _require(np.isfinite(k[0]) and np.isfinite(k[1]), "KPSS returned a nonfinite statistic")
            relation = _kpss_relation(float(k[1]), float(k[0]), k[3], captured)
            row.update(kpss_stat=float(k[0]), kpss_pvalue=float(k[1]), kpss_pvalue_relation=relation,
                       kpss_lags=int(k[2]), kpss_critical={key: float(value) for key, value in k[3].items()},
                       status="tested")
            ar, kr = bool(row["adf_pvalue"] <= alpha), _decision(row["kpss_pvalue"], relation, alpha)
            row.update(adf_reject_nominal=ar, kpss_reject_nominal=kr)
            row["classification"] = ("inconclusive" if kr is None else
                                     "stationarity_compatible" if ar and not kr else
                                     "unit_root_compatible" if not ar and kr else
                                     "conflicting" if ar and kr else "inconclusive")
        except Exception as error:
            row.update(status="test_error", skip_reason=f"{type(error).__name__}: {error}", classification="not_tested")
    return rows


def adjust_adf_holm(rows, *, alpha=.05):
    """Return copied rows with Holm corrections for this explicit ADF family.

    KPSS bounds and nominal classifications are unchanged. Call separately for
    prespecified test families; skipped ADF tests are not invented p-values.
    """
    _require(np.isfinite(alpha) and 0 < alpha < 1, "alpha must lie strictly between zero and one")
    result = copy.deepcopy(list(rows))
    selected = [n for n, row in enumerate(result) if row.get("adf_pvalue") is not None
                and np.isfinite(row["adf_pvalue"]) and 0 <= row["adf_pvalue"] <= 1]
    order = sorted(selected, key=lambda n: result[n]["adf_pvalue"])
    previous, count = 0., len(order)
    adjusted = {}
    for rank, index in enumerate(order):
        previous = min(1., max(previous, (count - rank) * result[index]["adf_pvalue"]))
        adjusted[index] = previous
    for n, row in enumerate(result):
        value = adjusted.get(n)
        row.update(adf_pvalue_holm=value, adf_holm_family_size=count,
                   adf_reject_holm=bool(value <= alpha) if value is not None else None)
    return result
