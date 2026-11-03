"""Strictly past full-book controls and common-cohort SIRS prediction panels.

Replays retain every native displayed order but expose a fixed 100 ms grid.
Grid states contain only records strictly before their forecast timestamp.
Completed BBO-1s observations use their interval-end ``ts_recv`` directly:
https://databento.com/docs/schemas/bbo . No extra second is added to this clock.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .mbo_book import F_BAD_TS_RECV, F_MAYBE_BAD_BOOK, F_SNAPSHOT, UNDEF_PRICE
from .sir_model import (SUSCEPTIBLE, INFECTIOUS, accepted_infections,
                        infection_labels, sirs_states)


CONTROL_NAMES = [
    "elapsed_fraction", "log_spread_bps", "depth_imbalance_1", "depth_imbalance_10",
    "log_depth_10", "log_order_count", "add_imbalance_100ms", "remove_imbalance_100ms",
    "log_add_shares_100ms", "log_removed_shares_100ms", "trade_imbalance_100ms",
    "log_trade_shares_100ms", "log_book_age_ms", "own_infections_1s",
    "spy_return_bps_1s", "spy_return_bps_10s", "market_median_return_bps_1s",
]
MARKET_NAMES = ["spy_return_bps_1s", "spy_return_bps_10s"]
_NS_PER_MS = 1_000_000
_NS_PER_SECOND = 1_000_000_000


def _lag_counts(events: np.ndarray, anchors: np.ndarray, window_ms: int) -> np.ndarray:
    """Count accepted timestamps in [t-window,t), matching the legacy clock."""
    return (np.searchsorted(events, anchors, side="left")
            - np.searchsorted(events, anchors - window_ms * _NS_PER_MS, side="left"))


def _ns(values) -> np.ndarray:
    array = np.asarray(values)
    if not array.size:
        return np.empty(0, dtype=np.int64)
    if array.dtype.kind in "iu":
        if array.dtype.kind == "u" and array.size and array.max() > np.iinfo(np.int64).max:
            raise ValueError("Timestamp outside signed nanosecond range.")
        return array.astype(np.int64, copy=False)
    if array.dtype.kind == "f":
        raise ValueError("Nanosecond timestamps must not be rounded floating-point values.")
    return pd.DatetimeIndex(values).asi8


def fit_thresholds(
    training_replays: Iterable[tuple[str, str, dict]], *,
    training_days: Sequence[str] | None = None,
    expected_datasets: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Freeze each venue's 75th-percentile absolute nonzero training jump.

    Explicit training dates are checked before reading each replay payload,
    so held-out data cannot enter calibration. Every intended venue/date must
    be present exactly once when both expected dimensions are supplied.
    """
    selected_days = set(training_days) if training_days is not None else None
    datasets = set(expected_datasets) if expected_datasets is not None else None
    if training_days is not None and len(selected_days) != len(training_days):
        raise ValueError("Training dates must be unique.")
    if expected_datasets is not None and len(datasets) != len(expected_datasets):
        raise ValueError("Expected venues must be unique.")
    arrays: dict[str, list[np.ndarray]] = {}
    seen = set()
    for dataset, day, replay in training_replays:
        if selected_days is not None and day not in selected_days:
            continue
        if datasets is not None and dataset not in datasets:
            raise ValueError(f"Unexpected training venue: {dataset}")
        key = dataset, day
        if key in seen:
            raise ValueError(f"Duplicate training replay: {key}")
        seen.add(key)
        jumps = replay["jumps"]["jump"].to_numpy(dtype=float)
        if not np.isfinite(jumps).all():
            raise ValueError("Training jump candidates must be finite.")
        arrays.setdefault(dataset, []).append(np.abs(jumps[jumps != 0]))
    if selected_days is not None and datasets is not None:
        missing = {(dataset, day) for dataset in datasets for day in selected_days} - seen
        if missing:
            raise ValueError(f"Missing intended training venue/date replays: {sorted(missing)}")
    if not seen:
        raise ValueError("No training replays supplied.")
    result = []
    for dataset in sorted(arrays):
        values = np.concatenate(arrays[dataset])
        quantile = float(np.quantile(values, .75)) if len(values) else 0.
        result.append({"dataset": dataset, "threshold": max(.005, quantile),
                       "training_days": len({day for venue, day in seen if venue == dataset}),
                       "training_jump_count": len(values), "quantile": .75, "floor_usd": .005})
    return pd.DataFrame(result)


def _market_frame(source: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(source, (str, Path)):
        import databento as db
        store = db.DBNStore.from_file(source)
        if str(store.metadata.schema) != "bbo-1s":
            raise ValueError("Market control file must use completed bbo-1s observations.")
        records = []
        for record in store:
            if not hasattr(record, "levels") or len(record.levels) != 1:
                raise ValueError("Market stream contains a non-BBO record.")
            level = record.levels[0]
            bid, ask = int(level.bid_px), int(level.ask_px)
            valid = (0 < bid < UNDEF_PRICE and bid <= ask < UNDEF_PRICE
                     and int(level.bid_sz) > 0 and int(level.ask_sz) > 0
                     and not int(record.flags) & (F_BAD_TS_RECV | F_MAYBE_BAD_BOOK | F_SNAPSHOT))
            records.append({"availability_ns": int(record.ts_recv),
                            "mid": (bid + ask) / 2e9 if valid else np.nan, "valid": valid})
        frame = pd.DataFrame(records, columns=["availability_ns", "mid", "valid"])
    else:
        frame = source.copy()
        if "availability_ns" not in frame:
            clock = frame["ts_recv"] if "ts_recv" in frame else frame.index
            frame["availability_ns"] = _ns(clock)
        if "mid" not in frame:
            # A raw DataFrame uses SDK price_type='float' (USD). A native DBN
            # path above preserves integer prices and performs explicit scaling.
            bid = frame["bid_px_00"].to_numpy(dtype=float)
            ask = frame["ask_px_00"].to_numpy(dtype=float)
            valid = (np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask >= bid)
                     & (frame["bid_sz_00"].to_numpy() > 0) & (frame["ask_sz_00"].to_numpy() > 0))
            frame["mid"] = (bid + ask) / 2
            frame["valid"] = valid & frame.get("valid", True)
        if "valid" not in frame:
            frame["valid"] = True
        if "flags" in frame:
            frame["valid"] &= (frame["flags"].to_numpy(dtype=np.int64)
                                & (F_BAD_TS_RECV | F_MAYBE_BAD_BOOK | F_SNAPSHOT)) == 0
    if frame.empty:
        return pd.DataFrame({"availability_ns": pd.Series(dtype=np.int64),
                             "mid": pd.Series(dtype=float), "valid": pd.Series(dtype=bool)})
    frame["availability_ns"] = _ns(frame.availability_ns)
    frame["valid"] = frame.valid.astype(bool) & np.isfinite(frame.mid) & frame.mid.gt(0)
    frame = frame.sort_values("availability_ns", kind="stable")
    # A conflicting interval-end tie is an unknown observation, not an
    # arbitrary choice of the most favorable quote. Keep that barrier.
    grouped = frame.groupby("availability_ns", sort=False)
    bad_ties = grouped.mid.nunique(dropna=False).gt(1) | ~grouped.valid.all()
    frame = frame.drop_duplicates("availability_ns", keep="last").copy()
    frame.loc[frame.availability_ns.isin(bad_ties.index[bad_ties]), "valid"] = False
    return frame[["availability_ns", "mid", "valid"]].reset_index(drop=True)


def market_features(
    source: pd.DataFrame | str | Path, anchors_ns: np.ndarray, *, max_age_seconds: float = 5.,
) -> dict:
    """Past SPY returns from complete interval ends, with a five-second age cap.

    Quotes at the forecast timestamp are unavailable. The same strict rule
    applies at each lagged reference timestamp. Invalid BBO intervals are kept
    as barriers, so an older valid quote cannot be carried through one.
    """
    if not np.isfinite(max_age_seconds) or max_age_seconds <= 0:
        raise ValueError("max_age_seconds must be finite and positive.")
    anchors = _ns(anchors_ns)
    frame = _market_frame(source)
    times = frame.availability_ns.to_numpy(dtype=np.int64)
    mids = frame.mid.to_numpy(dtype=float)
    valid_observation = frame.valid.to_numpy(dtype=bool)
    sampled = []
    support = []
    for seconds in (0, 1, 10):
        when = anchors - seconds * _NS_PER_SECOND
        index = np.searchsorted(times, when, side="left") - 1
        if not len(times):
            sampled.append(np.full(len(anchors), np.nan))
            support.append(np.zeros(len(anchors), dtype=bool))
            continue
        clipped = np.maximum(index, 0)
        age = when - times[clipped]
        good = ((index >= 0) & valid_observation[clipped]
                & (age > 0) & (age <= max_age_seconds * _NS_PER_SECOND))
        sampled.append(mids[clipped])
        support.append(good)
    x = np.column_stack([(sampled[0] / sampled[k] - 1) * 1e4 for k in (1, 2)])
    valid = np.logical_and.reduce(support) & np.isfinite(x).all(axis=1)
    return {"x": x, "names": MARKET_NAMES.copy(), "valid": valid,
            "diagnostics": {"completed_market_observations": len(frame),
                            "candidate_bins": len(anchors), "eligible_bins": int(valid.sum()),
                            "timestamp": "ts_recv is interval end; strictly before each anchor/reference time",
                            "maximum_observation_age_seconds": max_age_seconds}}


def _barriers(replay: dict) -> np.ndarray:
    arrays = [np.asarray(replay.get(name, []), dtype=np.int64)
              for name in ("reset_times_ns", "invalid_times_ns")]
    return np.unique(np.concatenate(arrays))


def _interval_clear(barriers: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    return (np.searchsorted(barriers, ends, side="left")
            == np.searchsorted(barriers, starts, side="left"))


def _past_point_valid(grid: pd.DataFrame, when: np.ndarray, barriers: np.ndarray) -> np.ndarray:
    times = grid.time_ns.to_numpy(dtype=np.int64)
    index = np.searchsorted(times, when, side="right") - 1
    clipped = np.maximum(index, 0)
    return ((index >= 0) & grid.valid.to_numpy(dtype=bool)[clipped]
            & _interval_clear(barriers, times[clipped], when))


def _history_support(grid: pd.DataFrame, steps: int, barriers: np.ndarray) -> np.ndarray:
    n = len(grid)
    result = np.zeros(n, dtype=bool)
    if n <= steps:
        return result
    indices = np.arange(steps, n)
    valid = grid.valid.to_numpy(dtype=bool)
    prefix = np.r_[0, np.cumsum(~valid)]
    segments = grid.segment.to_numpy(dtype=np.int64)
    times = grid.time_ns.to_numpy(dtype=np.int64)
    result[indices] = ((prefix[indices + 1] - prefix[indices - steps] == 0)
                       & (segments[indices] == segments[indices - steps])
                       & _interval_clear(barriers, times[indices] - _NS_PER_SECOND, times[indices]))
    return result


def _flow(grid: pd.DataFrame, prefix: str, sides: Sequence[str]) -> np.ndarray:
    cumulative = sum(grid[f"cum_{prefix}_{side}"].to_numpy(dtype=float) for side in sides)
    return np.r_[np.nan, np.diff(cumulative)]


def _imbalance(buy: np.ndarray, sell: np.ndarray, total: np.ndarray | None = None) -> np.ndarray:
    denominator = buy + sell if total is None else total
    return np.divide(buy - sell, denominator, out=np.zeros_like(denominator), where=denominator > 0)


def build_day_panels(
    replays: dict[str, dict], market_source: pd.DataFrame | str | Path,
    thresholds: dict[str, float], protocol: dict | None = None,
) -> dict[str, dict]:
    """Build identical forecast cohorts for all four fixed model families.

    Within each target, every model receives the exact same timestamps and
    accepted-infection labels. Current one-second book support is required on
    all venues, as is valid book support at the past-shift exposure timestamp.
    Source states are evaluated at t-2 ms and t-30002 ms, without time wrapping.
    """
    protocol = protocol or {}
    if protocol.get("controls", CONTROL_NAMES) != CONTROL_NAMES:
        raise ValueError("Controls differ from the frozen MBO/SIRS specification.")
    if protocol.get("forecast_anchor_ms", 100) != 100 or protocol.get("forecast_horizon_ms", 10) != 10:
        raise ValueError("Forecast grid or horizon differs from the frozen 100/10 ms specification.")
    venues = sorted(replays)
    if len(venues) < 2 or set(thresholds) != set(venues):
        raise ValueError("At least two venues and exactly one threshold per venue are required.")
    if isinstance(protocol.get("venues"), int) and protocol["venues"] != len(venues):
        raise ValueError("Venue count differs from the frozen protocol.")
    first = replays[venues[0]]["grid"]
    anchors = _ns(first.time_ns)
    if len(anchors) < 2:
        raise ValueError("A nonempty uniform forecast grid is required.")
    differences = np.diff(anchors)
    step_ns = int(differences[0])
    if step_ns != 100 * _NS_PER_MS or not np.all(differences == step_ns):
        raise ValueError("The frozen protocol requires a uniform 100 ms forecast grid.")
    n = len(anchors)
    start = int(replays[venues[0]].get("analysis_start_ns", anchors[0]))
    end = int(replays[venues[0]].get("analysis_end_ns", anchors[-1] + step_ns))
    if not end > start:
        raise ValueError("Analysis end must follow its start.")
    infectious_ms = protocol.get("susceptible_infected_refractory", {}).get("infectious_ms", 100)
    refractory_ms = protocol.get("susceptible_infected_refractory", {}).get("refractory_ms", 100)
    if (infectious_ms, refractory_ms) != (100, 100):
        raise ValueError("Infectious and refractory durations differ from the frozen 100/100 ms protocol.")
    market = market_features(market_source, anchors)
    common = (anchors >= start + 30 * _NS_PER_SECOND) & (anchors + 10 * _NS_PER_MS <= end) & market["valid"]
    infections, barriers_by_venue, support_by_venue = {}, {}, {}
    source_i, shifted_i, returns, flow_columns = {}, {}, [], {}
    for venue in venues:
        replay = replays[venue]
        grid = replay["grid"]
        if not np.array_equal(_ns(grid.time_ns), anchors):
            raise ValueError("Every venue must retain the identical complete forecast grid.")
        threshold = float(thresholds[venue])
        if not np.isfinite(threshold) or threshold < .005:
            raise ValueError("Frozen midpoint shock thresholds must be finite and at least $0.005.")
        barriers = _barriers(replay)
        barriers_by_venue[venue] = barriers
        jumps = replay["jumps"]
        magnitude = np.abs(jumps["jump"].to_numpy(dtype=float))
        if not np.isfinite(magnitude).all():
            raise ValueError("Jump candidates must be finite.")
        times = _ns(jumps.loc[magnitude >= threshold, "time_ns"])
        infections[venue] = accepted_infections(times, reset_times_ns=barriers)
        source_i[venue] = sirs_states(infections[venue], anchors, lag_ms=2, reset_times_ns=barriers) == INFECTIOUS
        shifted_i[venue] = sirs_states(infections[venue], anchors, lag_ms=30_002, reset_times_ns=barriers) == INFECTIOUS
        support = _history_support(grid, 10, barriers)
        shifted_support = _past_point_valid(grid, anchors - 30_002 * _NS_PER_MS, barriers)
        add_b, add_a = (_flow(grid, "add_shares", [side]) for side in ("B", "A"))
        remove_b, remove_a = (_flow(grid, "removed_shares", [side]) for side in ("B", "A"))
        trade_b, trade_a, trade_n = (_flow(grid, "trade_shares", [side]) for side in ("B", "A", "N"))
        flow = np.column_stack([add_b, add_a, remove_b, remove_a, trade_b, trade_a, trade_n])
        flow_good = np.isfinite(flow).all(axis=1) & (flow >= 0).all(axis=1)
        support_by_venue[venue] = {"history": support, "shifted": shifted_support, "flow": flow_good}
        common &= support & shifted_support & flow_good
        flow_columns[venue] = (add_b, add_a, remove_b, remove_a, trade_b, trade_a, trade_n)
        mids = grid.mid.to_numpy(dtype=float)
        lagged_mid = np.r_[np.full(10, np.nan), mids[:-10]] if n >= 10 else np.full(n, np.nan)
        returns.append((mids / lagged_mid - 1) * 1e4)
    market_return = np.median(np.column_stack(returns), axis=1)
    common &= np.isfinite(market_return)
    panels = {}
    for venue in venues:
        replay, grid = replays[venue], replays[venue]["grid"]
        add_b, add_a, remove_b, remove_a, trade_b, trade_a, trade_n = flow_columns[venue]
        add_total, remove_total, trade_total = add_b + add_a, remove_b + remove_a, trade_b + trade_a + trade_n
        age = grid.state_change_age_ms.to_numpy(dtype=float)
        values = [
            (anchors - start) / (end - start), np.log1p(grid.spread_bps.to_numpy(dtype=float)),
            grid.depth_imbalance_1.to_numpy(dtype=float), grid.depth_imbalance_10.to_numpy(dtype=float),
            np.log1p(grid.bid_depth_10.to_numpy(dtype=float) + grid.ask_depth_10.to_numpy(dtype=float)),
            np.log1p(grid.order_count.to_numpy(dtype=float)), _imbalance(add_b, add_a),
            _imbalance(remove_b, remove_a), np.log1p(add_total), np.log1p(remove_total),
            _imbalance(trade_b, trade_a, trade_total), np.log1p(trade_total), np.log1p(age),
            _lag_counts(infections[venue], anchors, 1000), market["x"][:, 0], market["x"][:, 1], market_return,
        ]
        x = np.column_stack(values).astype(np.float64)
        susceptible = sirs_states(infections[venue], anchors, reset_times_ns=barriers_by_venue[venue]) == SUSCEPTIBLE
        future = _interval_clear(barriers_by_venue[venue], anchors, anchors + 10 * _NS_PER_MS)
        intervals = np.asarray(replay.get("atomic_intervals_ns", []), dtype=np.int64).reshape(-1, 2)
        if len(intervals):
            # Atomic intervals can span positive receive time. A target outcome
            # touching one is unavailable even when endpoints look healthy.
            order = np.argsort(intervals[:, 0])
            intervals = intervals[order]
            prior_count = np.searchsorted(intervals[:, 0], anchors + 10 * _NS_PER_MS, side="left")
            ended = np.searchsorted(np.sort(intervals[:, 1]), anchors, side="left")
            future &= prior_count == ended
        eligible = common & susceptible & future & np.isfinite(x).all(axis=1) & (age >= 0)
        sources = [source for source in venues if source != venue]
        legacy, legacy_names = [], []
        for source in [venue] + sources:
            for milliseconds in (10, 100, 1000):
                legacy.append(_lag_counts(infections[source], anchors, milliseconds))
                label = "own" if source == venue else f"other_{source}"
                legacy_names.append(f"{label}_infections_{milliseconds}ms")
        panels[venue] = {
            "dataset": venue, "day": replay.get("day", pd.Timestamp(int(anchors[0]), tz="UTC").strftime("%Y-%m-%d")),
            "x": x[eligible], "names": CONTROL_NAMES.copy(),
            "infected": np.column_stack([source_i[source] for source in sources])[eligible],
            "shifted_infected": np.column_stack([shifted_i[source] for source in sources])[eligible],
            "source_names": sources, "y": infection_labels(infections[venue], anchors)[eligible],
            "legacy": np.column_stack(legacy).astype(np.float64)[eligible], "legacy_names": legacy_names,
            "time_ns": anchors[eligible], "infections_ns": infections[venue],
            "coverage": {"candidate_bins": n, "common_supported_bins": int(common.sum()),
                         "cohort_guard": "All venues: valid completed states and a continuous healthy prior second; valid carried state at t-30002ms with no intervening barrier. Every model shares this mask, target susceptibility, observed next-10ms support and finite controls.",
                         "initial_warmup_seconds": 30,
                         "source_lead_ms": 2, "shifted_source_lag_ms": 30_002,
                         "target_susceptible_bins": int((common & susceptible).sum()),
                         "target_future_supported_bins": int((common & susceptible & future).sum()),
                         "eligible_bins": int(eligible.sum()), "positive_bins": int(infection_labels(infections[venue], anchors)[eligible].sum()),
                         "market_control": market["diagnostics"],
                         "feed_support": {source: {key: int(mask.sum()) for key, mask in support_by_venue[source].items()} for source in venues}},
        }
    return panels
