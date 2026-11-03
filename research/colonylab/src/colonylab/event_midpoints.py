"""Verified native-event midpoint paths for saved full-session MBO event trials.

Only the replay's native receive-time midpoint changes drive matching. The
100-ms book grid validates price states and initial coverage; it never supplies
microsecond event times. These paths contain neither native bid/ask changes nor
exchange-clock observations and must not be reported as those measurements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from .latency import _ThresholdIndex, summarize_events as _summarize_matching


PRICE2_SCALE = 2_000_000_000
SEMANTIC_FAILURES = (
    "duplicate_adds", "unmatched_cancels", "cancel_identity_mismatches", "oversize_cancels",
    "unmatched_modifies", "unknown_modify_invalidations", "modify_upserts", "modify_side_mismatches",
    "snapshot_without_clear", "invalid_snapshot_actions", "invalid_order_fields", "invalid_fill_fields",
    "invalid_trade_fields", "unknown_actions", "non_native_records", "unmatched_fills", "invalid_timestamps",
)


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def file_sha256(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _restore_integer(values, name: str, *, positive=False) -> tuple[np.ndarray, float]:
    values = np.asarray(values, dtype=np.float64)
    _require(values.ndim == 1 and np.isfinite(values).all(), f"{name} must be a finite vector")
    scaled = values * PRICE2_SCALE
    _require((np.abs(scaled) < (1 << 62)).all(), f"{name} exceeds exact supported integer price range")
    rounded = np.rint(scaled)
    residual = float(np.max(np.abs(scaled - rounded))) if len(values) else 0.
    _require(residual < .01, f"{name} cannot be recovered on the native twice-midpoint integer lattice")
    result = rounded.astype(np.int64)
    if positive:
        _require((result > 0).all(), "Native midpoint states must be positive")
    return result, residual


@dataclass
class NativeMidpoints:
    """One receive-clock midpoint path, with explicit observation discontinuities.

    ``price2`` is bid+ask in integer nanodollars. One prior state may precede
    start_ns to establish the baseline; end_ns is exclusive. Quiet initialized
    MBO books remain observed without an arbitrary last-price-change age cut.
    """
    times_ns: np.ndarray
    price2: np.ndarray
    valid: np.ndarray
    segments: np.ndarray
    start_ns: int
    end_ns: int
    grid: pd.DataFrame | None = None
    _tree: _ThresholdIndex | None = field(default=None, init=False, repr=False)
    _barriers: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        for name in ("times_ns", "price2", "segments"):
            values = np.asarray(getattr(self, name))
            _require(values.ndim == 1 and values.dtype.kind in "iu", f"{name} must be an integer vector")
            _require(not (values.dtype.kind == "u" and np.any(values > np.iinfo(np.int64).max)),
                     f"{name} exceeds signed integer range")
            setattr(self, name, values.astype(np.int64, copy=False))
        self.valid = np.asarray(self.valid, dtype=bool)
        n = len(self.times_ns)
        _require(self.valid.ndim == 1 and all(len(getattr(self, name)) == n for name in ("price2", "valid", "segments")),
                 "All midpoint arrays must have equal lengths")
        _require((self.times_ns >= 0).all() and (np.diff(self.times_ns) > 0).all(),
                 "Native midpoint timestamps must be nonnegative and strictly increasing")
        _require((self.segments >= 0).all() and (self.price2[self.valid] > 0).all(), "Invalid midpoint segment or valid price")
        self.start_ns, self.end_ns = int(self.start_ns), int(self.end_ns)
        _require(0 <= self.start_ns < self.end_ns <= np.iinfo(np.int64).max, "Invalid half-open midpoint observation window")
        boundary = ~self.valid
        if n > 1:
            boundary = boundary | np.r_[False, self.segments[1:] != self.segments[:-1]]
        self._barriers = self.times_ns[boundary]

    def next_barrier(self, times):
        times = np.asarray(times, np.int64)
        index = np.searchsorted(self._barriers, times, side="right")
        result = np.full(len(times), self.end_ns, dtype=np.int64)
        present = index < len(self._barriers)
        result[present] = np.minimum(self._barriers[index[present]], self.end_ns)
        return result

    def threshold_index(self):
        if self._tree is None:
            self._tree = _ThresholdIndex(self.price2, self.valid)
        return self._tree


def slice_series(series: NativeMidpoints, start_ns: int, end_ns: int) -> NativeMidpoints:
    """Restrict support without losing the actual strictly prior baseline."""
    start_ns, end_ns = int(start_ns), int(end_ns)
    _require(series.start_ns <= start_ns < end_ns <= series.end_ns, "Slice lies outside loaded midpoint support")
    first = max(0, int(np.searchsorted(series.times_ns, start_ns, side="left")) - 1)
    stop = int(np.searchsorted(series.times_ns, end_ns, side="left"))
    grid = (series.grid.loc[series.grid.time_ns.ge(start_ns) & series.grid.time_ns.lt(end_ns)].copy()
            if series.grid is not None else None)
    return NativeMidpoints(series.times_ns[first:stop].copy(), series.price2[first:stop].copy(),
                           series.valid[first:stop].copy(), series.segments[first:stop].copy(), start_ns, end_ns, grid)


def match_events(leader: NativeMidpoints, follower: NativeMidpoints, *,
                 max_horizon_ns: int = 100_000_000, min_move2: int = 0) -> pd.DataFrame:
    """Native midpoint first passage, with full horizon/quality support recorded.

    The target is the leader's new midpoint, not either best quote. Equal-time
    followers already at/beyond it are excluded from positive-delay matches.
    Later leader reversals do not cancel the original target.
    """
    _require(isinstance(max_horizon_ns, (int, np.integer)) and max_horizon_ns > 0, "Horizon must be positive integer nanoseconds")
    _require(isinstance(min_move2, (int, np.integer)) and min_move2 >= 0, "Minimum move must be nonnegative integer price2 units")
    change = np.diff(leader.price2)
    keep = (leader.valid[:-1] & leader.valid[1:] & (leader.segments[:-1] == leader.segments[1:])
            & (np.abs(change) > min_move2) & (leader.times_ns[1:] >= leader.start_ns)
            & (leader.times_ns[1:] < leader.end_ns))
    position = np.flatnonzero(keep) + 1
    times, target = leader.times_ns[position], leader.price2[position]
    direction = np.sign(change[keep]).astype(np.int8)
    count = len(times)
    status = np.full(count, "no_state", dtype=object)
    lag = np.full(count, -1, dtype=np.int64)
    observed = np.zeros(count, dtype=np.int64)
    reason = np.full(count, "end", dtype=object)
    if count and len(follower.times_ns):
        prior = np.searchsorted(follower.times_ns, times, side="right") - 1
        known = (prior >= 0) & (times >= follower.start_ns) & (times < follower.end_ns)
        known &= follower.valid[np.clip(prior, 0, len(follower.valid) - 1)]
        indexes = np.flatnonzero(known)
        already = np.zeros(count, dtype=bool)
        already[indexes] = np.where(direction[indexes] > 0,
             follower.price2[prior[indexes]] >= target[indexes], follower.price2[prior[indexes]] <= target[indexes])
        status[already] = "already"
        behind = known & ~already
        status[behind] = "behind"
        stop = np.minimum(leader.next_barrier(times), follower.next_barrier(times))
        end = min(leader.end_ns, follower.end_ns)
        reason[:] = np.where(stop < end, "quality", "end")
        available = np.maximum(0, stop - times - 1)
        observed[known] = np.minimum(available[known], max_horizon_ns)
        reason[available >= max_horizon_ns] = "horizon"
        indexes = np.flatnonzero(behind)
        if len(indexes):
            lo = prior[indexes] + 1
            hi = np.searchsorted(follower.times_ns, times[indexes] + observed[indexes], side="right")
            hit = follower.threshold_index().first_hit(lo, hi, target[indexes], direction[indexes])
            found = hit >= 0
            lag[indexes[found]] = follower.times_ns[hit[found]] - times[indexes[found]]
    result = pd.DataFrame({"t": times, "price2": target, "change2": change[keep], "direction": direction,
                           "latency_ns": lag, "observed_ns": observed, "terminal_reason": reason, "status": status})
    result.attrs.update(max_horizon_ns=int(max_horizon_ns), metric="native_receive_midpoint_matching",
                        clock="ts_recv", price2_scale=PRICE2_SCALE)
    return result


def summarize_events(events: pd.DataFrame, horizon_ns: int) -> dict:
    result = _summarize_matching(events, horizon_ns)
    result.update(metric="native_receive_midpoint_matching", clock="ts_recv")
    return result


def _normalize_cache(payload: dict, start_ns: int, end_ns: int) -> tuple[NativeMidpoints, dict]:
    """Build a conservative native path and check every eligible grid state."""
    jumps, grid = payload["jumps"], payload["grid"]
    required = {"time_ns", "mid", "jump", "segment"}
    _require(required <= set(jumps) and {"time_ns", "bid", "ask", "valid", "segment"} <= set(grid),
             "Replay cache lacks native jumps or integer validation grid")
    _require(len(grid) > 0, "Replay validation grid is empty")
    jt = jumps.time_ns.to_numpy(np.int64)
    js = jumps.segment.to_numpy(np.int64)
    _require((np.diff(jt) >= 0).all(), "Native midpoint jumps reverse receive time")
    price, price_residual = _restore_integer(jumps.mid.to_numpy(), "Native midpoint", positive=True)
    delta, delta_residual = _restore_integer(jumps.jump.to_numpy(), "Native midpoint change")
    same = js[1:] == js[:-1]
    _require(np.array_equal(np.diff(price)[same], delta[1:][same]),
             "Stored jump changes disagree with adjacent exact midpoint states")
    invalid = np.asarray(payload["invalid_times_ns"], np.int64)
    reset = np.asarray(payload["reset_times_ns"], np.int64)
    atomic = np.asarray(payload["atomic_intervals_ns"], np.int64).reshape(-1, 2)
    _require((atomic[:, 1] >= atomic[:, 0]).all(), "Atomic quality intervals are reversed")
    barriers = np.unique(np.r_[invalid, reset, atomic[:, 0]])
    if len(atomic) and len(jt):
        for first, stop in atomic:
            _require(not np.any((jt >= first) & (jt < stop)), "Native jump occurs inside an unfinished atomic interval")
    gt = grid.time_ns.to_numpy(np.int64)
    _require((np.diff(gt) > 0).all(), "Validation grid timestamps are not increasing")
    gi = int(np.searchsorted(gt, start_ns, side="right")) - 1
    prior = int(np.searchsorted(jt, start_ns, side="left")) - 1
    seed = (prior >= 0 and gi >= 0 and bool(grid.valid.iloc[gi]) and int(grid.segment.iloc[gi]) == int(js[prior]))
    if seed:
        seed = not np.any((barriers > jt[prior]) & (barriers <= start_ns))
    first = int(np.searchsorted(jt, start_ns, side="left"))
    stop = int(np.searchsorted(jt, end_ns, side="left"))
    time = jt[first:stop].copy()
    values = price[first:stop].copy()
    segments = js[first:stop].copy()
    valid = np.ones(len(time), dtype=bool)
    if seed:
        time, values, segments, valid = (np.r_[array[prior], chosen] for array, chosen in
            ((jt, time), (price, values), (js, segments), (np.ones(len(jt), bool), valid)))
    else:
        initial_segment = int(grid.segment.iloc[gi]) if gi >= 0 else 0
        time = np.r_[start_ns - 1, time]
        values = np.r_[0, values]
        segments = np.r_[max(0, initial_segment), segments]
        valid = np.r_[False, valid]
    selected_barriers = barriers[(barriers >= start_ns) & (barriers < end_ns)]
    if len(selected_barriers):
        nearest = np.searchsorted(jt, selected_barriers, side="right") - 1
        barrier_segments = np.where(nearest >= 0, js[np.maximum(nearest, 0)] if len(js) else 0, 0)
        time = np.r_[time, selected_barriers]
        values = np.r_[values, np.zeros(len(selected_barriers), np.int64)]
        segments = np.r_[segments, barrier_segments]
        valid = np.r_[valid, np.zeros(len(selected_barriers), bool)]
    order = np.argsort(time, kind="stable")
    time, values, segments, valid = (array[order] for array in (time, values, segments, valid))
    groups = np.r_[0, np.flatnonzero(np.diff(time) != 0) + 1]
    lengths = np.diff(np.r_[groups, len(time)])
    repeated = np.repeat(groups, lengths)
    conflict = (values != values[repeated]) | (segments != segments[repeated])
    bad = np.logical_or.reduceat(~valid | conflict, groups)
    conflicting = int(np.logical_or.reduceat(conflict, groups).sum())
    duplicate_rows = len(time) - len(groups)
    time, values, segments, valid = time[groups], values[groups], segments[groups], ~bad
    values[~valid] = 0
    series = NativeMidpoints(time, values, valid, segments, start_ns, end_ns)
    window = (gt >= start_ns) & (gt < end_ns)
    anchors = gt[window]
    raw_valid = grid.valid.to_numpy(bool)[window]
    raw_segment = grid.segment.to_numpy(np.int64)[window]
    index = np.searchsorted(series.times_ns, anchors, side="left") - 1
    safe = np.maximum(index, 0)
    known = (index >= 0) & series.valid[safe]
    _require(not np.any(known & ~raw_valid), "Inferred midpoint support bridges a raw invalid-book interval")
    eligible = raw_valid & known & (series.segments[safe] == raw_segment)
    expected = (grid.bid.to_numpy(np.int64)[window][eligible] + grid.ask.to_numpy(np.int64)[window][eligible])
    actual = series.price2[safe[eligible]]
    _require(np.array_equal(actual, expected), "Native midpoint path disagrees with strict-prior integer BBO grid")
    _require(not np.any(raw_valid & known & (series.segments[safe] != raw_segment)),
             "Native midpoint support disagrees with raw book segment")
    grid_fields = [name for name in ("time_ns", "valid", "segment", "bid", "ask", "bid_size", "ask_size",
                  "mid", "spread_bps", "bid_count", "ask_count", "order_count", "bid_depth_1", "ask_depth_1") if name in grid]
    series.grid = grid.loc[window, grid_fields].reset_index(drop=True).copy()
    return series, {"native_midpoint_rows": stop - first, "initial_native_seed_available": bool(seed),
                    "window_native_quality_barriers": len(selected_barriers), "duplicate_rows_coalesced": duplicate_rows,
                    "ambiguous_timestamp_groups": conflicting, "price2_recovery_max_residual": price_residual,
                    "jump2_recovery_max_residual": delta_residual, "grid_anchors_checked": len(anchors),
                    "grid_healthy_anchors": int(raw_valid.sum()), "grid_eligible_anchors_matched": int(eligible.sum()),
                    "grid_healthy_unknown_midpoint_anchors": int((raw_valid & ~known).sum()),
                    "grid_price_mismatches": 0, "native_receive_times_only": True, "bid_ask_events_available": False}


def load_day(root: str | Path, day: str, start_ns: int, end_ns: int,
             datasets: list[str]) -> tuple[dict[str, NativeMidpoints], list[dict]]:
    """Freshly verify trusted replay/source bytes, then load one event hour.

    No API client or download is used. Cache pickles are deserialized only after
    matching their SHA-256 in the original local research archive manifest.
    """
    root = Path(root).resolve()
    start_ns, end_ns = int(start_ns), int(end_ns)
    _require(datasets and len(datasets) == len(set(datasets)), "Use explicit unique native datasets")
    replay_path = root / "out/mbo_sir/replay_manifest.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    acquisition = json.loads((root / "out/mbo_sir/data_manifest.json").read_text(encoding="utf-8"))
    archive = json.loads((root / "research-archive-manifest.json").read_text(encoding="utf-8"))
    final_audit = json.loads((root / "out/mbo_sir/result_audit.json").read_text(encoding="utf-8"))
    prior_audit_passed = final_audit.get("passed") is True or final_audit.get("status") == "passed"
    _require(replay.get("complete") is True and acquisition.get("complete") is True and prior_audit_passed,
             "Native event trial requires complete previously audited MBO acquisition/replay")
    archive_rows = {row["path"].replace("\\", "/"): row for row in archive["files"]}
    entries = {_digest(row["request"]): row for row in acquisition["requests"]}
    code = {name: file_sha256(root / "src/colonylab" / name) for name in ("mbo_book.py", "mbo_replay.py")}
    code_hash = _digest(code)
    selected = [task for task in replay["tasks"] if task["day"] == day and task["dataset"] in datasets]
    _require(len(selected) == len(datasets) and {task["dataset"] for task in selected} == set(datasets),
             "Saved replay lacks the exact requested venue/day grid")
    paths, facts = {}, []
    for task in sorted(selected, key=lambda item: item["dataset"]):
        provenance = task["provenance"]
        _require(provenance["code"] == code_hash, "Replay code differs from saved reconstruction provenance")
        _require(task["validation"].get("passed") is True, "Saved MBP-10 reconstruction check did not pass")
        _require(not any(task["diagnostics"].get(name, 0) for name in SEMANTIC_FAILURES), "Saved replay has order-semantics failures")
        cache_path = root / task["path"]
        relative = cache_path.relative_to(root).as_posix()
        expected = archive_rows.get(relative)
        _require(expected is not None, "Replay cache has no original research-archive identity")
        cache_hash = file_sha256(cache_path)
        _require(cache_path.stat().st_size == expected["bytes"] and cache_hash == expected["sha256"],
                 "Replay cache bytes differ from the original archived study")
        raw_facts = []
        for schema, request in provenance["requests"].items():
            _require(request["dataset"] == task["dataset"] and request["symbols"] == ["AAPL"]
                     and request["schema"] == schema and request["start"][:10] == day,
                     "Replay request schema/date/venue/symbol provenance differs")
            entry = entries.get(_digest(request))
            _require(entry is not None and entry.get("complete") is True and not entry.get("record_limit_reached"),
                     "Native source request is absent, incomplete or capped")
            raw = Path(entry["data_path"])
            if not raw.is_absolute():
                raw = root / raw
            if not raw.exists():
                from .mbo_sir_data import cache_directory
                raw = cache_directory(root / acquisition["configuration"]["data_dir"], request) / "sample.dbn.zst"
            receipt = json.loads((raw.parent / "fetch.json").read_text(encoding="utf-8"))
            marker = json.loads((raw.parent / "stream_complete.json").read_text(encoding="utf-8"))
            _require(receipt.get("request") == request == marker.get("request") and receipt.get("complete") is True
                     and not receipt.get("record_limit_reached") and receipt["rows"] == entry["rows"]
                     and 0 < entry["rows"] < request["limit"] and raw.stat().st_size == receipt["file_size_bytes"] == entry["file_size_bytes"],
                     "Native raw receipt/termination/row/size identity is inconsistent")
            raw_hash = file_sha256(raw)
            _require(raw_hash == provenance["sources"][schema], "Native raw bytes changed after the validated replay")
            archived = archive_rows.get(raw.relative_to(root).as_posix())
            _require(archived is not None and archived["sha256"] == raw_hash, "Native source disagrees with original archive identity")
            raw_facts.append({"path": str(raw), "schema": schema, "sha256": raw_hash, "rows": int(entry["rows"]), "request": request})
        # This is an existing first-party local research checkpoint, with its
        # full bytes checked before any pickle deserialization.
        with gzip.open(cache_path, "rb") as stream:
            payload = pickle.load(stream)
        _require(payload.get("provenance") == provenance and payload["validation"] == task["validation"],
                 "Embedded replay identity/validation differs from the manifest")
        _require(payload["day"] == day and int(payload["analysis_start_ns"]) <= start_ns < end_ns <= int(payload["analysis_end_ns"]),
                 "Requested event window lies outside initialized full-session analysis support")
        series, quality = _normalize_cache(payload, start_ns, end_ns)
        paths[task["dataset"]] = series
        facts.append({"day": day, "dataset": task["dataset"], "cache_path": str(cache_path), "cache_sha256": cache_hash,
                      "cache_bytes": cache_path.stat().st_size, "replay_code_sha256": code_hash,
                      "raw_sources": raw_facts, "book_validation": task["validation"],
                      "observation_start_ns": start_ns, "observation_end_ns": end_ns,
                      "metric": "native_receive_midpoint_matching", "clock": "ts_recv", **quality})
    return paths, facts
