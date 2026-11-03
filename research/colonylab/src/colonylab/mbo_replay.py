"""Full-day native MBO replay, causal RTH grids, and exact MBP-10 checks.

The source stream is replayed from UTC midnight.  Only a fixed grid and
completed-event midpoint changes are retained, not millions of book snapshots.
The legacy synthetic startup clear opt-in is gated on DBN request metadata;
its unreliable receive timestamp is never used as a feature or shock.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable

import databento as db
import numpy as np
import pandas as pd

from .mbo_book import (
    Book, F_LAST, F_SNAPSHOT, F_BAD_TS_RECV, F_MAYBE_BAD_BOOK,
    PRICE_SCALE, UNDEF_PRICE, UNDEF_TIMESTAMP,
)

LEVEL_FIELDS = ("bid_px", "ask_px", "bid_sz", "ask_sz", "bid_ct", "ask_ct")


def _day_bounds(day: str) -> tuple[int, int, int]:
    midnight = pd.Timestamp(day, tz="UTC").normalize()
    return midnight.value, (midnight + pd.Timedelta(hours=13, minutes=30)).value, (midnight + pd.Timedelta(hours=20)).value


def _metadata_guard(store: Any, *, day_start: int, end: int, schema: str) -> None:
    metadata = store.metadata
    if str(metadata.schema) != schema:
        raise ValueError(f"Expected {schema} schema, received {metadata.schema}")
    if schema == "mbo":
        if int(metadata.start) != day_start:
            raise ValueError("Full MBO replay requires request metadata starting exactly at UTC midnight")
        if metadata.end is None or int(metadata.end) < end:
            raise ValueError("Full MBO replay requires request metadata covering the RTH close")


def _reference_levels(record: Any) -> tuple[tuple[int, int, int, int, int, int], ...]:
    levels = tuple(tuple(int(getattr(level, name)) for name in LEVEL_FIELDS) for level in record.levels)
    if len(levels) != 10:
        raise ValueError(f"MBP-10 record has {len(levels)} levels")
    return levels


def _load_validation(path: str | Path) -> tuple[dict, dict, tuple[int, int] | None]:
    store = db.DBNStore.from_file(path)
    if str(store.metadata.schema) != "mbp-10":
        raise ValueError("Validation file must use mbp-10 schema")
    references = {}
    counts: Counter[str] = Counter()
    namespace = None
    for record in store:
        counts["provider_records"] += 1
        if not hasattr(record, "levels") or len(record.levels) != 10:
            raise ValueError("Validation stream contains a non-MBP-10 record")
        flags = int(record.flags)
        if not flags & F_LAST:
            counts["excluded_non_last_records"] += 1
            continue
        if flags & (F_SNAPSHOT | F_BAD_TS_RECV | F_MAYBE_BAD_BOOK):
            counts["excluded_quality_flag_records"] += 1
            continue
        this_namespace = (int(record.publisher_id), int(record.instrument_id))
        if namespace is None:
            namespace = this_namespace
        elif namespace != this_namespace:
            raise ValueError("Validation stream contains multiple publisher/instrument books")
        key = (int(record.ts_recv), int(record.sequence))
        if key in references:
            counts["duplicate_reference_keys"] += 1
        references[key] = _reference_levels(record)
    counts["reference_keys"] = len(references)
    return references, dict(counts), namespace


def _grid_arrays(anchors: np.ndarray) -> dict[str, np.ndarray]:
    template = Book().snapshot()
    arrays: dict[str, np.ndarray] = {"time_ns": anchors}
    omit = {"time_ns", "publisher_id", "instrument_id", "initialization_source"}
    floats = {"mid", "spread_bps", "depth_imbalance", "depth_imbalance_1", "depth_imbalance_5", "depth_imbalance_10",
              "cum_cancel_lifetime_ns", "cum_cancel_age_share_ns"}
    booleans = {"valid", "initialized", "event_complete"}
    for key in template:
        if key in omit:
            continue
        dtype = np.bool_ if key in booleans else np.float64 if key in floats else np.int64
        arrays[key] = np.empty(len(anchors), dtype=dtype)
    arrays["last_complete_age_ms"] = np.empty(len(anchors), dtype=np.float64)
    arrays["state_change_age_ms"] = np.empty(len(anchors), dtype=np.float64)
    return arrays


def _write_grid(arrays: dict[str, np.ndarray], position: int, book: Book, analytic_segment: int) -> None:
    anchor = int(arrays["time_ns"][position])
    state = book.snapshot()
    state["segment"] = analytic_segment
    for key, values in arrays.items():
        if key in {"time_ns", "last_complete_age_ms", "state_change_age_ms"}:
            continue
        value = state[key]
        if value is None:
            value = UNDEF_PRICE if key in {"bid", "ask"} else -1
        values[position] = value
    for field, output in (("last_complete_ns", "last_complete_age_ms"),
                          ("last_state_change_ns", "state_change_age_ms")):
        timestamp = state[field]
        arrays[output][position] = (anchor - timestamp) / 1e6 if timestamp is not None else np.nan


def _write_grid_span(arrays: dict[str, np.ndarray], first: int, stop: int, book: Book, analytic_segment: int) -> None:
    """Fill a quiet interval once; state is constant, observation age advances."""
    if stop == first + 1:
        _write_grid(arrays, first, book, analytic_segment)
        return
    if stop <= first:
        return
    state = book.snapshot()
    state["segment"] = analytic_segment
    for key, values in arrays.items():
        if key in {"time_ns", "last_complete_age_ms", "state_change_age_ms"}:
            continue
        value = state[key]
        if value is None:
            value = UNDEF_PRICE if key in {"bid", "ask"} else -1
        values[first:stop] = value
    anchors = arrays["time_ns"][first:stop]
    for field, output in (("last_complete_ns", "last_complete_age_ms"),
                          ("last_state_change_ns", "state_change_age_ms")):
        timestamp = state[field]
        arrays[output][first:stop] = (anchors - timestamp) / 1e6 if timestamp is not None else np.nan


def _validation_summary(references: dict, observed: dict, reference_counts: dict) -> dict[str, Any]:
    result = dict(reference_counts)
    result.update(matched_keys=0, missing_keys=0, unhealthy_matched_keys=0,
                  price_size_mismatch_keys=0, count_mismatch_keys=0,
                  mismatch_keys=0, first_differences=[])
    for key, expected in references.items():
        found = observed.get(key)
        if found is None:
            result["missing_keys"] += 1
            if len(result["first_differences"]) < 8:
                result["first_differences"].append({"time_ns": key[0], "sequence": key[1], "reason": "missing_mbo_event"})
            continue
        actual, healthy = found
        result["matched_keys"] += 1
        result["unhealthy_matched_keys"] += not healthy
        price_size_bad = any(a[:4] != e[:4] for a, e in zip(actual, expected))
        count_bad = any(a[4:] != e[4:] for a, e in zip(actual, expected))
        result["price_size_mismatch_keys"] += price_size_bad
        result["count_mismatch_keys"] += count_bad
        result["mismatch_keys"] += price_size_bad or count_bad
        if (price_size_bad or count_bad) and len(result["first_differences"]) < 8:
            differences = []
            for level, (a, e) in enumerate(zip(actual, expected)):
                for field, avalue, evalue in zip(LEVEL_FIELDS, a, e):
                    if avalue != evalue:
                        differences.append({"level": level, "field": field, "mbo": avalue, "provider": evalue})
            result["first_differences"].append({"time_ns": key[0], "sequence": key[1], "reason": "level_mismatch", "fields": differences[:12]})
    n = len(references)
    result["matched_fraction"] = result["matched_keys"] / n if n else 0.0
    result["mismatch_fraction"] = result["mismatch_keys"] / n if n else None
    result["passed"] = bool(n and result["matched_keys"] == n and not result["mismatch_keys"] and not result["unhealthy_matched_keys"])
    return result


def replay_file(
    mbo_path: str | Path,
    validation_path: str | Path,
    *,
    day: str,
    anchor_step_ms: int = 100,
    progress: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Replay one complete native venue/AAPL day and validate top ten levels.

    Forecast anchors contain only records with ``ts_recv < anchor``. Quiet
    books remain valid because the complete feed carries state; stale quotes
    are not imputed across bad-book flags, semantic failures, or unfinished
    atomic events. Raw integer bid/ask fields use UNDEF_PRICE for missing sides.
    All cumulative order counts/quantities stay integer; accumulated lifetime
    nanoseconds use float64 to avoid overflow in full-session aggregate sums.
    """
    if not isinstance(anchor_step_ms, int) or anchor_step_ms < 1 or 23_400_000 % anchor_step_ms:
        raise ValueError("anchor_step_ms must be a positive integral divisor of the RTH duration")
    day_start, start, end = _day_bounds(day)
    store = db.DBNStore.from_file(mbo_path)
    _metadata_guard(store, day_start=day_start, end=end, schema="mbo")
    references, reference_counts, reference_namespace = _load_validation(validation_path)
    anchors = np.arange(start, end, anchor_step_ms * 1_000_000, dtype=np.int64)
    arrays = _grid_arrays(anchors)
    book = Book(allow_legacy_startup_clear=True)
    observed = {}
    jump_times, jump_values, jump_segments, jump_mids = [], [], [], []
    invalid_times = []
    reset_times = []
    atomic_intervals = []
    analytic_segment = 0
    last_complete_valid = False
    atomic_start_ns = None
    previous_mid: int | None = None
    previous_segment: int | None = None
    position = 0
    record_count = 0
    first_record_time = last_record_time = None
    ref_first = min((key[0] for key in references), default=end)
    ref_last = max((key[0] for key in references), default=start)
    for record in store:
        if not hasattr(record, "order_id") or not hasattr(record, "channel_id"):
            raise ValueError("MBO stream contains a non-native MBO record")
        record_count += 1
        time_ns = int(record.ts_recv)
        flags = int(record.flags)
        synthetic_legacy = (record_count == 1 and str(record.action) == "R"
                            and flags in {F_BAD_TS_RECV, F_BAD_TS_RECV | F_LAST}
                            and int(record.sequence) == 0 and int(record.order_id) == 0
                            and int(record.price) == UNDEF_PRICE and int(record.size) == 0
                            and str(record.side) == "N")
        synthetic_undefined = bool(flags & F_SNAPSHOT and str(record.action) == "R" and time_ns == UNDEF_TIMESTAMP)
        if synthetic_legacy and not day_start <= time_ns < start:
            raise ValueError("Legacy startup clear must precede the RTH analysis window")
        if not (synthetic_legacy or synthetic_undefined):
            if not day_start <= time_ns < end:
                raise ValueError("MBO receive timestamp falls outside complete-session request bounds")
            first_record_time = time_ns if first_record_time is None else first_record_time
            last_record_time = time_ns
            # Process a whole timestamp strictly after its matching anchor.
            if position < len(anchors) and int(anchors[position]) <= time_ns:
                stop = int(np.searchsorted(anchors, time_ns, side="right"))
                _write_grid_span(arrays, position, stop, book, analytic_segment)
                position = stop
        was_healthy = book._healthy
        old_segment = book.segment
        book.apply(record, emit=False)
        if reference_namespace is not None and (book.publisher_id, book.instrument_id) != reference_namespace:
            raise ValueError("MBO and validation publisher/instrument namespaces differ")
        if was_healthy and not book._healthy:
            invalid_times.append(time_ns)
            analytic_segment += 1
            previous_mid = previous_segment = None
            last_complete_valid = False
        if str(record.action) == "R" and book._healthy and book.segment != old_segment:
            reset_times.append({"time_ns": None if synthetic_legacy or synthetic_undefined else time_ns,
                                "source": book.initialization_source})
            analytic_segment += 1
            previous_mid = previous_segment = None
            last_complete_valid = False
        if atomic_start_ns is None and not synthetic_legacy and not synthetic_undefined:
            atomic_start_ns = time_ns
        if flags & F_LAST and not synthetic_legacy:
            if atomic_start_ns is not None and atomic_start_ns < time_ns:
                atomic_intervals.append((atomic_start_ns, time_ns))
                invalid_times.append(atomic_start_ns)
                analytic_segment += 1
                previous_mid = previous_segment = None
            atomic_start_ns = None
            if book.valid:
                mid_raw_twice = book._bid_prices[-1] + book._ask_prices[0]
                if previous_mid is not None and previous_segment == analytic_segment and mid_raw_twice != previous_mid and start <= time_ns < end:
                    jump_times.append(time_ns)
                    jump_values.append((mid_raw_twice - previous_mid) / (2 * PRICE_SCALE))
                    jump_segments.append(analytic_segment)
                    jump_mids.append(mid_raw_twice / (2 * PRICE_SCALE))
                previous_mid = mid_raw_twice
                previous_segment = analytic_segment
            else:
                if last_complete_valid:
                    invalid_times.append(time_ns)
                    analytic_segment += 1
                previous_mid = previous_segment = None
            last_complete_valid = book.valid
            if ref_first <= time_ns <= ref_last:
                key = (time_ns, int(record.sequence))
                if key in references:
                    # Repeated same-key normalized final records overwrite so
                    # validation compares the final state for that atomic key.
                    observed[key] = (book.level_snapshot(10), book.initialized and book._healthy and book.event_complete)
        if progress is not None and record_count % 1_000_000 == 0:
            progress({"stage": "mbo_replay", "day": day, "records": record_count,
                      "grid_rows": position, "jumps": len(jump_times), "time_ns": time_ns})
    requested_limit = store.metadata.limit
    if requested_limit not in (None, 0) and record_count >= int(requested_limit):
        raise ValueError("MBO record count reached request limit; complete session cannot be established")
    if atomic_start_ns is not None:
        # A truncated final atomic event leaves its whole remainder unknown.
        # Keep the last grid invalid and prevent future labels spanning its
        # first partial record from being treated as a clean negative.
        invalid_times.append(atomic_start_ns)
        atomic_intervals.append((atomic_start_ns, end))
        analytic_segment += 1
    _write_grid_span(arrays, position, len(anchors), book, analytic_segment)
    position = len(anchors)
    grid = pd.DataFrame(arrays, copy=False)
    jumps = pd.DataFrame({"time_ns": np.asarray(jump_times, dtype=np.int64),
                          "jump": np.asarray(jump_values, dtype=np.float64),
                          "segment": np.asarray(jump_segments, dtype=np.int64),
                          "mid": np.asarray(jump_mids, dtype=np.float64)})
    validation = _validation_summary(references, observed, reference_counts)
    diagnostics: dict[str, Any] = dict(book.diagnostics)
    diagnostics.update({"day": day, "dataset": str(store.metadata.dataset),
                        "publisher_id": book.publisher_id, "instrument_id": book.instrument_id,
                        "channels_seen": sorted(book.channels_seen), "initialized": book.initialized,
                        "final_healthy": book._healthy, "final_event_complete": book.event_complete,
                        "initialization_source": book.initialization_source,
                        "records_replayed": record_count, "first_real_receive_ns": first_record_time,
                        "last_real_receive_ns": last_record_time, "grid_rows": len(grid),
                        "valid_grid_rows": int(grid["valid"].sum()), "midpoint_change_candidates": len(jumps),
                        "invalid_barrier_times_ns": invalid_times, "resets": reset_times,
                        "positive_duration_atomic_intervals": len(atomic_intervals),
                        "unfinished_atomic_event_at_eof": atomic_start_ns is not None,
                        "final_analytic_segment": analytic_segment,
                        "anchor_step_ms": anchor_step_ms, "strict_receive_time_features": True,
                        "quiet_book_coverage": "continuous full-session displayed state; no event-age cutoff",
                        "grid_memory_bytes": int(grid.memory_usage(deep=True).sum())})
    if progress is not None:
        progress({"stage": "mbo_replay_complete", "day": day, "records": record_count,
                  "grid_rows": len(grid), "valid_grid_rows": diagnostics["valid_grid_rows"],
                  "validation_passed": validation["passed"]})
    return {"grid": grid, "jumps": jumps, "validation": validation, "diagnostics": diagnostics,
            "reset_times_ns": np.unique(np.asarray([row["time_ns"] for row in reset_times if row["time_ns"] is not None] + invalid_times, dtype=np.int64)),
            "invalid_times_ns": np.unique(np.asarray(invalid_times, dtype=np.int64)),
            "atomic_intervals_ns": np.asarray(atomic_intervals, dtype=np.int64).reshape(-1, 2),
            "requested_start_ns": day_start, "requested_end_ns": end,
            "analysis_start_ns": start, "analysis_end_ns": end, "day": day}
