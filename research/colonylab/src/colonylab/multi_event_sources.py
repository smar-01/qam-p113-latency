"""Offline native MBP-1 paths assembled from exact contiguous source pieces.

Each DBN retains its own metadata and content identity. Only normalization uses
a virtual union interval; no DBN is represented as having union metadata.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .event_midpoints import NativeMidpoints
from .latency_data import _clock_integers, _metadata_check, file_sha256, normalize_frame


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _empty(start, end):
    return NativeMidpoints(np.array([], np.int64), np.array([], np.int64), np.array([], bool),
                           np.array([], np.int64), start, end)


def _partition(entries, group, datasets):
    start, end = pd.Timestamp(group["start"]), pd.Timestamp(group["end"])
    _require(start.tzinfo is not None and end.tzinfo is not None and start < end, "Invalid source-group interval")
    _require(len(datasets) == len(set(datasets)) and datasets, "Use unique explicit source-group venues")
    by_venue = {dataset: [] for dataset in datasets}
    for entry in entries:
        request = entry["request"]
        _require(request["dataset"] in by_venue and request["schema"] == "mbp-1"
                 and request["symbols"] == ["AAPL"] and request["stype_in"] == "raw_symbol",
                 "Foreign source fragment dataset/schema/symbol")
        first, last = pd.Timestamp(request["start"]), pd.Timestamp(request["end"])
        _require(first.tzinfo is not None and last.tzinfo is not None and start <= first < last <= end,
                 "Source fragment lies outside its group interval")
        _require(type(request["limit"]) is int and request["limit"] > 0, "Invalid source-fragment record limit")
        by_venue[request["dataset"]].append(entry)
    for dataset, pieces in by_venue.items():
        pieces.sort(key=lambda row: pd.Timestamp(row["request"]["start"]).value)
        cursor = start.value
        for entry in pieces:
            request = entry["request"]
            _require(pd.Timestamp(request["start"]).value == cursor,
                     f"Source fragments must exactly partition {dataset}: gap, overlap or duplicate")
            cursor = pd.Timestamp(request["end"]).value
        _require(pieces and cursor == end.value,
                 f"Missing source fragment: {dataset} does not cover the complete group interval")
    return by_venue, start.value, end.value


def _decode(entry):
    import databento as db

    request = entry["request"]
    raw = Path(entry["data_path"])
    _require(raw.is_file() and raw.stat().st_size == entry["file_size_bytes"]
             and file_sha256(raw) == entry["content_sha256"], "Source fragment bytes changed after acquisition")
    store = db.DBNStore.from_file(raw)
    _metadata_check(store, request)
    frame = store.to_df(price_type="fixed", pretty_ts=False, map_symbols=True).reset_index()
    _require(type(entry["rows"]) is int and 0 <= len(frame) == entry["rows"] < request["limit"],
             "Decoded source-fragment row count differs or reaches its cap")
    required = {"ts_recv", "ts_event", "symbol", "flags", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"}
    _require(required <= set(frame), "Native source fragment lacks quote fields")
    _require(all(frame[column].dtype.kind in "iu" for column in ("bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00")),
             "Native fixed source prices/sizes must remain integer")
    receives, valid_clock = _clock_integers(frame.ts_recv.to_numpy())
    first, last = pd.Timestamp(request["start"]).value, pd.Timestamp(request["end"]).value
    _require(valid_clock.all() and ((receives >= first) & (receives < last)).all(),
             "Source fragment has raw receive timestamps outside its exact interval")
    # ts_event may legitimately precede a fragment's receive-time boundary.
    # It is normalized once against the union, avoiding an artificial seam.
    return frame


def decode_group(entries, group, protocol):
    """Verify a venue-wise exact tiling and normalize native quotes jointly.

    Missing/gapped/overlapping request registries raise. Explicit unavailable,
    failed or unanalyzable positive-row pieces quarantine their entire venue.
    A completed empty piece is an observed interval without updates: ordinary
    initial-state and freshness rules still determine quote support.
    """
    by_venue, start, end = _partition(entries, group, protocol["datasets"])
    paths, facts = {}, []
    for dataset, entries in by_venue.items():
        frames, fragments, reasons = [], [], []
        raw_records = 0
        for entry in entries:
            request = entry["request"]
            complete = entry.get("complete") is True
            rows = int(entry.get("rows", 0))
            fragment = {"request": request, "request_sha256": _signature(request),
                        "start_ns": pd.Timestamp(request["start"]).value, "end_ns": pd.Timestamp(request["end"]).value,
                        "content_sha256": entry.get("content_sha256"), "rows": rows,
                        "status": entry.get("status", "complete" if complete else "unavailable"),
                        "complete": complete, "analyzable": bool(entry.get("analyzable", rows > 0)),
                        "data_path": entry.get("data_path")}
            if complete:
                frame = _decode(entry)
                frames.append(frame)
                raw_records += len(frame)
            if (not complete or fragment["status"] in {"unavailable", "failed"}
                    or (rows > 0 and not fragment["analyzable"])):
                reason = (entry.get("availability_reason") or entry.get("error")
                          or "source_unavailable_or_unanalyzable_fragment")
                fragment["reason"] = reason
                reasons.append(f"{request['start']}..{request['end']}: {reason}")
            fragments.append(fragment)
        union_request = {"dataset": dataset, "schema": "mbp-1", "symbols": ["AAPL"], "stype_in": "raw_symbol",
                         "start": group["start"], "end": group["end"],
                         "limit": sum(entry["request"]["limit"] for entry in entries)}
        common = {"dataset": dataset, "date": pd.Timestamp(group["start"]).date().isoformat(), "symbol": "AAPL",
                  "raw_records": raw_records, "source_available": not reasons,
                  "source_analyzable": not reasons and raw_records > 0, "source_chunk_count": len(fragments),
                  "source_hashes": [row["content_sha256"] for row in fragments if row["content_sha256"]],
                  "source_fragments": fragments, "source_fragments_sha256": _signature(fragments),
                  "virtual_union_request": union_request, "virtual_union_has_dbn_metadata": False,
                  "requested_start_ns": start, "requested_end_ns": end}
        for age in protocol["quote_age_ms"]:
            if reasons or raw_records == 0:
                for clock in protocol["clocks"]:
                    paths[(dataset, age, clock)] = _empty(start, end)
                    facts.append({**common, "clock": clock, "age_ms": age,
                                  "reason": "; ".join(reasons) if reasons else "all_source_fragments_empty_no_initial_quote_state"})
                continue
            frame = pd.concat(frames, ignore_index=True)
            normalized, quality = normalize_frame(frame, union_request, clocks=protocol["clocks"], max_age_ms=age)
            for (symbol, clock), quote in normalized.items():
                paths[(dataset, age, clock)] = NativeMidpoints(quote.times_ns, quote.bid + quote.ask,
                    quote.valid, quote.segments, quote.start_ns, quote.end_ns)
            facts.extend({**row, **common, "age_ms": age} for row in quality)
    return paths, facts
