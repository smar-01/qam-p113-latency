"""Offline billing, provenance, completion and exact-window acquisition tests."""
from __future__ import annotations

from datetime import date
import copy
import json
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import databento as db
import numpy as np
import pandas as pd
import pytest

from colonylab import config
from colonylab import multi_event_data as data


@pytest.fixture
def request_rows():
    return [{"dataset": dataset, "schema": "mbp-1", "symbols": ["AAPL"], "stype_in": "raw_symbol",
             "start": "2026-06-08T16:20:00+00:00", "end": "2026-06-08T17:40:00+00:00", "limit": 100}
            for dataset in ("XNAS.ITCH", "BATS.PITCH")]


class FakeStore:
    def __init__(self, request, rows=3):
        start = pd.Timestamp(request["start"])
        self.metadata = SimpleNamespace(dataset=request["dataset"], schema="mbp-1", start=start.value,
            end=pd.Timestamp(request["end"]).value, limit=request["limit"], stype_in="raw_symbol",
            symbols=["AAPL"], partial=[], not_found=[], mappings={"AAPL": [{
                "start_date": start.date(), "end_date": (start + pd.Timedelta(days=1)).date(), "symbol": "42"}]})
        self.records = np.zeros(rows, dtype=[("ts_recv", "u8"), ("ts_event", "u8"), ("instrument_id", "u4"),
             ("publisher_id", "u2"), ("rtype", "u1"), ("flags", "u1"), ("bid_px_00", "i8"),
             ("ask_px_00", "i8"), ("bid_sz_00", "u4"), ("ask_sz_00", "u4")])
        self.records["ts_recv"] = start.value + np.arange(rows) + 1
        self.records["ts_event"] = self.records["ts_recv"] - 1
        self.records["instrument_id"], self.records["publisher_id"] = 42, 2
        self.records["rtype"], self.records["flags"] = 1, 128
        self.records["bid_px_00"], self.records["ask_px_00"] = 200_000_000_000, 200_010_000_000
        self.records["bid_sz_00"], self.records["ask_sz_00"] = 100, 200
        self.chunk_size = None

    def to_ndarray(self, *, count):
        self.chunk_size = count
        yield self.records[:1]
        yield self.records[1:]


class FakeAPI:
    def __init__(self):
        self.calls, self.downloads, self.stores = [], [], {}
        self.count, self.size, self.cost, self.rows = 10, 1000, .01, 3
        self.condition, self.schema_start = "available", "2018-05-01T00:00Z"
        self.fail_cost = None
        self.fail_stream = False
        self.before_download = lambda request: None
        self.after_download = lambda store: None
        self.metadata = SimpleNamespace(list_schemas=self.schemas, get_dataset_range=self.ranges,
            get_dataset_condition=self.conditions, get_record_count=self.get_count,
            get_billable_size=self.get_size, get_cost=self.get_cost)
        self.timeseries = SimpleNamespace(get_range=self.get_range)

    def schemas(self, **kwargs):
        self.calls.append(("schemas", kwargs))
        return ["mbp-1", "mbo"]

    def ranges(self, **kwargs):
        self.calls.append(("range", kwargs))
        return {"start": "2018-05-01T00:00Z", "end": "2030-01-01T00:00Z",
                "schema": {"mbp-1": {"start": self.schema_start, "end": "2030-01-01T00:00Z"}}}

    def conditions(self, **kwargs):
        self.calls.append(("conditions", kwargs))
        return [{"date": day.date().isoformat(), "condition": self.condition, "last_modified_date": "2026-06-10"}
                for day in pd.date_range(kwargs["start_date"], kwargs["end_date"], freq="D")]

    def get_count(self, **kwargs):
        self.calls.append(("count", kwargs))
        return self.count

    def get_size(self, **kwargs):
        self.calls.append(("size", kwargs))
        return self.size

    def get_cost(self, **kwargs):
        self.calls.append(("cost", kwargs))
        if self.fail_cost:
            raise RuntimeError(self.fail_cost)
        return self.cost

    def get_range(self, **kwargs):
        request = {key: value for key, value in kwargs.items() if key != "path"}
        self.before_download(request)
        self.downloads.append(request)
        path = Path(kwargs["path"])
        path.write_bytes(b"fake complete native MBP-1 stream")
        if self.fail_stream:
            raise RuntimeError("db-pretend-secret interrupted stream")
        store = FakeStore(request, self.rows)
        self.after_download(store)
        self.stores[str(path.parent.resolve())] = store
        return store


@pytest.fixture
def api(monkeypatch):
    # No actual key is read or live API created, including during safe errors.
    monkeypatch.setattr(config, "load_key", lambda: "unit-test-placeholder")
    fake = FakeAPI()
    monkeypatch.setattr(data, "client", lambda: fake)
    monkeypatch.setattr(db.DBNStore, "from_file", lambda path: fake.stores[str(Path(path).parent.resolve())])
    return fake


def plan(request_rows, tmp_path, **kwargs):
    return data.plan_requests(request_rows, tmp_path, "data", "out", workers=2, progress=lambda _: None, **kwargs)


def fetch(plan, tmp_path, budget=1.):
    return data.fetch_requests(plan, tmp_path, "data", "out", budget, workers=2, progress=lambda _: None)


def test_plan_uses_only_free_metadata_and_full_uncapped_windows(api, request_rows, tmp_path):
    p = plan(request_rows, tmp_path)
    assert p["ready_to_download"] and p["incremental_estimated_cost_usd"] == .02
    assert p["expected_request_count"] == 2 and not p["time_series_downloaded"]
    assert not api.downloads
    assert all("limit" not in request for kind, request in api.calls if kind in {"count", "size", "cost"})
    conditions = [request for kind, request in api.calls if kind == "conditions"]
    assert all(request["start_date"] == request["end_date"] == "2026-06-08" for request in conditions)


@pytest.mark.parametrize("change", [{"schema": "mbo"}, {"symbols": ["ALL_SYMBOLS"]}, {"dataset": "EQUS.MINI"},
    {"limit": 10_000_001}, {"limit": True}, {"start": "2026-06-08T16:20:00"}, {"end": "2026-06-10T16:20Z"}])
def test_invalid_requests_stop_before_api(api, request_rows, tmp_path, change):
    with pytest.raises(ValueError):
        plan([dict(request_rows[0], **change)], tmp_path)
    assert not api.calls and not api.downloads


def test_duplicate_and_outside_paths_stop_before_api(api, request_rows, tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        plan([request_rows[0], request_rows[0]], tmp_path)
    with pytest.raises(ValueError, match="inside the workspace"):
        data.plan_requests(request_rows, tmp_path, tmp_path.parent / "outside", "out")
    assert not api.calls


def test_schema_range_cannot_be_replaced_by_overall_dataset_range(api, request_rows, tmp_path):
    api.schema_start = "2026-07-01T00:00Z"
    p = plan(request_rows, tmp_path)
    assert p["ready_to_download"] and p["unavailable_requests"] == 2
    assert all(row["status"] == "unavailable" for row in p["requests"])
    assert not any(kind == "cost" for kind, _ in api.calls)


@pytest.mark.parametrize("condition", ["degraded", "pending", "missing"])
def test_known_bad_dataset_day_is_not_downloaded(api, request_rows, tmp_path, condition):
    api.condition = condition
    p = plan(request_rows, tmp_path)
    assert p["ready_to_download"] and not api.downloads
    m = fetch(p, tmp_path, budget=0)
    assert not m["complete"] and m["acquisition_complete_for_available_requests"]
    assert m["unavailable_requests"] == 2 and not api.downloads
    assert all(row["status"] == "unavailable" and not row["analyzable"] for row in m["requests"])


def test_cost_error_is_explicit_redacted_and_never_inferred_zero(api, request_rows, tmp_path):
    api.fail_cost = "unavailable for db-pretend-secret"
    p = plan(request_rows, tmp_path)
    assert not p["ready_to_download"] and p["incremental_estimated_cost_usd"] is None
    text = (tmp_path / "out/download_plan.json").read_text()
    assert "db-pretend-secret" not in text and "[REDACTED]" in text
    api.fail_cost = None
    p = plan(request_rows, tmp_path)
    assert p["ready_to_download"]  # Failed free quotes are retried on explicit replan.


def test_recent_free_quotes_resume_without_repeating_api_calls(api, request_rows, tmp_path):
    plan(request_rows, tmp_path)
    count = len(api.calls)
    p = plan(request_rows, tmp_path)
    assert len(api.calls) == count and all(row["metadata_quote_reused"] for row in p["requests"])


@pytest.mark.parametrize("count,size", [(100, 1000), (10, 1_000_000_001)])
def test_whole_window_size_estimate_cannot_hide_behind_cap(api, request_rows, tmp_path, count, size):
    api.count, api.size = count, size
    p = plan(request_rows, tmp_path)
    assert not p["ready_to_download"] and all(row["status"] == "blocked_size" for row in p["requests"])


def test_budget_rechecked_for_entire_grid_before_first_stream(api, request_rows, tmp_path):
    p = plan(request_rows, tmp_path)
    api.cost = 1.
    with pytest.raises(ValueError, match="exceeds ceiling"):
        fetch(p, tmp_path, budget=.02)
    assert not api.downloads


def test_complete_stream_retains_exact_native_identity_and_hash(api, request_rows, tmp_path):
    p = plan(request_rows, tmp_path)
    api.before_download = lambda _: sum(kind == "count" for kind, _ in api.calls) == 4 or pytest.fail("Incomplete fresh grid preflight")
    m = fetch(p, tmp_path)
    assert m["complete"] and m["downloaded_rows"] == 6 and m["analyzable_requests"] == 2
    for row in m["requests"]:
        raw = Path(row["data_path"])
        assert row["content_sha256"] == data._sha(raw)
        assert row["stream_complete"] and row["analyzable"] and not row["cost_check_skipped"]
        assert row["instrument_ids"] == [42] and row["receive_timestamp_min_ns"] > pd.Timestamp(row["request"]["start"]).value
        assert json.loads((raw.parent / "stream_complete.json").read_text())["content_sha256"] == row["content_sha256"]
    assert all(store.chunk_size == 100_000 for store in api.stores.values())


def test_reusing_completed_sources_never_calls_api_or_rebills(api, request_rows, tmp_path, monkeypatch):
    p = plan(request_rows, tmp_path)
    first = fetch(p, tmp_path)
    monkeypatch.setattr(data, "client", lambda: pytest.fail("Completed-cache reuse must be offline"))
    second = fetch(p, tmp_path, budget=0)
    assert first["complete"] and second["complete"] and len(api.downloads) == 2
    assert all(row["cache_reused"] and row["incremental_estimated_cost_usd"] == 0 for row in second["requests"])


def test_empty_success_is_complete_but_not_analyzable(api, request_rows, tmp_path):
    api.rows = 0
    m = fetch(plan(request_rows, tmp_path), tmp_path)
    assert m["complete"] and m["downloaded_rows"] == m["analyzable_requests"] == 0
    assert all(row["complete"] and not row["analyzable"] for row in m["requests"])


def test_uncertain_paid_stream_preserves_prefix_and_blocks_retry(api, request_rows, tmp_path):
    api.fail_stream = True
    p = plan(request_rows[:1], tmp_path)
    m = fetch(p, tmp_path)
    assert not m["complete"] and len(api.downloads) == 1
    directory = data.cache_directory(tmp_path / "data", request_rows[0])
    assert list(directory.glob("stream.*.dbn.zst")) and not (directory / "stream_complete.json").exists()
    assert "db-pretend-secret" not in (directory / "failed_attempt.json").read_text()
    with pytest.raises(ValueError, match="explicit repair"):
        fetch(p, tmp_path)
    assert len(api.downloads) == 1


def test_successful_stream_marker_recovers_local_decode_without_api(api, request_rows, tmp_path, monkeypatch):
    original = data._parse_stream
    monkeypatch.setattr(data, "_parse_stream", lambda *_: (_ for _ in ()).throw(ValueError("local decode interrupted")))
    p = plan(request_rows[:1], tmp_path)
    first = fetch(p, tmp_path)
    assert not first["complete"] and len(api.downloads) == 1
    monkeypatch.setattr(data, "_parse_stream", original)
    monkeypatch.setattr(data, "client", lambda: pytest.fail("Decode recovery must be offline"))
    second = fetch(p, tmp_path, budget=0)
    assert second["complete"] and len(api.downloads) == 1


def test_raw_hash_tamper_blocks_reuse_before_any_paid_call(api, request_rows, tmp_path):
    p = plan(request_rows[:1], tmp_path)
    m = fetch(p, tmp_path)
    Path(m["requests"][0]["data_path"]).write_bytes(b"changed raw stream")
    with pytest.raises(ValueError, match="bytes differ"):
        fetch(p, tmp_path)
    assert len(api.downloads) == 1


@pytest.mark.parametrize("corruption", ["instrument", "rtype", "outside", "cap", "metadata"])
def test_decoded_source_corruption_never_becomes_complete(api, request_rows, tmp_path, corruption):
    def corrupt(store):
        if corruption == "instrument": store.records["instrument_id"][-1] = 99
        elif corruption == "rtype": store.records["rtype"][-1] = 160
        elif corruption == "outside": store.records["ts_recv"][-1] = store.metadata.end
        elif corruption == "metadata": store.metadata.dataset = "MEMX.MEMOIR"
    api.after_download = corrupt
    if corruption == "cap": api.rows = 100
    m = fetch(plan(request_rows[:1], tmp_path), tmp_path)
    assert not m["complete"] and len(api.downloads) == 1
    assert not (data.cache_directory(tmp_path / "data", request_rows[0]) / "fetch.json").exists()


def test_each_stream_cost_increase_cannot_exceed_aggregate_allocation(api, request_rows, tmp_path):
    p = plan(request_rows[:1], tmp_path)
    original = api.get_cost
    def change_after_grid(**kwargs):
        value = original(**kwargs)
        if sum(kind == "cost" for kind, _ in api.calls) >= 3:
            return 1.
        return value
    api.metadata.get_cost = change_after_grid
    m = fetch(p, tmp_path)
    assert not m["complete"] and not api.downloads


def test_midnight_exclusive_end_does_not_request_next_days_condition(api, request_rows, tmp_path):
    request = dict(request_rows[0], start="2026-06-08T23:00Z", end="2026-06-09T00:00Z")
    plan([request], tmp_path)
    calls = [value for kind, value in api.calls if kind == "conditions"]
    assert calls[0]["start_date"] == calls[0]["end_date"] == "2026-06-08"


def test_unavailable_rows_are_retained_while_available_grid_completes(api, request_rows, tmp_path):
    original = api.conditions
    def conditions(**kwargs):
        rows = original(**kwargs)
        if kwargs["dataset"] == "XNAS.ITCH":
            rows[0]["condition"] = "degraded"
        return rows
    api.metadata.get_dataset_condition = conditions
    p = plan(request_rows, tmp_path)
    assert p["ready_to_download"] and p["incremental_estimated_cost_usd"] == .01
    m = fetch(p, tmp_path)
    assert not m["complete"] and m["acquisition_complete_for_available_requests"]
    assert len(m["requests"]) == 2 and len(api.downloads) == 1 and api.downloads[0]["dataset"] == "BATS.PITCH"
    assert m["requests"][0]["status"] == "unavailable"
    assert m["requests"][0]["dataset_conditions"][0]["condition"] == "degraded"


def test_stale_plan_cannot_start_new_retrieval(api, request_rows, tmp_path):
    p = plan(request_rows, tmp_path)
    p["created_utc"] = "2020-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="stale"):
        fetch(p, tmp_path)
    assert not api.downloads


def test_free_fetch_preflight_errors_are_redacted(api, request_rows, tmp_path):
    p = plan(request_rows, tmp_path)
    api.fail_cost = "failure with db-another-secret"
    with pytest.raises(ValueError, match="REDACTED") as found:
        fetch(p, tmp_path)
    assert "db-another-secret" not in str(found.value) and not api.downloads


def test_fetch_preflight_quotes_are_parallel_and_finish_before_streams(api, request_rows, tmp_path, monkeypatch):
    p = plan(request_rows, tmp_path)
    rendezvous = Barrier(2)
    original = data._estimate
    def paired(request):
        rendezvous.wait(timeout=5)
        return original(request)
    monkeypatch.setattr(data, "_estimate", paired)
    api.before_download = lambda _: sum(kind == "count" for kind, _ in api.calls) == 4 or pytest.fail("A quote was still pending")
    m = fetch(p, tmp_path)
    assert m["complete"] and len(api.downloads) == 2


def test_canonical_cache_folder_keeps_full_hash_without_windows_path_overhead(request_rows, tmp_path):
    request = request_rows[0]
    path = data.cache_directory(tmp_path / "data", request)
    assert path.name == data._signature(request) and len(path.name) == 64
    assert path.parent == tmp_path / "data"


def test_completed_legacy_pilot_folder_is_reused_offline_without_rebilling(api, request_rows, tmp_path, monkeypatch):
    request = request_rows[0]
    base = tmp_path / "data"
    token = data._signature(request)
    legacy = base / (request["start"][:10] + "_" + request["dataset"] + "_" + token)
    legacy.mkdir(parents=True)
    assert data.cache_directory(base, request) == legacy
    p = plan(request_rows[:1], tmp_path)
    first = fetch(p, tmp_path)
    assert first["complete"] and Path(first["requests"][0]["data_path"]).parent == legacy
    monkeypatch.setattr(data, "client", lambda: pytest.fail("Legacy pilot cache reuse must be offline"))
    second = fetch(p, tmp_path, budget=0)
    assert second["complete"] and second["requests"][0]["cache_reused"] and len(api.downloads) == 1
    assert not (base / token).exists()
