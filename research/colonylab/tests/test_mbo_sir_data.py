"""Offline acquisition checks: exact grid, included access, raw completeness."""
from __future__ import annotations

import copy
from datetime import date
import json
from pathlib import Path
from types import SimpleNamespace
from threading import Barrier
import weakref

import databento as db
import numpy as np
import pandas as pd
import pytest

from colonylab import config as key_config
from colonylab import mbo_sir_data as data


@pytest.fixture(autouse=True)
def no_live_api(monkeypatch):
    monkeypatch.setattr(data, "client", lambda: pytest.fail("Live API access is forbidden in offline acquisition tests"))
    monkeypatch.setattr(data.shutil, "disk_usage", lambda _: SimpleNamespace(free=1_000_000_000_000))
    monkeypatch.setattr(key_config, "load_key", lambda: "unit-test-placeholder-key")


@pytest.fixture
def study(tmp_path):
    return {
        "name": "offline-mbo-sirs", "datasets": ["XNAS.ITCH", "BATS.PITCH"],
        "symbol": "AAPL", "market_dataset": "XNAS.ITCH", "market_symbol": "SPY",
        "training_days": ["2026-06-01"], "held_out_days": ["2026-06-02"],
        "download_start_utc": "00:00:00", "session_start_utc": "13:30:00",
        "session_end_utc": "20:00:00", "validation_start_utc": "14:00:00",
        "validation_minutes": 5, "anchor_step_ms": 100,
        "data_dir": str(tmp_path / "data"), "out_dir": str(tmp_path / "out"),
        "workers": 2, "record_limit_mbo": 100, "record_limit_aux": 100,
        "max_bytes_mbo": 100_000, "billing_assumed_included": True,
    }


class FakeStore:
    def __init__(self, request, *, rows=3, extra_id=False, mapped=True):
        self.schema = request["schema"]
        day = date.fromisoformat(request["start"][:10])
        next_day = (pd.Timestamp(day) + pd.Timedelta(days=1)).date()
        instrument = 42 if request["symbols"] == ["AAPL"] else 99
        self.metadata = SimpleNamespace(mappings={request["symbols"][0]: [{
            "start_date": day, "end_date": next_day, "symbol": str(instrument),
        }]} if mapped else {})
        dtype = [("rtype", "u1"), ("instrument_id", "u4"), ("ts_event", "u8"), ("ts_recv", "u8"), ("flags", "u1")]
        self.records = np.zeros(rows, dtype=dtype)
        self.records["rtype"] = {"mbo": 160, "mbp-10": 10, "bbo-1s": 196}[request["schema"]]
        self.records["instrument_id"] = instrument
        stamps = pd.Timestamp(request["start"]).value + (np.arange(rows) + 1) * 1_000_000
        self.records["ts_event"] = stamps
        self.records["ts_recv"] = stamps
        self.records["flags"] = 128
        if extra_id and rows:
            self.records["instrument_id"][-1] = 12345
        self.chunk_size = None

    def to_ndarray(self, *, count):
        self.chunk_size = count
        split = max(1, len(self.records) // 2)
        yield self.records[:split]
        yield self.records[split:]

    def to_df(self, **_):
        pytest.fail("Full MBO acquisition must never materialize a DataFrame")


class FakeAPI:
    def __init__(self, *, rows=3, count=10, size=1000):
        self.rows, self.count, self.size = rows, count, size
        self.calls, self.downloads, self.stores = [], [], {}
        self.fail_downloads = 0
        self.always_fail = False
        self.extra_id = False
        self.mapped = True
        self.fail_metadata = False
        self.before_download = lambda request: None
        self.metadata = SimpleNamespace(get_record_count=self.get_count,
                                        get_billable_size=self.get_size,
                                        get_cost=lambda **_: pytest.fail("No cost metadata is authorized"))
        self.timeseries = SimpleNamespace(get_range=self.get_range)

    def get_count(self, **request):
        self.calls.append(("count", request))
        if self.fail_metadata:
            raise RuntimeError("504 metadata timeout")
        return self.count

    def get_size(self, **request):
        self.calls.append(("size", request))
        return self.size

    def get_range(self, **kwargs):
        path = Path(kwargs["path"])
        request = {key: value for key, value in kwargs.items() if key != "path"}
        self.before_download(request)
        self.downloads.append(request)
        path.write_bytes(b"preserved fake DBN bytes")
        if self.always_fail or self.fail_downloads:
            self.fail_downloads = max(0, self.fail_downloads - 1)
            raise RuntimeError("Response ended prematurely")
        store = FakeStore(request, rows=self.rows, extra_id=self.extra_id, mapped=self.mapped)
        self.stores[str(path.parent.resolve())] = store
        return store


def fake_api(monkeypatch, **kwargs):
    api = FakeAPI(**kwargs)
    monkeypatch.setattr(db.DBNStore, "from_file", lambda path: api.stores[str(Path(path).parent.resolve())])
    return api


def fetch(study, api):
    return data.fetch_study(study, progress=lambda _: None, api_factory=lambda: api)


def test_frozen_production_configuration_has_full_exact_510_grid():
    path = Path(__file__).resolve().parents[1] / "mbo_sir.toml"
    frozen = data.read_config(path)
    requests = data.build_requests(frozen)
    assert len(frozen["datasets"]) == 8
    assert len(frozen["training_days"]) == 10
    assert len(frozen["held_out_days"]) == 20
    assert len(requests) == 510
    assert sum(request["schema"] == "mbo" for request in requests) == 240
    assert sum(request["schema"] == "mbp-10" for request in requests) == 240
    assert sum(request["schema"] == "bbo-1s" for request in requests) == 30
    assert all(request["symbols"] == ["AAPL"] for request in requests if request["schema"] != "bbo-1s")
    assert all(request["symbols"] == ["SPY"] for request in requests if request["schema"] == "bbo-1s")
    assert all(request["start"].endswith("T00:00:00+00:00") and request["end"].endswith("T20:00:00+00:00")
               for request in requests if request["schema"] == "mbo")
    assert frozen["workers"] == 8


@pytest.mark.parametrize("change", [
    {"billing_assumed_included": False}, {"extra": True}, {"datasets": ["XNAS.ITCH", "XNAS.ITCH"]},
    {"datasets": ["EQUS.MINI", "BATS.PITCH"]}, {"symbol": "ALL_SYMBOLS"},
    {"market_symbol": "AAPL"}, {"workers": 9}, {"record_limit_mbo": 50_000_001},
    {"record_limit_aux": 5_000_001}, {"max_bytes_mbo": 3_000_000_001},
    {"training_days": ["2026-06-06"]}, {"held_out_days": ["2026-06-01"]},
    {"session_start_utc": "21:00:00"}, {"validation_start_utc": "12:00:00"},
    {"download_start_utc": "25:00:00"}, {"anchor_step_ms": 0},
])
def test_invalid_configuration_stops_before_api(study, change):
    with pytest.raises(ValueError):
        data.validate_config(dict(study, **change))


def test_metadata_only_sizes_mbo_and_never_calls_cost_or_capped_estimates(study, monkeypatch):
    api = fake_api(monkeypatch)
    plan = data.plan_study(study, progress=lambda _: None, api_factory=lambda: api)
    assert plan["ready_to_download"]
    assert len(api.calls) == 8  # Four MBO streams, count and size each.
    assert all(request["schema"] == "mbo" and "limit" not in request for _, request in api.calls)
    assert not api.downloads
    assert len(plan["requests"]) == 10
    assert plan["estimated_total_mbo_billable_bytes"] == 4000
    assert plan["download_workers"] == 2
    assert plan["metadata_workers"] == 4


def test_entire_grid_estimated_before_first_download(study, monkeypatch):
    api = fake_api(monkeypatch)
    api.before_download = lambda _: len(api.calls) == 8 or pytest.fail("A stream started before complete preflight")
    result = fetch(study, api)
    assert result["complete"]
    assert len(api.downloads) == 10
    assert len(result["requests"]) == 10
    assert result["downloaded_rows"] == 30
    assert result["expected_request_count"] == 10
    assert result["configuration_sha256"] == data._signature(study)
    assert all(entry["rows"] == 3 and entry["complete"] for entry in result["requests"])
    assert all(entry["instrument_ids"] in ([42], [99]) for entry in result["requests"])
    assert all(store.chunk_size == 1_000_000 for store in api.stores.values())
    assert all(Path(entry["data_path"]).is_file() for entry in result["requests"])
    persisted = json.loads((Path(study["out_dir"]) / "data_manifest.json").read_text())
    assert persisted["complete"]


def test_exact_cached_run_issues_no_metadata_or_new_streams(study, monkeypatch):
    api = fake_api(monkeypatch)
    first = fetch(study, api)
    api.calls.clear()
    api.downloads.clear()
    second = fetch(study, api)
    assert second["complete"] and first["request_grid_sha256"] == second["request_grid_sha256"]
    assert not api.calls and not api.downloads
    assert all(entry["cache_reused"] for entry in second["requests"])


def test_free_metadata_checkpoint_resumes_without_repeating_quotes(study, monkeypatch):
    api = fake_api(monkeypatch)
    first = data.plan_study(study, progress=lambda _: None, api_factory=lambda: api)
    api.calls.clear()
    second = data.plan_study(study, progress=lambda _: None, api_factory=lambda: api)
    assert first["requests"] == second["requests"]
    assert not api.calls


@pytest.mark.parametrize("count,size", [(100, 1000), (101, 1000), (10, 100_001)])
def test_oversize_stops_all_downloads_without_shrinking_dates(study, monkeypatch, count, size):
    api = fake_api(monkeypatch, count=count, size=size)
    with pytest.raises(ValueError, match="safety bound"):
        fetch(study, api)
    assert not api.downloads
    plan = json.loads((Path(study["out_dir"]) / "download_plan.json").read_text())
    assert len(plan["requests"]) == 10
    assert plan["configuration"]["training_days"] == study["training_days"]


def test_disk_guard_uses_remaining_raw_plus_explicit_feature_reserve(study, monkeypatch):
    api = fake_api(monkeypatch)
    monkeypatch.setattr(data.shutil, "disk_usage", lambda _: SimpleNamespace(free=20_000_003_999))
    with pytest.raises(ValueError, match="free disk"):
        fetch(study, api)
    assert not api.downloads


def test_transient_metadata_failure_preserves_partial_plan_and_downloads_nothing(study, monkeypatch):
    api = fake_api(monkeypatch)
    api.fail_metadata = True
    with pytest.raises(ValueError, match="Free sizing failed"):
        fetch(study, api)
    assert not api.downloads
    state = json.loads((Path(study["out_dir"]) / "metadata_progress.json").read_text())
    assert len(state["failures"]) == 4
    assert len(state["requests"]) == 6  # All auxiliary requests need no metadata.
    api.fail_metadata = False
    result = fetch(study, api)
    assert result["complete"]


def test_failed_stream_retries_preserve_every_partial_prefix(study, monkeypatch):
    api = fake_api(monkeypatch)
    api.fail_downloads = 1
    result = fetch(study, api)
    assert result["complete"]
    assert len(api.downloads) == 11
    prefixes = list(Path(study["data_dir"]).rglob("interrupted.*.dbn.zst"))
    assert len(prefixes) == 1
    assert prefixes[0].read_bytes() == b"preserved fake DBN bytes"


def test_three_failed_attempts_leave_incomplete_manifest_and_preserved_raw(study, monkeypatch):
    api = fake_api(monkeypatch)
    api.always_fail = True
    with pytest.raises(ValueError, match="grid is incomplete"):
        fetch(study, api)
    assert len(api.downloads) == 30
    assert len(list(Path(study["data_dir"]).rglob("interrupted.*.dbn.zst"))) == 30
    manifest = json.loads((Path(study["out_dir"]) / "data_manifest.json").read_text())
    assert not manifest["complete"]
    assert len(manifest["requests"]) == 10
    assert all(not entry["complete"] for entry in manifest["requests"])


@pytest.mark.parametrize("rows", [0, 100, 101])
def test_empty_or_capped_actual_stream_is_never_complete_and_is_not_retried(study, monkeypatch, rows):
    api = fake_api(monkeypatch, rows=rows)
    with pytest.raises(ValueError, match="grid is incomplete"):
        fetch(study, api)
    assert len(api.downloads) == 10
    assert len(list(Path(study["data_dir"]).rglob("sample.dbn.zst"))) == 10
    assert len(list(Path(study["data_dir"]).rglob("stream_complete.json"))) == 10


@pytest.mark.parametrize("extra_id,mapped", [(True, True), (False, False)])
def test_wrong_or_unresolved_instrument_rejects_actual_stream(study, monkeypatch, extra_id, mapped):
    api = fake_api(monkeypatch)
    api.extra_id, api.mapped = extra_id, mapped
    with pytest.raises(ValueError, match="grid is incomplete"):
        fetch(study, api)
    assert len(api.downloads) == 10
    manifest = json.loads((Path(study["out_dir"]) / "data_manifest.json").read_text())
    assert not manifest["complete"]


def test_stream_marker_is_durable_before_export_and_export_recovers_without_download(study, monkeypatch):
    api = fake_api(monkeypatch)
    original_export = data._export_stream
    def crash_export(store, entry, directory, **kwargs):
        marker = json.loads((directory / "stream_complete.json").read_text())
        assert marker["request"] == entry["request"]
        assert (directory / "sample.dbn.zst").exists()
        raise RuntimeError("simulate receipt-export interruption")
    monkeypatch.setattr(data, "_export_stream", crash_export)
    with pytest.raises(ValueError):
        fetch(study, api)
    assert len(api.downloads) == 10
    monkeypatch.setattr(data, "_export_stream", original_export)
    api.downloads.clear()
    result = fetch(study, api)
    assert result["complete"]
    assert not api.downloads
    assert all(entry["recovered_from_local_dbn"] for entry in result["requests"])


def test_unconfirmed_old_canonical_prefix_is_preserved_and_refetched(study, monkeypatch):
    api = fake_api(monkeypatch)
    request = data.build_requests(study)[0]
    directory = data.cache_directory(Path(study["data_dir"]), request)
    directory.mkdir(parents=True)
    (directory / "request.json").write_text(json.dumps(request))
    (directory / "sample.dbn.zst").write_bytes(b"unconfirmed original partial")
    result = fetch(study, api)
    assert result["complete"]
    prefixes = list(directory.glob("interrupted.*.dbn.zst"))
    assert len(prefixes) == 1 and prefixes[0].read_bytes() == b"unconfirmed original partial"


def test_success_marker_recovers_file_when_crash_preceded_canonical_rename(study, monkeypatch):
    api = fake_api(monkeypatch)
    request = data.build_requests(study)[0]
    directory = data.cache_directory(Path(study["data_dir"]), request)
    directory.mkdir(parents=True)
    pending = directory / "stream.success.dbn.zst"
    pending.write_bytes(b"finished raw bytes")
    (directory / "request.json").write_text(json.dumps(request))
    (directory / "stream_complete.json").write_text(json.dumps({"request": request, "download_file": pending.name}))
    api.stores[str(directory.resolve())] = FakeStore(request)
    result = fetch(study, api)
    assert result["complete"]
    assert len(api.downloads) == 9
    assert not pending.exists()
    assert (directory / "sample.dbn.zst").read_bytes() == b"finished raw bytes"


def test_changed_limit_probe_hash_is_excluded_from_exact_study_grid(study, monkeypatch):
    api = fake_api(monkeypatch)
    request = data.build_requests(study)[0]
    probe = dict(request, limit=2)
    directory = data.cache_directory(Path(study["data_dir"]), probe)
    directory.mkdir(parents=True)
    (directory / "sample.dbn.zst").write_bytes(b"100-record probe is not study data")
    result = fetch(study, api)
    assert result["complete"]
    assert len(api.downloads) == 10
    assert all(entry["request"] != probe for entry in result["requests"])
    assert (directory / "sample.dbn.zst").read_bytes() == b"100-record probe is not study data"


def test_tampered_completed_cache_is_rejected_without_new_retrieval(study, monkeypatch):
    api = fake_api(monkeypatch)
    result = fetch(study, api)
    Path(result["requests"][0]["data_path"]).write_bytes(b"changed bytes")
    api.downloads.clear()
    with pytest.raises(ValueError, match="receipt size"):
        fetch(study, api)
    assert not api.downloads


def test_out_of_range_clocks_are_audited_and_flagged_snapshot_clocks_retained(study, tmp_path):
    request = data.build_requests(study)[0]
    directory = tmp_path / "raw"
    directory.mkdir()
    (directory / "sample.dbn.zst").write_bytes(b"raw")
    store = FakeStore(request)
    store.records["ts_recv"][0] = pd.Timestamp(request["start"]).value - 1
    store.records["flags"][0] = 32
    store.records["ts_recv"][1] = pd.Timestamp(request["end"]).value + 1
    entry = data._export_stream(store, {"request": request}, directory)
    assert entry["rows"] == 3
    assert entry["unflagged_records_outside_requested_clock_range"] == 1
    assert entry["receive_timestamp_min_ns"] == int(store.records["ts_recv"][0])


@pytest.mark.parametrize("bad_field", ["dataset", "schema"])
def test_dbn_metadata_must_match_requested_dataset_and_schema(study, tmp_path, bad_field):
    request = data.build_requests(study)[0]
    directory = tmp_path / "wrong_header"
    directory.mkdir()
    (directory / "sample.dbn.zst").write_bytes(b"raw")
    store = FakeStore(request)
    if bad_field == "dataset":
        store.metadata.dataset = "XNYS.PILLAR"
    else:
        store.schema = "mbp-1"
    with pytest.raises(ValueError, match=bad_field):
        data._export_stream(store, {"request": request}, directory)


def test_foreign_stream_marker_cannot_be_recovered(study, tmp_path):
    request = data.build_requests(study)[0]
    directory = tmp_path / "foreign_marker"
    directory.mkdir()
    raw = directory / "sample.dbn.zst"
    raw.write_bytes(b"foreign raw")
    (directory / "stream_complete.json").write_text(json.dumps({"request": dict(request, limit=1)}))
    with pytest.raises(ValueError, match="another request"):
        data._recover_completed(directory, request)
    assert raw.read_bytes() == b"foreign raw"


def test_metadata_checkpoint_rejects_foreign_request(study, monkeypatch):
    api = fake_api(monkeypatch)
    data.plan_study(study, progress=lambda _: None, api_factory=lambda: api)
    path = Path(study["out_dir"]) / "metadata_progress.json"
    state = json.loads(path.read_text())
    state["requests"][0]["request"]["limit"] = 1
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="foreign request"):
        data.plan_study(study, progress=lambda _: None, api_factory=lambda: api)


def test_sdk_truncation_flag_cannot_be_silently_accepted(study, tmp_path):
    request = data.build_requests(study)[0]
    directory = tmp_path / "truncated"
    directory.mkdir()
    raw = directory / "sample.dbn.zst"
    raw.write_bytes(b"truncated bytes preserved")
    store = FakeStore(request)
    class TruncatedChunks:
        _close_on_next = True
        def __iter__(self):
            return iter([store.records])
    store.to_ndarray = lambda **_: TruncatedChunks()
    with pytest.raises(ValueError, match="trailing incomplete"):
        data._export_stream(store, {"request": request}, directory)
    assert raw.read_bytes() == b"truncated bytes preserved"
    assert not (directory / "fetch.json").exists()


def test_independent_audit_reparses_hashes_full_grid_and_preserves_receipts(study, monkeypatch):
    api = fake_api(monkeypatch)
    manifest = fetch(study, api)
    before = {entry["data_path"]: (Path(entry["data_path"]).parent / "fetch.json").read_bytes()
              for entry in manifest["requests"]}
    api.calls.clear()
    api.downloads.clear()
    report = data.audit_study(study, progress=lambda _: None, workers=2)
    assert report["complete"] and len(report["requests"]) == 10
    assert report["total_rows"] == 30
    assert len(report["validator_code_sha256"]) == 64
    assert all(len(entry["sha256"]) == 64 for entry in report["requests"])
    assert not api.calls and not api.downloads
    assert all((Path(path).parent / "fetch.json").read_bytes() == original for path, original in before.items())


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "configuration", "signature"])
def test_independent_audit_requires_exact_complete_manifest_grid(study, monkeypatch, corruption):
    api = fake_api(monkeypatch)
    manifest = fetch(study, api)
    if corruption == "missing":
        manifest["requests"].pop()
    elif corruption == "duplicate":
        manifest["requests"].append(manifest["requests"][0])
    elif corruption == "configuration":
        manifest["configuration"]["name"] = "different study"
    else:
        manifest["configuration_sha256"] = "wrong"
    path = Path(study["out_dir"]) / "data_manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exact configuration and request grid"):
        data.audit_study(study, progress=lambda _: None)


def test_independent_audit_catches_actual_parser_row_mismatch(study, monkeypatch):
    api = fake_api(monkeypatch)
    manifest = fetch(study, api)
    directory = str(Path(manifest["requests"][0]["data_path"]).parent.resolve())
    store = api.stores[directory]
    store.records = np.r_[store.records, store.records[-1:]]
    with pytest.raises(ValueError, match="integrity audit failed"):
        data.audit_study(study, progress=lambda _: None, workers=2)
    report = json.loads((Path(study["out_dir"]) / "raw_audit.json").read_text())
    assert not report["complete"] and len(report["failures"]) == 1
    assert "record count differs" in report["failures"][0]["error"]


def test_independent_audit_catches_truncation_even_when_old_receipt_was_complete(study, monkeypatch):
    api = fake_api(monkeypatch)
    manifest = fetch(study, api)
    directory = str(Path(manifest["requests"][0]["data_path"]).parent.resolve())
    store = api.stores[directory]
    class TruncatedChunks:
        _close_on_next = True
        def __iter__(self):
            return iter([store.records])
    store.to_ndarray = lambda **_: TruncatedChunks()
    with pytest.raises(ValueError, match="integrity audit failed"):
        data.audit_study(study, progress=lambda _: None, workers=2)
    report = json.loads((Path(study["out_dir"]) / "raw_audit.json").read_text())
    assert "trailing incomplete" in report["failures"][0]["error"]


def test_atomic_json_retries_transient_onedrive_rename_with_bounded_backoff(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    target.write_text('{"old": true}')
    replace = Path.replace
    attempts, waits = [], []
    def temporary_lock(path, destination):
        attempts.append(str(path))
        if len(attempts) <= 6:
            raise PermissionError("WinError 5: OneDrive temporary sharing violation")
        return replace(path, destination)
    monkeypatch.setattr(Path, "replace", temporary_lock)
    monkeypatch.setattr(data.time, "sleep", waits.append)
    data._atomic_json(target, {"new": True})
    assert len(attempts) == 7
    assert waits == [.1, .2, .4, .8, 1., 1.]
    assert json.loads(target.read_text()) == {"new": True}
    assert len(set(attempts)) == 1  # Retry the same durable temp, not its contents.


def test_atomic_json_preserves_old_checkpoint_and_new_temp_after_twelve_failures(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    original = '{"old": true}'
    target.write_text(original)
    attempts, waits = [], []
    def permanent_lock(path, destination):
        attempts.append(str(path))
        raise PermissionError("WinError 5: permanent sharing violation")
    monkeypatch.setattr(Path, "replace", permanent_lock)
    monkeypatch.setattr(data.time, "sleep", waits.append)
    with pytest.raises(PermissionError):
        data._atomic_json(target, {"new": True})
    assert len(attempts) == 12
    assert len(waits) == 11 and max(waits) == 1.
    assert target.read_text() == original
    pending = list(tmp_path.glob("state.json.*.tmp"))
    assert len(pending) == 1 and json.loads(pending[0].read_text()) == {"new": True}


@pytest.mark.parametrize("workers", [0, 33, True, 1.5])
def test_invalid_runtime_worker_override_stops_before_any_api_or_plan(study, workers):
    def forbidden():
        pytest.fail("Invalid worker override reached the API factory")
    with pytest.raises(ValueError, match="download_workers"):
        data.fetch_study(study, progress=lambda _: None, api_factory=forbidden, download_workers=workers)
    assert not (Path(study["out_dir"]) / "download_plan.json").exists()


def test_runtime_transport_override_changes_only_actual_download_concurrency(study, monkeypatch):
    api = fake_api(monkeypatch)
    executor = data.ThreadPoolExecutor
    pool_sizes = []
    def recorded_pool(*, max_workers):
        pool_sizes.append(max_workers)
        return executor(max_workers=max_workers)
    monkeypatch.setattr(data, "ThreadPoolExecutor", recorded_pool)
    original = copy.deepcopy(study)
    result = data.fetch_study(study, progress=lambda _: None, api_factory=lambda: api, download_workers=24)
    assert result["complete"]
    assert pool_sizes == [4, 24]
    assert study == original
    assert result["configuration"] == original
    assert result["configuration_sha256"] == data._signature(original)
    assert result["request_grid_sha256"] == data._signature({"requests": data.build_requests(original)})
    assert result["configured_download_workers"] == original["workers"] == 2
    assert result["download_workers"] == result["download_workers_override"] == 24
    assert result["metadata_workers"] == 4
    assert len(api.downloads) == 10
    assert len(api.calls) == 8 and all(request["schema"] == "mbo" for _, request in api.calls)
    plan = json.loads((Path(study["out_dir"]) / "download_plan.json").read_text())
    assert plan["download_workers"] == 24 and plan["configuration"]["workers"] == 2


def test_changing_runtime_workers_reuses_all_exact_completed_requests(study, monkeypatch):
    api = fake_api(monkeypatch)
    first = fetch(study, api)
    api.calls.clear()
    api.downloads.clear()
    second = data.fetch_study(study, progress=lambda _: None, api_factory=lambda: api, download_workers=24)
    assert second["complete"] and second["download_workers"] == 24
    assert second["configuration_sha256"] == first["configuration_sha256"]
    assert second["request_grid_sha256"] == first["request_grid_sha256"]
    assert not api.calls and not api.downloads
    assert all(entry["cache_reused"] for entry in second["requests"])


def test_raw_rename_retries_same_prefix_and_destination_with_bounded_backoff(tmp_path, monkeypatch):
    source, destination = tmp_path / "stream.dbn.zst", tmp_path / "sample.dbn.zst"
    source.write_bytes(b"complete raw bytes")
    replace = Path.replace
    attempts, waits = [], []
    def temporary_lock(path, target):
        attempts.append((path, target))
        if len(attempts) <= 6:
            raise PermissionError("WinError 32: temporary sharing violation")
        return replace(path, target)
    monkeypatch.setattr(Path, "replace", temporary_lock)
    monkeypatch.setattr(data.time, "sleep", waits.append)
    data._replace_with_retry(source, destination)
    assert attempts == [(source, destination)] * 7
    assert waits == [.1, .2, .4, .8, 1., 1.]
    assert not source.exists() and destination.read_bytes() == b"complete raw bytes"


def test_raw_rename_permanent_lock_preserves_both_existing_files(tmp_path, monkeypatch):
    source, destination = tmp_path / "stream.dbn.zst", tmp_path / "sample.dbn.zst"
    source.write_bytes(b"new complete bytes")
    destination.write_bytes(b"previous durable bytes")
    attempts, waits = [], []
    def permanent_lock(path, target):
        attempts.append((path, target))
        raise PermissionError("WinError 32: permanent sharing violation")
    monkeypatch.setattr(Path, "replace", permanent_lock)
    monkeypatch.setattr(data.time, "sleep", waits.append)
    with pytest.raises(PermissionError, match="permanent sharing"):
        data._replace_with_retry(source, destination)
    assert attempts == [(source, destination)] * 12
    assert len(waits) == 11 and max(waits) == 1.
    assert source.read_bytes() == b"new complete bytes"
    assert destination.read_bytes() == b"previous durable bytes"


def test_real_sdk_failed_writer_is_released_before_prefix_rename(study, monkeypatch):
    from databento.common import http

    api = fake_api(monkeypatch)
    transport = http.BentoHttpAPI("unit-test-placeholder-key", "https://offline.invalid")
    references = []
    real_open = open
    def observed_open(*args, **kwargs):
        writer = real_open(*args, **kwargs)
        references.append(weakref.ref(writer))
        return writer
    class OfflineResponse:
        headers = {}
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def iter_content(self, **_):
            yield b"prefix from real SDK writer"
            raise RuntimeError("original offline transfer failure")
    monkeypatch.setattr(http, "open", observed_open, raising=False)
    monkeypatch.setattr(http.requests, "post", lambda **_: OfflineResponse())
    monkeypatch.setattr(http, "check_http_error", lambda _: None)
    def failed_download(**kwargs):
        api.downloads.append(kwargs)
        return transport._stream(url="https://offline.invalid", data={}, basic_auth=False, path=kwargs["path"])
    api.timeseries.get_range = failed_download
    replace = Path.replace
    renames = []
    def checked_replace(path, target):
        if path.name.startswith("stream.") and target.name.startswith("interrupted."):
            assert references and all(ref() is None or ref().closed for ref in references)
            renames.append((path, target))
        return replace(path, target)
    monkeypatch.setattr(Path, "replace", checked_replace)
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    request = data.build_requests(study)[0]
    directory = data.cache_directory(Path(study["data_dir"]), request)
    directory.mkdir(parents=True)
    with pytest.raises(http.BentoError, match="original offline transfer failure"):
        data._download_one({"request": request}, Path(study["data_dir"]), lambda: api, lambda _: None)
    assert len(renames) == len(references) == len(api.downloads) == 3
    assert all(path.read_bytes() == b"prefix from real SDK writer" for path in directory.glob("interrupted.*"))
    history = json.loads((directory / "download_attempts.json").read_text())["attempts"]
    assert len(history) == 3
    assert all("original offline transfer failure" in item["error"] and item["preservation_error"] is None
               for item in history)


def test_locked_failed_prefix_records_original_error_and_preserves_each_source(study, monkeypatch):
    api = fake_api(monkeypatch)
    api.always_fail = True
    replace = Path.replace
    renames = []
    def externally_locked(path, target):
        if path.name.startswith("stream.") and target.name.startswith("interrupted."):
            renames.append(path)
            raise PermissionError("WinError 32: externally held failed prefix")
        return replace(path, target)
    monkeypatch.setattr(Path, "replace", externally_locked)
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    with pytest.raises(ValueError, match="grid is incomplete"):
        fetch(study, api)
    manifest = json.loads((Path(study["out_dir"]) / "data_manifest.json").read_text())
    assert len(api.downloads) == len(manifest["requests"]) * 3 == 30
    assert len(renames) == 30 * 12
    assert len(set(renames)) == 30
    assert not list(Path(study["data_dir"]).rglob("interrupted.*"))
    for entry in manifest["requests"]:
        assert entry["error"] == "RuntimeError: Response ended prematurely"
        directory = data.cache_directory(Path(study["data_dir"]), entry["request"])
        history = json.loads((directory / "download_attempts.json").read_text())["attempts"]
        assert len(history) == 3
        for attempt in history:
            assert attempt["error"] == entry["error"]
            assert attempt["preserved_in_place"] is True
            assert "externally held failed prefix" in attempt["preservation_error"]
            assert attempt["preserved_prefix"] == attempt["source_filename"]
            assert (directory / attempt["source_filename"]).read_bytes() == b"preserved fake DBN bytes"


def test_transient_failed_prefix_lock_retries_then_downloads_to_new_filename(study, monkeypatch):
    api = fake_api(monkeypatch)
    api.fail_downloads = 1
    replace = Path.replace
    attempts = []
    def temporary_lock(path, target):
        if path.name.startswith("stream.") and target.name.startswith("interrupted."):
            attempts.append((path, target))
            if len(attempts) < 7:
                raise PermissionError("WinError 32: temporary prefix lock")
        return replace(path, target)
    monkeypatch.setattr(Path, "replace", temporary_lock)
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    request = data.build_requests(study)[0]
    directory = data.cache_directory(Path(study["data_dir"]), request)
    directory.mkdir(parents=True)
    result = data._download_one({"request": request}, Path(study["data_dir"]), lambda: api, lambda _: None)
    assert result["complete"] and len(api.downloads) == 2
    assert len(attempts) == 7 and len(set(attempts)) == 1
    history = json.loads((directory / "download_attempts.json").read_text())["attempts"]
    assert not history[0]["preserved_in_place"] and history[0]["preservation_error"] is None
    assert history[0]["source_filename"] != history[1]["download_file"]


def test_failed_attempt_checkpoint_does_not_mask_original_transfer_error(study, monkeypatch):
    api = fake_api(monkeypatch)
    api.always_fail = True
    atomic_json = data._atomic_json
    def failed_checkpoint(path, value):
        if path.name == "download_attempts.json":
            raise PermissionError("attempt checkpoint locked")
        return atomic_json(path, value)
    monkeypatch.setattr(data, "_atomic_json", failed_checkpoint)
    request = data.build_requests(study)[0]
    directory = data.cache_directory(Path(study["data_dir"]), request)
    directory.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Response ended prematurely") as captured:
        data._download_one({"request": request}, Path(study["data_dir"]), lambda: api, lambda _: None)
    assert any("attempt checkpoint locked" in note for note in captured.value.__notes__)
    assert len(list(directory.glob("interrupted.*"))) == 1


def test_successful_download_rename_lock_recovers_from_marker_without_refetch(study, monkeypatch):
    api = fake_api(monkeypatch)
    replace = Path.replace
    attempts = []
    def permanent_final_lock(path, target):
        if path.name.startswith("stream.") and target.name == "sample.dbn.zst":
            attempts.append((path, target))
            raise PermissionError("WinError 32: successful prefix temporarily held")
        return replace(path, target)
    monkeypatch.setattr(Path, "replace", permanent_final_lock)
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    request = data.build_requests(study)[0]
    directory = data.cache_directory(Path(study["data_dir"]), request)
    directory.mkdir(parents=True)
    with pytest.raises(PermissionError, match="successful prefix temporarily held"):
        data._download_one({"request": request}, Path(study["data_dir"]), lambda: api, lambda _: None)
    assert len(attempts) == 12 and len(api.downloads) == 1
    marker = json.loads((directory / "stream_complete.json").read_text())
    assert (directory / marker["download_file"]).is_file()
    monkeypatch.setattr(Path, "replace", replace)
    result = data._download_one({"request": request}, Path(study["data_dir"]), lambda: api, lambda _: None)
    assert result["complete"] and result["recovered_from_local_dbn"]
    assert len(api.downloads) == 1 and (directory / "sample.dbn.zst").is_file()


@pytest.mark.parametrize("completed", [False, True])
def test_recovery_rename_retries_transient_lock_without_replacing_prefix_bytes(study, tmp_path, monkeypatch, completed):
    request = data.build_requests(study)[0]
    directory = tmp_path / "recovery"
    directory.mkdir()
    source = directory / ("stream.finished.dbn.zst" if completed else "sample.dbn.zst")
    source.write_bytes(b"original durable prefix")
    if completed:
        (directory / "stream_complete.json").write_text(json.dumps({"request": request, "download_file": source.name}))
    replace = Path.replace
    attempts, waits = [], []
    def temporary_lock(path, target):
        attempts.append((path, target))
        if len(attempts) < 7:
            raise PermissionError("WinError 32: transient recovery lock")
        return replace(path, target)
    monkeypatch.setattr(Path, "replace", temporary_lock)
    monkeypatch.setattr(data.time, "sleep", waits.append)
    recovered = data._recover_completed(directory, request)
    assert len(attempts) == 7 and len(set(attempts)) == 1
    assert waits == [.1, .2, .4, .8, 1., 1.]
    if completed:
        assert recovered == directory / "sample.dbn.zst"
        assert recovered.read_bytes() == b"original durable prefix"
    else:
        assert recovered is None and not source.exists()
        prefixes = list(directory.glob("interrupted.*"))
        assert len(prefixes) == 1 and prefixes[0].read_bytes() == b"original durable prefix"


@pytest.mark.parametrize("completed", [False, True])
def test_recovery_permanent_lock_preserves_source_without_starting_download(study, monkeypatch, completed):
    api = fake_api(monkeypatch)
    request = data.build_requests(study)[0]
    directory = data.cache_directory(Path(study["data_dir"]), request)
    directory.mkdir(parents=True)
    (directory / "request.json").write_text(json.dumps(request))
    source = directory / ("stream.finished.dbn.zst" if completed else "sample.dbn.zst")
    source.write_bytes(b"original durable prefix")
    if completed:
        (directory / "stream_complete.json").write_text(json.dumps({"request": request, "download_file": source.name}))
    replace = Path.replace
    attempts = []
    def permanent_lock(path, target):
        if path == source:
            attempts.append((path, target))
            raise PermissionError("WinError 32: persistent recovery lock")
        return replace(path, target)
    monkeypatch.setattr(Path, "replace", permanent_lock)
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError, match="persistent recovery lock"):
        data._download_one({"request": request}, Path(study["data_dir"]), lambda: api, lambda _: None)
    assert len(attempts) == 12 and len(set(attempts)) == 1
    assert source.read_bytes() == b"original durable prefix" and not api.downloads


def failed_manifest(study, api, *, failed_indices=(0,)):
    plan = data.plan_study(study, progress=lambda _: None, api_factory=lambda: api)
    rows = [dict(plan["requests"][index], complete=False, error="original completed-future failure")
            for index in failed_indices]
    manifest = dict(plan, requests=rows, complete=False, downloaded_rows=0)
    data._atomic_json(Path(study["out_dir"]) / "data_manifest.json", manifest)
    api.calls.clear()
    return plan, manifest


def test_failed_repair_runs_only_recorded_failures_in_parallel_and_preserves_locked_prefixes(study, monkeypatch):
    api = fake_api(monkeypatch)
    plan, manifest = failed_manifest(study, api, failed_indices=(0, 2, 4))
    completed = dict(plan["requests"][1], complete=True, rows=3)
    manifest["requests"].append(completed)
    manifest_path = Path(study["out_dir"]) / "data_manifest.json"
    data._atomic_json(manifest_path, manifest)
    manifest_before = manifest_path.read_bytes()
    plan_path = Path(study["out_dir"]) / "download_plan.json"
    plan_before = plan_path.read_bytes()
    old_prefixes = []
    for entry in manifest["requests"][:3]:
        directory = data.cache_directory(Path(study["data_dir"]), entry["request"])
        directory.mkdir(parents=True)
        (directory / "request.json").write_text(json.dumps(entry["request"]))
        prefix = directory / "stream.original_locked.dbn.zst"
        prefix.write_bytes(b"original locked partial bytes")
        old_prefixes.append(prefix)
    replace = Path.replace
    def locked_old_source(path, target):
        if path in old_prefixes:
            pytest.fail("A repair tried to move an unrelated old locked prefix")
        return replace(path, target)
    monkeypatch.setattr(Path, "replace", locked_old_source)
    barrier = Barrier(3)
    api.before_download = lambda _: barrier.wait(timeout=10)
    report = data.repair_failed_study(study, progress=lambda _: None, api_factory=lambda: api, workers=4)
    assert report["complete"] and report["selected_failed_request_count"] == 3
    assert report["repair_workers"] == 3 and report["requested_repair_workers"] == 4
    assert len(api.downloads) == 3 and not api.calls
    assert {data._signature(request) for request in api.downloads} == {
        data._signature(manifest["requests"][index]["request"]) for index in range(3)}
    assert all(prefix.read_bytes() == b"original locked partial bytes" for prefix in old_prefixes)
    assert all(entry["complete"] and entry["original_manifest_error"] == "original completed-future failure"
               for entry in report["requests"])
    assert manifest_path.read_bytes() == manifest_before and plan_path.read_bytes() == plan_before
    assert json.loads((Path(study["out_dir"]) / "repair_report.json").read_text()) == report


def test_failed_repair_reuses_canonical_success_without_metadata_or_download(study, monkeypatch):
    api = fake_api(monkeypatch)
    failed_manifest(study, api)
    first = data.repair_failed_study(study, progress=lambda _: None, api_factory=lambda: api)
    assert first["complete"] and len(api.downloads) == 1
    api.downloads.clear()
    second = data.repair_failed_study(study, progress=lambda _: None, api_factory=lambda: api)
    assert second["complete"] and second["requests"][0]["cache_reused"]
    assert not api.downloads and not api.calls


def test_failed_repair_empty_snapshot_never_constructs_api_or_touches_queued_requests(study, monkeypatch):
    api = fake_api(monkeypatch)
    _, manifest = failed_manifest(study, api, failed_indices=())
    manifest_path = Path(study["out_dir"]) / "data_manifest.json"
    before = manifest_path.read_bytes()
    def forbidden():
        pytest.fail("An empty failed-future snapshot constructed a live client")
    result = data.repair_failed_study(study, progress=lambda _: None, api_factory=forbidden)
    assert result["complete"] and not result["requests"]
    assert result["selected_failed_request_count"] == result["repair_workers"] == 0
    assert result["expected_study_request_count"] == len(data.build_requests(study))
    assert manifest_path.read_bytes() == before and not api.downloads


def test_failed_repair_records_failed_retry_without_mutating_main_manifest(study, monkeypatch):
    api = fake_api(monkeypatch)
    failed_manifest(study, api)
    manifest_path = Path(study["out_dir"]) / "data_manifest.json"
    before = manifest_path.read_bytes()
    api.always_fail = True
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    result = data.repair_failed_study(study, progress=lambda _: None, api_factory=lambda: api)
    assert not result["complete"] and len(api.downloads) == 3
    assert result["requests"][0]["error"] == "RuntimeError: Response ended prematurely"
    assert result["requests"][0]["original_manifest_error"] == "original completed-future failure"
    assert manifest_path.read_bytes() == before
    assert len(list(Path(study["data_dir"]).rglob("interrupted.*"))) == 3


@pytest.mark.parametrize("document,key,value", [
    ("manifest", "configuration_sha256", "foreign"),
    ("manifest", "request_grid_sha256", "foreign"),
    ("manifest", "expected_request_count", 11),
    ("plan", "configuration_sha256", "foreign"),
    ("plan", "request_grid_sha256", "foreign"),
    ("plan", "expected_request_count", 11),
    ("plan", "ready_to_download", False),
])
def test_failed_repair_rejects_foreign_plan_or_manifest_before_api(study, monkeypatch, document, key, value):
    api = fake_api(monkeypatch)
    failed_manifest(study, api)
    path = Path(study["out_dir"]) / ("data_manifest.json" if document == "manifest" else "download_plan.json")
    snapshot = json.loads(path.read_text())
    snapshot[key] = value
    data._atomic_json(path, snapshot)
    def forbidden():
        pytest.fail("Invalid repair provenance reached the API")
    with pytest.raises(ValueError):
        data.repair_failed_study(study, progress=lambda _: None, api_factory=forbidden)
    assert not (Path(study["out_dir"]) / "repair_report.json").exists()
    assert not api.downloads


@pytest.mark.parametrize("defect", ["duplicate", "foreign", "missing_status", "missing_error", "invalid_entry", "wrong_configuration", "missing_plan_request"])
def test_failed_repair_rejects_ambiguous_completed_future_or_grid(study, monkeypatch, defect):
    api = fake_api(monkeypatch)
    plan, manifest = failed_manifest(study, api)
    if defect == "duplicate":
        manifest["requests"].append(copy.deepcopy(manifest["requests"][0]))
    elif defect == "foreign":
        manifest["requests"][0]["request"]["limit"] -= 1
    elif defect == "missing_status":
        del manifest["requests"][0]["complete"]
    elif defect == "missing_error":
        del manifest["requests"][0]["error"]
    elif defect == "invalid_entry":
        manifest["requests"] = [None]
    elif defect == "wrong_configuration":
        manifest["configuration"] = dict(study, name="foreign-study")
    elif defect == "missing_plan_request":
        plan["requests"].pop()
    data._atomic_json(Path(study["out_dir"]) / "data_manifest.json", manifest)
    data._atomic_json(Path(study["out_dir"]) / "download_plan.json", plan)
    with pytest.raises(ValueError):
        data.repair_failed_study(study, progress=lambda _: None, api_factory=lambda: pytest.fail("Invalid repair reached API"))
    assert not api.downloads and not (Path(study["out_dir"]) / "repair_report.json").exists()


@pytest.mark.parametrize("workers", [0, 9, True, 1.5])
def test_failed_repair_rejects_invalid_parallelism_before_reading_files(study, workers):
    with pytest.raises(ValueError, match="Repair workers"):
        data.repair_failed_study(study, progress=lambda _: None, workers=workers,
                                api_factory=lambda: pytest.fail("Invalid workers reached API"))


@pytest.mark.parametrize("complete", [False, True])
def test_repair_cli_dispatches_only_to_failed_repair_and_sets_exit_status(study, monkeypatch, capsys, complete):
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_mbo_sir.py"
    spec = importlib.util.spec_from_file_location("offline_mbo_repair_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(data, "read_config", lambda _: study)
    def repair(config):
        calls.append(config)
        return {"complete": complete, "selected_failed_request_count": 1}
    monkeypatch.setattr(data, "repair_failed_study", repair)
    monkeypatch.setattr(data, "fetch_study", lambda *_, **__: pytest.fail("Repair CLI started the main acquisition"))
    monkeypatch.setattr(sys, "argv", [str(path), "--repair-failed-only"])
    if complete:
        module.main()
    else:
        with pytest.raises(SystemExit) as captured:
            module.main()
        assert captured.value.code == 1
    assert calls == [study]
    assert json.loads(capsys.readouterr().out)["complete"] is complete
