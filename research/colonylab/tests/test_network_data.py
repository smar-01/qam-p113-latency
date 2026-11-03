"""Offline checks of the bounded, aligned network download protocol."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from colonylab import config, network_data


@pytest.fixture(autouse=True)
def no_credentials_or_network(monkeypatch):
    monkeypatch.setattr(config, "load_key", lambda: "unit-test-placeholder")

    def forbidden_client():
        pytest.fail("A network-data unit test attempted to access the real API")

    monkeypatch.setattr(network_data, "client", forbidden_client)


@pytest.fixture
def tmp_path(tmp_path_factory):
    # Keep hashed cache paths under Windows' legacy path limit even when the
    # sandbox's temporary root is long. Each test still has an isolated root.
    return tmp_path_factory.mktemp("networkdata")


def tiny_frame(request):
    index = pd.DatetimeIndex(
        pd.Timestamp(request["start"]) + pd.to_timedelta([0, 10, 20], unit="s"),
        name="ts_recv",
    )
    return pd.DataFrame({"symbol": list(network_data.SYMBOLS), "flags": [128] * 3,
                         "ts_event": index, "bid_px_00": [100.] * 3,
                         "ask_px_00": [100.02] * 3}, index=index)


class FakeAPI:
    def __init__(self, *, count=1000, billable_bytes=10000, cost=.001):
        self.count, self.billable_bytes, self.cost = count, billable_bytes, cost
        self.metadata_calls = []
        self.download_calls = []
        self.stores = {}
        self.before_download = lambda request: None
        self.metadata = SimpleNamespace(get_record_count=self.record_count,
                                        get_billable_size=self.billable_size,
                                        get_cost=self.get_cost)
        self.timeseries = SimpleNamespace(get_range=self.get_range)

    def record_count(self, **request):
        self.metadata_calls.append(("count", request))
        return self.count(request) if callable(self.count) else self.count

    def billable_size(self, **request):
        self.metadata_calls.append(("bytes", request))
        return self.billable_bytes(request) if callable(self.billable_bytes) else self.billable_bytes

    def get_cost(self, **request):
        self.metadata_calls.append(("cost", request))
        return self.cost(request) if callable(self.cost) else self.cost

    def get_range(self, **request):
        self.before_download(request)
        self.download_calls.append(request)
        path = Path(request["path"])
        path.write_bytes(b"offline mock DBN")
        store = SimpleNamespace(to_df=lambda **_: tiny_frame(request))
        self.stores[str(path.resolve())] = store
        return store


def fake_api(monkeypatch, **kwargs):
    api = FakeAPI(**kwargs)
    monkeypatch.setattr(network_data, "client", lambda: api)
    return api


def fetch(tmp_path, **kwargs):
    try:
        return network_data.fetch_network(tmp_path / "data", tmp_path / "out",
                                          progress=lambda _: None, **kwargs)
    except ValueError as exc:
        path = tmp_path / "out/data_manifest.json"
        if path.exists():
            errors = [entry["error"] for entry in json.loads(path.read_text())["requests"] if "error" in entry]
            if errors:
                raise ValueError(f"{exc}; individual errors: {errors}") from exc
        raise


def install_store_reader(monkeypatch, api):
    import databento as db
    monkeypatch.setattr(db.DBNStore, "from_file", lambda path: api.stores[str(Path(path).resolve())])


def test_request_grid_is_exactly_the_aligned_eighteen_feed_windows():
    requests = network_data.requests_for(5)
    expected = {(dataset, schema, day) for dataset in network_data.DATASETS
                for schema in network_data.SCHEMAS for day in network_data.DAYS}
    assert len(requests) == 18
    assert {(r["dataset"], r["schema"], r["start"][:10]) for r in requests} == expected
    for request in requests:
        assert request["symbols"] == ["AAPL", "MSFT", "SPY"]
        assert request["stype_in"] == "raw_symbol"
        assert pd.Timestamp(request["end"]) - pd.Timestamp(request["start"]) == pd.Timedelta(minutes=5)
        assert pd.Timestamp(request["start"]).strftime("%H:%M:%S") == "14:00:00"


@pytest.mark.parametrize("minutes", [0, 16, 1.0, True, "5", None])
def test_invalid_window_rejected_before_any_client_access(minutes):
    with pytest.raises(ValueError):
        network_data.requests_for(minutes)


@pytest.mark.parametrize("budget", [-1, float("nan"), float("inf")])
def test_invalid_budget_rejected_before_any_client_access(tmp_path, budget):
    with pytest.raises(ValueError, match="Budget"):
        fetch(tmp_path, budget=budget)


def test_all_eighteen_cost_and_size_checks_complete_before_first_download(tmp_path, monkeypatch):
    api = fake_api(monkeypatch)

    def inspect_plan_before_download(request):
        assert len(api.metadata_calls) == 54
        plan = json.loads((tmp_path / "out/download_plan.json").read_text())
        assert plan["ready_to_download"]
        assert len(plan["requests"]) == 18
        assert plan["incremental_estimated_cost_usd"] <= plan["authorized_estimate_ceiling_usd"]

    api.before_download = inspect_plan_before_download
    manifest = fetch(tmp_path, minutes=5)
    assert manifest["complete"]
    assert len(api.download_calls) == 18
    # Count/size metadata have no record cap that could conceal truncation.
    assert all("limit" not in request for kind, request in api.metadata_calls if kind in {"count", "bytes"})


def test_budget_failure_blocks_every_download_and_persists_reviewable_plan(tmp_path, monkeypatch):
    api = fake_api(monkeypatch, cost=.04)
    with pytest.raises(ValueError, match="exceeds pilot ceiling"):
        fetch(tmp_path, budget=.5)
    assert not api.download_calls
    plan = json.loads((tmp_path / "out/download_plan.json").read_text())
    assert len(plan["requests"]) == 18
    assert not plan["ready_to_download"]


@pytest.mark.parametrize("bound", ["records", "bytes"])
def test_oversized_one_minute_plan_blocks_every_download(tmp_path, monkeypatch, bound):
    api = fake_api(monkeypatch, count=network_data.MAX_RECORDS if bound == "records" else 100,
                   billable_bytes=network_data.MAX_REQUEST_BYTES + 1 if bound == "bytes" else 100)
    with pytest.raises(ValueError, match="size bound"):
        fetch(tmp_path, minutes=1)
    assert not api.download_calls


def test_all_venues_replan_together_when_one_window_exceeds_size_bound(tmp_path, monkeypatch):
    def count(request):
        duration = pd.Timestamp(request["end"]) - pd.Timestamp(request["start"])
        if request["dataset"] == "XNAS.ITCH" and duration > pd.Timedelta(minutes=1):
            return network_data.MAX_RECORDS
        return 100

    api = fake_api(monkeypatch, count=count)
    manifest = fetch(tmp_path, minutes=5)
    assert manifest["window_minutes"] == 1
    assert len(api.download_calls) == 18
    assert all(pd.Timestamp(r["end"]) - pd.Timestamp(r["start"]) == pd.Timedelta(minutes=1)
               for r in api.download_calls)


def test_documented_metadata_overreport_does_not_falsely_mark_stream_incomplete(tmp_path, monkeypatch):
    api = fake_api(monkeypatch, count=1000)
    manifest = fetch(tmp_path)
    assert manifest["complete"]
    assert all(entry["rows"] == 3 < entry["expected_records_without_limit"] for entry in manifest["requests"])
    assert all(entry["fewer_rows_than_metadata"] for entry in manifest["requests"])


def test_complete_cache_is_reused_without_metadata_or_download_calls(tmp_path, monkeypatch):
    api = fake_api(monkeypatch)
    fetch(tmp_path)
    first_metadata, first_downloads = len(api.metadata_calls), len(api.download_calls)
    manifest = fetch(tmp_path)
    assert manifest["complete"]
    assert manifest["incremental_estimated_cost_usd"] == 0
    assert all(entry["cache_reused"] for entry in manifest["requests"])
    assert len(api.metadata_calls) == first_metadata
    assert len(api.download_calls) == first_downloads


@pytest.mark.parametrize("bad_flag", [True, False])
def test_capped_cache_manifest_cannot_bypass_planning_even_if_marked_complete(tmp_path, monkeypatch, bad_flag):
    api = fake_api(monkeypatch)
    manifest = fetch(tmp_path)
    request = manifest["requests"][0]["request"]
    directory = network_data.cache_directory(tmp_path / "data", request)
    path = directory / "fetch.json"
    cached = json.loads(path.read_text())
    cached["record_limit_reached"] = bad_flag
    if not bad_flag:
        cached["rows"] = request["limit"]
    path.write_text(json.dumps(cached))
    with pytest.raises(ValueError, match="cap"):
        network_data._read_cached(tmp_path / "data", request)
    assert len(api.download_calls) == 18


def test_loader_returns_exact_canonical_keys_and_requested_coverage(tmp_path, monkeypatch):
    api = fake_api(monkeypatch)
    fetch(tmp_path)
    install_store_reader(monkeypatch, api)
    frames = network_data.load_network_frames(tmp_path / "out/data_manifest.json")
    assert len(frames) == 18
    assert set(frames) == {(dataset, schema, day) for dataset in network_data.DATASETS
                           for schema in network_data.SCHEMAS for day in network_data.DAYS}
    assert all(frame.index.name == "ts_recv" for frame in frames.values())
    assert all("requested_start" in frame.attrs and "requested_end" in frame.attrs for frame in frames.values())


@pytest.mark.parametrize("mutation", ["overall_incomplete", "entry_incomplete", "entry_capped", "row_mismatch", "missing_key", "duplicate_key"])
def test_loader_rejects_incomplete_capped_or_inconsistent_cache(tmp_path, monkeypatch, mutation):
    api = fake_api(monkeypatch)
    fetch(tmp_path)
    install_store_reader(monkeypatch, api)
    path = tmp_path / "out/data_manifest.json"
    manifest = json.loads(path.read_text())
    if mutation == "overall_incomplete":
        manifest["complete"] = False
    elif mutation == "entry_incomplete":
        manifest["requests"][0]["complete"] = False
    elif mutation == "entry_capped":
        manifest["requests"][0]["record_limit_reached"] = True
    elif mutation == "row_mismatch":
        manifest["requests"][0]["rows"] += 1
    elif mutation == "missing_key":
        manifest["requests"].pop()
    else:
        manifest["requests"].append(manifest["requests"][0])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        network_data.load_network_frames(path)


def test_loader_checks_actual_record_cap_even_if_manifest_false_flag(tmp_path, monkeypatch):
    api = fake_api(monkeypatch)
    fetch(tmp_path)
    install_store_reader(monkeypatch, api)
    path = tmp_path / "out/data_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["requests"][0]["request"]["limit"] = 3
    manifest["requests"][0]["record_limit_reached"] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="cap|limit"):
        network_data.load_network_frames(path)


def test_existing_partial_stream_is_not_promoted_to_complete_or_redownloaded(tmp_path, monkeypatch):
    api = fake_api(monkeypatch)
    request = network_data.requests_for()[0]
    directory = network_data.cache_directory(tmp_path / "data", request)
    directory.mkdir(parents=True)
    (directory / "sample.dbn.zst").write_bytes(b"valid prefix of interrupted stream")
    (directory / "request.json").write_text(json.dumps(request))
    entry = {"request": request, "expected_records_without_limit": 1000,
             "estimated_billable_bytes_without_limit": 10000,
             "estimated_cost_usd": .001, "incremental_estimated_cost_usd": .001}
    api.stores[str((directory / "sample.dbn.zst").resolve())] = SimpleNamespace(to_df=lambda **_: tiny_frame(request))
    install_store_reader(monkeypatch, api)
    with pytest.raises(ValueError, match="stream|interrupt|verif|complet"):
        network_data._download_entry(entry, tmp_path / "data")
    assert not api.download_calls
