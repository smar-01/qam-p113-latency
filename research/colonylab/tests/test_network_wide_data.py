"""Offline validation of the explicit wide-study grid, budget and cache gates."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from colonylab import config, network_data, network_wide_data as wide


@pytest.fixture(autouse=True)
def no_live_api(monkeypatch):
    monkeypatch.setattr(config, "load_key", lambda: "wide-unit-placeholder")

    def forbidden():
        pytest.fail("Wide-study test attempted live API access")

    monkeypatch.setattr(network_data, "client", forbidden)


@pytest.fixture
def study():
    return {
        "name": "offline-wide", "datasets": ["XNAS.ITCH", "BATS.PITCH"],
        "symbols": ["AAPL", "MSFT", "SPY"], "schemas": ["mbp-1"],
        "training_days": ["2026-10-05"], "held_out_days": ["2026-10-06"],
        "start_time_utc": "14:00:00", "minutes": 10, "max_cost_usd": 5.,
        "workers": 2, "record_limit_per_request": 1_200_000,
        "max_billable_bytes_per_request": 75_000_000,
        "data_dir": "data/wide_network", "out_dir": "out/wide_network",
    }


@pytest.fixture
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("wide")


class FakeAPI:
    def __init__(self, *, count=10, size=1000, cost=.001):
        self.count, self.size, self.cost = count, size, cost
        self.metadata_calls, self.downloads, self.stores = [], [], {}
        self.before_download = lambda request: None
        self.metadata = SimpleNamespace(get_record_count=self.get_count,
                                        get_billable_size=self.get_size, get_cost=self.get_cost)
        self.timeseries = SimpleNamespace(get_range=self.get_range)

    def get_count(self, **request):
        self.metadata_calls.append(("count", request))
        return self.count

    def get_size(self, **request):
        self.metadata_calls.append(("size", request))
        return self.size

    def get_cost(self, **request):
        self.metadata_calls.append(("cost", request))
        return self.cost

    def get_range(self, **request):
        self.before_download(request)
        self.downloads.append(request)
        path = Path(request["path"])
        path.write_bytes(b"mock DBN")
        frame = pd.DataFrame({"symbol": ["AAPL", "MSFT", "SPY"], "flags": [128] * 3},
                             index=pd.DatetimeIndex(pd.Timestamp(request["start"]) + pd.to_timedelta([1, 2, 3], unit="s"), name="ts_recv"))
        store = SimpleNamespace(to_df=lambda **_: frame)
        self.stores[str(path.resolve())] = store
        return store


def fake_api(monkeypatch, **kwargs):
    api = FakeAPI(**kwargs)
    monkeypatch.setattr(network_data, "client", lambda: api)
    import databento as db
    monkeypatch.setattr(db.DBNStore, "from_file", lambda path: api.stores[str(Path(path).resolve())])
    return api


def fetch(study, tmp_path):
    try:
        return wide.fetch_wide_network(study, tmp_path / "data", tmp_path / "out", progress=lambda _: None)
    except ValueError as exc:
        path = tmp_path / "out/data_manifest.json"
        if path.exists():
            errors = [entry["error"] for entry in json.loads(path.read_text())["requests"] if "error" in entry]
            if errors:
                raise ValueError(f"{exc}; individual errors: {errors}") from exc
        raise


def test_fixed_design_has_exactly_fifty_complete_ten_minute_requests():
    frozen = wide.read_wide_config(Path(__file__).resolve().parents[1] / "wide_network.toml")
    requests = wide.requests_for(frozen)
    assert len(requests) == 50
    assert len({request["dataset"] for request in requests}) == 5
    assert len({request["start"][:10] for request in requests}) == 10
    assert len(frozen["training_days"]) == len(frozen["held_out_days"]) == 5
    assert all(request["symbols"] == ["AAPL", "MSFT", "SPY"] for request in requests)
    assert all(pd.Timestamp(request["end"]) - pd.Timestamp(request["start"]) == pd.Timedelta(minutes=10)
               for request in requests)


def interrupted_cache(study, tmp_path, *, error="Error streaming response: Response ended prematurely"):
    included = dict(study, billing_assumed_included=True)
    request = wide.requests_for(included)[0]
    data_dir = tmp_path / "data"
    directory = network_data.cache_directory(data_dir, request)
    directory.mkdir(parents=True)
    (directory / "request.json").write_text(json.dumps(request))
    (directory / "sample.dbn.zst").write_bytes(b"original interrupted bytes")
    (directory / "fetch.json").write_text(json.dumps({"request": request, "complete": False}))
    entry = {"request": request, "complete": False, "error": error}
    manifest = {"configuration": included, "configuration_sha256": wide._signature(included),
                "requests": [entry], "complete": False}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return included, data_dir, directory, manifest_path, manifest


def test_explicit_interrupted_recovery_preserves_all_bytes_and_matching_request(study, tmp_path):
    included, data_dir, directory, path, manifest = interrupted_cache(study, tmp_path)
    original_receipt = (directory / "fetch.json").read_bytes()
    report = wide.recover_interrupted_wide_downloads(included, data_dir, path)
    assert len(report["prepared_for_retry"]) == 1
    assert not (directory / "sample.dbn.zst").exists() and not (directory / "fetch.json").exists()
    preserved = [Path(value) for value in report["prepared_for_retry"][0]["preserved_paths"]]
    assert all(value.parent == directory.resolve() for value in preserved)
    assert [value.read_bytes() for value in preserved] == [b"original interrupted bytes", original_receipt]
    assert json.loads((directory / "request.json").read_text()) == manifest["requests"][0]["request"]
    assert json.loads(path.read_text()) == manifest
    again = wide.recover_interrupted_wide_downloads(included, data_dir, path)
    assert not again["prepared_for_retry"] and len(again["no_partial_file"]) == 1


@pytest.mark.parametrize("error", ["Timeout: socket stalled", "ConnectionError: reset", "HTTP 502", "HTTP 503", "HTTP 504"])
def test_interrupted_recovery_accepts_only_known_transient_failures(study, tmp_path, error):
    included, data_dir, _, path, _ = interrupted_cache(study, tmp_path, error=error)
    assert len(wide.recover_interrupted_wide_downloads(included, data_dir, path)["prepared_for_retry"]) == 1


@pytest.mark.parametrize("mutation", ["no_included", "nontransient", "wrong_config", "duplicate", "wrong_grid",
                                      "request_mismatch", "receipt_mismatch", "external_path", "wrong_cache_path",
                                      "marker_mismatch"])
def test_interrupted_recovery_refuses_unsafe_changes_without_mutating_files(study, tmp_path, mutation):
    included, data_dir, directory, path, manifest = interrupted_cache(study, tmp_path)
    if mutation == "no_included":
        included["billing_assumed_included"] = False
    elif mutation == "nontransient":
        manifest["requests"][0]["error"] = "ValueError: unsupported schema"
    elif mutation == "wrong_config":
        manifest["configuration"]["name"] = "wrong"
    elif mutation == "duplicate":
        manifest["requests"].append(copy.deepcopy(manifest["requests"][0]))
    elif mutation == "wrong_grid":
        manifest["requests"][0]["request"]["end"] = "2026-10-05T14:00:30+00:00"
    elif mutation == "request_mismatch":
        (directory / "request.json").write_text("{}")
    elif mutation == "receipt_mismatch":
        (directory / "fetch.json").write_text(json.dumps({"request": {}, "complete": False}))
    elif mutation == "external_path":
        manifest["requests"][0]["data_path"] = str(tmp_path / "external.dbn.zst")
    elif mutation == "wrong_cache_path":
        manifest["requests"][0]["data_path"] = str(directory / "another.dbn.zst")
    else:
        (directory / "stream_complete.json").write_text(json.dumps({"request": {}}))
    path.write_text(json.dumps(manifest))
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(ValueError):
        wide.recover_interrupted_wide_downloads(included, data_dir, path)
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}


@pytest.mark.parametrize("kind", ["complete_manifest", "complete_receipt", "complete_stream"])
def test_interrupted_recovery_never_moves_a_successful_cache(study, tmp_path, kind):
    included, data_dir, directory, path, manifest = interrupted_cache(study, tmp_path)
    if kind == "complete_manifest":
        manifest["requests"][0]["complete"] = True
        path.write_text(json.dumps(manifest))
    elif kind == "complete_receipt":
        (directory / "fetch.json").write_text(json.dumps({"request": manifest["requests"][0]["request"], "complete": True}))
    else:
        (directory / "stream_complete.json").write_text(json.dumps({"request": manifest["requests"][0]["request"]}))
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    report = wide.recover_interrupted_wide_downloads(included, data_dir, path)
    assert not report["prepared_for_retry"]
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
    assert len(report["export_recovery"]) == (kind == "complete_stream")


def test_interrupted_recovery_checks_all_failed_entries_before_first_rename(study, tmp_path):
    included, data_dir, directory, path, manifest = interrupted_cache(study, tmp_path)
    second = {"request": wide.requests_for(included)[1], "complete": False, "error": "No entitlement"}
    manifest["requests"].append(second)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="nontransient"):
        wide.recover_interrupted_wide_downloads(included, data_dir, path)
    assert (directory / "sample.dbn.zst").read_bytes() == b"original interrupted bytes"
    assert (directory / "fetch.json").exists()


@pytest.mark.parametrize("key,value", [
    ("datasets", ["EQUS.MINI"]), ("schemas", ["mbo"]), ("symbols", ["ALL_SYMBOLS"]),
    ("symbols", ["AAPL,MSFT"]), ("symbols", ["AAPL", "AAPL"]),
    ("training_days", []), ("held_out_days", ["2026-10-05"]),
    ("held_out_days", ["2026-10-03"]), ("held_out_days", ["2026-10-32"]),
    ("training_days", ["2026-10-06", "2026-10-05"]),
    ("start_time_utc", "14:00"), ("start_time_utc", "25:00:00"),
    ("start_time_utc", "23:59:59"), ("minutes", True), ("workers", 5),
    ("max_cost_usd", float("nan")), ("max_cost_usd", -1),
    ("record_limit_per_request", 0), ("max_billable_bytes_per_request", 0),
])
def test_invalid_design_rejected_before_any_api(study, tmp_path, key, value):
    study[key] = value
    with pytest.raises(ValueError):
        fetch(study, tmp_path)


def test_all_grid_cost_and_size_metadata_finish_before_first_pull(study, tmp_path, monkeypatch):
    api = fake_api(monkeypatch)

    def check_before_download(request):
        assert len(api.metadata_calls) == 12
        assert json.loads((tmp_path / "out/download_plan.json").read_text())["ready_to_download"]

    api.before_download = check_before_download
    manifest = fetch(study, tmp_path)
    assert manifest["complete"] and len(api.downloads) == 4
    assert manifest["configuration"] == study
    assert manifest["training_days"] == study["training_days"]
    assert manifest["held_out_days"] == study["held_out_days"]
    assert manifest["window_minutes"] == 10
    assert all("limit" not in request for kind, request in api.metadata_calls if kind in {"count", "size"})


@pytest.mark.parametrize("kind", ["cost", "count", "size"])
def test_budget_or_size_failure_preserves_full_quote_without_any_shortening(study, tmp_path, monkeypatch, kind):
    api = fake_api(monkeypatch, **{"cost": 2.0} if kind == "cost" else
                   {"count": study["record_limit_per_request"]} if kind == "count" else
                   {"size": study["max_billable_bytes_per_request"] + 1})
    with pytest.raises(ValueError, match="no download performed"):
        fetch(study, tmp_path)
    assert not api.downloads
    plan = json.loads((tmp_path / "out/download_plan.json").read_text())
    assert len(plan["requests"]) == 4 and plan["window_minutes"] == 10
    assert not plan["ready_to_download"]


def test_complete_cache_reuse_has_no_repeated_metadata_or_retrieval(study, tmp_path, monkeypatch):
    api = fake_api(monkeypatch)
    first = fetch(study, tmp_path)
    second = fetch(study, tmp_path)
    assert len(api.downloads) == 4 and len(api.metadata_calls) == 12
    assert second["incremental_estimated_cost_usd"] == 0
    assert second["original_estimated_cost_usd"] == first["original_estimated_cost_usd"]
    frames = wide.load_wide_frames(tmp_path / "out/data_manifest.json", expected_config=study)
    assert set(frames) == {(dataset, "mbp-1", day) for dataset in study["datasets"]
                           for day in study["training_days"] + study["held_out_days"]}
    assert all(frame.attrs["requested_start"].endswith("14:00:00+00:00") for frame in frames.values())


def test_transport_worker_override_preserves_frozen_request_config(study, tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    api = fake_api(monkeypatch)
    observed_workers = []

    def observed_pool(*, max_workers):
        observed_workers.append(max_workers)
        return ThreadPoolExecutor(max_workers=max_workers)

    monkeypatch.setattr(wide, "ThreadPoolExecutor", observed_pool)
    original = copy.deepcopy(study)
    result = wide.fetch_wide_network(study, tmp_path / "data", tmp_path / "out",
                                     download_workers=8, progress=lambda _: None)
    assert observed_workers == [study["workers"], 8]
    assert result["download_workers"] == 8
    assert result["configuration"] == original and study == original
    assert len(api.downloads) == len(wide.requests_for(study))
    assert {json.dumps(entry["request"], sort_keys=True) for entry in result["requests"]} == {
        json.dumps(request, sort_keys=True) for request in wide.requests_for(study)}


def test_default_transport_workers_come_from_original_config(study, tmp_path, monkeypatch):
    fake_api(monkeypatch)
    assert fetch(study, tmp_path)["download_workers"] == study["workers"]


@pytest.mark.parametrize("workers", [0, 17, -1, True, 1.5, "8"])
def test_invalid_transport_workers_rejected_before_metadata(study, tmp_path, workers):
    with pytest.raises(ValueError, match="download_workers"):
        wide.fetch_wide_network(study, tmp_path / "data", tmp_path / "out",
                                download_workers=workers, progress=lambda _: None)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "changed_request", "bad_config_hash", "incomplete", "capped", "wrong_rows"])
def test_loader_rejects_grid_or_provenance_corruption(study, tmp_path, monkeypatch, mutation):
    fake_api(monkeypatch)
    fetch(study, tmp_path)
    path = tmp_path / "out/data_manifest.json"
    manifest = json.loads(path.read_text())
    if mutation == "missing":
        manifest["requests"].pop()
    elif mutation == "extra":
        extra = copy.deepcopy(manifest["requests"][0])
        extra["request"]["start"] = "2026-10-07T14:00:00+00:00"
        manifest["requests"].append(extra)
    elif mutation == "duplicate":
        manifest["requests"][1] = manifest["requests"][0]
    elif mutation == "changed_request":
        manifest["requests"][0]["request"]["end"] = "2026-10-05T14:01:00+00:00"
    elif mutation == "bad_config_hash":
        manifest["configuration"]["name"] = "altered"
    elif mutation == "incomplete":
        manifest["complete"] = False
    elif mutation == "capped":
        manifest["requests"][0]["record_limit_reached"] = True
    else:
        manifest["requests"][0]["rows"] += 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        wide.load_wide_frames(path, expected_config=study)


@pytest.mark.parametrize("mutation", ["too_early", "too_late", "wrong_symbol", "missing_symbol", "capped"])
def test_loader_validates_actual_dbn_bounds_and_symbols(study, tmp_path, monkeypatch, mutation):
    if mutation == "capped":
        study["record_limit_per_request"] = 4
    api = fake_api(monkeypatch, count=3)
    manifest = fetch(study, tmp_path)
    entry = manifest["requests"][0]
    store = api.stores[entry["data_path"]]
    frame = store.to_df().copy()
    if mutation in {"too_early", "too_late"}:
        frame.index = frame.index - pd.Timedelta(days=1) if mutation == "too_early" else frame.index + pd.Timedelta(minutes=10)
    elif mutation == "wrong_symbol":
        frame["symbol"] = ["AAPL", "MSFT", "NVDA"]
    elif mutation == "missing_symbol":
        frame["symbol"] = ["AAPL", "MSFT", "MSFT"]
    else:
        frame = pd.concat([frame, frame.iloc[:1]])
    store.to_df = lambda **_: frame
    with pytest.raises(ValueError):
        wide.load_wide_frames(tmp_path / "out/data_manifest.json", expected_config=study)


def test_loader_refuses_analysis_against_changed_design(study, tmp_path, monkeypatch):
    fake_api(monkeypatch)
    fetch(study, tmp_path)
    study["minutes"] = 9
    with pytest.raises(ValueError, match="differs"):
        wide.load_wide_frames(tmp_path / "out/data_manifest.json", expected_config=study)


def test_recent_exact_plan_can_be_pulled_without_repeated_metadata(study, tmp_path, monkeypatch):
    api = fake_api(monkeypatch)
    plan = wide.plan_wide_network(study, tmp_path / "data", tmp_path / "out", progress=lambda _: None)
    result = wide.fetch_wide_network(study, tmp_path / "data", tmp_path / "out", plan=plan, progress=lambda _: None)
    assert result["complete"] and len(api.downloads) == 4
    assert len(api.metadata_calls) == 12
    assert result["original_estimated_cost_usd"] == plan["original_estimated_cost_usd"]


@pytest.mark.parametrize("mutation", ["stale", "missing_request", "changed_config", "over_budget", "size"])
def test_supplied_plan_must_pass_freshness_grid_and_budget_gates(study, tmp_path, monkeypatch, mutation):
    api = fake_api(monkeypatch)
    plan = wide.plan_wide_network(study, tmp_path / "data", tmp_path / "out", progress=lambda _: None)
    if mutation == "stale":
        plan["created_utc"] = "2000-01-01T00:00:00+00:00"
    elif mutation == "missing_request":
        plan["requests"].pop()
    elif mutation == "changed_config":
        study["name"] = "changed"
    elif mutation == "over_budget":
        plan["requests"][0]["incremental_estimated_cost_usd"] = 10.
        plan["incremental_estimated_cost_usd"] = 10.003
    else:
        plan["requests"][0]["expected_records_without_limit"] = study["record_limit_per_request"]
    with pytest.raises(ValueError):
        wide.fetch_wide_network(study, tmp_path / "data", tmp_path / "out", plan=plan, progress=lambda _: None)
    assert not api.downloads


def test_metadata_resume_keeps_original_quote_age_instead_of_refreshing_it(study, tmp_path, monkeypatch):
    api = fake_api(monkeypatch)
    wide.plan_wide_network(study, tmp_path / "data", tmp_path / "out", progress=lambda _: None)
    progress_path = tmp_path / "out/metadata_progress.json"
    state = json.loads(progress_path.read_text())
    old = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    state["created_utc"] = old
    progress_path.write_text(json.dumps(state))
    resumed = wide.plan_wide_network(study, tmp_path / "data", tmp_path / "out", progress=lambda _: None)
    assert len(api.metadata_calls) == 12
    assert resumed["created_utc"] == old
    assert json.loads(progress_path.read_text())["created_utc"] == old


def test_expired_metadata_progress_forces_all_free_quotes_to_be_refreshed(study, tmp_path, monkeypatch):
    api = fake_api(monkeypatch)
    wide.plan_wide_network(study, tmp_path / "data", tmp_path / "out", progress=lambda _: None)
    progress_path = tmp_path / "out/metadata_progress.json"
    state = json.loads(progress_path.read_text())
    state["created_utc"] = "2000-01-01T00:00:00+00:00"
    progress_path.write_text(json.dumps(state))
    wide.plan_wide_network(study, tmp_path / "data", tmp_path / "out", progress=lambda _: None)
    assert len(api.metadata_calls) == 24


def test_serial_directory_setup_retries_one_permission_error_before_any_download(study, tmp_path, monkeypatch):
    api = fake_api(monkeypatch)
    data_dir = tmp_path / "data"
    request = wide.requests_for(study)[0]
    contested = network_data.cache_directory(data_dir, request)
    original_mkdir = Path.mkdir
    attempts = []

    def flaky_mkdir(path, *args, **kwargs):
        if path == contested:
            attempts.append(path)
            if len(attempts) == 1:
                assert not api.downloads
                raise PermissionError("simulated transient OneDrive directory contention")
        return original_mkdir(path, *args, **kwargs)

    def assert_all_directories_prepared(request):
        assert all(network_data.cache_directory(data_dir, candidate).is_dir()
                   for candidate in wide.requests_for(study))

    monkeypatch.setattr(Path, "mkdir", flaky_mkdir)
    api.before_download = assert_all_directories_prepared
    assert fetch(study, tmp_path)["complete"]
    # Further idempotent mkdir calls also occur when writing cache metadata.
    assert len(attempts) >= 2


def test_persistent_directory_permission_failure_is_bounded_and_blocks_downloads(study, tmp_path, monkeypatch):
    api = fake_api(monkeypatch)
    data_dir = tmp_path / "data"
    original_mkdir = Path.mkdir
    attempts = []

    def denied_mkdir(path, *args, **kwargs):
        if path == data_dir:
            attempts.append(path)
            raise PermissionError("persistent directory permission failure")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", denied_mkdir)
    with pytest.raises(PermissionError, match="persistent"):
        fetch(study, tmp_path)
    assert len(attempts) == 3
    assert not api.downloads


def test_included_access_skips_price_metadata_and_monetary_ceiling(study, tmp_path, monkeypatch):
    study.update(billing_assumed_included=True, max_cost_usd=0.)
    api = fake_api(monkeypatch, cost=999.)

    def forbidden_cost(**request):
        pytest.fail("Included historical access must not request cost metadata")

    api.metadata.get_cost = forbidden_cost

    def before_download(request):
        assert len(api.metadata_calls) == 8
        assert {kind for kind, _ in api.metadata_calls} == {"count", "size"}

    api.before_download = before_download
    manifest = fetch(study, tmp_path)
    assert manifest["complete"] and len(api.downloads) == 4
    assert manifest["cost_check_skipped"]
    assert "User reports" in manifest["billing_assumption"]
    assert manifest["incremental_estimated_cost_usd"] is None
    assert manifest["original_estimated_cost_usd"] is None
    assert manifest["authorized_estimate_ceiling_usd"] is None
    assert all(entry["estimated_cost_usd"] is None and entry["cost_check_skipped"]
               for entry in manifest["requests"])
    assert all("limit" not in request for _, request in api.metadata_calls)


def test_included_access_does_not_require_budget_and_reuses_recent_size_plan(study, tmp_path, monkeypatch):
    study["billing_assumed_included"] = True
    del study["max_cost_usd"]
    api = fake_api(monkeypatch)
    plan = wide.plan_wide_network(study, tmp_path / "data", tmp_path / "out", progress=lambda _: None)
    resumed = wide.plan_wide_network(study, tmp_path / "data", tmp_path / "out", progress=lambda _: None)
    manifest = wide.fetch_wide_network(study, tmp_path / "data", tmp_path / "out",
                                       plan=resumed, progress=lambda _: None)
    assert manifest["complete"]
    assert plan["created_utc"] == resumed["created_utc"]
    assert len(api.metadata_calls) == 8 and len(api.downloads) == 4


@pytest.mark.parametrize("kind", ["count", "size"])
def test_included_access_retains_all_request_size_bounds(study, tmp_path, monkeypatch, kind):
    study["billing_assumed_included"] = True
    api = fake_api(monkeypatch, **{"count": study["record_limit_per_request"]} if kind == "count" else
                   {"size": study["max_billable_bytes_per_request"] + 1})
    with pytest.raises(ValueError, match="size bound"):
        fetch(study, tmp_path)
    assert not api.downloads
    assert len(api.metadata_calls) == 8


def test_included_access_reuses_old_exact_cache_and_frozen_manifest(study, tmp_path, monkeypatch):
    api = fake_api(monkeypatch)
    original = fetch(study, tmp_path)
    study["billing_assumed_included"] = True
    del study["max_cost_usd"]
    frames = wide.load_wide_frames(tmp_path / "out/data_manifest.json", expected_config=study)
    assert len(frames) == 4
    manifest = fetch(study, tmp_path)
    assert manifest["complete"]
    assert len(api.metadata_calls) == 12 and len(api.downloads) == 4
    assert manifest["incremental_estimated_cost_usd"] is None
    assert all(entry["cache_reused"] for entry in manifest["requests"])
    assert manifest["configuration"] == study
    assert original["configuration"] != study


def test_budgeted_mode_can_reuse_cache_without_original_price_quote(study, tmp_path, monkeypatch):
    study["billing_assumed_included"] = True
    api = fake_api(monkeypatch)
    fetch(study, tmp_path)
    study["billing_assumed_included"] = False
    study["max_cost_usd"] = 0.
    manifest = fetch(study, tmp_path)
    assert manifest["complete"] and manifest["incremental_estimated_cost_usd"] == 0
    assert manifest["original_estimated_cost_usd"] is None
    assert len(api.metadata_calls) == 8 and len(api.downloads) == 4


def test_included_access_policy_requires_boolean(study):
    study["billing_assumed_included"] = "true"
    with pytest.raises(ValueError, match="boolean"):
        wide.validate_config(study)
