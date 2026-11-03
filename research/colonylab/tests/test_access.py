from __future__ import annotations

import base64
import csv
from datetime import date
import json
from pathlib import Path
import sys
from threading import Barrier
from types import SimpleNamespace

import pandas as pd
import pytest

from colonylab import access, config


DAY = date(2026, 10, 23)
SYNTHETIC_KEY = "db-test-only-never-a-real-key"


@pytest.fixture(autouse=True)
def no_credentials_or_network(monkeypatch):
    # safe_error must exercise its actual redactor without reading the local .env.
    monkeypatch.setattr(config, "load_key", lambda: SYNTHETIC_KEY)

    def unexpected_client():
        pytest.fail("An access unit test must use its injected fake API")

    monkeypatch.setattr(access, "client", unexpected_client)


class FakeAPI:
    def __init__(self, *, datasets=None, schemas=None, costs=None, frames=None, failures=None):
        self.datasets = ["XNAS.ITCH"] if datasets is None else datasets
        self.schemas = ["mbo"] if schemas is None else schemas
        self.costs = costs or {}
        self.frames = frames or {}
        self.failures = failures or {}
        self.cost_requests = []
        self.download_requests = []
        self.before_download = lambda request: None
        self.metadata = SimpleNamespace(
            list_datasets=lambda: self.datasets,
            list_publishers=lambda: [{"publisher_id": 2, "dataset": "XNAS.ITCH", "venue": "XNAS"}],
            list_schemas=lambda **_: self.schemas,
            get_dataset_range=lambda **_: {"start": "2020-01-01T00:00:00Z", "end": "2027-01-01T00:00:00Z"},
            get_cost=self.get_cost,
        )
        self.timeseries = SimpleNamespace(get_range=self.get_range)

    def get_cost(self, **request):
        self.cost_requests.append(request)
        value = self.costs.get((request["dataset"], request["schema"]), 0.01)
        if isinstance(value, Exception):
            raise value
        return value

    def get_range(self, **request):
        self.before_download(request)
        self.download_requests.append(request)
        key = request["dataset"], request["schema"]
        if key in self.failures:
            raise self.failures[key]
        Path(request["path"]).write_bytes(b"mock local DBN data")
        return self.store(request)

    def store(self, request):
        key = request["dataset"], request["schema"]
        frame = self.frames.get(key, pd.DataFrame({"publisher_id": [2], "price": [100.0]}))
        return SimpleNamespace(dataset=request["dataset"], schema=request["schema"], to_df=lambda **_: frame)


def run_audit(tmp_path, api, **kwargs):
    return access.audit_access(
        day=DAY, api=api, data_dir=tmp_path / "data", out_dir=tmp_path / "out",
        progress=lambda _: None, **kwargs,
    )


def probe(result, schema="mbo", dataset="XNAS.ITCH"):
    return next(p for p in result["feeds"][dataset]["probes"] if p["schema"] == schema)


@pytest.mark.parametrize("budget", [-0.01, 0.100001, float("nan"), float("inf")])
def test_invalid_budget_rejected_before_api_access(tmp_path, budget):
    with pytest.raises(ValueError, match="Audit budget"):
        run_audit(tmp_path, object(), budget=budget)
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("symbols", [[], ["A", "B", "C", "D"], ["ALL_SYMBOLS"], [" AAPL"], ["AAPL,MSFT"], [None]])
def test_explicit_symbol_validation_precedes_api_access(tmp_path, symbols):
    with pytest.raises(ValueError, match="explicit ticker"):
        run_audit(tmp_path, object(), symbols=symbols)
    assert not (tmp_path / "data").exists()


def test_cost_sum_is_reserved_and_persisted_before_each_download(tmp_path):
    api = FakeAPI(schemas=list(access.PROBE_SCHEMAS), costs={
        ("XNAS.ITCH", "mbo"): 0.04,
        ("XNAS.ITCH", "mbp-1"): 0.04,
        ("XNAS.ITCH", "trades"): 0.04,
        ("XNAS.ITCH", "ohlcv-1d"): 0.01,
    })
    reservations = []

    def assert_reserved_before_call(request):
        report = json.loads((tmp_path / "out/databento_access.json").read_text())
        reservations.append(report["estimated_budget_reserved_usd"])
        requested = {k: v for k, v in request.items() if k != "path"}
        _, manifest_path = access._request_paths(tmp_path / "data", requested)
        manifest = json.loads(manifest_path.read_text())
        assert manifest["request"] == requested
        assert manifest["retrieval_status"] == "started"

    api.before_download = assert_reserved_before_call
    result = run_audit(tmp_path, api, budget=0.1)
    assert reservations == pytest.approx([0.04, 0.08, 0.09])
    assert result["estimated_budget_reserved_usd"] == pytest.approx(0.09)
    assert probe(result, "trades")["retrieval_status"] == "blocked_by_budget"
    assert [r["schema"] for r in api.download_requests] == ["mbo", "mbp-1", "ohlcv-1d"]


def test_failed_download_still_consumes_reserved_estimate(tmp_path):
    api = FakeAPI(schemas=["mbo", "mbp-1"], costs={
        ("XNAS.ITCH", "mbo"): 0.06, ("XNAS.ITCH", "mbp-1"): 0.06,
    }, failures={("XNAS.ITCH", "mbo"): RuntimeError("connection interrupted")})
    result = run_audit(tmp_path, api, budget=0.1)
    assert result["estimated_budget_reserved_usd"] == pytest.approx(0.06)
    assert probe(result)["retrieval_status"] == "error"
    assert probe(result, "mbp-1")["retrieval_status"] == "blocked_by_budget"
    assert len(api.download_requests) == 1
    manifest = json.loads(next((tmp_path / "data").glob("*.json")).read_text())
    assert manifest["retrieval_status"] == "error"
    assert manifest["estimated_cost_usd"] == 0.06


def test_unsupported_schemas_are_skipped_without_cost_or_download_calls(tmp_path):
    api = FakeAPI(schemas=["trades"])
    result = run_audit(tmp_path, api)
    assert [r["schema"] for r in api.cost_requests] == ["trades"]
    assert [r["schema"] for r in api.download_requests] == ["trades"]
    for schema in ("mbo", "mbp-1", "ohlcv-1d"):
        row = probe(result, schema)
        assert row["support_status"] == "unsupported"
        assert row["cost_status"] == "not_requested"
        assert row["retrieval_status"] == "not_attempted"


def test_supported_but_out_of_range_probe_does_not_request_cost(tmp_path):
    api = FakeAPI()
    api.metadata.get_dataset_range = lambda **_: {
        "start": "2020-01-01T00:00:00Z", "end": "2025-01-01T00:00:00Z",
    }
    result = run_audit(tmp_path, api)
    assert probe(result)["retrieval_status"] == "outside_available_range"
    assert api.cost_requests == api.download_requests == []


def test_empty_data_is_successful_retrieval_not_access_denial(tmp_path):
    api = FakeAPI(frames={("XNAS.ITCH", "mbo"): pd.DataFrame()})
    result = run_audit(tmp_path, api)
    row = probe(result)
    assert row["retrieval_status"] == "downloaded_empty"
    assert row["records"] == 0
    assert "retrieval_error" not in row
    assert result["retrieval_status_counts"]["downloaded_empty"] == 1
    assert Path(row["data_path"]).exists()


def test_estimate_and_download_receive_identical_explicit_request_parameters(tmp_path):
    api = FakeAPI(schemas=list(access.PROBE_SCHEMAS))
    run_audit(tmp_path, api, symbols=["AAPL", "MSFT", "SPY"])
    assert len(api.cost_requests) == len(api.download_requests) == 4
    for estimated, downloaded in zip(api.cost_requests, api.download_requests):
        assert {k: v for k, v in downloaded.items() if k != "path"} == estimated
        assert estimated["stype_in"] == "raw_symbol"
        if estimated["schema"] == "ohlcv-1d":
            assert estimated["symbols"] == ["AAPL", "MSFT", "SPY"]
            assert estimated["start"] == "2026-10-23T00:00:00Z"
            assert estimated["end"] == "2026-10-24T00:00:00Z"
            assert estimated["limit"] == 100
        else:
            assert estimated["symbols"] == ["AAPL"]
            assert estimated["start"] == "2026-10-23T14:00:00Z"
            assert estimated["end"] == "2026-10-23T14:00:01Z"
            assert estimated["limit"] == (2000 if estimated["schema"] == "mbo" else 500)


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), -0.01])
def test_invalid_cost_estimates_never_trigger_download(tmp_path, cost):
    api = FakeAPI(costs={("XNAS.ITCH", "mbo"): cost})
    result = run_audit(tmp_path, api)
    assert api.download_requests == []
    assert probe(result)["cost_status"] == "error"
    assert result["estimated_budget_reserved_usd"] == 0


@pytest.mark.parametrize("failure_stage", ["metadata", "cost", "download"])
def test_errors_use_shared_redactor_and_never_persist_key(tmp_path, monkeypatch, failure_stage):
    encoded = base64.b64encode((SYNTHETIC_KEY + ":").encode()).decode()
    error = RuntimeError(f"mock failure {SYNTHETIC_KEY} Authorization: Basic {encoded}")
    api = FakeAPI()
    if failure_stage == "metadata":
        def fail_metadata(**_):
            raise error
        api.metadata.list_schemas = fail_metadata
    elif failure_stage == "cost":
        api.costs[("XNAS.ITCH", "mbo")] = error
    else:
        api.failures[("XNAS.ITCH", "mbo")] = error
    redacted_errors = []
    actual_redactor = config.safe_error

    def record_redaction(exc):
        redacted_errors.append(exc)
        return actual_redactor(exc)

    monkeypatch.setattr(access, "safe_error", record_redaction)
    result = run_audit(tmp_path, api)
    assert redacted_errors == [error]
    serialized = json.dumps(result)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            serialized += path.read_text()
    assert SYNTHETIC_KEY not in serialized
    assert encoded not in serialized
    assert "[REDACTED]" in serialized


@pytest.mark.parametrize("empty", [False, True])
def test_matching_cached_manifest_is_decoded_without_new_estimate_or_download(tmp_path, monkeypatch, empty):
    api = FakeAPI(frames={("XNAS.ITCH", "mbo"): pd.DataFrame()} if empty else None)
    request = access.make_probe_request("XNAS.ITCH", "mbo", DAY, ["AAPL", "MSFT", "SPY"])
    dbn_path, manifest_path = access._request_paths(tmp_path / "data", request)
    dbn_path.parent.mkdir()
    dbn_path.write_bytes(b"existing cached bytes")
    config.write_json(manifest_path, {"request": request, "estimated_cost_usd": 0.02, "retrieval_status": "downloaded_empty" if empty else "downloaded_nonempty"})
    opened = []

    def from_file(path):
        opened.append(path)
        return api.store(request)

    monkeypatch.setitem(sys.modules, "databento", SimpleNamespace(DBNStore=SimpleNamespace(from_file=from_file)))
    result = run_audit(tmp_path, api, budget=0)
    row = probe(result)
    assert opened == [dbn_path]
    assert api.cost_requests == api.download_requests == []
    assert row["retrieval_status"] == ("cached_empty" if empty else "cached_nonempty")
    assert row["cost_status"] == "cached_no_new_download"
    assert row["original_estimated_cost_usd"] == 0.02
    assert result["estimated_budget_reserved_usd"] == 0
    assert dbn_path.read_bytes() == b"existing cached bytes"


@pytest.mark.parametrize("manifest", [None, "{malformed", {}, {"request": {"dataset": "XNAS.ITCH", "symbols": ["MSFT"]}}])
def test_unverified_existing_download_never_billed_again_or_overwritten(tmp_path, manifest):
    api = FakeAPI()
    request = access.make_probe_request("XNAS.ITCH", "mbo", DAY, ["AAPL", "MSFT", "SPY"])
    dbn_path, manifest_path = access._request_paths(tmp_path / "data", request)
    dbn_path.parent.mkdir()
    dbn_path.write_bytes(b"do not overwrite these bytes")
    if manifest is not None:
        manifest_path.write_text(manifest if isinstance(manifest, str) else json.dumps(manifest))
    result = run_audit(tmp_path, api)
    assert api.cost_requests == api.download_requests == []
    assert probe(result)["retrieval_status"] != "downloaded_nonempty"
    assert dbn_path.read_bytes() == b"do not overwrite these bytes"


def test_equity_selection_and_feed_scope_do_not_treat_reporting_feeds_as_exchange_books():
    assert access.select_equity_datasets([
        "OPRA.PILLAR", "GLBX.MDP3", "XNAS.ITCH", "EQUS.MINI", "FINN.NLS", "XNAS.ITCH",
    ]) == ["EQUS.MINI", "FINN.NLS", "XNAS.ITCH"]
    assert access.feed_scope("EQUS.MINI") == "composite_feed_not_an_execution_venue"
    assert access.feed_scope("FINN.NLS") == "trade_reporting_facility_not_exchange_book"
    assert access.feed_scope("XNAS.BASIC").startswith("combined_exchange_and_off_exchange")
    assert access.feed_scope("XNAS.ITCH") == "individual_exchange_feed"


def test_summary_preserves_snapshot_flags_publishers_and_record_limit_warning():
    request = access.make_probe_request("XNAS.ITCH", "mbo", DAY, ["AAPL"])
    request["limit"] = 3
    frame = pd.DataFrame({
        "publisher_id": [2, 2, 999], "action": ["A", "T", "A"], "flags": [32, 0, 160],
    }, index=pd.date_range("2026-10-23T14:00:00Z", periods=3, freq="ms", name="ts_recv"))
    api = FakeAPI(frames={("XNAS.ITCH", "mbo"): frame})
    result = access.summarize_store(api.store(request), request, [{"publisher_id": 2, "venue": "XNAS"}])
    assert result["snapshot_records"] == 2
    assert result["trade_action_records"] == 1
    assert result["record_limit_reached"] is True
    assert result["timestamp_basis"] == "ts_recv"
    assert result["publisher_mappings"] == [{"publisher_id": 2, "venue": "XNAS"}, {"publisher_id": 999}]


@pytest.mark.parametrize("download_fails", [False, True])
def test_concurrent_estimates_cannot_both_spend_the_same_budget(tmp_path, monkeypatch, download_fails):
    datasets = ["XNAS.ITCH", "XNYS.PILLAR"]
    api = FakeAPI(
        datasets=datasets,
        costs={(dataset, "mbo"): 0.06 for dataset in datasets},
        failures={
            (dataset, "mbo"): RuntimeError("mock interrupted download") for dataset in datasets
        } if download_fails else None,
    )
    both_estimates_in_flight = Barrier(2, timeout=5)
    original_estimate = api.get_cost

    def overlapping_estimate(**request):
        value = original_estimate(**request)
        # Neither worker can reserve its cost until both estimates are in flight.
        both_estimates_in_flight.wait()
        return value

    api.metadata.get_cost = overlapping_estimate
    persisted = []
    original_save = access._save_reports

    def remember_persisted_reservations(result, out_dir):
        original_save(result, out_dir)
        persisted.append(json.loads(json.dumps(result)))

    monkeypatch.setattr(access, "_save_reports", remember_persisted_reservations)
    observations_before_download = []

    def before_download(request):
        observations_before_download.append(bool(persisted and any(
            item["estimated_budget_reserved_usd"] == pytest.approx(0.06)
            for item in persisted
        )))
        _, manifest_path = access._request_paths(
            tmp_path / "data", {k: v for k, v in request.items() if k != "path"},
        )
        manifest = json.loads(manifest_path.read_text())
        assert manifest["retrieval_status"] == "started"
        assert manifest["estimated_cost_usd"] == 0.06

    api.before_download = before_download
    result = run_audit(tmp_path, api, workers=2, budget=0.1)

    assert len(api.cost_requests) == 2
    assert len(api.download_requests) == 1
    assert observations_before_download == [True]
    assert result["estimated_budget_reserved_usd"] == pytest.approx(0.06)
    expected_retrieved_status = "error" if download_fails else "downloaded_nonempty"
    rows = [probe(result, dataset=dataset) for dataset in datasets]
    assert sorted(row["retrieval_status"] for row in rows) == sorted([
        expected_retrieved_status, "blocked_by_budget",
    ])
    assert result["retrieval_status_counts"] == {
        expected_retrieved_status: 1, "blocked_by_budget": 1, "not_attempted": 6,
    }
    # The final JSON and CSV must agree even when completion order varies.
    saved = json.loads((tmp_path / "out/databento_access.json").read_text())
    assert saved == result
    with (tmp_path / "out/databento_access.csv").open(newline="") as file:
        csv_rows = [row for row in csv.DictReader(file) if row["schema"] == "mbo"]
    assert sorted(row["retrieval_status"] for row in csv_rows) == sorted(
        row["retrieval_status"] for row in rows
    )


@pytest.mark.parametrize("invalid_cost", [float("nan"), float("inf"), -0.01])
def test_concurrent_invalid_estimate_and_low_budget_never_start_retrieval(tmp_path, invalid_cost):
    api = FakeAPI(datasets=["XNAS.ITCH", "XNYS.PILLAR"], costs={
        ("XNAS.ITCH", "mbo"): invalid_cost,
        ("XNYS.PILLAR", "mbo"): 0.06,
    })
    both_estimates_in_flight = Barrier(2, timeout=5)
    original_estimate = api.get_cost

    def overlapping_estimate(**request):
        value = original_estimate(**request)
        both_estimates_in_flight.wait()
        return value

    api.metadata.get_cost = overlapping_estimate
    result = run_audit(tmp_path, api, workers=2, budget=0.01)
    assert len(api.cost_requests) == 2
    assert api.download_requests == []
    assert result["estimated_budget_reserved_usd"] == 0
    assert probe(result, dataset="XNAS.ITCH")["cost_status"] == "error"
    assert probe(result, dataset="XNYS.PILLAR")["retrieval_status"] == "blocked_by_budget"
    assert not list((tmp_path / "data").glob("*.dbn.zst"))


@pytest.mark.parametrize("workers", [0, 5, 1.5, True])
def test_invalid_worker_count_is_rejected_before_api_access(tmp_path, workers):
    with pytest.raises(ValueError, match="[Ww]orkers"):
        run_audit(tmp_path, object(), workers=workers)
    assert not (tmp_path / "data").exists()


def test_report_persistence_recovers_from_transient_windows_file_locks(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    request = access.make_probe_request("XNAS.ITCH", "mbo", DAY, ["AAPL"])
    result = {"feeds": {"XNAS.ITCH": {
        "feed_scope": "individual_exchange_feed",
        "probes": [{"schema": "mbo", "request": request, "retrieval_status": "downloaded_nonempty"}],
    }}}
    original_replace = Path.replace
    locked_targets = set()

    def transient_lock(source, target):
        target = Path(target)
        if source.parent == out_dir and target.name not in locked_targets:
            locked_targets.add(target.name)
            raise PermissionError("mock temporary Windows reader lock")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", transient_lock)
    access._save_reports(result, out_dir)
    assert locked_targets == {
        "databento_access.json", "databento_access.csv", "databento_access_daily_volumes.csv",
    }
    assert json.loads((out_dir / "databento_access.json").read_text()) == result
    with (out_dir / "databento_access.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["retrieval_status"] == "downloaded_nonempty"
    assert not list(out_dir.glob("*.tmp"))
