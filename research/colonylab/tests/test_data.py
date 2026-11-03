from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from colonylab import data


REQUEST = {
    "dataset": "EQUS.MINI", "schema": "mbp-1", "symbols": ["AAPL"],
    "start": "2025-01-06T14:30:00Z", "end": "2025-01-06T14:31:00Z",
    "limit": 10000, "stype_in": "raw_symbol",
}


class FakeAPI:
    def __init__(self, cost=0.1):
        self.cost = cost
        self.cost_requests = []
        self.download_requests = []
        self.metadata = SimpleNamespace(get_cost=self.get_cost, list_schemas=lambda **_: ["mbp-1", "trades"])
        self.timeseries = SimpleNamespace(get_range=self.get_range)

    def get_cost(self, **request):
        self.cost_requests.append(request)
        return self.cost

    def get_range(self, **request):
        self.download_requests.append(request)
        return SimpleNamespace(
            dataset="EQUS.MINI", schema="mbp-1",
            to_df=lambda **_: pd.DataFrame({"price": [100.0], "size": [10]}),
        )


def plan_file(tmp_path, request=None):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"request": REQUEST if request is None else request, "estimated_cost_usd": 0.001}), encoding="utf-8")
    return plan


@pytest.mark.parametrize("cost", [0.100001, float("nan"), float("inf"), -0.01])
def test_current_cost_above_budget_or_invalid_blocks_billable_call(tmp_path, monkeypatch, cost):
    api = FakeAPI(cost)
    monkeypatch.setattr(data, "client", lambda: api)
    with pytest.raises(ValueError, match="Download blocked"):
        data.fetch(plan_file(tmp_path), 0.1, tmp_path / "download")
    assert api.cost_requests == [REQUEST]
    assert api.download_requests == []


@pytest.mark.parametrize("budget", [-1.0, float("nan"), float("inf")])
def test_invalid_authorized_budget_is_rejected_before_api_access(tmp_path, monkeypatch, budget):
    def unexpected_client():
        pytest.fail("An invalid budget must be rejected before creating a client")
    monkeypatch.setattr(data, "client", unexpected_client)
    with pytest.raises(ValueError, match="Budget"):
        data.fetch(plan_file(tmp_path), budget, tmp_path / "download")


def test_estimate_and_download_use_identical_parameters_and_reestimate_cost(tmp_path, monkeypatch):
    api = FakeAPI(0.05)
    monkeypatch.setattr(data, "client", lambda: api)
    config = tmp_path / "pilot.toml"
    config.write_text("\n".join(f"{key} = {json.dumps(value)}" for key, value in REQUEST.items() if key != "stype_in"), encoding="utf-8")
    plan = tmp_path / "plan.json"
    data.estimate(config, plan)
    # The estimate in the saved plan is deliberately older and cheaper.
    api.cost = 0.1
    result = data.fetch(plan, 0.1, tmp_path / "download")
    assert api.cost_requests == [REQUEST, REQUEST]
    assert len(api.download_requests) == 1
    downloaded = dict(api.download_requests[0])
    assert downloaded.pop("path") == tmp_path / "download" / "sample.dbn.zst"
    assert downloaded == REQUEST
    assert result["estimated_cost_usd"] == 0.1
    assert result["authorized_budget_usd"] == 0.1


def test_existing_download_is_not_overwritten_or_billed_again(tmp_path, monkeypatch):
    api = FakeAPI()
    monkeypatch.setattr(data, "client", lambda: api)
    output = tmp_path / "download"
    output.mkdir()
    (output / "sample.dbn.zst").write_bytes(b"existing sample")
    (output / "fetch.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="already has a download"):
        data.fetch(plan_file(tmp_path), 1.0, output)
    assert api.cost_requests == []
    assert api.download_requests == []
    assert (output / "sample.dbn.zst").read_bytes() == b"existing sample"


def test_partial_download_is_exported_from_local_cache_without_api_access(tmp_path, monkeypatch):
    def unexpected_client():
        pytest.fail("Recovering an existing DBN must not access the API")
    monkeypatch.setattr(data, "client", unexpected_client)
    output = tmp_path / "download"
    output.mkdir()
    cached_file = output / "sample.dbn.zst"
    cached_file.write_bytes(b"existing sample")
    (output / "request.json").write_text(json.dumps({
        "request": REQUEST, "max_cost_usd": 0.1, "estimated_cost_usd": 0.05,
    }), encoding="utf-8")
    opened = []
    store = SimpleNamespace(
        dataset="EQUS.MINI", schema="mbp-1",
        to_df=lambda **_: pd.DataFrame({"price": [100.0], "size": [10]}),
    )
    def from_file(path):
        opened.append(path)
        return store
    monkeypatch.setitem(sys.modules, "databento", SimpleNamespace(DBNStore=SimpleNamespace(from_file=from_file)))
    result = data.fetch(plan_file(tmp_path), 0.1, output)
    assert opened == [cached_file]
    assert result["recovered_from_local_cache"] is True
    assert result["estimated_cost_usd"] == 0.05
    assert (output / "sample.csv").exists()
    assert (output / "fetch.json").exists()
    assert cached_file.read_bytes() == b"existing sample"


@pytest.mark.parametrize("manifest", [
    None, "{malformed", "{}", json.dumps({"request": dict(REQUEST, symbols=["MSFT"])}),
])
def test_unverifiable_partial_download_is_refused_without_api_access(tmp_path, monkeypatch, manifest):
    def unexpected_client():
        pytest.fail("An unverifiable existing download must not trigger another API request")
    monkeypatch.setattr(data, "client", unexpected_client)
    output = tmp_path / "download"
    output.mkdir()
    cached_file = output / "sample.dbn.zst"
    cached_file.write_bytes(b"existing sample")
    if manifest is not None:
        (output / "request.json").write_text(manifest, encoding="utf-8")
    with pytest.raises(ValueError):
        data.fetch(plan_file(tmp_path), 0.1, output)
    assert cached_file.read_bytes() == b"existing sample"


@pytest.mark.parametrize("changes", [
    {"symbols": ["ALL_SYMBOLS"]}, {"stype_in": "instrument_id"},
    {"start": "2025-01-06T14:30:00"}, {"limit": 1000001},
])
def test_hand_edited_plan_is_revalidated_before_api_access(tmp_path, monkeypatch, changes):
    api = FakeAPI()
    monkeypatch.setattr(data, "client", lambda: api)
    with pytest.raises(ValueError):
        data.fetch(plan_file(tmp_path, dict(REQUEST, **changes)), 1.0, tmp_path / "download")
    assert api.cost_requests == []
    assert api.download_requests == []
