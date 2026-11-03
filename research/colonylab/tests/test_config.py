from __future__ import annotations

import json

import pytest

from colonylab.config import read_request


def request_config(tmp_path, **changes):
    request = {
        "dataset": "EQUS.MINI", "schema": "mbp-1", "symbols": ["AAPL"],
        "start": "2025-01-06T14:30:00Z", "end": "2025-01-06T14:31:00Z", "limit": 10000,
    }
    request.update(changes)
    path = tmp_path / "pilot.toml"
    path.write_text("\n".join(f"{key} = {json.dumps(value)}" for key, value in request.items()), encoding="utf-8")
    return path


def test_explicit_bounded_request_gets_raw_symbol_input_type(tmp_path):
    request = read_request(request_config(tmp_path))
    assert request["stype_in"] == "raw_symbol"
    assert request["symbols"] == ["AAPL"]


@pytest.mark.parametrize("symbols", [["ALL_SYMBOLS"], ["all_symbols"], [" ALL_SYMBOLS "], ["ALL_SYMBOLS,"], [",ALL_SYMBOLS"], ["AAPL,MSFT"], [""], [], ["AAPL"] * 11])
def test_bulk_or_empty_symbol_requests_are_rejected(tmp_path, symbols):
    with pytest.raises(ValueError):
        read_request(request_config(tmp_path, symbols=symbols))


@pytest.mark.parametrize("changes", [
    {"start": "2025-01-06T14:30:00"},
    {"end": "2025-01-06T14:31:00"},
    {"end": "2025-01-06T14:30:00Z"},
    {"end": "2025-01-06T15:30:00.001Z"},
    {"limit": True}, {"limit": 0}, {"limit": 1000001}, {"schema": "mbo"},
])
def test_unbounded_ambiguous_or_unsupported_requests_are_rejected(tmp_path, changes):
    with pytest.raises(ValueError):
        read_request(request_config(tmp_path, **changes))
