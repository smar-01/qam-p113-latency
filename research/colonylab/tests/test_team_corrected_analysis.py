"""Integration checks for the explicit ColonyLab quality adapter."""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "comparisons/qam_p113_corrected/p113"
NAME = "team_corrected_quality_test"
SPEC = importlib.util.spec_from_file_location(NAME, PACKAGE / "__init__.py", submodule_search_locations=[str(PACKAGE)])
module = importlib.util.module_from_spec(SPEC)
sys.modules[NAME] = module
SPEC.loader.exec_module(module)
quality = importlib.import_module(NAME + ".quality")


def raw_quotes(bad=False):
    rows = []
    for venue, times, prices, flags in (
        ("XNAS", [0, 100, 200, 500], [100., 100.01, 100.02, 100.02], [128] * 4),
        ("BATS", [0, 120, 150, 250, 500], [100., 100., 100.01, 100.02, 100.02], [128, 132 if bad else 128, 128, 128, 128]),
    ):
        for timestamp, price, flag in zip(times, prices, flags):
            rows.append({"ts_recv": timestamp, "ts_event": timestamp, "bid": price, "ask": price + .02,
                         "bid_raw": int(round(price * 1e9)), "ask_raw": int(round((price + .02) * 1e9)),
                         "bid_sz": 100, "ask_sz": 100, "venue": venue, "symbol": "TEST", "flags": flag})
    result = pd.DataFrame(rows)
    result.attrs.update(requested_start_ns=0, requested_end_ns=1000, price_changes_only=False)
    return result


@pytest.mark.parametrize("clock", ["ts_recv", "ts_event"])
def test_adapter_recovers_exact_known_lag_on_complete_observed_cohort(clock):
    prepared, facts = quality.prepare(raw_quotes(), clock)
    events = quality.pair_latencies(prepared, horizon_ns=60)
    summary = quality.summarize(events, horizon_ns=60)
    row = summary.loc[summary.leader.eq("XNAS") & summary.follower.eq("BATS")].iloc[0]
    assert row.n_events == row.n_eligible == row.n_matched == 4
    assert row.median_us == pytest.approx(.05)
    assert row.match_rate == 1
    assert set(facts.clock) == {clock}


def test_adapter_quality_gap_is_not_counted_as_slow_or_successful_match():
    prepared, _ = quality.prepare(raw_quotes(bad=True), "ts_recv")
    events = quality.pair_latencies(prepared, horizon_ns=60)
    summary = quality.summarize(events, horizon_ns=60)
    row = summary.loc[summary.leader.eq("XNAS") & summary.follower.eq("BATS")].iloc[0]
    assert row.n_events == row.n_behind == 4
    assert row.n_quality_truncated == 2
    assert row.n_eligible == row.n_matched == 2
    assert row.median_us == pytest.approx(.05)


def test_adapter_refuses_compressed_states_for_freshness():
    quotes = raw_quotes()
    quotes.attrs["price_changes_only"] = True
    with pytest.raises(ValueError, match="raw size/state updates"):
        quality.prepare(quotes, "ts_recv")


def test_adapter_leadlag_and_event_study_emit_finite_supported_values():
    prepared, _ = quality.prepare(raw_quotes(), "ts_recv")
    curves, summary = quality.pair_leadlag(prepared)
    assert {"symbol", "x", "y", "corr", "lag_us"} <= set(curves)
    assert len(summary) == 1
    study = quality.event_study(prepared, "XNAS", "TEST", np.array([0, 60]))
    follower = study.loc[study.venue.eq("BATS")]
    assert follower.n.eq(2).all()
    assert follower.share.tolist() == pytest.approx([0, 1])
