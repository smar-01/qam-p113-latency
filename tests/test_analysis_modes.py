"""The standalone driver separates numerical fixes from support policies."""
import builtins
import json

import numpy as np
import pandas as pd
import pytest

import analyze


def raw_quotes():
    rows = []
    for venue, times, levels, delay in (
        ("XNAS", [0, 100, 200, 500], [0, 1, 2, 2], 0),
        ("BATS", [0, 120, 150, 250, 500], [0, 0, 1, 2, 2], 20),
    ):
        for timestamp, level in zip(times, levels):
            bid = 100_000_000_000 + level * 10_000_000
            rows.append({"venue": venue, "symbol": "SYN", "ts_event": timestamp,
                         "ts_recv": timestamp + delay, "bid_raw": bid, "ask_raw": bid + 20_000_000,
                         "bid": bid / 1e9, "ask": (bid + 20_000_000) / 1e9,
                         "bid_sz": 100, "ask_sz": 100, "flags": 128, "valid": True})
    result = pd.DataFrame(rows)
    result.attrs.update(requested_start_ns=0, requested_end_ns=1000, price_changes_only=False)
    return result


def test_default_and_complete_modes_are_standalone_and_preserve_known_bid_ask_lags(tmp_path, monkeypatch):
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name == "colonylab" or name.startswith("colonylab."):
            pytest.fail("Standalone numerical corrections must not import ColonyLab")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    for name in ("heatmap", "distributions", "regime_bars", "leadlag", "event_study"):
        monkeypatch.setattr(analyze.figures, name, lambda *args, **kwargs: None)
    observed = []
    original_volatility = analyze.regimes.realized_vol
    def volatility(quotes, **kwargs):
        observed.append((kwargs["clock"], len(quotes), quotes.attrs["price_changes_only"]))
        return original_volatility(quotes, **kwargs)
    monkeypatch.setattr(analyze.regimes, "realized_vol", volatility)
    raw = raw_quotes()
    for support in ("legacy", "complete"):
        out = tmp_path / support
        mode = analyze.run(raw, out, "XNAS", ["ts_recv", "ts_event"], 10_000_000, .75, 0., event_support=support)
        assert mode["mode"] == "numerical_only" and not mode["quality_aware_reaction_latency"]
        assert mode["event_study_support"] == support
        assert mode["latency_max_quote_age_ms"] is None
        assert json.loads((out / "analysis_mode.json").read_text()) == mode
        for clock, expected in (("ts_event", .05), ("ts_recv", .07)):
            summary = pd.read_csv(out / f"latency_summary_{clock}.csv")
            matched = summary.loc[summary.leader.eq("XNAS") & summary.follower.eq("BATS")].iloc[0]
            assert matched.n_matched == 4 and matched.median_us == pytest.approx(expected)
            assert (out / f"realized_vol_{clock}.csv").is_file()
    assert observed == [("ts_recv", 9, False), ("ts_event", 9, False)] * 2
    legacy = pd.read_csv(tmp_path / "legacy/event_study_XNAS_SYN_ts_recv.csv")
    complete = pd.read_csv(tmp_path / "complete/event_study_XNAS_SYN_ts_recv.csv")
    assert legacy.loc[legacy.offset_us.eq(10_000), "n"].eq(2).all()
    assert complete.n.eq(0).all()


def test_price_compression_cannot_silently_remove_freshness_updates(tmp_path):
    raw = analyze.bbo_changes(raw_quotes())
    with pytest.raises(ValueError, match="raw size/state"):
        analyze.run(raw, tmp_path, "XNAS", ["ts_recv"], 10_000_000, .75, 0.)
