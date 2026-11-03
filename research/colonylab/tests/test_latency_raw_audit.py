"""Cross-check the standalone raw audit with explicit states and known lags."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from colonylab.latency import QuoteSeries, reaction_events, summarize_events
from colonylab.latency_data import normalize_frame


SPEC = importlib.util.spec_from_file_location("latency_raw_audit", Path(__file__).resolve().parents[1] / "scripts" / "audit_latency_raw.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

BASE = pd.Timestamp("2026-08-03T14:00:00Z").value


def series(times, bid, *, valid=None, end=1000):
    valid = np.ones(len(times), dtype=bool) if valid is None else np.asarray(valid, bool)
    return QuoteSeries(np.asarray(times, np.int64), np.asarray(bid, np.int64), np.asarray(bid, np.int64) + 10,
                       valid, np.cumsum(~valid), 0, end)


@pytest.mark.parametrize("side", ["bid", "ask"])
def test_independent_full_scan_and_summary_match_known_lags(side):
    leader = series([0, 100, 300, 700], [100, 101, 99, 102])
    follower = series([0, 150, 390, 750], [100, 101, 99, 102])
    direct = audit.independently_match(leader, follower, side, 200)
    assert direct["latency_ns"].tolist() == [50, 90, 50]
    produced = reaction_events(leader, follower, side, max_horizon_ns=200)
    for h in (40, 60, 100, 200):
        audit.compare_summary(audit.independently_summarize(direct, h), pd.Series(summarize_events(produced, h)), str(h))


def test_independent_scan_respects_quality_barriers_and_known_early_match():
    leader = series([0, 100, 170, 300], [100, 101, 0, 103], valid=[True, True, False, True])
    follower = series([0, 150, 210, 400], [100, 101, 103, 103])
    direct = audit.independently_match(leader, follower, "bid", 200)
    assert direct["latency_ns"].tolist() == [50]
    assert direct["observed_ns"].tolist() == [69]
    result = audit.independently_summarize(direct, 100)
    assert result["n_matched"] == 0
    assert result["known_matches_truncated"] == 1
    assert result["n_quality_truncated"] == 1


def test_independent_pandas_state_rebuild_agrees_at_every_boundary():
    offsets = np.array([101, 201, 201, 301, 2001, 2101, 2201, 2301], dtype=np.uint64)
    request = {"dataset": "XNAS.ITCH", "symbols": ["AAPL"],
               "start": pd.Timestamp(BASE, tz="UTC").isoformat(),
               "end": pd.Timestamp(BASE + 10_000, tz="UTC").isoformat()}
    frame = pd.DataFrame({"ts_recv": offsets + np.uint64(BASE), "ts_event": offsets + np.uint64(BASE),
                          "symbol": "AAPL", "bid_px_00": [100, 101, 102, 103, 104, 105, 106, 107],
                          "ask_px_00": [110] * 8, "bid_sz_00": [100] * 8, "ask_sz_00": [100] * 8,
                          "flags": [128, 128, 128, 128, 128, 4, 0, 128]})
    produced, _ = normalize_frame(frame, request, max_age_ms=.001)
    for clock in ("ts_recv", "ts_event"):
        expected = audit.independently_normalize(frame, request, "AAPL", clock, .001)
        assert audit.assert_state_equivalence(expected, produced[("AAPL", clock)], clock) > 5


def test_raw_auditor_reversal_interval_preserves_single_nanoseconds():
    offsets = np.array([101, 403, 202, 303, 501], dtype=np.uint64)
    request = {"dataset": "XNAS.ITCH", "symbols": ["AAPL"],
               "start": pd.Timestamp(BASE, tz="UTC").isoformat(),
               "end": pd.Timestamp(BASE + 10_000, tz="UTC").isoformat()}
    frame = pd.DataFrame({"ts_recv": np.arange(5, dtype=np.uint64) * 100 + np.uint64(BASE + 100),
                          "ts_event": offsets + np.uint64(BASE), "symbol": "AAPL",
                          "bid_px_00": np.arange(100, 105), "ask_px_00": [110] * 5,
                          "bid_sz_00": [100] * 5, "ask_sz_00": [100] * 5, "flags": [128] * 5})
    produced, _ = normalize_frame(frame, request)
    expected = audit.independently_normalize(frame, request, "AAPL", "ts_event", 1000)
    audit.assert_state_equivalence(expected, produced[("AAPL", "ts_event")], "nanosecond reversal")
    assert (expected.times_ns - BASE).tolist() == [101, 202, 303, 403]


def test_state_equivalence_detects_missing_quality_barrier():
    expected = series([0, 100, 200], [100, 0, 100], valid=[True, False, True])
    actual = series([0], [100])
    with pytest.raises(ValueError, match="reconstruction differs"):
        audit.assert_state_equivalence(expected, actual, "bad barrier")


def test_summary_comparison_detects_changed_count():
    with pytest.raises(ValueError, match="n_matched differs"):
        audit.compare_summary({"n_matched": 2}, pd.Series({"n_matched": 3}), "changed")


def test_prepared_raw_facts_roundtrip_is_content_bound(tmp_path):
    identity = {"sources": [{"sha256": "original"}], "code": "same", "protocol": "same"}
    facts = pd.DataFrame({"n_events": [3], "median_us": [np.nan]})
    audit.write_prepared(tmp_path, identity, facts, {"raw_requests_checked": 1})
    metadata, actual = audit.read_prepared(tmp_path, identity)
    assert metadata["status"] == "raw_facts_prepared"
    pd.testing.assert_frame_equal(actual, facts)


def test_prepared_raw_csv_tampering_is_rejected(tmp_path):
    identity = {"source_sha256": "original"}
    audit.write_prepared(tmp_path, identity, pd.DataFrame({"n_events": [3]}), {})
    path = tmp_path / audit.PREPARED_CSV
    path.write_text(path.read_text().replace("3", "4"), encoding="utf-8")
    with pytest.raises(ValueError, match="CSV hash differs"):
        audit.read_prepared(tmp_path, identity)


def test_prepared_raw_stale_protocol_or_source_identity_is_rejected(tmp_path):
    audit.write_prepared(tmp_path, {"source_sha256": "original", "age_ms": 250}, pd.DataFrame({"n_events": [3]}), {})
    with pytest.raises(ValueError, match="identity differs"):
        audit.read_prepared(tmp_path, {"source_sha256": "changed", "age_ms": 250})


def test_prepared_verification_checks_pooled_and_both_sides(tmp_path, monkeypatch):
    protocol = {"clocks": ["ts_recv"], "horizons_ms": [1]}
    provenance = {"studies": [{"name": "tiny", "configuration": {"symbols": ["AAPL"]}}]}
    identity = {"selected_dates": {"tiny": ["2026-08-03"]}, "sources": [{"sha256": "raw"}]}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    (tmp_path / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(audit, "prepared_identity", lambda *args: identity)
    lead, follow = series([0, 100, 300], [100, 101, 102], end=2_000_000), series([0, 150, 350], [100, 101, 102], end=2_000_000)
    events = audit.independently_match(lead, follow, "bid", 1_000_000)
    metrics = audit.independently_summarize(events, 1_000_000)
    facts, daily, sides = [], [], []
    for leader, follower in (("XNAS.ITCH", "BATS.PITCH"), ("BATS.PITCH", "XNAS.ITCH")):
        key = {"study": "tiny", "date": "2026-08-03", "symbol": "AAPL", "clock": "ts_recv",
               "leader": leader, "follower": follower, "horizon_ms": 1}
        daily.append({**key, "regime": "all", **metrics})
        for side in ("pooled", "bid", "ask"):
            facts.append({**key, "side": side, **metrics})
            if side != "pooled":
                sides.append({**key, "side": side, **metrics})
    pd.DataFrame(daily).to_csv(tmp_path / "daily_latency.csv", index=False)
    pd.DataFrame(sides).to_csv(tmp_path / "daily_latency_by_side.csv", index=False)
    audit.write_prepared(tmp_path, identity, pd.DataFrame(facts), {"raw_requests_checked": 1})
    result = audit.verify_prepared(tmp_path, root=tmp_path, progress=lambda *a, **k: None)
    assert result["full_pair_horizon_summaries_checked"] == 2
    assert result["full_bid_ask_summaries_checked"] == 4
    assert result["mismatches"] == 0
