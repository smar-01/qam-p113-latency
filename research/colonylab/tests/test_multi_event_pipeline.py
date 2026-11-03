"""Fixed cohorts, native prices, clock identity and event-window support."""
from __future__ import annotations

from concurrent.futures import Future
from itertools import permutations
import json
from pathlib import Path
from types import SimpleNamespace

import databento as db
import numpy as np
import pandas as pd
import pytest

from colonylab import multi_event_pipeline as pipeline
from colonylab.event_midpoints import NativeMidpoints, match_events


@pytest.fixture
def protocol():
    return {"datasets": ["XNAS.ITCH", "ARCX.PILLAR", "BATS.PITCH", "EDGX.PITCH"],
        "windows": [{"name": "pre30", "start_minute": -1, "end_minute": 0},
                    {"name": "post30", "start_minute": 0, "end_minute": 1}],
        "clocks": ["ts_recv", "ts_event"], "quote_age_ms": [5000], "horizons_ms": [10],
        "minimum_coverage_fraction": .9, "minimum_matched_events": 20,
        "primary_shock_threshold_bp": 1, "shock_thresholds_bp": [1], "shock_cluster_separation_ms": 100,
        "measurement": "native_mbp1_midpoint_first_passage"}


@pytest.fixture
def trial():
    return {"trial_id": "wwdc", "bundle_id": "wwdc", "event_id": "wwdc", "event_type": "wwdc",
        "role": "event", "control_for": None, "year": 2024, "day": "2024-06-10", "session": "RTH",
        "anchor_utc": "2024-06-10T17:00:00Z"}


def metric_frames(protocol):
    pairs, venues = [], []
    for w in protocol["windows"]:
        for n, (leader, follower) in enumerate(permutations(protocol["datasets"], 2), 1):
            pairs.append({"window": w["name"], "horizon_ms": 10, "leader": leader, "follower": follower,
                          "n_eligible": 100, "n_matched": 20, "median_us": n * 10.})
        for dataset in protocol["datasets"]:
            venues.append({"window": w["name"], "venue": dataset, "shock_threshold_bp": 1,
                           "coverage_fraction": 1., "clustered_shocks_per_minute": 2.})
    return pd.DataFrame(pairs), pd.DataFrame(venues)


def aggregate(pairs, venues, trial, protocol):
    return pd.DataFrame(pipeline.aggregate_metrics(pairs, venues, trial, protocol, "ts_event", 5000))


def test_all_twelve_directed_pairs_and_four_venues_form_fixed_summary(protocol, trial):
    pairs, venues = metric_frames(protocol)
    result = aggregate(pairs, venues, trial, protocol)
    assert len(result) == 6 and result.complete.all()
    assert result.expected_pairs.eq(12).all()
    assert result.clock.eq("ts_event").all() and result.age_ms.eq(5000).all()
    assert result.loc[result.metric.eq("latency_us"), "value"].eq(65.).all()
    assert result.loc[result.metric.eq("match_rate"), "value"].eq(.2).all()
    assert result.loc[result.metric.eq("shock_rate"), "value"].eq(2.).all()


def test_missing_pair_cannot_shrink_successful_latency_or_match_rate_universe(protocol, trial):
    pairs, venues = metric_frames(protocol)
    pairs = pairs.drop(pairs.index[0])
    result = aggregate(pairs, venues, trial, protocol)
    pre = result.loc[result.window.eq("pre30")]
    invalid = pre.loc[pre.metric.isin(["latency_us", "match_rate"])]
    assert not invalid.complete.any() and invalid.value.isna().all()
    assert pre.loc[pre.metric.eq("shock_rate"), "complete"].all()
    assert result.loc[result.window.eq("post30"), "complete"].all()


@pytest.mark.parametrize("identity", ["pair", "venue"])
def test_duplicate_rows_cannot_substitute_for_missing_fixed_identity(protocol, trial, identity):
    pairs, venues = metric_frames(protocol)
    if identity == "pair":
        pairs.loc[1, ["leader", "follower"]] = pairs.loc[0, ["leader", "follower"]].to_numpy()
    else:
        venues.loc[1, "venue"] = venues.loc[0, "venue"]
    with pytest.raises(ValueError, match="[Dd]uplicate|[Uu]nique|identit"):
        aggregate(pairs, venues, trial, protocol)


def test_sparse_successes_do_not_hide_observed_zero_match_probability(protocol, trial):
    pairs, venues = metric_frames(protocol)
    pairs.loc[0, ["n_matched", "median_us"]] = [0, np.nan]
    result = aggregate(pairs, venues, trial, protocol)
    pre = result.loc[result.window.eq("pre30")].set_index("metric")
    assert not pre.loc["latency_us", "complete"] and pd.isna(pre.loc["latency_us", "value"])
    assert pre.loc["match_rate", "complete"]
    assert pre.loc["match_rate", "value"] == pytest.approx(220 / 1200)


def test_pair_without_complete_followup_events_is_not_a_zero_match_rate(protocol, trial):
    pairs, venues = metric_frames(protocol)
    pairs.loc[0, ["n_eligible", "n_matched", "median_us"]] = [0, 0, np.nan]
    result = aggregate(pairs, venues, trial, protocol)
    row = result.loc[result.window.eq("pre30") & result.metric.eq("match_rate")].iloc[0]
    assert not row.complete and pd.isna(row.value)


def test_one_uncovered_venue_suppresses_every_metric_in_that_window(protocol, trial):
    pairs, venues = metric_frames(protocol)
    venues.loc[0, "coverage_fraction"] = .89
    result = aggregate(pairs, venues, trial, protocol)
    pre = result.loc[result.window.eq("pre30")]
    assert not pre.complete.any() and pre.value.isna().all()
    assert pre.reason.str.contains("coverage").all()
    assert result.loc[result.window.eq("post30"), "complete"].all()


def test_nonfinite_coverage_cannot_pass_the_coverage_threshold(protocol, trial):
    pairs, venues = metric_frames(protocol)
    venues.loc[0, "coverage_fraction"] = np.nan
    result = aggregate(pairs, venues, trial, protocol)
    assert not result.loc[result.window.eq("pre30"), "complete"].any()


def test_unavailable_or_empty_source_is_quarantined_for_all_clocks_and_ages(protocol, monkeypatch):
    protocol = {**protocol, "quote_age_ms": [250, 1000, 5000]}
    group = {"start": "2024-06-10T16:20:00Z", "end": "2024-06-10T17:40:00Z"}
    entries = [{"request": {"dataset": dataset, "schema": "mbp-1", "symbols": ["AAPL"],
                "stype_in": "raw_symbol", "start": group["start"], "end": group["end"], "limit": 1000},
                "complete": False, "status": "unavailable",
                "availability_reason": "degraded_provider_day", "rows": 0} for dataset in protocol["datasets"]]
    monkeypatch.setattr(db.DBNStore, "from_file", lambda _: pytest.fail("Unavailable sources must not decode"))
    paths, facts = pipeline.decode_sources(entries, group, protocol)
    assert len(paths) == len(facts) == 4 * 3 * 2
    assert all(len(path.times_ns) == 0 for path in paths.values())
    assert all(not fact["source_available"] and "degraded_provider_day" in fact["reason"] for fact in facts)
    sample = next(iter(paths.values()))
    assert sample.start_ns == pd.Timestamp(group["start"]).value and sample.end_ns == pd.Timestamp(group["end"]).value


@pytest.fixture
def source_data(tmp_path, protocol, monkeypatch):
    group = {"group_id": "g", "start": "2024-06-10T16:59:00+00:00", "end": "2024-06-10T17:01:00+00:00",
             "trial_ids": ["wwdc"]}
    start, end = pd.Timestamp(group["start"]).value, pd.Timestamp(group["end"]).value
    entries, stores = [], {}
    for n, dataset in enumerate(protocol["datasets"]):
        raw = tmp_path / (dataset + ".dbn.zst")
        raw.write_bytes(("verified-native-" + dataset).encode())
        request = {"dataset": dataset, "schema": "mbp-1", "symbols": ["AAPL"], "stype_in": "raw_symbol",
                   "start": group["start"], "end": group["end"], "limit": 1000}
        receives = start + np.arange(1, 120, dtype=np.int64) * 1_000_000_000 + n * 1000
        bid = 200_000_000_000 + (np.arange(119, dtype=np.int64) % 2) * 10_000_000
        frame = pd.DataFrame({"ts_recv": receives, "ts_event": receives - 500, "symbol": "AAPL", "flags": 128,
                              "bid_px_00": bid, "ask_px_00": bid + 10_000_001,
                              "bid_sz_00": 100, "ask_sz_00": 200}).set_index("ts_recv")
        metadata = SimpleNamespace(dataset=dataset, schema="mbp-1", start=start, end=end, limit=1000,
                                   stype_in="raw_symbol", symbols=["AAPL"], partial=[], not_found=[])
        def to_df(frame=frame, **kwargs):
            assert kwargs == {"price_type": "fixed", "pretty_ts": False, "map_symbols": True}
            return frame.copy()
        stores[str(raw)] = SimpleNamespace(metadata=metadata, to_df=to_df)
        entries.append({"request": request, "complete": True, "analyzable": True, "data_path": str(raw),
                        "rows": len(frame), "file_size_bytes": raw.stat().st_size, "content_sha256": pipeline.file_sha(raw)})
    monkeypatch.setattr(db.DBNStore, "from_file", lambda raw: stores[str(raw)])
    return group, entries


def test_native_twice_midpoint_preserves_odd_integer_nanodollar_without_float_roundtrip(protocol, source_data):
    group, entries = source_data
    paths, quality = pipeline.decode_sources(entries, group, protocol)
    receive = paths[(protocol["datasets"][0], 5000, "ts_recv")]
    event = paths[(protocol["datasets"][0], 5000, "ts_event")]
    assert receive.price2.dtype == np.int64
    assert receive.price2[receive.valid][0] == 400_010_000_001
    assert (receive.price2[receive.valid] % 2 == 1).all()
    assert event.times_ns[0] == receive.times_ns[0] - 500
    assert {row["clock"] for row in quality} == {"ts_recv", "ts_event"}


def test_changed_raw_source_is_rejected_before_decode(protocol, source_data):
    group, entries = source_data
    Path(entries[0]["data_path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="bytes changed"):
        pipeline.decode_sources(entries, group, protocol)


def test_process_group_preserves_clock_labels_and_emits_twelve_pairs_per_window(tmp_path, protocol, trial, source_data):
    group, entries = source_data
    protocol = {**protocol, "trials": [trial]}
    directory = tmp_path / "analysis"
    directory.mkdir()
    result = pipeline.process_group({"directory": str(directory), "protocol": protocol,
                                     "group": group, "entries": entries, "identity": {"test": "native"}})
    assert not result["reused"]
    pairs = pd.read_csv(directory / "pair_metrics.csv")
    metrics = pd.read_csv(directory / "metrics.csv")
    assert set(pairs.clock) == set(metrics.clock) == {"ts_recv", "ts_event"}
    assert pairs.groupby(["clock", "window", "horizon_ms"]).size().eq(12).all()
    assert metrics.groupby(["clock", "window", "horizon_ms"]).size().eq(3).all()
    assert metrics.expected_pairs.eq(12).all()
    assert metrics.loc[metrics.clock.eq("ts_event"), "clock"].eq("ts_event").all()


def test_each_window_excludes_matches_and_followup_at_or_after_its_end():
    source = NativeMidpoints(np.array([1, 10], np.int64), np.array([100, 110], np.int64),
                             np.array([True, True]), np.array([0, 0], np.int64), 1, 100)
    follower = NativeMidpoints(np.array([1, 20], np.int64), np.array([100, 110], np.int64),
                               np.array([True, True]), np.array([0, 0], np.int64), 1, 100)
    all_events = match_events(source, follower, max_horizon_ns=30)
    assert all_events.latency_ns.tolist() == [10]
    pre = pipeline.window_events(all_events, 1, 20)
    post = pipeline.window_events(all_events, 20, 100)
    assert pre.latency_ns.tolist() == [-1] and pre.observed_ns.tolist() == [9]
    assert pre.terminal_reason.tolist() == ["end"] and post.empty
    # Even an earlier observed match is excluded from a primary full-horizon
    # statistic when the window has insufficient remaining follow-up.
    early_match = pipeline.window_events(all_events, 1, 21)
    summary = pipeline.summarize_events(early_match, 20)
    assert summary["n_eligible"] == 0 and summary["n_right_truncated"] == 1


@pytest.fixture
def checkpoint_study(tmp_path, protocol, trial, monkeypatch):
    """Run real registry/task selection with an in-process checkpoint worker."""
    groups = [{"group_id": "g1", "start": "2024-06-10T16:24:00+00:00", "end": "2024-06-10T17:36:00+00:00", "trial_ids": [trial["trial_id"]]},
              {"group_id": "g2", "start": "2024-06-17T16:24:00+00:00", "end": "2024-06-17T17:36:00+00:00", "trial_ids": []}]
    frozen = {**protocol, "source_groups": groups, "trials": [trial], "record_limit_per_request": 1000}
    entries = [{"request": {"dataset": venue, "schema": "mbp-1", "symbols": ["AAPL"], "stype_in": "raw_symbol",
                            "start": group["start"], "end": group["end"], "limit": 1000},
                "complete": True, "analyzable": True, "status": "complete", "rows": 3,
                "content_sha256": "test-source-identity"}
               for group in groups for venue in protocol["datasets"]]
    out = tmp_path / "checkpoints"
    out.mkdir()
    (out / "protocol.json").write_text(json.dumps(frozen), encoding="utf-8")
    calls, executor_starts = [], []
    def checkpoint(task):
        calls.append(task["group"]["group_id"])
        (Path(task["directory"]) / "test_checkpoint.json").write_text(json.dumps(task["identity"]), encoding="utf-8")
        return {"group_id": task["group"]["group_id"], "reused": False}
    class InlineExecutor:
        def __init__(self, **kwargs):
            executor_starts.append(kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def submit(self, function, task):
            future = Future()
            try:
                future.set_result(function(task))
            except Exception as exc:
                future.set_exception(exc)
            return future
    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr(pipeline, "process_group", checkpoint)
    monkeypatch.setattr(pipeline, "summarize_multi_events", lambda *_: pytest.fail("Partial acquisition cannot publish event inference"))
    def run(selected, *, done=False, available_only=True):
        manifest = {"requests": selected, "complete": done, "acquisition_complete_for_available_requests": done}
        (out / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return pipeline.analyze(Path(__file__).resolve().parents[1], out, available_only=available_only,
                                progress=lambda *args, **kwargs: None)
    return SimpleNamespace(out=out, run=run, entries=entries, calls=calls, executor_starts=executor_starts)


def assert_no_published_results(out):
    for name in (*pipeline.DAY_FILES, "summary.json", "analysis_provenance.json", "event_changes.csv", "event_excess.csv", "class_summary.csv", "class_comparisons.csv"):
        assert not (out / name).exists(), name


@pytest.mark.parametrize("received", [0, 2])
def test_empty_or_incomplete_source_group_produces_no_summaries(checkpoint_study, received):
    s = checkpoint_study
    result = s.run(s.entries[:received])
    assert result == {"status": "partial_checkpoints_only", "completed_source_windows": 0,
                      "planned_source_windows": 2, "event_results_published": False}
    assert not s.calls
    assert_no_published_results(s.out)


def test_only_complete_groups_checkpoint_while_partial_acquisition_never_publishes(checkpoint_study):
    s = checkpoint_study
    result = s.run(s.entries[:5])  # Four venues in g1; one of four in g2.
    assert result["status"] == "partial_checkpoints_only" and result["completed_source_windows"] == 1
    assert result["event_results_published"] is False and s.calls == ["g1"]
    assert (s.out / "groups/g1/test_checkpoint.json").is_file()
    assert not (s.out / "groups/g2").exists()
    assert_no_published_results(s.out)


def test_unavailable_source_remains_explicit_quarantine_in_partial_checkpoint(checkpoint_study):
    s = checkpoint_study
    selected = [dict(row) for row in s.entries[:4]]
    selected[0] = {**selected[0], "complete": False, "analyzable": False, "status": "unavailable", "availability_reason": "degraded"}
    result = s.run(selected)
    assert result["completed_source_windows"] == 1 and s.calls == ["g1"]
    checkpoint = json.loads((s.out / "groups/g1/test_checkpoint.json").read_text())
    assert any(row["status"] == "unavailable" and not row["complete"] for row in checkpoint["sources"])
    assert_no_published_results(s.out)


def test_failed_available_source_is_not_checkpointed_as_a_complete_group(checkpoint_study):
    s = checkpoint_study
    selected = [dict(row) for row in s.entries[:4]]
    selected[0] = {**selected[0], "complete": False, "analyzable": False, "status": "failed"}
    result = s.run(selected)
    assert result["completed_source_windows"] == 0 and not s.calls
    assert_no_published_results(s.out)


def test_partial_acquisition_without_opt_in_stops_before_work(checkpoint_study):
    s = checkpoint_study
    with pytest.raises(ValueError, match="Finish every available"):
        s.run(s.entries[:4], available_only=False)
    assert not s.calls and not s.executor_starts
    assert_no_published_results(s.out)


@pytest.mark.parametrize("received", [0, 4, 7])
def test_full_acquisition_claim_missing_any_expected_request_stops_before_work(checkpoint_study, received):
    s = checkpoint_study
    with pytest.raises(ValueError, match="exact frozen source grid"):
        s.run(s.entries[:received], done=True)
    assert not s.calls and not s.executor_starts
    assert_no_published_results(s.out)


@pytest.mark.parametrize("corruption", ["duplicate", "foreign_venue", "foreign_window", "wrong_limit"])
def test_duplicate_or_foreign_request_cannot_enter_partial_checkpoint_registry(checkpoint_study, corruption):
    s = checkpoint_study
    selected = [dict(row, request=dict(row["request"])) for row in s.entries[:4]]
    if corruption == "duplicate":
        selected.append(selected[0])
    elif corruption == "foreign_venue":
        selected[0]["request"]["dataset"] = "XNYS.PILLAR"
    elif corruption == "foreign_window":
        selected[0]["request"]["start"] = "2024-06-10T16:25:00+00:00"
    else:
        selected[0]["request"]["limit"] = 999
    with pytest.raises(ValueError, match="exact frozen source grid"):
        s.run(selected)
    assert not s.calls and not s.executor_starts
    assert_no_published_results(s.out)
