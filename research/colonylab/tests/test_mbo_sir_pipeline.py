"""Independent integration checks for common cohorts and model provenance."""
from __future__ import annotations

import copy
from concurrent.futures import Future
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from colonylab import mbo_sir_pipeline as pipeline
from colonylab import mbo_sir_data as acquisition


def minimal_config(tmp_path):
    return {"name": "offline-pipeline", "datasets": ["A", "B"], "symbol": "AAPL",
            "training_days": ["2026-06-01", "2026-06-02"], "held_out_days": ["2026-06-03"],
            "data_dir": str(tmp_path / "data"), "out_dir": str(tmp_path / "out")}


def panel(day, *, size=4):
    return {"x": np.arange(size * 2, dtype=float).reshape(size, 2),
            "infected": np.tile([[0.], [1.]], (size // 2, 1)),
            "shifted_infected": np.tile([[1.], [0.]], (size // 2, 1)),
            "legacy": np.arange(size, dtype=float).reshape(-1, 1), "y": np.tile([0, 1], size // 2),
            "names": ["depth", "market"], "source_names": ["B"], "legacy_names": ["other_100ms"],
            "time_ns": np.arange(size, dtype=np.int64) + 1000 * int(day[-2:]),
            "coverage": {"eligible_bins": size}}


def write_panel(tmp_path, day, dataset="A", *, size=4):
    path = tmp_path / f"{day}_{dataset}.pkl"
    value = panel(day, size=size)
    pipeline._dump(path, value)
    return {"day": day, "dataset": dataset, "path": str(path), "signature": day + dataset}, value


def test_training_concatenates_only_explicit_training_days_in_declared_column_order(tmp_path):
    first, value1 = write_panel(tmp_path, "2026-06-01")
    second, value2 = write_panel(tmp_path, "2026-06-02")
    heldout, _ = write_panel(tmp_path, "2026-06-03")
    target = pipeline._training([first, second, heldout], "A", ["2026-06-01", "2026-06-02"])
    assert len(target["y"]) == 8
    np.testing.assert_array_equal(target["x"], np.r_[value1["x"], value2["x"]])
    assert target["source_names"] == ["B"]


@pytest.mark.parametrize("corruption", ["missing", "duplicate_missing", "empty", "column_order"])
def test_training_requires_each_date_once_and_consistent_nonempty_columns(tmp_path, corruption):
    first, _ = write_panel(tmp_path, "2026-06-01")
    second, value2 = write_panel(tmp_path, "2026-06-02")
    panels = [first, second]
    if corruption == "missing":
        panels = [first]
    elif corruption == "duplicate_missing":
        panels = [first, dict(first)]
    elif corruption == "empty":
        for key in ("x", "infected", "shifted_infected", "legacy", "y", "time_ns"):
            value2[key] = value2[key][:0]
        pipeline._dump(Path(second["path"]), value2)
    else:
        value2["names"] = value2["names"][::-1]
        pipeline._dump(Path(second["path"]), value2)
    with pytest.raises(ValueError):
        pipeline._training(panels, "A", ["2026-06-01", "2026-06-02"])


def test_four_real_fits_share_training_labels_and_scale_from_training_only():
    rng = np.random.default_rng(118)
    n = 1800
    x = rng.normal(size=(n, 2))
    source = rng.binomial(1, .3, size=(n, 1))
    y = rng.binomial(1, -np.expm1(-.01 * (np.exp(1 + .4 * x[:, 0]) + 6 * source[:, 0])))
    train = {"x": x, "infected": source, "shifted_infected": np.roll(source, 300, axis=0),
             "legacy": rng.normal(size=(n, 1)), "y": y, "names": ["own_depth", "market_return"],
             "source_names": ["B"], "legacy_names": ["B_count_100ms"]}
    protocol = {"forecast_horizon_ms": 10, "optimization": {"hazard_ridge": 1}}
    models, metadata = pipeline._fit_target(train, protocol)
    assert set(models) == set(pipeline.MODELS)
    assert all(item["converged"] for item in metadata.values())
    assert all(item["training_rows"] == n and item["training_infections"] == int(y.sum()) for item in metadata.values())
    for name in ("independent_mbo_hazard", "sirs_network", "past_shifted_sirs"):
        np.testing.assert_allclose(models[name].control_mean_, x.mean(axis=0))
    np.testing.assert_allclose(models["logistic_network"][0].mean_[:2], x.mean(axis=0))
    assert np.all(models["sirs_network"].beta_rates_ >= 0)


def test_heldout_task_scores_preserve_same_cohort_and_labels_for_all_four_models(tmp_path):
    config = minimal_config(tmp_path)
    config["datasets"] = ["A"]
    panels = []
    for day in config["training_days"] + config["held_out_days"]:
        row, _ = write_panel(tmp_path, day, size=40)
        panels.append(row)
    protocol = {"forecast_horizon_ms": 10, "optimization": {"hazard_ridge": 1}}
    scores, edges, metadata = pipeline._fit_and_score(config, protocol, panels)
    assert len(scores) == 4
    assert set(scores.model) == set(pipeline.MODELS)
    assert scores.cohort_sha256.nunique() == 1
    assert scores.rows.tolist() == [40] * 4
    assert scores.positive_count.tolist() == [20] * 4
    assert set(metadata) == {"A"}
    assert all(value["converged"] for value in metadata["A"].values())
    assert edges.beta_per_second.ge(0).all()


def test_score_uses_paired_loss_sums_not_unweighted_task_means():
    score = pipeline._score(np.asarray([0, 1]), np.asarray([.25, .8]))
    assert score["rows"] == 2 and score["positive_count"] == 1
    assert score["log_loss_sum"] == pytest.approx(-np.log(.75) - np.log(.8))
    assert score["brier_sum"] == pytest.approx(.25 ** 2 + .2 ** 2)
    assert score["probability_clip_rows"] == 0


@pytest.mark.parametrize("y,p", [
    ([0], [np.nan]), ([0], [np.inf]), ([0], [-.1]), ([0], [1.1]),
    ([], []), ([0, 1], [.5]), ([0, np.nan], [.5, .5]), ([0, .5], [.5, .5]),
])
def test_score_rejects_nonfinite_nonbinary_or_unpaired_observations(y, p):
    with pytest.raises(ValueError):
        pipeline._score(np.asarray(y), np.asarray(p))


def test_score_probability_clipping_is_explicit_and_brier_uses_actual_probability():
    score = pipeline._score(np.asarray([0, 1]), np.asarray([0., 1.]))
    assert score["probability_clip_rows"] == 2
    assert score["brier_sum"] == 0
    assert np.isfinite(score["log_loss_sum"])


def test_fit_cache_requires_its_stored_signature_to_match_expected_provenance(tmp_path, monkeypatch):
    config = minimal_config(tmp_path)
    config["datasets"] = ["A"]
    signature = "expected-cache-signature"
    monkeypatch.setattr(pipeline, "_digest", lambda _: signature)
    monkeypatch.setattr(pipeline, "_code_hash", lambda *_: "code")
    path = Path(config["out_dir"]) / "fit_cache" / f"A_{signature[:20]}.pkl.gz"
    pipeline._dump(path, {"signature": "foreign-cache-signature", "models": {}, "metadata": {}})
    with pytest.raises(ValueError):
        pipeline._fit_and_score(config, {"frozen": True}, [])


def valid_replay(day, dataset):
    return {"day": day, "dataset": dataset,
            "validation": {"passed": True, "matched_keys": 1, "mismatch_keys": 0, "missing_keys": 0},
            "diagnostics": {"records_replayed": 10, "valid_grid_rows": 10, "initialization_source": "reset"}}


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "extra", "failed_validation", "incomplete"])
def test_analysis_gates_exact_replay_grid_before_calibration_or_fitting(tmp_path, monkeypatch, corruption):
    config = minimal_config(tmp_path)
    Path(config["out_dir"]).mkdir()
    tasks = [valid_replay(day, dataset) for day in config["training_days"] + config["held_out_days"]
             for dataset in config["datasets"]]
    replays = {"complete": True, "tasks": tasks}
    if corruption == "missing":
        tasks.pop()
    elif corruption == "duplicate":
        tasks.append(copy.deepcopy(tasks[0]))
    elif corruption == "extra":
        tasks.append(valid_replay("2026-06-04", "A"))
    elif corruption == "failed_validation":
        tasks[0]["validation"]["passed"] = False
    else:
        replays["complete"] = False
    monkeypatch.setattr(pipeline, "_build_panels", lambda *_: pytest.fail("Invalid grid reached training calibration"))
    with pytest.raises(ValueError):
        pipeline.analyze_study(config, {}, replays)


def test_receipt_checks_exact_success_marker_and_original_record_cap(tmp_path):
    config = {"data_dir": str(tmp_path / "data"), "out_dir": str(tmp_path / "out")}
    request = {"dataset": "XNAS.ITCH", "schema": "mbo", "symbols": ["AAPL"], "stype_in": "raw_symbol",
               "start": "2026-06-01T00:00:00+00:00", "end": "2026-06-01T20:00:00+00:00", "limit": 100}
    directory = acquisition.cache_directory(Path(config["data_dir"]), request)
    directory.mkdir(parents=True)
    raw = directory / "sample.dbn.zst"
    raw.write_bytes(b"raw")
    (directory / "fetch.json").write_text(json.dumps({"request": request, "rows": 1, "complete": True,
                                                        "file_size_bytes": 3, "record_limit_reached": False}))
    (directory / "stream_complete.json").write_text(json.dumps({"request": dict(request, limit=1)}))
    with pytest.raises(ValueError):
        pipeline._receipt(config, request)


def test_raw_byte_hash_is_content_sensitive(tmp_path):
    path = tmp_path / "raw"
    path.write_bytes(b"first")
    assert pipeline._file_hash(path) == hashlib.sha256(b"first").hexdigest()
    path.write_bytes(b"other")
    assert pipeline._file_hash(path) == hashlib.sha256(b"other").hexdigest()


@pytest.mark.parametrize("counter", pipeline.SEMANTIC_FAILURES)
def test_full_session_semantic_failure_gates_before_any_training_fit(tmp_path, monkeypatch, counter):
    config = minimal_config(tmp_path)
    out = Path(config["out_dir"])
    out.mkdir()
    tasks = [valid_replay(day, dataset) for day in config["training_days"] + config["held_out_days"]
             for dataset in config["datasets"]]
    tasks[0]["diagnostics"][counter] = 1
    monkeypatch.setattr(pipeline, "_build_panels", lambda *_: pytest.fail("Semantic failure reached feature calibration"))
    with pytest.raises(ValueError, match="semantics/clock audit failed"):
        pipeline.analyze_study(config, {}, {"complete": True, "tasks": tasks})
    errors = json.loads((out / "semantic_failures.json").read_text())
    assert not errors["passed"]
    assert errors["tasks"][0]["counts"][counter] == 1
    assert "diag_" + counter in (out / "book_audit.csv").read_text()


@pytest.mark.parametrize("counter", ["maybe_bad_book_flags", "bad_receive_timestamp_flags"])
def test_native_quality_flags_remain_explicit_barriers_and_diagnostics(tmp_path, monkeypatch, counter):
    config = minimal_config(tmp_path)
    out = Path(config["out_dir"])
    out.mkdir()
    tasks = [valid_replay(day, dataset) for day in config["training_days"] + config["held_out_days"]
             for dataset in config["datasets"]]
    tasks[0]["diagnostics"][counter] = 1
    class ReachedFeatureCalibration(RuntimeError):
        pass
    def reached(*_):
        raise ReachedFeatureCalibration
    monkeypatch.setattr(pipeline, "_build_panels", reached)
    with pytest.raises(ReachedFeatureCalibration):
        pipeline.analyze_study(config, {}, {"complete": True, "tasks": tasks})
    assert "diag_" + counter in (out / "book_audit.csv").read_text()
    assert not (out / "semantic_failures.json").exists()


def duplicate_history_panel(day, *, size=200, include_own_control=True):
    rng = np.random.default_rng(811 + int(day[-2:]))
    x = np.column_stack([rng.poisson(2, size), rng.normal(size=size)])
    legacy = rng.poisson(2, size=(size, 4)).astype(float)
    legacy[:, 2] = x[:, 0]
    return {"x": x, "names": ["own_infections_1s" if include_own_control else "unrelated_control", "depth"],
            "infected": rng.binomial(1, .25, size=(size, 1)),
            "shifted_infected": rng.binomial(1, .25, size=(size, 1)), "source_names": ["B"],
            "legacy": legacy, "legacy_names": ["own_infections_10ms", "own_infections_100ms",
                                                       "own_infections_1000ms", "other_B_infections_100ms"],
            "y": rng.binomial(1, .2, size), "time_ns": np.arange(size, dtype=np.int64),
            "coverage": {"eligible_bins": size}}


@pytest.mark.parametrize("include_own_control", [True, False])
def test_logistic_history_dedup_fits_one_own_one_second_column_and_freezes_indices(include_own_control):
    train = duplicate_history_panel("2026-06-01", include_own_control=include_own_control)
    protocol = {"forecast_horizon_ms": 10, "optimization": {"hazard_ridge": 1}}
    models, metadata = pipeline._fit_target(train, protocol)
    logistic_meta = metadata["logistic_network"]
    keep = [0, 1, 3] if include_own_control else [0, 1, 2, 3]
    assert logistic_meta["legacy_column_indices"] == keep
    scaler, logistic = models["logistic_network"]
    assert scaler.n_features_in_ == logistic.n_features_in_ == train["x"].shape[1] + len(keep)
    if include_own_control:
        assert "own_infections_1s" in logistic_meta["coefficients"]
        assert "own_infections_1000ms" not in logistic_meta["coefficients"]
    else:
        assert "own_infections_1000ms" in logistic_meta["coefficients"]


def test_heldout_logistic_prediction_uses_frozen_training_legacy_indices(tmp_path):
    config = minimal_config(tmp_path)
    config["datasets"] = ["A"]
    panels, values = [], {}
    for day in config["training_days"] + config["held_out_days"]:
        value = duplicate_history_panel(day)
        if day in config["held_out_days"]:
            # Extreme values in the excluded duplicate must never enter held-
            # out preprocessing or change the fitted column-selection list.
            value["legacy"][:, 2] = 1_000_000.
        path = tmp_path / f"dedup_{day}.pkl"
        pipeline._dump(path, value)
        panels.append({"dataset": "A", "day": day, "path": str(path), "signature": day})
        values[day] = value
    protocol = {"forecast_horizon_ms": 10, "optimization": {"hazard_ridge": 1}}
    scores, _, metadata = pipeline._fit_and_score(config, protocol, panels)
    assert metadata["A"]["logistic_network"]["legacy_column_indices"] == [0, 1, 3]
    cached = pipeline._load(next((Path(config["out_dir"]) / "fit_cache").glob("*.pkl.gz")))
    scaler, logistic = cached["models"]["logistic_network"]
    heldout = values[config["held_out_days"][0]]
    joined = np.column_stack([heldout["x"], heldout["legacy"][:, [0, 1, 3]]])
    probabilities = logistic.predict_proba(scaler.transform(joined))[:, 1]
    expected = pipeline._score(heldout["y"], probabilities)
    actual = scores.loc[scores.model.eq("logistic_network")].iloc[0]
    assert actual.log_loss_sum == pytest.approx(expected["log_loss_sum"])
    assert actual.brier_sum == pytest.approx(expected["brier_sum"])


@pytest.mark.parametrize("filename", ["checkpoint.pkl.gz", "checkpoint.pkl"])
def test_dump_compresses_and_load_roundtrips_numpy_payload_independent_of_extension(tmp_path, filename):
    path = tmp_path / filename
    value = {"array": np.zeros((1000, 17), dtype=np.float64), "names": ["depth", "imbalance"],
             "provenance": {"source": "raw-hash", "code": "code-hash"}}
    pipeline._dump(path, value)
    assert path.read_bytes()[:2] == b"\x1f\x8b"
    restored = pipeline._load(path)
    np.testing.assert_array_equal(restored["array"], value["array"])
    assert restored["array"].dtype == np.float64
    assert restored["names"] == value["names"] and restored["provenance"] == value["provenance"]
    assert path.stat().st_size < value["array"].nbytes / 10


@pytest.mark.parametrize("filename", ["legacy.pkl", "legacy.pkl.gz"])
def test_load_detects_legacy_plain_pickle_from_magic_bytes(tmp_path, filename):
    path = tmp_path / filename
    value = {"array": np.arange(80, dtype=np.int64).reshape(10, 8), "provenance": {"legacy": True}}
    with path.open("wb") as stream:
        pickle.dump(value, stream, protocol=5)
    assert path.read_bytes()[:2] != b"\x1f\x8b"
    restored = pipeline._load(path)
    np.testing.assert_array_equal(restored["array"], value["array"])
    assert restored["provenance"] == value["provenance"]


@pytest.mark.parametrize("origin", ["cached", "new"])
@pytest.mark.parametrize("failure", ["depth_validation", "semantics"])
def test_replay_rejects_bad_new_and_cached_results_before_proceeding(tmp_path, monkeypatch, origin, failure):
    config = minimal_config(tmp_path)
    config["datasets"] = ["A"]
    config["anchor_step_ms"] = 100
    day = config["training_days"][0]
    pair = {schema: {"dataset": "A", "schema": schema, "start": day + "T00:00:00+00:00"}
            for schema in ("mbo", "mbp-10")}
    monkeypatch.setattr(pipeline, "build_requests", lambda _: list(pair.values()))
    monkeypatch.setattr(pipeline, "_receipt", lambda *_: {"data_path": "unused", "rows": 10, "sha256": "raw-hash"})
    monkeypatch.setattr(pipeline, "_code_hash", lambda *_: "book-code")
    monkeypatch.setattr(pipeline, "_digest", lambda _: "replay-token")
    provenance = {"sources": {"mbo": "raw-hash", "mbp-10": "raw-hash"}, "code": "book-code",
                  "anchor_step_ms": 100, "requests": pair, "cache_version": 1}
    path = Path(config["out_dir"]) / "replay_cache" / f"{day}_A_replay-token.pkl.gz"
    result = valid_replay(day, "A")
    result.update(path=str(path), provenance=provenance)
    if failure == "depth_validation":
        result["validation"]["passed"] = False
    else:
        result["diagnostics"]["unmatched_cancels"] = 1
    submissions = []
    class ImmediatePool:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def submit(self, function, job):
            submissions.append(job)
            future = Future()
            future.set_result(result)
            return future
    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", lambda **_: ImmediatePool())
    if origin == "cached":
        pipeline._dump(path, result)
    with pytest.raises(ValueError):
        pipeline.replay_study(config, workers=1)
    assert len(submissions) == (0 if origin == "cached" else 1)
    if failure == "semantics":
        report = json.loads((Path(config["out_dir"]) / "early_semantic_failure.json").read_text())
        assert report["counts"]["unmatched_cancels"] == 1
        assert report["path"] == str(path)
