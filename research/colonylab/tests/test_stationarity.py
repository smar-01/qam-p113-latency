"""Contiguous observation boundaries and known-process stationarity checks."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from colonylab.event_midpoints import NativeMidpoints
from colonylab.stationarity import adjust_adf_holm, sample_contiguous, test_pairs as run_tests


def path(times, prices, *, valid=None, segments=None, start=0, end=100):
    n = len(times)
    return NativeMidpoints(np.asarray(times, np.int64), np.asarray(prices, np.int64),
                           np.ones(n, bool) if valid is None else np.asarray(valid, bool),
                           np.zeros(n, np.int64) if segments is None else np.asarray(segments, np.int64), start, end)


def ar1(n=1600, seed=42):
    noise = np.random.default_rng(seed).normal(size=n)
    values = np.zeros(n)
    for i in range(1, n):
        values[i] = .4 * values[i - 1] + noise[i]
    return values


def prices(logs):
    return np.rint(200_000_000_000 * np.exp(np.asarray(logs))).astype(np.int64)


def row(rows, transformation="log_level", regression="c"):
    return next(item for item in rows if item["transformation"] == transformation and item["regression"] == regression)


def test_strict_prior_half_open_grid_and_output_copies():
    source = path([0, 10, 20, 30], [100, 110, 120, 130], end=40)
    result = sample_contiguous(source, 10, 40, 10)
    assert result["times_ns"].tolist() == [10, 20, 30]
    assert result["price2"].tolist() == [100, 110, 120]
    assert result["total_grid_points"] == result["longest_points"] == 3
    assert result["run_start_ns"] == 10 and result["run_end_ns"] == 30
    result["price2"][0] = 999
    assert source.price2[0] == 100


def test_sampling_clips_true_support_and_never_uses_future_initial_state():
    source = path([0, 15], [100, 110], start=10, end=25)
    result = sample_contiguous(source, 0, 40, 10)
    assert result["times_ns"].tolist() == [10, 20]
    assert result["price2"].tolist() == [100, 110]
    assert result["valid_grid_points"] == 2 and result["total_grid_points"] == 4
    assert result["fraction_window"] == result["valid_fraction_window"] == .5
    no_seed = path([15], [100], start=10, end=40)
    assert sample_contiguous(no_seed, 10, 40, 10)["times_ns"].tolist() == [20, 30]


def test_invalid_interval_between_grid_points_splits_even_with_reused_segment_id():
    source = path([0, 15, 16], [100, 0, 100], valid=[True, False, True], end=60)
    result = sample_contiguous(source, 10, 60, 10)
    assert result["valid_grid_points"] == 5 and result["run_count"] == 2
    assert result["times_ns"].tolist() == [20, 30, 40, 50]
    assert result["fraction_window"] == .8 and result["valid_fraction_window"] == 1


def test_hidden_next_barrier_and_segment_transition_are_both_respected(monkeypatch):
    source = path([0], [100], end=60)
    monkeypatch.setattr(source, "next_barrier", lambda queries: np.where(np.asarray(queries) < 15, 15, 60))
    assert sample_contiguous(source, 10, 60, 10)["times_ns"].tolist() == [20, 30, 40, 50]
    changed = path([0, 15], [100, 110], segments=[0, 1], end=60)
    assert sample_contiguous(changed, 10, 60, 10)["times_ns"].tolist() == [20, 30, 40, 50]


def test_barrier_at_grid_timestamp_enters_only_the_following_strict_prior_sample():
    source = path([0, 20, 21], [100, 0, 100], valid=[True, False, True], end=60)
    result = sample_contiguous(source, 10, 60, 10)
    assert result["times_ns"].tolist() == [30, 40, 50]
    assert result["run_count"] == 2


def test_equal_longest_runs_choose_earliest_without_concatenating_them():
    source = path([0, 20, 21, 40, 41], [100, 0, 100, 0, 100],
                  valid=[True, False, True, False, True], end=70)
    result = sample_contiguous(source, 10, 70, 10)
    assert result["run_count"] == 3 and result["longest_points"] == 2
    assert result["times_ns"].tolist() == [10, 20]
    assert result["fraction_window"] == 1 / 3


def test_empty_and_wholly_invalid_paths_have_no_invented_run():
    for source in (path([], [], end=50), path([0], [0], valid=[False], end=50)):
        result = sample_contiguous(source, 0, 50, 10)
        assert result["run_count"] == result["longest_points"] == result["valid_grid_points"] == 0
        assert result["run_start_ns"] is result["run_end_ns"] is None
        assert result["price2"].dtype == np.int64 and result["price2"].size == 0


def test_stationary_ar_process_is_compatible_and_preserves_input_and_pvalue_bounds():
    original = prices(ar1() * .0001)
    before = original.copy()
    results = run_tests(original)
    assert [(item["transformation"], item["regression"]) for item in results] == [("log_level", "c"), ("log_level", "ct"), ("log_return", "c")]
    assert all(item["status"] == "tested" and item["classification"] == "stationarity_compatible" for item in results)
    assert all(item["adf_maxlag"] >= item["adf_lags"] and item["adf_nobs"] > 0 for item in results)
    assert row(results)["kpss_pvalue_relation"] == ">="
    assert any(warning["test"] == "kpss" for warning in row(results)["warnings"])
    assert np.array_equal(original, before)
    json.dumps(results, allow_nan=False)


def test_random_walk_levels_and_returns_produce_different_recommendation_evidence():
    log_walk = np.cumsum(np.random.default_rng(123).normal(0, .0001, 1600))
    results = run_tests(prices(log_walk))
    assert row(results)["classification"] == "unit_root_compatible"
    assert row(results)["kpss_pvalue_relation"] == "<="
    assert row(results, "log_return")["classification"] == "stationarity_compatible"


def test_deterministic_trend_specification_is_not_selected_by_a_winning_pvalue():
    logs = np.arange(1600) * .00001 + ar1(seed=1234) * .00005
    results = run_tests(prices(logs))
    assert row(results, regression="c")["classification"] == "unit_root_compatible"
    assert row(results, regression="ct")["classification"] == "stationarity_compatible"
    assert len(results) == 3  # Both prespecified level hypotheses remain reported.


def test_minimum_applies_to_transformed_observations_including_lost_return_endpoint():
    results = run_tests(prices(ar1(300) * .0001))
    assert row(results)["status"] == "tested" and row(results)["n_observations"] == 300
    assert row(results, "log_return")["status"] == "skipped"
    assert row(results, "log_return")["n_observations"] == 299
    assert row(results, "log_return")["skip_reason"] == "too_short"
    assert all(item["skip_reason"] == "too_short" for item in run_tests([]))


@pytest.mark.parametrize("values,reason", [
    (np.full(500, 200_000_000_000, np.int64), "constant_series"),
    (np.r_[np.full(499, 200_000_000_000), 0].astype(np.int64), "nonpositive_price2"),
    (np.array([100., np.nan]), "nonfinite_price2"),
    (np.array([100., 101.]), "noninteger_price2"),
    (np.array([2**63], dtype=np.uint64), "price2_out_of_range"),
])
def test_unsupported_or_degenerate_inputs_are_explicit_skips(values, reason):
    results = run_tests(values)
    assert all(item["status"] == "skipped" and item["classification"] == "not_tested" for item in results)
    assert all(item["skip_reason"] == reason for item in results)
    assert all(item["adf_pvalue"] is item["kpss_pvalue"] is None for item in results)
    json.dumps(results, allow_nan=False)


def test_holm_correction_preserves_nominal_results_and_does_not_adjust_kpss_bounds():
    original = [{"adf_pvalue": p, "classification": "inconclusive", "kpss_pvalue": .01,
                 "kpss_pvalue_relation": "<="} for p in (.04, .01, .03, None)]
    before = copy.deepcopy(original)
    result = adjust_adf_holm(original)
    assert original == before
    assert [item["adf_pvalue_holm"] for item in result] == [.06, .03, .06, None]
    assert [item["adf_reject_holm"] for item in result] == [False, True, False, None]
    assert all(item["adf_holm_family_size"] == 3 and item["kpss_pvalue_relation"] == "<=" for item in result)
    assert all(item["classification"] == "inconclusive" for item in result)
