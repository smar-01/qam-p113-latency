"""Known-answer and corruption checks for the independent final-result audit."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("independent_mbo_audit", ROOT / "scripts" / "audit_mbo_sir.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def hazard_example():
    panel = {"x": np.asarray([[2., 1.], [4., 5.]]), "names": ["z", "a"],
             "infected": np.asarray([[0], [1]]), "shifted_infected": np.asarray([[1], [0]]),
             "source_names": ["B"]}
    parameters = {"intercept": np.log(2), "converged": True,
                  "control_coefficients": {"a": .2, "z": .5},
                  "control_mean": {"a": 1., "z": 2.}, "control_scale": {"a": 4., "z": 2.},
                  "dt_seconds": .01, "ridge": 1., "log_baseline_rate_numerical_clip": [-60., 60.],
                  "beta_rates_per_second": {"B": 3.}}
    return panel, parameters


def logistic_example():
    panel = {"x": np.asarray([[2., 1.], [4., 5.]]), "names": ["z", "own_infections_1s"],
             "legacy": np.asarray([[9999., 3.], [-9999., 5.]]),
             "legacy_names": ["own_infections_1000ms", "other_B_100ms"]}
    parameters = {"converged": True, "intercept": -.3, "C": 1,
                  "legacy_column_indices": [1], "control_mean": [2., 1., 3.], "control_scale": [2., 4., 2.],
                  "coefficients": {"other_B_100ms": .7, "own_infections_1s": .2, "z": .5},
                  "feature_names": ["z", "own_infections_1s", "other_B_100ms"]}
    return panel, parameters


def test_hazard_probabilities_use_explicit_panel_names_not_sorted_json_dict_order():
    panel, parameters = hazard_example()
    p, guard = audit.probability_from_json(panel, parameters, "sirs_network")
    expected = 1 - np.exp(-.01 * np.asarray([2., 2 * np.exp(.7) + 3]))
    np.testing.assert_allclose(p, expected, rtol=1e-13)
    assert guard == 0


def test_shifted_exposure_and_exogenous_baseline_are_distinct_known_hazards():
    panel, parameters = hazard_example()
    shifted, _ = audit.probability_from_json(panel, parameters, "past_shifted_sirs")
    expected = 1 - np.exp(-.01 * np.asarray([5., 2 * np.exp(.7)]))
    np.testing.assert_allclose(shifted, expected)
    parameters["beta_rates_per_second"] = {}
    independent, _ = audit.probability_from_json(panel, parameters, "independent_mbo_hazard")
    np.testing.assert_allclose(independent, 1 - np.exp(-.01 * np.asarray([2., 2 * np.exp(.7)])))


def test_hazard_guard_count_is_recomputed_separately_from_finite_probability():
    panel, parameters = hazard_example()
    parameters["intercept"] = 100.
    result, guards = audit.probability_from_json(panel, parameters, "sirs_network")
    assert guards == 2
    assert np.isfinite(result).all() and np.all(result == 1)


@pytest.mark.parametrize("change", ["negative_beta", "bad_scale", "nonboolean_status", "wrong_names", "nonbinary_sources", "wrong_duration"])
def test_invalid_hazard_json_or_source_state_is_rejected(change):
    panel, parameters = hazard_example()
    if change == "negative_beta":
        parameters["beta_rates_per_second"]["B"] = -1
    elif change == "bad_scale":
        parameters["control_scale"]["z"] = 0
    elif change == "nonboolean_status":
        parameters["converged"] = 1
    elif change == "wrong_names":
        parameters["control_coefficients"]["unknown"] = 0
    elif change == "wrong_duration":
        parameters["dt_seconds"] = .1
    else:
        panel["infected"] = np.asarray([[.5], [0.]])
    with pytest.raises(audit.AuditError):
        audit.probability_from_json(panel, parameters, "sirs_network")


@pytest.mark.parametrize("model", ["sirs_network", "logistic_network"])
def test_finite_json_predictions_are_reconstructed_when_convergence_is_honestly_false(model):
    panel, parameters = logistic_example() if model == "logistic_network" else hazard_example()
    expected, expected_guards = audit.probability_from_json(panel, parameters, model)
    parameters["converged"] = False
    actual, guards = audit.probability_from_json(panel, parameters, model)
    np.testing.assert_array_equal(actual, expected)
    assert guards == expected_guards


def convergence_fixture():
    parameters = {
        f"V{target}": {model: {"converged": True, "iterations": [12] if model == "logistic_network" else 12,
                              **({} if model == "logistic_network" else {"optimizer_message": "Converged"})}
                      for model in audit.MODELS}
        for target in range(8)
    }
    return parameters, {"all_models_converged": True}


def test_all_32_convergence_statuses_are_independently_aggregated():
    parameters, summary = convergence_fixture()
    result = audit.validate_model_convergence(parameters, summary)
    assert result == {"all_models_converged": True, "fitted_model_count": 32, "nonconverged_models": []}


def test_consistent_nonconvergence_is_retained_with_exact_targets_iterations_and_messages():
    parameters, summary = convergence_fixture()
    parameters["V2"]["sirs_network"].update(converged=False, iterations=1000, optimizer_message="Iteration limit reached")
    parameters["V5"]["logistic_network"].update(converged=False, iterations=[1000])
    summary["all_models_converged"] = False
    result = audit.validate_model_convergence(parameters, summary)
    assert result == {
        "all_models_converged": False, "fitted_model_count": 32,
        "nonconverged_models": [
            {"target": "V2", "model": "sirs_network", "iterations": 1000, "optimizer_message": "Iteration limit reached"},
            {"target": "V5", "model": "logistic_network", "iterations": [1000], "optimizer_message": None},
        ],
    }


@pytest.mark.parametrize("status", [0, 1, None, "false", []])
def test_each_model_convergence_status_must_be_an_actual_bool(status):
    parameters, summary = convergence_fixture()
    parameters["V7"]["past_shifted_sirs"]["converged"] = status
    with pytest.raises(audit.AuditError, match="Invalid convergence status"):
        audit.validate_model_convergence(parameters, summary)


@pytest.mark.parametrize("fitted_status,claimed_status", [(False, True), (True, False)])
def test_mismatched_convergence_aggregates_are_rejected(fitted_status, claimed_status):
    parameters, summary = convergence_fixture()
    parameters["V0"]["sirs_network"]["converged"] = fitted_status
    summary["all_models_converged"] = claimed_status
    with pytest.raises(audit.AuditError, match="convergence aggregate differs"):
        audit.validate_model_convergence(parameters, summary)


@pytest.mark.parametrize("claimed", [0, 1, None, "true"])
def test_summary_convergence_aggregate_must_be_an_actual_bool(claimed):
    parameters, summary = convergence_fixture()
    summary["all_models_converged"] = claimed
    with pytest.raises(audit.AuditError, match="convergence aggregate differs"):
        audit.validate_model_convergence(parameters, summary)


def test_missing_fitted_model_cannot_disappear_from_convergence_aggregate():
    parameters, summary = convergence_fixture()
    del parameters["V4"]["past_shifted_sirs"]
    with pytest.raises(audit.AuditError, match="Fitted model coverage differs"):
        audit.validate_model_convergence(parameters, summary)


def test_nonconvergence_forces_scientific_ineligibility_despite_positive_statistics():
    result = audit.validate_evidence_eligibility({"supports_incremental_prediction": False},
        all_models_converged=False, guards=0, primary_statistical_support=True)
    assert result == {"scientifically_eligible": False,
                      "scientific_ineligibility_reasons": ["optimization_nonconvergence"],
                      "supports_incremental_prediction": False}


@pytest.mark.parametrize("statistical_support", [False, True])
def test_eligible_fits_still_require_the_primary_statistical_support_criterion(statistical_support):
    result = audit.validate_evidence_eligibility({"supports_incremental_prediction": statistical_support},
        all_models_converged=True, guards=0, primary_statistical_support=statistical_support)
    assert result["scientifically_eligible"] is True
    assert result["scientific_ineligibility_reasons"] == []
    assert result["supports_incremental_prediction"] is statistical_support


@pytest.mark.parametrize("converged,guards,claimed", [(False, 0, True), (True, 1, True), (True, 0, False), (True, 0, 1)])
def test_inconsistent_overall_support_flags_are_rejected(converged, guards, claimed):
    with pytest.raises(audit.AuditError, match="Overall evidence flag differs"):
        audit.validate_evidence_eligibility({"supports_incremental_prediction": claimed},
            all_models_converged=converged, guards=guards, primary_statistical_support=True)


def test_logistic_json_order_frozen_indices_and_deduplicated_history():
    panel, parameters = logistic_example()
    p, guard = audit.probability_from_json(panel, parameters, "logistic_network")
    expected_eta = np.asarray([-.3, -.3 + .5 + .2 + .7])
    np.testing.assert_allclose(p, 1 / (1 + np.exp(-expected_eta)))
    assert guard == 0
    panel["legacy"][:, 0] *= 1000  # Excluded duplicate cannot affect inference.
    np.testing.assert_array_equal(audit.probability_from_json(panel, parameters, "logistic_network")[0], p)


@pytest.mark.parametrize("change", ["include_duplicate", "wrong_order", "wrong_scale"])
def test_inconsistent_logistic_column_metadata_is_rejected(change):
    panel, parameters = logistic_example()
    if change == "include_duplicate":
        parameters["legacy_column_indices"] = [0, 1]
    elif change == "wrong_order":
        parameters["feature_names"] = parameters["feature_names"][::-1]
    else:
        parameters["control_scale"] = [1.]
    with pytest.raises(audit.AuditError):
        audit.probability_from_json(panel, parameters, "logistic_network")


def test_loss_sums_are_known_values_and_clipping_is_visible():
    values = audit.loss_sums(np.asarray([0, 1]), np.asarray([.25, .8]))
    assert values["log_loss_sum"] == pytest.approx(-np.log(.75) - np.log(.8))
    assert values["brier_sum"] == pytest.approx(.25 ** 2 + .2 ** 2)
    assert values["positive_count"] == 1 and values["rows"] == 2
    clipped = audit.loss_sums(np.asarray([0, 1]), np.asarray([0., 1.]))
    assert clipped["probability_clip_rows"] == 2 and clipped["brier_sum"] == 0


def test_cohort_hash_catches_changed_labels_with_equal_counts_and_changed_timestamps():
    times, y = np.asarray([100, 200, 300, 400], dtype=np.int64), np.asarray([0, 1, 0, 1], dtype=np.int8)
    row = {"cohort_sha256": hashlib.sha256(times.tobytes() + y.tobytes()).hexdigest(), "rows": 4, "positive_count": 2}
    assert audit.validate_score_cohort(row, times, y) == row["cohort_sha256"]
    with pytest.raises(audit.AuditError):
        audit.validate_score_cohort(row, times, 1 - y)
    with pytest.raises(audit.AuditError):
        audit.validate_score_cohort(row, times + 1, y)


@pytest.mark.parametrize("corruption", ["zero_rows", "duplicate_time", "invalid_label", "wrong_rows", "wrong_positives"])
def test_invalid_or_zero_task_cohorts_are_rejected(corruption):
    times, y = np.asarray([100, 200], dtype=np.int64), np.asarray([0, 1], dtype=np.int8)
    row = {"cohort_sha256": hashlib.sha256(times.tobytes() + y.tobytes()).hexdigest(), "rows": 2, "positive_count": 1}
    if corruption == "zero_rows":
        times, y = times[:0], y[:0]
    elif corruption == "duplicate_time":
        times[1] = times[0]
    elif corruption == "invalid_label":
        y[1] = 2
    elif corruption == "wrong_rows":
        row["rows"] = 3
    else:
        row["positive_count"] = 0
    with pytest.raises(audit.AuditError):
        audit.validate_score_cohort(row, times, y)


@pytest.mark.parametrize("records", [[{"day": "A"}], [{"day": "A"}, {"day": "A"}], [{"day": "A"}, {"day": "C"}]])
def test_grid_holes_duplicates_and_extra_dates_are_rejected(records):
    with pytest.raises(audit.AuditError):
        audit.exact_grid(records, ("day",), {("A",), ("B",)}, "example")


def test_independent_request_grid_matches_frozen_full_session_dimensions():
    config = tomllib.loads((ROOT / "mbo_sir.toml").read_text())
    requests = audit.request_grid(config)
    assert len(requests) == 510
    assert sum(request["schema"] == "mbo" for request in requests) == 240
    assert sum(request["schema"] == "mbp-10" for request in requests) == 240
    assert sum(request["schema"] == "bbo-1s" for request in requests) == 30
    assert all(request["start"].endswith("T00:00:00+00:00") and request["end"].endswith("T20:00:00+00:00")
               for request in requests if request["schema"] == "mbo")
    assert len({audit.digest(request) for request in requests}) == 510


@pytest.mark.parametrize("change", ["venue", "date", "session", "validation", "horizon"])
def test_actual_frozen_dates_venues_session_and_horizon_cannot_silently_drift(change):
    config = tomllib.loads((ROOT / "mbo_sir.toml").read_text())
    protocol = json.loads((ROOT / "mbo_sir_protocol.json").read_text())
    audit.validate_specification(config, protocol)
    if change == "venue":
        config["datasets"][0] = "XCHI.PILLAR"
    elif change == "date":
        config["held_out_days"][-1] = "2026-07-15"
    elif change == "session":
        config["session_end_utc"] = "21:00:00"
    elif change == "validation":
        config["validation_minutes"] = 10
    else:
        protocol["forecast_horizon_ms"] = 100
    with pytest.raises(audit.AuditError):
        audit.validate_specification(config, protocol)


def test_streaming_training_moments_match_population_std_and_keep_constant_features():
    rng = np.random.default_rng(123)
    x = np.column_stack([rng.normal(size=1000), np.ones(1000), rng.normal(1000, .01, size=1000)])
    moments = audit.Moments()
    for part in np.array_split(x, 7):
        moments.add(part)
    np.testing.assert_allclose(moments.mean, x.mean(axis=0), rtol=1e-13, atol=1e-13)
    expected = x.std(axis=0)
    expected[1] = 1
    np.testing.assert_allclose(moments.scale(), expected, rtol=1e-10)
    assert moments.n == 1000


def test_paired_block_bootstrap_preserves_constant_ratios_and_denominators():
    rows = np.arange(1, 21)
    totals = np.column_stack([.2 * rows, .18 * rows, .1 * rows, .09 * rows, rows])
    intervals = audit.bootstrap_intervals(totals, replicates=1000, seed=9, block_length=5)
    assert intervals["log_loss"]["absolute_gain_ci95"] == pytest.approx([.02, .02])
    assert intervals["log_loss"]["relative_gain_percent_ci95"] == pytest.approx([10, 10])
    assert intervals["brier"]["absolute_gain_ci95"] == pytest.approx([.01, .01])


def test_independent_bootstrap_matches_known_production_fixture_without_calling_it_in_auditor():
    from colonylab.confirmation_stats import _bootstrap
    rng = np.random.default_rng(19)
    rows = rng.integers(100, 1000, size=20)
    reference = rng.uniform(.1, .3, size=20) * rows
    candidate = reference * rng.uniform(.8, 1.2, size=20)
    totals = np.column_stack([reference, candidate, reference / 4, candidate / 4, rows])
    independent = audit.bootstrap_intervals(totals, replicates=10_000, seed=20261027, block_length=5)
    published = _bootstrap(totals, replicates=10_000, seed=20261027, block_length=5)
    for metric in ("log_loss", "brier"):
        for field in ("absolute_gain_ci95", "relative_gain_percent_ci95"):
            np.testing.assert_allclose(independent[metric][field], published[metric][field], rtol=1e-12, atol=1e-12)


def comparison_fixture():
    from colonylab.mbo_sir_stats import summarize_sir_comparison
    dates = pd.bdate_range("2026-06-01", periods=20).strftime("%Y-%m-%d").tolist()
    records = []
    factors = {"independent_mbo_hazard": 1.1, "sirs_network": 1., "logistic_network": .95, "past_shifted_sirs": 1.05}
    for i, day in enumerate(dates):
        for model in audit.MODELS:
            records.append({"day": day, "model": model, "rows": 100 + i, "positive_count": 10,
                            "log_loss_sum": (20 + i / 5) * factors[model], "brier_sum": (5 + i / 20) * factors[model]})
    daily = pd.DataFrame(records)
    protocol = {"inference": {"replicates": 10_000, "seed": 20261026, "block_sessions": 5}}
    summary = {"held_out_days": dates, "comparisons": {reference: summarize_sir_comparison(
        daily, reference_model=reference, candidate_model="sirs_network", expected_days=dates)
        for reference in audit.REFERENCES}}
    return daily, summary, protocol


def test_all_three_published_comparisons_and_intervals_are_independently_recomputed():
    daily, summary, protocol = comparison_fixture()
    result = audit.compare_summary_statistics(daily, summary, protocol)
    assert set(result) == set(audit.REFERENCES)
    assert result["independent_mbo_hazard"]["supports_positive_incremental_prediction"]
    assert not result["logistic_network"]["supports_positive_incremental_prediction"]


@pytest.mark.parametrize("corruption", ["pooled", "interval", "support", "zero_day", "unpaired"])
def test_bad_published_statistics_or_coverage_are_rejected(corruption):
    daily, summary, protocol = comparison_fixture()
    if corruption == "pooled":
        summary["comparisons"]["independent_mbo_hazard"]["pooled"]["log_loss"]["relative_gain_percent"] += 1
    elif corruption == "interval":
        summary["comparisons"]["past_shifted_sirs"]["five_session_block_bootstrap"]["brier"]["absolute_gain_ci95"][0] += .1
    elif corruption == "support":
        summary["comparisons"]["independent_mbo_hazard"]["supports_positive_incremental_prediction"] = False
    elif corruption == "zero_day":
        daily = daily.loc[daily.day.ne(summary["held_out_days"][0])]
    else:
        daily.loc[daily.model.eq("sirs_network"), "rows"] += 1
    with pytest.raises(audit.AuditError):
        audit.compare_summary_statistics(daily, summary, protocol)


def test_public_audit_writes_explicit_failed_report_for_unfinished_results(tmp_path):
    source = (ROOT / "mbo_sir.toml").read_text()
    source = source.replace('out_dir = "out/mbo_sir"', 'out_dir = "' + str(tmp_path / "missing_results").replace("\\", "/") + '"')
    config_path = tmp_path / "mbo_sir.toml"
    config_path.write_text(source)
    (tmp_path / "mbo_sir_protocol.json").write_text((ROOT / "mbo_sir_protocol.json").read_text())
    result = audit.audit_results(config_path, progress=lambda _: None)
    assert not result["passed"] and len(result["errors"]) == 1
    assert result["arithmetic_verified"] is False and result["scientifically_eligible"] is None
    assert "summary.json" in result["errors"][0]
    saved = json.loads((tmp_path / "missing_results" / "result_audit.json").read_text())
    assert not saved["passed"] and saved["mode"] == "independent_bin_predictions"
