from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.optimize import check_grad

from colonylab.sir_model import (
    INFECTIOUS, REFRACTORY, SUSCEPTIBLE, accepted_infections,
    fit_sirs, fit_sirs_models, hazard_objective, infection_labels, sirs_states,
)


ORIGIN = 1_800_000_000_000_000_000


def ns(milliseconds):
    return ORIGIN + np.asarray(milliseconds, dtype=np.int64) * 1_000_000


def test_reinfection_requires_full_infectious_and_refractory_period():
    # Repeated shocks do not prolong infection or restart recovery.
    raw = ns([200, 0, 50, 99, 100, 199, 200, 250, 399, 400])
    np.testing.assert_array_equal(accepted_infections(raw), ns([0, 200, 400]))


def test_sirs_state_boundaries_use_only_strict_past_information():
    states = sirs_states(ns([0, 200]), ns([-1, 0, 1, 99, 100, 199, 200, 201, 300, 400]))
    assert states.tolist() == [SUSCEPTIBLE, SUSCEPTIBLE, INFECTIOUS, INFECTIOUS,
                               REFRACTORY, REFRACTORY, SUSCEPTIBLE, INFECTIOUS,
                               REFRACTORY, SUSCEPTIBLE]


def test_two_ms_source_lag_excludes_equal_cutoff_and_synchronous_events():
    assert sirs_states(ns([10]), ns([10, 11, 12, 13, 111, 112]), lag_ms=2).tolist() == [0, 0, 0, 1, 1, 2]


def test_future_infections_cannot_change_historical_state():
    anchors = ns([20, 100, 200])
    original = sirs_states(ns([10]), anchors, lag_ms=2)
    appended = sirs_states(ns([10, 200, 201, 1000]), anchors, lag_ms=2)
    np.testing.assert_array_equal(original, appended)


def test_future_candidate_shocks_cannot_change_accepted_history():
    history = accepted_infections(ns([0, 100, 199, 200]))
    extended = accepted_infections(ns([0, 100, 199, 200, 250, 400]))
    np.testing.assert_array_equal(extended[extended <= ns([200])[0]], history)


def test_resets_clear_refractory_acceptance_and_sampled_state():
    reset = ns([75])
    chosen = accepted_infections(ns([0, 80]), reset_times_ns=reset)
    np.testing.assert_array_equal(chosen, ns([0, 80]))
    assert sirs_states(chosen, ns([74, 75, 76, 80, 81]), reset_times_ns=reset).tolist() == [1, 0, 0, 0, 1]


def test_reset_at_infection_timestamp_precedes_new_infection():
    chosen = accepted_infections(ns([0, 50]), reset_times_ns=ns([50]))
    np.testing.assert_array_equal(chosen, ns([0, 50]))
    assert sirs_states(chosen, ns([50, 51]), reset_times_ns=ns([50])).tolist() == [0, 1]


def test_labels_are_half_open_and_include_equal_anchor():
    assert infection_labels(ns([10, 200]), ns([0, 10, 190, 200, 210])).tolist() == [0, 1, 0, 1, 0]


def test_zero_refractory_duration_returns_directly_to_susceptible():
    np.testing.assert_array_equal(accepted_infections(ns([0, 99, 100]), refractory_ms=0), ns([0, 100]))
    assert sirs_states(ns([0]), ns([99, 100]), refractory_ms=0).tolist() == [1, 0]


@pytest.mark.parametrize("function,kwargs", [
    (accepted_infections, {"infectious_ms": 0}),
    (accepted_infections, {"refractory_ms": -1}),
    (accepted_infections, {"infectious_ms": float("nan")}),
    (sirs_states, {"lag_ms": -1}),
    (infection_labels, {"horizon_ms": 0}),
])
def test_invalid_state_durations_raise(function, kwargs):
    with pytest.raises(ValueError):
        if function is accepted_infections:
            function(ns([0]), **kwargs)
        else:
            function(ns([0]), ns([1]), **kwargs)


def test_noninteger_timestamp_and_overflow_are_rejected():
    with pytest.raises(ValueError):
        accepted_infections(np.asarray([1.5]))
    with pytest.raises(ValueError):
        infection_labels(np.asarray([], dtype=np.int64), np.asarray([np.iinfo(np.int64).max]), horizon_ms=1)
    with pytest.raises(ValueError):
        sirs_states(np.asarray([], dtype=np.int64), np.asarray([np.iinfo(np.int64).min]), lag_ms=1)


def test_hazard_likelihood_and_penalty_match_hand_computation():
    design = np.asarray([[1., 0.], [1., 1.]])
    infected = np.asarray([[0.], [1.]])
    params = np.asarray([np.log(2), np.log(3), 4.])
    y = np.asarray([0., 1.])
    value, gradient = hazard_objective(params, design, infected, y, dt_seconds=.01, ridge=2)
    expected = (.02 - np.log(1 - np.exp(-.1)) + np.log(3) ** 2 + 16) / 2
    assert value == pytest.approx(expected)
    assert gradient.shape == params.shape
    assert np.isfinite(gradient).all()


def test_analytic_gradient_matches_finite_difference():
    rng = np.random.default_rng(4)
    design = np.column_stack([np.ones(100), rng.normal(size=(100, 3))])
    infected = rng.binomial(1, .25, size=(100, 2))
    y = rng.binomial(1, .2, size=100)
    params = np.asarray([1., -.2, .3, -.1, 2., 3.])
    def value(p):
        return hazard_objective(p, design, infected, y, ridge=1.2)[0]
    def gradient(p):
        return hazard_objective(p, design, infected, y, ridge=1.2)[1]
    assert check_grad(value, gradient, params) < 1e-6


def test_rare_and_extreme_hazards_are_numerically_finite():
    for intercept in (-1000., -60., 60., 1000.):
        value, gradient = hazard_objective(np.asarray([intercept]), np.ones((2, 1)),
                                           np.empty((2, 0)), np.asarray([0., 1.]))
        assert np.isfinite(value)
        assert np.isfinite(gradient).all()


def test_intercept_only_fit_recovers_bin_prevalence():
    y = np.r_[np.ones(200), np.zeros(800)]
    fit = fit_sirs(np.empty((1000, 0)), np.empty((1000, 0)), y)
    assert fit.converged
    assert fit.intercept_ == pytest.approx(np.log(-np.log(.8) / .01), abs=1e-7)
    assert fit.predict_proba(np.empty((3, 0))) == pytest.approx([.2, .2, .2])
    assert fit.numerical_guard_rows == 0
    assert json.loads(json.dumps(fit.to_dict()))["training_infections"] == 200


def test_nested_models_use_identical_training_rows_and_training_only_scaling():
    x = np.arange(1000, dtype=float).reshape(-1, 1)
    y = ((np.arange(1000) % 7) == 0).astype(int)
    infected = ((np.arange(1000) % 3) == 0).astype(int).reshape(-1, 1)
    fits = fit_sirs_models(x, infected, y, source_names=["A"], control_names=["own_depth"])
    baseline, network = fits["exogenous"], fits["network"]
    assert baseline.converged and network.converged
    assert baseline.training_rows == network.training_rows == 1000
    assert baseline.training_infections == network.training_infections == y.sum()
    np.testing.assert_array_equal(baseline.control_mean_, network.control_mean_)
    np.testing.assert_array_equal(baseline.control_scale_, network.control_scale_)
    center = network.control_mean_.copy()
    pred = network.predict_proba(np.asarray([[2000.], [3000.]]), np.asarray([[0.], [1.]]))
    assert np.all((pred > 0) & (pred < 1))
    np.testing.assert_array_equal(network.control_mean_, center)
    assert network.beta_rates_.min() >= 0
    assert network.to_dict()["beta_rates_per_second"].keys() == {"A"}


def test_network_recovers_a_planted_directed_rate_and_improves_heldout_likelihood():
    rng = np.random.default_rng(723)
    n = 60_000
    x = rng.normal(size=(n, 1))
    infected = rng.binomial(1, .3, size=(n, 2))
    actual = -np.expm1(-.01 * (np.exp(np.log(2) + .4 * x[:, 0]) + 12 * infected[:, 0]))
    y = rng.binomial(1, actual)
    split = 45_000
    fits = fit_sirs_models(x[:split], infected[:split], y[:split], source_names=["connected", "null"])
    network = fits["network"]
    assert fits["exogenous"].converged and network.converged
    assert network.beta_rates_[0] > 6
    assert network.beta_rates_[0] > network.beta_rates_[1] + 5
    assert np.all(network.beta_rates_ >= 0)
    def loss(prob):
        target = y[split:]
        return np.mean(-target * np.log(prob) - (1 - target) * np.log1p(-prob))
    assert loss(network.predict_proba(x[split:], infected[split:])) < loss(fits["exogenous"].predict_proba(x[split:]))


def test_negative_association_cannot_be_fitted_as_a_negative_transmission_edge():
    source = np.tile([0., 1.], 5000).reshape(-1, 1)
    y = np.zeros(10_000)
    y[np.arange(0, 10_000, 10)] = 1  # Every outcome occurs when source is absent.
    fit = fit_sirs(np.empty((len(y), 0)), source, y)
    assert fit.converged
    assert fit.beta_rates_[0] == pytest.approx(0., abs=1e-10)


@pytest.mark.parametrize("changed", [
    {"dt_seconds": 0}, {"dt_seconds": float("nan")}, {"ridge": -1},
    {"ridge": float("inf")}, {"max_iter": 0}, {"source_names": ["A", "B"]},
    {"control_names": []},
])
def test_invalid_fit_options_raise(changed):
    with pytest.raises(ValueError):
        fit_sirs(np.ones((10, 1)), np.ones((10, 1)), np.tile([0, 1], 5), **changed)


def test_invalid_training_arrays_and_single_class_raise():
    valid_x, valid_i, y = np.ones((10, 1)), np.ones((10, 1)), np.tile([0, 1], 5)
    for x, infected, outcome in [
        (np.full((10, 1), np.nan), valid_i, y),
        (valid_x, np.ones((9, 1)), y),
        (valid_x, np.full((10, 1), .5), y),
        (valid_x, valid_i, np.full(10, 2)),
        (valid_x, valid_i, np.zeros(10)),
        (valid_x, valid_i, np.ones(10)),
    ]:
        with pytest.raises(ValueError):
            fit_sirs(x, infected, outcome)


def test_prediction_rejects_missing_wrong_or_nonbinary_source_states():
    fit = fit_sirs(np.ones((10, 1)), np.ones((10, 1)), np.tile([0, 1], 5))
    with pytest.raises(ValueError):
        fit.predict_proba(np.ones((2, 1)))
    with pytest.raises(ValueError):
        fit.predict_proba(np.ones((2, 1)), np.full((2, 1), .5))
    with pytest.raises(ValueError):
        fit.predict_proba(np.ones((2, 2)), np.ones((2, 1)))


def test_nonconvergence_is_visible_in_serialized_diagnostics():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(1000, 3))
    infected = rng.binomial(1, .3, size=(1000, 2))
    y = rng.binomial(1, .2, size=1000)
    fit = fit_sirs(x, infected, y, max_iter=1)
    assert not fit.converged
    assert not fit.to_dict()["converged"]
