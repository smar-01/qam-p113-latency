"""Observed, recurrent SIRS venue states and a constrained infection hazard.

This is an observable-state approximation, not a claim that venues transmit
news causally. A susceptible venue's price shock starts a fixed infectious
period, followed by a fixed refractory period and renewed susceptibility.
The next-bin Bernoulli probability comes from an additive nonnegative hazard,
not a logistic regression relabelled as an epidemic model::

    p(t) = 1 - exp(-dt * (exp(a + x(t) @ w) + I(t - lag) @ beta))

Each beta is a nonnegative directed source-to-target rate, in seconds^-1.
Hazards and source states are held fixed over the forecast bin. All supplied
rows must be susceptible at the anchor and share observed book support.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits


SUSCEPTIBLE = 0
INFECTIOUS = 1
REFRACTORY = 2
INFECTIOUS_MS = 100
REFRACTORY_MS = 100
SOURCE_LAG_MS = 2
FORECAST_MS = 10
_NS_PER_MS = 1_000_000
_LOG_RATE_LIMIT = 60.0  # An explicit numerical guard, never data-calibrated.


def _duration_ns(value: float, name: str, *, allow_zero: bool = False) -> int:
    if not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}.")
    result = int(round(float(value) * _NS_PER_MS))
    if result == 0 and not allow_zero:
        raise ValueError(f"{name} must be at least one nanosecond.")
    return result


def _timestamps(values: np.ndarray, name: str) -> np.ndarray:
    original = np.asarray(values)
    if original.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional timestamp array.")
    if original.size and original.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer nanosecond timestamps.")
    if original.dtype.kind == "u" and original.size and original.max() > np.iinfo(np.int64).max:
        raise ValueError(f"{name} is outside signed nanosecond range.")
    return original.astype(np.int64, copy=False)


def accepted_infections(
    events_ns: np.ndarray,
    *,
    infectious_ms: float = INFECTIOUS_MS,
    refractory_ms: float = REFRACTORY_MS,
    reset_times_ns: np.ndarray | None = None,
) -> np.ndarray:
    """Greedily accept shocks only while S; repeated I/R shocks do not reset it.

    An event exactly I+R after the previous infection is eligible. A supplied
    observation reset clears prior state before processing any event at that
    timestamp. Callers must mask unsupported history after each reset.
    """
    infectious = _duration_ns(infectious_ms, "infectious_ms")
    refractory = _duration_ns(refractory_ms, "refractory_ms", allow_zero=True)
    events = np.unique(_timestamps(events_ns, "events_ns"))
    resets = (np.sort(_timestamps(reset_times_ns, "reset_times_ns"))
              if reset_times_ns is not None else np.empty(0, dtype=np.int64))
    chosen: list[int] = []
    previous: int | None = None
    reset_index = 0
    for raw_event in events:
        event = int(raw_event)
        if reset_index < len(resets) and int(resets[reset_index]) <= event:
            reset_index = int(np.searchsorted(resets, event, side="right"))
            previous = None
        if previous is None or event - previous >= infectious + refractory:
            chosen.append(event)
            previous = event
    return np.asarray(chosen, dtype=np.int64)


def sirs_states(
    infections_ns: np.ndarray,
    anchors_ns: np.ndarray,
    *,
    lag_ms: float = 0,
    infectious_ms: float = INFECTIOUS_MS,
    refractory_ms: float = REFRACTORY_MS,
    reset_times_ns: np.ndarray | None = None,
) -> np.ndarray:
    """Return strict-past S=0/I=1/R=2 state at anchor minus optional lag.

    At an infection's exact timestamp its information is not yet observable.
    At exact I expiry the state is R; at exact I+R expiry it is S. Unknown
    observation support is the caller's mask, never implicitly marked valid.
    """
    infectious = _duration_ns(infectious_ms, "infectious_ms")
    refractory = _duration_ns(refractory_ms, "refractory_ms", allow_zero=True)
    lag = _duration_ns(lag_ms, "lag_ms", allow_zero=True)
    infections = np.sort(_timestamps(infections_ns, "infections_ns"))
    anchors = _timestamps(anchors_ns, "anchors_ns")
    if anchors.size and int(anchors.min()) < np.iinfo(np.int64).min + lag:
        raise ValueError("Lag would overflow nanosecond timestamps.")
    evaluation = anchors - lag
    states = np.full(len(anchors), SUSCEPTIBLE, dtype=np.int8)
    if not len(infections):
        return states
    prior = np.searchsorted(infections, evaluation, side="left") - 1
    available = prior >= 0
    previous = infections[np.maximum(prior, 0)]
    # Ages use float64 differences only after subtracting nanosecond integers;
    # normal exchange timestamps and millisecond windows stay exact.
    age = evaluation - previous
    if reset_times_ns is not None:
        resets = np.sort(_timestamps(reset_times_ns, "reset_times_ns"))
        if resets.size:
            reset_index = np.searchsorted(resets, evaluation, side="right") - 1
            latest_reset = resets[np.maximum(reset_index, 0)]
            available &= (reset_index < 0) | (previous >= latest_reset)
    states[available & (age > 0) & (age < infectious)] = INFECTIOUS
    states[available & (age >= infectious) & (age < infectious + refractory)] = REFRACTORY
    return states


def infection_labels(
    infections_ns: np.ndarray,
    anchors_ns: np.ndarray,
    *,
    horizon_ms: float = FORECAST_MS,
) -> np.ndarray:
    """Accepted infection in [anchor, anchor+horizon); no future predictors."""
    horizon = _duration_ns(horizon_ms, "horizon_ms")
    events = np.sort(_timestamps(infections_ns, "infections_ns"))
    anchors = _timestamps(anchors_ns, "anchors_ns")
    if anchors.size and int(anchors.max()) > np.iinfo(np.int64).max - horizon:
        raise ValueError("Horizon would overflow nanosecond timestamps.")
    return (np.searchsorted(events, anchors + horizon, side="left") >
            np.searchsorted(events, anchors, side="left")).astype(np.int8)


def _matrix(values: np.ndarray, name: str, *, n: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite two-dimensional matrix.")
    if n is not None and len(array) != n:
        raise ValueError(f"{name} has the wrong number of rows.")
    return array


def hazard_objective(
    parameters: np.ndarray,
    design: np.ndarray,
    infected: np.ndarray,
    y: np.ndarray,
    *,
    dt_seconds: float = FORECAST_MS / 1000,
    ridge: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Mean exact negative Bernoulli log likelihood and analytic gradient.

    ``design`` includes its own intercept column. Its intercept is unpenalized;
    all other coefficients, including beta rates, receive ridge/2 * coef^2 in
    the summed objective (therefore ridge/(2*n) in the returned mean objective).
    This low-level optimizer helper assumes already validated arrays.
    """
    d = design.shape[1]
    theta, beta = parameters[:d], parameters[d:]
    eta = design @ theta
    baseline = np.exp(np.clip(eta, -_LOG_RATE_LIMIT, _LOG_RATE_LIMIT))
    integrated = dt_seconds * (baseline + infected @ beta)
    # Exponentials are guarded above; constrained nonnegative beta makes this
    # strictly positive, and expm1 preserves rare-event precision.
    event_loss = -np.log(-np.expm1(-integrated))
    losses = np.where(y == 1, event_loss, integrated)
    inv_expm1 = np.zeros_like(integrated)
    safe = integrated < 700
    inv_expm1[safe] = 1.0 / np.expm1(integrated[safe])
    derivative = (1.0 - y) - y * inv_expm1
    # Derivative of the explicit clipping guard is zero outside its bounds.
    baseline_derivative = baseline * ((eta > -_LOG_RATE_LIMIT) & (eta < _LOG_RATE_LIMIT))
    grad_theta = design.T @ (derivative * dt_seconds * baseline_derivative)
    grad_beta = infected.T @ (derivative * dt_seconds)
    gradient = np.r_[grad_theta, grad_beta]
    penalty = parameters.copy()
    penalty[0] = 0
    n = len(y)
    value = (float(losses.sum()) + 0.5 * ridge * float(penalty @ penalty)) / n
    gradient = (gradient + ridge * penalty) / n
    return value, gradient


@dataclass
class SIRSFit:
    """Frozen target-specific hazard fit; scaling is from training rows only."""

    intercept_: float
    control_coefficients_: np.ndarray
    beta_rates_: np.ndarray
    control_mean_: np.ndarray
    control_scale_: np.ndarray
    control_names_: tuple[str, ...]
    source_names_: tuple[str, ...]
    dt_seconds: float
    ridge: float
    training_rows: int
    training_infections: int
    converged: bool
    optimizer_message: str
    iterations: int
    objective: float
    numerical_guard_rows: int

    def predict_proba(self, x: np.ndarray, infected: np.ndarray | None = None) -> np.ndarray:
        """Return a one-dimensional infection probability, never refitting."""
        controls = _matrix(x, "x")
        if controls.shape[1] != len(self.control_coefficients_):
            raise ValueError("x has the wrong number of controls.")
        if len(self.beta_rates_):
            if infected is None:
                raise ValueError("Network prediction requires source infection states.")
            sources = _matrix(infected, "infected", n=len(controls))
            if sources.shape[1] != len(self.beta_rates_) or not np.all((sources == 0) | (sources == 1)):
                raise ValueError("infected must have one binary column per fitted source.")
            contact = sources @ self.beta_rates_
        else:
            contact = 0.0
        eta = self.intercept_ + ((controls - self.control_mean_) / self.control_scale_) @ self.control_coefficients_
        baseline = np.exp(np.clip(eta, -_LOG_RATE_LIMIT, _LOG_RATE_LIMIT))
        return -np.expm1(-self.dt_seconds * (baseline + contact))

    def to_dict(self) -> dict:
        return {
            "model": "observed_state_SIRS_additive_hazard" if len(self.beta_rates_) else "exogenous_hazard",
            "hazard_equation": "exp(intercept + standardized_controls @ weights) + infectious_sources @ beta",
            "probability_equation": "1 - exp(-dt_seconds * hazard)",
            "intercept": self.intercept_,
            "control_coefficients": dict(zip(self.control_names_, map(float, self.control_coefficients_))),
            "control_mean": dict(zip(self.control_names_, map(float, self.control_mean_))),
            "control_scale": dict(zip(self.control_names_, map(float, self.control_scale_))),
            "beta_rates_per_second": dict(zip(self.source_names_, map(float, self.beta_rates_))),
            "dt_seconds": self.dt_seconds,
            "ridge": self.ridge,
            "penalty": "summed_negative_log_likelihood + ridge/2 * squared_nonintercept_coefficients",
            "training_rows": self.training_rows,
            "training_infections": self.training_infections,
            "converged": self.converged,
            "optimizer_message": self.optimizer_message,
            "iterations": self.iterations,
            "penalized_mean_objective": self.objective,
            "training_numerical_guard_rows": self.numerical_guard_rows,
            "log_baseline_rate_numerical_clip": [-_LOG_RATE_LIMIT, _LOG_RATE_LIMIT],
            "interpretation": "conditional predictive association; not causal transmission or investor identity",
        }


def fit_sirs(
    x: np.ndarray,
    infected: np.ndarray,
    y: np.ndarray,
    *,
    dt_seconds: float = FORECAST_MS / 1000,
    ridge: float = 1.0,
    source_names: Sequence[str] | None = None,
    control_names: Sequence[str] | None = None,
    max_iter: int = 1000,
) -> SIRSFit:
    """Fit an exponential exogenous hazard plus nonnegative network rates.

    Both outcome classes are required. A one-class training task has no finite
    unconstrained intercept MLE and must be reported as insufficient by callers.
    Fit convergence is returned explicitly and must be checked before reporting.
    """
    controls = _matrix(x, "x")
    sources = _matrix(infected, "infected", n=len(controls))
    if not len(controls):
        raise ValueError("At least one training row is required.")
    outcome = np.asarray(y, dtype=float)
    if outcome.ndim != 1 or len(outcome) != len(controls) or not np.all((outcome == 0) | (outcome == 1)):
        raise ValueError("y must be a binary vector matching x rows.")
    if len(np.unique(outcome)) != 2:
        raise ValueError("Both infection and noninfection training outcomes are required.")
    if not np.all((sources == 0) | (sources == 1)):
        raise ValueError("infected must contain binary source infection states.")
    if not np.isfinite(dt_seconds) or dt_seconds <= 0:
        raise ValueError("dt_seconds must be finite and positive.")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative.")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")
    cn = tuple(control_names) if control_names is not None else tuple(f"control_{i}" for i in range(controls.shape[1]))
    sn = tuple(source_names) if source_names is not None else tuple(f"source_{i}" for i in range(sources.shape[1]))
    if len(cn) != controls.shape[1] or len(sn) != sources.shape[1] or len(set(cn)) != len(cn) or len(set(sn)) != len(sn):
        raise ValueError("Control and source names must be unique and match their columns.")
    center = controls.mean(axis=0)
    scale = controls.std(axis=0)
    scale[scale == 0] = 1.0
    design = np.column_stack([np.ones(len(controls)), (controls - center) / scale])
    initial = np.zeros(design.shape[1] + sources.shape[1])
    initial[0] = np.log(-np.log1p(-outcome.mean()) / dt_seconds)
    bounds = [(None, None)] * design.shape[1] + [(0.0, None)] * sources.shape[1]
    with threadpool_limits(limits=1):
        fit = minimize(
            lambda params: hazard_objective(params, design, sources, outcome, dt_seconds=dt_seconds, ridge=ridge),
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": int(max_iter), "ftol": 1e-12, "gtol": 1e-8, "maxls": 40},
        )
    theta = fit.x[:design.shape[1]]
    guarded = int(np.count_nonzero(np.abs(design @ theta) >= _LOG_RATE_LIMIT))
    return SIRSFit(
        intercept_=float(theta[0]), control_coefficients_=theta[1:].copy(),
        beta_rates_=fit.x[design.shape[1]:].copy(), control_mean_=center,
        control_scale_=scale, control_names_=cn, source_names_=sn,
        dt_seconds=float(dt_seconds), ridge=float(ridge), training_rows=len(outcome),
        training_infections=int(outcome.sum()), converged=bool(fit.success),
        optimizer_message=str(fit.message), iterations=int(fit.nit),
        objective=float(fit.fun), numerical_guard_rows=guarded,
    )


def fit_sirs_models(
    x_train: np.ndarray,
    infected_train: np.ndarray,
    y_train: np.ndarray,
    *,
    dt_seconds: float = FORECAST_MS / 1000,
    ridge: float = 1.0,
    source_names: Sequence[str] | None = None,
    control_names: Sequence[str] | None = None,
    max_iter: int = 1000,
) -> dict[str, SIRSFit]:
    """Fit nested exogenous-only and SIRS-network models on identical rows."""
    controls = _matrix(x_train, "x_train")
    common = dict(dt_seconds=dt_seconds, ridge=ridge, control_names=control_names, max_iter=max_iter)
    return {
        "exogenous": fit_sirs(controls, np.empty((len(controls), 0)), y_train, source_names=(), **common),
        "network": fit_sirs(controls, infected_train, y_train, source_names=source_names, **common),
    }
