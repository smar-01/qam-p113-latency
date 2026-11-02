"""Reproduce the team-code numerical failures and independently check fixes."""
from __future__ import annotations


import numpy as np
import pandas as pd
import pytest


from p113 import leadlag as fixed


def frame(times, bid, ask=None, venue="XNAS", raw=False):
    ask = np.asarray(bid) + .02 if ask is None else np.asarray(ask)
    result = pd.DataFrame({"ts_recv": np.asarray(times, np.int64), "ts_event": np.asarray(times, np.int64),
                           "bid": np.asarray(bid), "ask": ask, "venue": venue, "symbol": "TEST"})
    if raw:
        result["bid_raw"] = np.rint(result.bid * 1e9).astype(np.int64)
        result["ask_raw"] = np.rint(result.ask * 1e9).astype(np.int64)
    return result


def test_phantom_float_midpoint_counterexample_is_removed():
    q = frame([0, 1000], [100.04, 100.03], [100.06, 100.07])
    floating = np.diff(((q.bid + q.ask) / 2).to_numpy())
    assert floating[0] != 0
    t, dm = fixed.mid_changes(q)
    assert len(t) == len(dm) == 0
    diagnostics = fixed.midpoint_diagnostics(q)
    assert diagnostics["float_only_phantom_midpoint_changes"] == 1
    assert diagnostics["integer_midpoint_changes"] == 0


def test_native_integer_legs_take_precedence_over_display_floats():
    q = frame([0, 1000, 2000], [100., 100.01, 100.02], raw=True)
    q[["bid", "ask"]] = 999.
    t, dm = fixed.mid_changes(q)
    assert t.tolist() == [1000, 2000]
    assert dm.tolist() == pytest.approx([.01, .01])


def test_sentinel_mask_precedes_integer_midpoint_sum():
    q = frame([0, 1000, 2000], [100., 100.01, 100.02], raw=True)
    q.loc[1, "bid_raw"] = np.iinfo(np.int64).max
    assert not len(fixed.mid_changes(q)[0])


def test_numerical_only_mid_changes_do_not_implicitly_apply_quality_rules():
    q = frame([0, 1000, 2000], [100., 101., 102.], [99., 100., 101.], raw=True)
    q["valid"] = [True, False, True]
    q["segment"] = [0, 1, 1]
    assert fixed.mid_changes(q)[1].tolist() == pytest.approx([1., 1.])
    assert not len(fixed.mid_changes_integer(q, quality=True)[0])


def test_crosscorr_aggregates_same_bin_before_normalization():
    # Starter event-energy normalization returns 2 for this example. The two
    # same-bin unit increments are one bin increment of 2, whose self score=1.
    t = np.array([0, 4], dtype=np.int64)
    cc = fixed.crosscorr(t, [1., 1.], t, [1., 1.], dt_ns=10, max_lag_ns=10)
    assert cc.loc[cc.lag_us.eq(0), "corr"].iloc[0] == pytest.approx(1.)
    assert cc["corr"].abs().max() <= 1


def test_sparse_crosscorr_matches_independent_dense_grid():
    tx, dx = np.array([0, 4, 20, 31]), np.array([1., 2., -.5, 1.])
    ty, dy = np.array([12, 16, 30, 44]), np.array([2., 1., -.25, 1.])
    cc = fixed.crosscorr(tx, dx, ty, dy, dt_ns=10, max_lag_ns=20)
    x, y = np.zeros(5), np.zeros(5)
    np.add.at(x, tx // 10, dx)
    np.add.at(y, ty // 10, dy)
    expected = []
    for lag in range(-2, 3):
        total = sum(x[i] * y[i + lag] for i in range(5) if 0 <= i + lag < 5)
        expected.append(total / np.sqrt(np.dot(x, x) * np.dot(y, y)))
    assert cc["corr"].tolist() == pytest.approx(expected)


def test_crosscorr_recovers_known_signed_lag():
    tx = np.arange(10, dtype=np.int64) * 1000
    dx = np.array([1., -1., 2., -.5, 1., -2., .75, 1., -.25, -1.])
    result = fixed.summarize_cc(fixed.crosscorr(tx, dx, tx + 50, dx, dt_ns=10, max_lag_ns=100))
    assert result["peak_lag_us"] == pytest.approx(.05)
    assert result["peak_corr"] == pytest.approx(1.)


def test_no_increment_energy_is_undefined_without_nanargmax_crash():
    cc = fixed.crosscorr(np.array([], np.int64), [], np.array([0], np.int64), [1.], dt_ns=10, max_lag_ns=10)
    assert cc["corr"].isna().all()
    assert all(np.isnan(value) for value in fixed.summarize_cc(cc).values())


def test_signed_ratio_uses_integer_numerator_and_denominator():
    leader = frame([0, 100, 200, 500], [100., 100.01, 100.02, 100.02], raw=True)
    follower = frame([0, 150, 250, 500], [100., 100.01, 100.02, 100.02], venue="BATS", raw=True)
    q = pd.concat([leader, follower], ignore_index=True)
    result = fixed.event_study(q, "XNAS", "TEST", np.array([0, 60]), support="complete").set_index(["venue", "offset_us"])
    assert result.loc[("XNAS", 0), "share"] == pytest.approx(1.)
    assert result.loc[("BATS", 0), "share"] == pytest.approx(0.)
    assert result.loc[("BATS", .06), "share"] == pytest.approx(1.)
    assert result.loc[("BATS", 0), "n"] == result.loc[("BATS", .06), "n"] == 2


def test_complete_support_does_not_extrapolate_past_last_observation():
    q = pd.concat([frame([0, 100, 200], [100., 100.01, 100.02], raw=True),
                   frame([0, 110, 210], [100., 100.01, 100.02], venue="BATS", raw=True)], ignore_index=True)
    complete = fixed.event_study(q, "XNAS", "TEST", np.array([0, 150]), support="complete")
    legacy = fixed.event_study(q, "XNAS", "TEST", np.array([0, 150]), support="legacy")
    assert complete.n.eq(0).all()
    assert legacy.n.eq(2).all()
    pd.testing.assert_frame_equal(fixed.event_study(q, "XNAS", "TEST", np.array([0, 150])), legacy)


def test_request_end_allows_known_quiet_state_but_not_beyond_request():
    q = pd.concat([frame([0, 100, 200], [100., 100.01, 100.02], raw=True),
                   frame([0, 110, 210], [100., 100.01, 100.02], venue="BATS", raw=True)], ignore_index=True)
    q.attrs.update(start_ns=0, end_ns=500)
    allowed = fixed.event_study(q, "XNAS", "TEST", np.array([0, 150]), support="complete")
    excluded = fixed.event_study(q, "XNAS", "TEST", np.array([0, 400]), support="complete")
    assert allowed.n.eq(2).all()
    assert excluded.n.eq(0).all()


def test_loader_nanosecond_bounds_attributes_are_honored():
    q = pd.concat([frame([0, 100, 200], [100., 100.01, 100.02], raw=True),
                   frame([0, 110, 210], [100., 100.01, 100.02], venue="BATS", raw=True)], ignore_index=True)
    q.attrs.update(requested_start_ns=0, requested_end_ns=500)
    result = fixed.event_study(q, "XNAS", "TEST", np.array([0, 150]), support="complete")
    assert result.n.eq(2).all()


def test_complete_support_never_crosses_invalid_state_barrier():
    leader = frame([0, 100, 200, 500], [100., 100.01, 100.02, 100.02], raw=True)
    follower = frame([0, 120, 150, 250, 500], [100., 100., 100.01, 100.02, 100.02], venue="BATS", raw=True)
    leader["valid"] = True
    follower["valid"] = [True, False, True, True, True]
    q = pd.concat([leader, follower], ignore_index=True)
    result = fixed.event_study(q, "XNAS", "TEST", np.array([0, 60]), support="complete")
    assert result.loc[result.venue.eq("BATS"), "n"].eq(1).all()


def test_zero_real_midpoint_move_creates_no_ratio_even_in_legacy_mode():
    q = frame([0, 1000], [100.04, 100.03], [100.06, 100.07], raw=True)
    result = fixed.event_study(q, "XNAS", "TEST", np.array([0, 100]), support="legacy")
    assert result.n.eq(0).all()
    assert result.share.isna().all()


def test_legacy_same_timestamp_individual_moves_remain_visible_not_coalesced():
    q = frame([0, 100, 100, 300], [100., 100.01, 100.02, 100.02], raw=True)
    result = fixed.event_study(q, "XNAS", "TEST", np.array([0]), support="legacy")
    # Two individual changes share one timestamp. Each strict baseline is the
    # state before the whole timestamp, so the residual ratio is 2, not 1.
    assert result.n.iloc[0] == 2
    assert result.share.iloc[0] == pytest.approx(2.)


@pytest.mark.parametrize("kwargs", [{"dt_ns": 0}, {"dt_ns": 10, "max_lag_ns": 15}, {"chunk": 0}])
def test_invalid_grid_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        fixed.crosscorr(np.array([0]), [1.], np.array([0]), [1.], **kwargs)
