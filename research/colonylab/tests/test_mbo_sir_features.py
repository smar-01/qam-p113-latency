"""Independent timing, training-only calibration and matched-cohort checks."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from colonylab.mbo_sir_features import (CONTROL_NAMES, build_day_panels,
                                       fit_thresholds, market_features)


ORIGIN = pd.Timestamp("2026-06-01T13:30:00Z").value
MS = 1_000_000
SECOND = 1_000_000_000


def replay(events_ms=(), *, length=450):
    index = np.arange(length)
    grid = pd.DataFrame({
        "time_ns": ORIGIN + index * 100 * MS, "valid": True, "initialized": True,
        "event_complete": True, "segment": 1, "mid": 100., "spread_bps": 2.,
        "depth_imbalance_1": .2, "depth_imbalance_10": -.1,
        "bid_depth_10": 2000, "ask_depth_10": 3000, "order_count": 40,
        "state_change_age_ms": 1000. + index * 100,
    })
    for prefix, multiples in (("add_shares", (2, 1)), ("removed_shares", (1, 3)),
                              ("trade_shares", (4, 2, 1))):
        for side, multiple in zip(("B", "A", "N"), multiples):
            grid[f"cum_{prefix}_{side}"] = index * multiple
    return {"grid": grid,
            "jumps": pd.DataFrame({"time_ns": ORIGIN + np.asarray(events_ms, dtype=np.int64) * MS,
                                    "jump": np.full(len(events_ms), .01)}),
            "reset_times_ns": np.array([], dtype=np.int64),
            "invalid_times_ns": np.array([], dtype=np.int64),
            "atomic_intervals_ns": np.empty((0, 2), dtype=np.int64),
            "analysis_start_ns": ORIGIN, "analysis_end_ns": ORIGIN + length * 100 * MS,
            "day": "2026-06-01"}


def market():
    index = np.arange(-12, 46)
    return pd.DataFrame({"availability_ns": ORIGIN + index * SECOND,
                         "mid": 100. + index * .01, "valid": True})


def panels(replays=None, **kwargs):
    replays = replays or {"A": replay(), "B": replay()}
    return build_day_panels(replays, market(), {venue: .005 for venue in replays}, **kwargs)


def row(panel, milliseconds):
    selected = np.flatnonzero(panel["time_ns"] == ORIGIN + milliseconds * MS)
    assert len(selected) == 1
    return int(selected[0])


def test_frozen_controls_flows_and_book_age_have_known_values():
    panel = panels()["A"]
    position = row(panel, 32_000)
    features = dict(zip(panel["names"], panel["x"][position]))
    assert panel["names"] == CONTROL_NAMES and panel["x"].shape[1] == 17
    assert features["add_imbalance_100ms"] == pytest.approx(1 / 3)
    assert features["remove_imbalance_100ms"] == pytest.approx(-.5)
    assert features["trade_imbalance_100ms"] == pytest.approx(2 / 7)
    assert features["log_add_shares_100ms"] == pytest.approx(np.log1p(3))
    assert features["log_removed_shares_100ms"] == pytest.approx(np.log1p(4))
    assert features["log_trade_shares_100ms"] == pytest.approx(np.log1p(7))
    assert features["log_book_age_ms"] == pytest.approx(np.log1p(33_000))
    assert features["log_depth_10"] == pytest.approx(np.log1p(5000))


def test_quiet_initialized_book_remains_eligible_with_old_update_age():
    replays = {"A": replay(), "B": replay()}
    for item in replays.values():
        item["grid"]["state_change_age_ms"] = 60_000.
    result = panels(replays)
    assert all(len(panel["y"]) for panel in result.values())
    assert result["A"]["time_ns"].min() >= ORIGIN + 30 * SECOND


@pytest.mark.parametrize("infection_ms,expected", [(30_097, True), (30_098, False), (30_100, False)])
def test_source_infection_requires_strict_two_millisecond_lead(infection_ms, expected):
    panel = panels({"A": replay(), "B": replay([infection_ms])})["A"]
    position = row(panel, 30_100)
    assert bool(panel["infected"][position, 0]) == expected
    assert panel["source_names"] == ["B"]


def test_shifted_sources_are_thirty_seconds_in_the_past_without_wrapping():
    panel = panels({"A": replay(), "B": replay([97])})["A"]
    position = row(panel, 30_100)
    assert panel["shifted_infected"][position, 0]
    assert not panel["infected"][position, 0]
    assert panel["time_ns"].min() >= ORIGIN + 30 * SECOND


def test_future_infection_changes_label_but_not_predictors_at_its_anchor():
    before = panels()["A"]
    after = panels({"A": replay([32_005]), "B": replay()})["A"]
    a, b = row(before, 32_000), row(after, 32_000)
    assert before["y"][a] == 0 and after["y"][b] == 1
    np.testing.assert_array_equal(before["x"][a], after["x"][b])
    np.testing.assert_array_equal(before["legacy"][a], after["legacy"][b])
    np.testing.assert_array_equal(before["infected"][a], after["infected"][b])


def test_forecast_remains_ten_milliseconds_on_one_hundred_millisecond_grid():
    panel = panels({"A": replay([32_055]), "B": replay()})["A"]
    assert panel["y"][row(panel, 32_000)] == 0
    assert not panel["y"].any()
    assert np.all(panel["time_ns"] % (100 * MS) == 0)


def test_current_infection_is_a_label_and_does_not_enter_past_legacy_counts():
    panel = panels({"A": replay([32_000]), "B": replay([32_000])})["A"]
    position = row(panel, 32_000)
    assert panel["y"][position] == 1
    assert not panel["legacy"][position].any()
    assert not panel["infected"][position].any()


def test_all_models_share_single_row_identity_and_legacy_width():
    panel = panels()["A"]
    lengths = {len(panel[name]) for name in ("x", "y", "time_ns", "infected", "shifted_infected", "legacy")}
    assert lengths == {len(panel["time_ns"])}
    assert panel["legacy"].shape[1] == 6
    assert panel["legacy_names"] == ["own_infections_10ms", "own_infections_100ms", "own_infections_1000ms",
                                      "other_B_infections_10ms", "other_B_infections_100ms", "other_B_infections_1000ms"]


def test_reset_and_invalidity_exclude_full_past_history_on_every_target():
    replays = {"A": replay(), "B": replay()}
    replays["B"]["reset_times_ns"] = np.array([ORIGIN + 31_550 * MS], dtype=np.int64)
    replays["B"]["grid"].loc[replays["B"]["grid"].time_ns.ge(ORIGIN + 31_600 * MS), "segment"] = 2
    result = panels(replays)
    for panel in result.values():
        assert ORIGIN + 32_000 * MS not in panel["time_ns"]
        row(panel, 32_700)


def test_quality_barrier_inside_target_future_bin_excludes_that_target_only():
    replays = {"A": replay(), "B": replay()}
    replays["A"]["invalid_times_ns"] = np.array([ORIGIN + 32_005 * MS], dtype=np.int64)
    result = panels(replays)
    assert ORIGIN + 32_000 * MS not in result["A"]["time_ns"]
    row(result["B"], 32_000)


def test_atomic_event_spanning_target_horizon_is_not_a_known_outcome():
    replays = {"A": replay(), "B": replay()}
    replays["A"]["atomic_intervals_ns"] = np.array([[ORIGIN + 32_002 * MS, ORIGIN + 32_003 * MS]])
    assert ORIGIN + 32_000 * MS not in panels(replays)["A"]["time_ns"]


def test_unobserved_shifted_book_is_not_invented_as_susceptible():
    replays = {"A": replay(), "B": replay()}
    replays["B"]["grid"].loc[0, "valid"] = False
    result = panels(replays)
    assert ORIGIN + 30_100 * MS not in result["A"]["time_ns"]


def test_market_interval_end_is_available_only_strictly_after_its_timestamp():
    source = market()
    source.loc[source.availability_ns.eq(ORIGIN + 32 * SECOND), "mid"] = 200.
    features = market_features(source, np.array([ORIGIN + 32 * SECOND, ORIGIN + 32_001 * MS]))
    assert features["valid"].all()
    assert features["x"][0, 0] < 10
    assert features["x"][1, 0] > 9000


def test_future_market_quotes_cannot_change_current_features():
    source = market()
    anchors = np.array([ORIGIN + 32 * SECOND])
    before = market_features(source, anchors)
    changed = source.copy()
    changed.loc[changed.availability_ns.ge(anchors[0]), "mid"] = 2000.
    after = market_features(changed, anchors)
    np.testing.assert_array_equal(before["x"], after["x"])
    np.testing.assert_array_equal(before["valid"], after["valid"])


def test_bad_or_stale_market_observation_blocks_carried_older_quote():
    source = market()
    source.loc[source.availability_ns.eq(ORIGIN + 31 * SECOND), "valid"] = False
    assert not market_features(source, np.array([ORIGIN + 32 * SECOND]))["valid"].any()
    old = source.loc[source.availability_ns.lt(ORIGIN + 20 * SECOND)]
    assert not market_features(old, np.array([ORIGIN + 32 * SECOND]))["valid"].any()


def test_invalid_or_conflicting_market_interval_ties_are_barriers():
    source = market()
    tie = source.loc[source.availability_ns.eq(ORIGIN + 31 * SECOND)].copy()
    tie["mid"] += 1
    source = pd.concat([source, tie], ignore_index=True)
    assert not market_features(source, np.array([ORIGIN + 32 * SECOND]))["valid"].any()


def test_calibration_uses_only_training_payloads_and_known_quantile():
    class ForbiddenHeldout(dict):
        def __getitem__(self, key):
            raise AssertionError("Held-out payload read during calibration")
    item = replay()
    item["jumps"] = pd.DataFrame({"jump": [.01, -.02, .03, 0.]})
    result = fit_thresholds([("A", "2026-06-01", item), ("A", "2026-06-15", ForbiddenHeldout())],
                            training_days=["2026-06-01"], expected_datasets=["A"])
    assert result.threshold.iloc[0] == pytest.approx(.025)
    assert result.training_jump_count.iloc[0] == 3


def test_calibration_rejects_duplicate_or_missing_intended_training_replay():
    item = ("A", "2026-06-01", replay())
    with pytest.raises(ValueError, match="Duplicate"):
        fit_thresholds([item, item], training_days=[item[1]], expected_datasets=["A"])
    with pytest.raises(ValueError, match="Missing"):
        fit_thresholds([item], training_days=[item[1]], expected_datasets=["A", "B"])


def test_empty_training_jumps_keep_frozen_half_cent_floor():
    result = fit_thresholds([("A", "2026-06-01", replay())], training_days=["2026-06-01"], expected_datasets=["A"])
    assert result.threshold.iloc[0] == .005 and result.training_jump_count.iloc[0] == 0


def test_mismatched_venue_grids_or_protocol_settings_fail_instead_of_intersecting():
    replays = {"A": replay(), "B": replay()}
    replays["B"]["grid"] = replays["B"]["grid"].iloc[1:].reset_index(drop=True)
    with pytest.raises(ValueError, match="identical complete"):
        panels(replays)
    with pytest.raises(ValueError, match="Controls differ"):
        panels(protocol={"controls": CONTROL_NAMES[:-1]})


def test_future_appended_jumps_do_not_change_earlier_features_or_cohort():
    earlier = panels({"A": replay(), "B": replay([30_097])})["A"]
    later = panels({"A": replay([40_005]), "B": replay([30_097, 41_005])})["A"]
    cutoff = ORIGIN + 39 * SECOND
    a, b = earlier["time_ns"] < cutoff, later["time_ns"] < cutoff
    np.testing.assert_array_equal(earlier["time_ns"][a], later["time_ns"][b])
    for name in ("x", "legacy", "infected", "shifted_infected", "y"):
        np.testing.assert_array_equal(earlier[name][a], later[name][b])
