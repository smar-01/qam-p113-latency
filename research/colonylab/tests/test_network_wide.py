"""Independent whole-day split and calibration checks for the wider study."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from colonylab import network


DAYS = [stamp.date().isoformat() for stamp in pd.bdate_range("2026-10-05", "2026-10-16")]
TRAINING_DAYS, HELD_OUT_DAYS = DAYS[:5], DAYS[5:]


def wide_frames(*, test_jump=0.02):
    frames = {}
    offsets = np.arange(0, 32100, 100)
    for day in DAYS:
        for dataset in ("A", "B"):
            origin = pd.Timestamp(f"{day}T14:00:00Z")
            jump = test_jump if day in HELD_OUT_DAYS else 0.01
            mid = 100 + (np.arange(len(offsets)) % 2) * jump
            frames[(dataset, "mbp-1", day)] = pd.DataFrame({
                "ts_recv": origin + pd.to_timedelta(offsets, unit="ms"),
                "ts_event": origin + pd.to_timedelta(offsets, unit="ms"),
                "symbol": "TEST", "bid_px_00": mid - 0.005, "ask_px_00": mid + 0.005,
                "bid_sz_00": 100, "ask_sz_00": 100, "flags": 128,
            })
    return frames


def run_wide(frames, directory, **kwargs):
    with patch.object(network, "_plot"):
        return network.run_network(frames, directory, **kwargs)


def test_five_whole_heldout_days_are_after_five_training_days(tmp_path):
    summary = run_wide(wide_frames(), tmp_path, held_out_days=HELD_OUT_DAYS)
    assert summary["training_days"] == TRAINING_DAYS
    assert summary["held_out_days"] == HELD_OUT_DAYS
    assert summary["held_out_day"] is None
    events = pd.read_csv(tmp_path / "shock_events.csv")
    expected = {day: "held_out" if day in HELD_OUT_DAYS else "training" for day in DAYS}
    assert events.groupby("day")["split"].nunique().eq(1).all()
    assert events.groupby("day")["split"].first().to_dict() == expected
    edges = pd.read_csv(tmp_path / "edge_response.csv")
    assert set(edges.loc[edges.split == "held_out", "day"]) == set(HELD_OUT_DAYS)
    assert edges.groupby("day")["split"].nunique().eq(1).all()


def test_heldout_jump_changes_cannot_change_thresholds_or_training_events(tmp_path):
    before_dir, after_dir = tmp_path / "before", tmp_path / "after"
    run_wide(wide_frames(test_jump=0.02), before_dir, held_out_days=HELD_OUT_DAYS)
    run_wide(wide_frames(test_jump=4.0), after_dir, held_out_days=HELD_OUT_DAYS)
    before = pd.read_csv(before_dir / "shock_thresholds.csv")
    after = pd.read_csv(after_dir / "shock_thresholds.csv")
    pd.testing.assert_frame_equal(before, after)
    np.testing.assert_allclose(before.threshold_usd, 0.01, rtol=0, atol=1e-10)
    before_events = pd.read_csv(before_dir / "shock_events.csv")
    after_events = pd.read_csv(after_dir / "shock_events.csv")
    pd.testing.assert_frame_equal(
        before_events.loc[before_events.split == "training"].reset_index(drop=True),
        after_events.loc[after_events.split == "training"].reset_index(drop=True),
    )


def test_daily_event_totals_reproduce_pooled_heldout_counts(tmp_path):
    summary = run_wide(wide_frames(), tmp_path, held_out_days=HELD_OUT_DAYS)
    daily = pd.DataFrame(summary["held_out_100ms_by_day"])
    assert daily.day.tolist() == HELD_OUT_DAYS
    pooled = summary["held_out_100ms"]
    for column in ("matched_edge_event_count", "hits", "placebo_mean_hits"):
        assert daily[column].sum() == pytest.approx(pooled[column])
    denominator = daily.matched_edge_event_count.sum()
    assert pooled["response_probability"] == pytest.approx(daily.hits.sum() / denominator)
    assert pooled["placebo_response_probability"] == pytest.approx(daily.placebo_mean_hits.sum() / denominator)


@pytest.mark.parametrize("held_out_days", [
    [], [DAYS[-1], DAYS[-1]], ["2026-10-19"], DAYS, [DAYS[0]], [DAYS[3], DAYS[-1]],
])
def test_invalid_or_nonchronological_explicit_holdout_is_rejected(tmp_path, held_out_days):
    with pytest.raises(ValueError):
        run_wide(wide_frames(), tmp_path, held_out_days=held_out_days)


def test_default_old_pilot_still_uses_only_last_day(tmp_path):
    summary = run_wide(wide_frames(), tmp_path)
    assert summary["training_days"] == DAYS[:-1]
    assert summary["held_out_day"] == DAYS[-1]
    assert summary["held_out_days"] == [DAYS[-1]]


def test_reverse_custom_date_order_cannot_turn_future_days_into_training(tmp_path):
    with pytest.raises(ValueError):
        run_wide(wide_frames(), tmp_path, day_order=list(reversed(DAYS)))
