"""Integration and independent-audit checks for the offline latency runner."""
import json

import numpy as np
import pandas as pd
import pytest

from colonylab.latency import QuoteSeries, reaction_events
from colonylab.latency_pipeline import audit_pair, _day_complete, sha256_file, DAY_FILES


def quotes(times, prices, valid=None, segments=None, end=1000):
    n = len(times)
    return QuoteSeries(np.array(times), np.array(prices), np.array(prices) + 2,
                       np.ones(n, dtype=bool) if valid is None else np.array(valid),
                       np.zeros(n, dtype=int) if segments is None else np.array(segments), 0, end)


def test_plain_scan_checks_both_sides_with_missing_state_and_barriers():
    leader = quotes([5, 10, 30, 40, 70, 80], [100, 101, 102, 103, 104, 105],
                    [True, True, False, True, True, True], [0, 0, 1, 1, 1, 1], end=100)
    follower = quotes([20, 25, 60, 75, 85], [100, 101, 104, 104, 105], end=100)
    for side in ("bid", "ask"):
        events = reaction_events(leader, follower, side, max_horizon_ns=50)
        result = audit_pair(leader, follower, side, events, 50)
        assert result["mismatches"] == 0
        assert result["cases"] == len(events)


def test_plain_scan_rejects_tampered_first_passage():
    leader = quotes([0, 10], [100, 101])
    follower = quotes([0, 20], [100, 101])
    events = reaction_events(leader, follower, max_horizon_ns=100)
    events.loc[0, "latency_ns"] += 1
    with pytest.raises(ValueError, match="first-passage mismatch"):
        audit_pair(leader, follower, "bid", events, 100)


def test_plain_scan_rejects_missing_leader_events():
    leader = quotes([0, 10], [100, 101])
    follower = quotes([0, 20], [100, 101])
    events = reaction_events(leader, follower, max_horizon_ns=100)
    with pytest.raises(ValueError, match="leader-event count"):
        audit_pair(leader, follower, "bid", events.iloc[:0], 100)


def test_day_checkpoint_binds_all_outputs_and_identity(tmp_path):
    identity = {"request": "exact", "code": "abc"}
    for name in DAY_FILES:
        pd.DataFrame({"value": [1]}).to_csv(tmp_path / name, index=False)
    marker = {"identity": identity, "outputs": {name: sha256_file(tmp_path / name) for name in DAY_FILES}}
    (tmp_path / "complete.json").write_text(json.dumps(marker))
    assert _day_complete(tmp_path, identity)
    assert not _day_complete(tmp_path, {"request": "other", "code": "abc"})
    (tmp_path / DAY_FILES[0]).write_text("tampered")
    assert not _day_complete(tmp_path, identity)
