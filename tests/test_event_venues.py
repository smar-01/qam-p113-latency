"""event_venues.py: known answers on hand-built quotes, synthetic data and real DBN bytes."""
import datetime as dt
from types import SimpleNamespace as NS

import databento as db
import databento_dbn as dbn
import numpy as np
import pandas as pd
import pytest

import event_venues as ev

S, MS, US = 1_000_000_000, 1_000_000, 1_000


def _states(**venues):
    """venue=[(t, bid, ask), ...] -> venue_states-style arrays."""
    return {v: tuple(np.array(c, dtype=np.int64 if i == 0 else float) for i, c in enumerate(zip(*rows)))
            for v, rows in venues.items()}


def test_find_moves_picks_largest_non_overlapping_windows_above_threshold():
    grid = np.arange(100) * S
    cons = np.full(100, 100.0)
    cons[30:] = 101.0    # +100 bps step at 30
    cons[70:] = 100.5    # about -50 bps step at 70
    moves = ev.find_moves(grid, cons, window_s=5, n_moves=10, min_bps=10)
    assert len(moves) == 2
    assert moves["move_bps"].round().tolist() == [100, -50]
    for (a, b) in zip(moves["start_ns"], moves["end_ns"]):
        assert b - a == 5 * S
    assert ev.find_moves(grid, cons, window_s=5, n_moves=10, min_bps=200).empty


def test_halfway_crossing_order_lag_already_and_no_cross():
    states = _states(
        LEAD=[(0, 99.99, 100.01), (2 * S, 100.99, 101.01)],
        SLOW=[(0, 99.99, 100.01), (2 * S + 300 * US, 100.99, 101.01)],
        AHEAD=[(0, 100.99, 101.01)],           # already at the new price when the window opens
        STUCK=[(0, 99.99, 100.01)],            # never moves
    )
    moves = pd.DataFrame({"move": [1], "start_ns": [1 * S], "end_ns": [6 * S], "p0": [100.0], "p1": [101.0]})
    c = ev.halfway_crossings(states, moves, 5 * S).set_index("venue")
    assert c.loc["AHEAD", "status"] == "already" and c.loc["AHEAD", "lag_us"] == 0
    assert c.loc["LEAD", "lag_us"] == pytest.approx(1e6)    # 1 s after the window opened
    assert c.loc["SLOW", "lag_us"] - c.loc["LEAD", "lag_us"] == pytest.approx(300)
    assert c.loc["STUCK", "status"] == "no_cross" and np.isnan(c.loc["STUCK", "lag_us"])
    fm = ev.first_mover_summary(c.reset_index()).set_index("venue")
    assert fm.loc["AHEAD", "first_share"] == 1 and fm.loc["STUCK", "n_reached_halfway"] == 0


def test_nbbo_time_shares_are_time_weighted():
    # A is best bid for the first 3 s, B improves the bid for the last 1 s; same ask throughout.
    states = _states(A=[(0, 10.00, 10.02)], B=[(0, 9.99, 10.02), (3 * S, 10.01, 10.02)])
    moves = pd.DataFrame({"start_ns": [3 * S], "end_ns": [4 * S]})
    s = ev.nbbo_shares(states, 0, 4 * S, moves, chunk=1).set_index(["window", "venue"])
    assert s.loc[("session", "A"), "pct_time_at_best_bid"] == pytest.approx(75)
    assert s.loc[("session", "B"), "pct_time_at_best_bid"] == pytest.approx(25)
    assert s.loc[("session", "A"), "pct_time_at_best_ask"] == pytest.approx(100)
    # B is 1 cent behind for 3 of 4 s -> 0.75 cents on average
    assert s.loc[("session", "B"), "avg_bid_behind_best_cents"] == pytest.approx(0.75)
    assert s.loc[("big_moves", "B"), "pct_time_at_best_bid"] == pytest.approx(100)
    assert s.loc[("session", "A"), "pct_time_nbbo_locked_or_crossed"] == 0


def test_invalid_states_are_gaps_not_bridged():
    q = pd.DataFrame({"ts_recv": [0, 1 * S, 2 * S], "ts_event": [0, 1 * S, 2 * S], "venue": "X",
                      "symbol": "M", "bid": [10.0, 10.05, 10.1], "ask": [10.02, 10.0, 10.12]})
    t, bid, ask = ev.venue_states(q, "ts_recv")["X"]
    assert np.isnan(bid[1]) and np.isnan(ask[1])    # crossed venue quote at 1 s is not a price
    assert bid[2] == 10.1


def test_synthetic_run_recovers_planted_venue_order(tmp_path):
    from p113.synthetic import make_quotes
    q = make_quotes(symbol="META", seconds=900, volatile_lag_mult=1.0,
                    lags_ns={"XNYS": 50 * US, "BATS": 150 * US, "IEXG": 400 * US})
    ev.run(q, tmp_path, symbol="META", day="2025-10-15", session="all", clocks=["ts_event"],
           move_window_s=10, n_moves=10, min_move_bps=10, horizon_ns=10 * MS)
    fm = pd.read_csv(tmp_path / "first_mover_ts_event.csv").set_index("venue")
    assert fm.index.tolist() == ["XNAS", "XNYS", "BATS", "IEXG"]
    assert fm.loc["XNAS", "first_share"] == 1
    for venue, lag in {"XNYS": 50, "BATS": 150, "IEXG": 400}.items():
        assert fm.loc[venue, "median_lag_behind_first_us"] == pytest.approx(lag, abs=10)
    lead = pd.read_csv(tmp_path / "leadership_share_ts_event.csv", index_col=0)
    assert lead["session"].idxmax() == "XNAS"
    for name in ("summary.md", "price_paths_ts_event.png", "biggest_move_ts_event.png",
                 "nbbo_time_share_ts_event.png", "latency_heatmap_ts_event.png"):
        assert (tmp_path / name).is_file()


def _publisher(dataset):
    return next(p for p in db.Publisher if p.dataset.value == dataset).to_int()


def _write_dbn(path, dataset, rows, t0, t1):
    """rows: (ts_event, ts_recv, bid_dollars, ask_dollars)."""
    day = dt.date(2026, 9, 9)
    meta = dbn.Metadata(dataset=dataset, start=t0, end=t1, stype_in=dbn.SType.RAW_SYMBOL,
                        stype_out=dbn.SType.INSTRUMENT_ID, schema=dbn.Schema.MBP_1, symbols=["META"],
                        mappings=[NS(raw_symbol="META", intervals=[
                            NS(start_date=day, end_date=day + dt.timedelta(days=1), symbol="1")])])
    buf = bytes(meta.encode())
    pub = _publisher(dataset)
    for seq, (te, tr, b, a) in enumerate(rows, 1):
        bid, ask = int(round(b * 1e9)), int(round(a * 1e9))
        lv = dbn.BidAskPair(bid_px=bid, ask_px=ask, bid_sz=100, ask_sz=100, bid_ct=1, ask_ct=1)
        buf += bytes(dbn.MBP1Msg(publisher_id=pub, instrument_id=1, ts_event=te, price=bid, size=100,
                                 action=dbn.Action.ADD, side=dbn.Side.BID, depth=0, flags=130,
                                 ts_recv=tr, ts_in_delta=0, sequence=seq, levels=lv))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf)


def test_real_dbn_files_end_to_end_through_main(tmp_path):
    t0 = pd.Timestamp("2026-09-09 10:00", tz=ev.NY).value
    t1 = t0 + 60 * S
    jump = t0 + 20 * S + 123_456
    lead = [(t0 + 1, t0 + 1, 650.00, 650.02), (jump, jump + 10 * US, 652.00, 652.02)]
    follow = [(t0 + 2, t0 + 2, 650.00, 650.02), (jump + 250 * US, jump + 300 * US, 652.00, 652.02)]
    _write_dbn(tmp_path / "raw/XNAS.ITCH/mbp-1_2026-09-09_2026-09-10_abc.dbn.zst", "XNAS.ITCH", lead, t0, t1)
    _write_dbn(tmp_path / "raw/BATS.PITCH/mbp-1_2026-09-09_2026-09-10_def.dbn.zst", "BATS.PITCH", follow, t0, t1)
    _write_dbn(tmp_path / "raw/BATS.PITCH/mbp-1_2026-09-08_2026-09-09_xyz.dbn.zst", "BATS.PITCH", follow,
               t0 - 86_400 * S, t1 - 86_400 * S)    # another day: must be ignored
    assert len(ev.day_files(tmp_path / "raw", "2026-09-09")) == 2
    out = tmp_path / "out"
    info = ev.main(["--data", str(tmp_path / "raw"), "--day", "2026-09-09", "--session", "all",
                    "--out", str(out), "--n-moves", "1"])
    assert info["venues"] == ["BATS", "XNAS"]
    for clock, lag in (("ts_recv", 290), ("ts_event", 250)):
        fm = pd.read_csv(out / f"first_mover_{clock}.csv").set_index("venue")
        assert fm.loc["XNAS", "first_share"] == 1
        assert fm.loc["BATS", "median_lag_behind_first_us"] == pytest.approx(lag)
    moves = pd.read_csv(out / "big_moves_ts_recv.csv")
    assert len(moves) == 1 and moves["move_bps"].iloc[0] == pytest.approx(30.77, abs=.01)


def test_streamed_latency_summaries_equal_the_team_engine():
    from p113 import latency
    from p113.load import bbo_changes
    from p113.synthetic import make_quotes
    rq = bbo_changes(make_quotes(symbol="META", seconds=300, lags_ns={"XNYS": 50 * US, "BATS": 150 * US}))
    moves = pd.DataFrame({"start_ns": [rq["ts_recv"].min() + 100 * S], "end_ns": [rq["ts_recv"].min() + 110 * S]})
    overall, by_window, n = ev.pair_latency_summaries(rq, "ts_recv", 10 * MS, moves)
    team = latency.pair_latencies(rq, clock="ts_recv", horizon_ns=10 * MS)
    assert n == len(team)
    expected = latency.summarize(team)
    pd.testing.assert_frame_equal(overall.sort_values(["leader", "follower"]).reset_index(drop=True),
                                  expected.sort_values(["leader", "follower"]).reset_index(drop=True))
    assert set(by_window["window"]) == {"big_move", "other"}
    assert by_window.groupby(["leader", "follower"])["n_events"].sum().tolist() == \
        expected.set_index(["leader", "follower"])["n_events"].tolist()


def _with_thin_stub_venue(q, venue="XPSX", seed=3):
    """Add a venue whose bid/ask are far-away placeholders: its midpoint jumps around randomly."""
    rng = np.random.default_rng(seed)
    lead = q[q["venue"] == "XNAS"].sort_values("ts_recv")
    times = np.arange(lead["ts_recv"].min(), lead["ts_recv"].max(), 200 * MS)
    mid = np.interp(times, lead["ts_recv"], (lead["bid"] + lead["ask"]) / 2)
    stub = pd.DataFrame({"ts_recv": times, "ts_event": times, "venue": venue, "symbol": q["symbol"].iloc[0],
                         "bid": np.round(mid - rng.uniform(0.5, 6, len(times)), 2),
                         "ask": np.round(mid + rng.uniform(0.5, 6, len(times)), 2), "bid_sz": 100, "ask_sz": 100})
    return pd.concat([q, stub], ignore_index=True).sort_values("ts_recv", kind="stable").reset_index(drop=True)


def test_thin_venue_with_placeholder_quotes_is_not_called_the_first_mover(tmp_path):
    from p113.synthetic import make_quotes
    q = _with_thin_stub_venue(make_quotes(symbol="META", seconds=900, volatile_lag_mult=1.0,
                                          lags_ns={"XNYS": 50 * US, "BATS": 150 * US}))
    kw = dict(symbol="META", day="2025-10-15", session="all", clocks=["ts_event"], move_window_s=10,
              n_moves=10, min_move_bps=10, horizon_ns=10 * MS, skip_open_close_min=0)
    ev.run(q, tmp_path / "raw_mid", max_spread_bps=None, **kw)       # the original behaviour
    unfiltered = pd.read_csv(tmp_path / "raw_mid/first_mover_ts_event.csv").set_index("venue")
    assert unfiltered.loc["XPSX", "first_share"] > 0.3                # placeholder noise looked like leading
    ev.run(q, tmp_path / "fixed", **kw)                               # default 25 bps spread limit
    fm = pd.read_csv(tmp_path / "fixed/first_mover_ts_event.csv").set_index("venue")
    assert fm.loc["XNAS", "first_share"] == 1
    assert fm.loc["XPSX", "n_reached_halfway"] == 0
    assert fm.loc["BATS", "median_lag_behind_first_us"] == pytest.approx(150, abs=10)
    lead = pd.read_csv(tmp_path / "fixed/leadership_share_ts_event.csv", index_col=0)
    assert "XPSX" not in lead["session_core_venues"].dropna().index
    assert lead["session_core_venues"].idxmax() == "XNAS"


def test_open_and_close_windows_are_skipped_when_finding_moves():
    grid = np.arange(100) * S
    cons = np.full(100, 100.0)
    cons[30:] = 102.0                     # biggest jump, inside the excluded interval
    cons[70:] = 102.5                     # smaller jump outside it
    moves = ev.find_moves(grid, cons, 5, 5, 10, exclude=[(28 * S, 33 * S)])
    assert len(moves) == 1 and moves["start_ns"].iloc[0] >= 64 * S


def test_wide_quotes_are_dropped_only_when_a_spread_limit_is_given():
    q = pd.DataFrame({"ts_recv": [0, S], "ts_event": [0, S], "venue": "X", "symbol": "M",
                      "bid": [99.99, 90.0], "ask": [100.01, 110.0]})
    assert np.isfinite(ev.venue_states(q, "ts_recv")["X"][1]).all()
    _, bid, _ = ev.venue_states(q, "ts_recv", max_spread_bps=25)["X"]
    assert np.isfinite(bid[0]) and np.isnan(bid[1])
