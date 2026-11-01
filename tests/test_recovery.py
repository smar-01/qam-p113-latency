"""Synthetic streams with a known lag: the code must recover it."""
import numpy as np
import pytest

from p113 import latency, leadlag, regimes
from p113.load import bbo_changes
from p113.synthetic import make_quotes

US = 1_000


@pytest.fixture(scope="module")
def quotes():
    # XNYS copies XNAS 50 µs later, BATS 200 µs later, both +-5 µs noise; no regime effect
    return bbo_changes(make_quotes(seconds=1_800, jitter_ns=5 * US,
                                   feed_ns={"XNAS": 0, "XNYS": 20 * US, "BATS": 30 * US}))


@pytest.fixture(scope="module")
def summary_event(quotes):
    return latency.summarize(latency.pair_latencies(quotes, clock="ts_event"))


def cell(summ, a, b, col="median_us"):
    return summ.set_index(["leader", "follower"]).loc[(a, b), col]


def test_latency_recovers_true_lag_on_exchange_clock(summary_event):
    assert cell(summary_event, "XNAS", "XNYS") == pytest.approx(50, abs=2)
    assert cell(summary_event, "XNAS", "BATS") == pytest.approx(200, abs=2)
    assert cell(summary_event, "XNYS", "BATS") == pytest.approx(150, abs=3)


def test_almost_every_leader_move_is_matched(summary_event):
    assert cell(summary_event, "XNAS", "XNYS", "match_rate") > 0.99
    assert cell(summary_event, "XNAS", "BATS", "match_rate") > 0.99


def test_reverse_direction_is_not_counted_as_leading(summary_event):
    # when XNYS moves, XNAS is already there: almost nothing to match
    behind = cell(summary_event, "XNYS", "XNAS", "n_behind")
    events = cell(summary_event, "XNYS", "XNAS", "n_events")
    assert behind / events < 0.02


def test_capture_clock_adds_feed_delay_difference(quotes):
    s = latency.summarize(latency.pair_latencies(quotes, clock="ts_recv"))
    assert cell(s, "XNAS", "XNYS") == pytest.approx(50 + 20, abs=2)
    assert cell(s, "XNAS", "BATS") == pytest.approx(200 + 30, abs=2)


def test_noise_spread_shows_in_quantiles(summary_event):
    p10, p90 = cell(summary_event, "XNAS", "XNYS", "p10_us"), cell(summary_event, "XNAS", "XNYS", "p90_us")
    # normal(0, 5 µs): p10..p90 is about +-6.4 µs
    assert 40 < p10 < 48 and 52 < p90 < 60


def test_leadership_share_orders_venues(summary_event):
    share = latency.leadership_share(summary_event)
    assert share["XNAS"] > 0.98
    assert share["BATS"] < 0.02
    assert 0.4 < share["XNYS"] < 0.6


def test_heatmap_matrix_shape(summary_event):
    m = latency.latency_matrix(summary_event)
    assert list(m.index) == list(m.columns) == ["BATS", "XNAS", "XNYS"]
    assert np.isnan(m.loc["XNAS", "XNAS"])


def test_crosscorr_peak_at_true_lag(quotes):
    mc = {v: leadlag.mid_changes(g, "ts_event") for v, g in quotes.groupby("venue")}
    cc = leadlag.crosscorr(*mc["XNAS"], *mc["XNYS"], dt_ns=10 * US, max_lag_ns=1_000 * US)
    s = leadlag.summarize_cc(cc)
    assert s["peak_lag_us"] == pytest.approx(50, abs=10)
    assert s["peak_corr"] > 0.5
    assert s["llr"] > 10
    cc2 = leadlag.crosscorr(*mc["XNYS"], *mc["BATS"], dt_ns=10 * US, max_lag_ns=1_000 * US)
    assert leadlag.summarize_cc(cc2)["peak_lag_us"] == pytest.approx(150, abs=10)


def test_crosscorr_is_antisymmetric_in_sign(quotes):
    mc = {v: leadlag.mid_changes(g, "ts_event") for v, g in quotes.groupby("venue")}
    cc = leadlag.crosscorr(*mc["XNYS"], *mc["XNAS"], dt_ns=10 * US, max_lag_ns=1_000 * US)
    assert leadlag.summarize_cc(cc)["peak_lag_us"] == pytest.approx(-50, abs=10)


def test_event_study_catch_up(quotes):
    offs = np.array([0, 20, 100, 300]) * US
    es = leadlag.event_study(quotes, "XNAS", "SYN", offs, clock="ts_event").set_index(["venue", "offset_us"])
    assert es.loc[("XNAS", 0), "share"] == pytest.approx(1, abs=0.01)
    assert es.loc[("XNYS", 20), "share"] == pytest.approx(0, abs=0.02)
    assert es.loc[("XNYS", 100), "share"] == pytest.approx(1, abs=0.02)
    assert es.loc[("BATS", 100), "share"] == pytest.approx(0, abs=0.02)
    assert es.loc[("BATS", 300), "share"] == pytest.approx(1, abs=0.02)


def test_regime_split_recovers_slower_volatile_lag():
    q = bbo_changes(make_quotes(seconds=3_600, volatile_lag_mult=2.0, lags_ns={"XNYS": 50 * US}))
    rv = regimes.label(regimes.realized_vol(q), q=0.5)
    lat = latency.pair_latencies(q, clock="ts_event")
    lat["regime"] = regimes.tag(lat, rv)
    s = latency.summarize(lat, by=("leader", "follower", "regime")).set_index(["leader", "follower", "regime"])
    assert s.loc[("XNAS", "XNYS", "calm"), "median_us"] == pytest.approx(50, abs=2)
    assert s.loc[("XNAS", "XNYS", "volatile"), "median_us"] == pytest.approx(100, abs=3)


def test_censoring_beyond_horizon():
    q = bbo_changes(make_quotes(seconds=300, lags_ns={"XNYS": 5_000 * US}))
    lat = latency.pair_latencies(q, clock="ts_event", horizon_ns=1_000 * US)
    a = lat[(lat["leader"] == "XNAS") & (lat["follower"] == "XNYS")]
    assert (a["status"] == "censored").mean() > 0.95
