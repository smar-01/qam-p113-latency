from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from colonylab.context import build_maker_context, build_venue_context


FRIDAY = "2026-10-23"
MONDAY = "2026-10-26"
EXCHANGES = ["NYSE (N)", "Nasdaq (Q)"]
TRF = "FINRA / Nasdaq TRF Carteret (DQ)"


def history_frame() -> pd.DataFrame:
    days = {
        FRIDAY: [(30.125, 20.25, 10.125), (10.375, 30.5, 20.125), (59.5, 49.25, 69.75)],
        MONDAY: [(180.0, 80.0, 20.0), (10.0, 30.0, 40.0), (10.0, 90.0, 140.0)],
    }
    counts = [(3, 2, 1), (1, 3, 2), (6, 5, 7)]
    rows = []
    # Deliberately use reverse source-date order: lags must follow dates.
    for date in (MONDAY, FRIDAY):
        for participant, shares, trades in zip([*EXCHANGES, TRF], days[date], counts):
            row = {"Day": date, "Market Participant": participant}
            for tape, volume, count in zip("ABC", shares, trades):
                row[f"Tape {tape} Shares"] = volume
                row[f"Tape {tape} Notional"] = volume * 100
                row[f"Tape {tape} Trade Count"] = count
            row["Total Shares"] = sum(shares)
            row["Total Notional"] = sum(shares) * 100
            row["Total Trade Count"] = sum(trades)
            rows.append(row)
    return pd.DataFrame(rows)


def write_history(tmp_path, frame=None):
    path = tmp_path / "market_history_2026.csv"
    (history_frame() if frame is None else frame).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def venue_context(tmp_path_factory):
    root = tmp_path_factory.mktemp("venue-context")
    output = root / "context"
    summary = build_venue_context(write_history(root), MONDAY, output)
    return {
        "summary": summary,
        "daily": pd.read_csv(output / "daily_context.csv"),
        "shares": pd.read_csv(output / "daily_market_shares.csv"),
        "snapshot": pd.read_csv(output / "venue_snapshot.csv"),
    }


def test_daily_context_preserves_fractional_trf_volume_and_fraction_scaled_hhi(venue_context):
    friday = venue_context["daily"].query("date == @FRIDAY").set_index("tape")
    assert set(friday.index) == {"A", "B", "C", "ALL"}
    expected_volumes = {
        "A": [30.125, 10.375, 59.5], "B": [20.25, 30.5, 49.25],
        "C": [10.125, 20.125, 69.75], "ALL": [60.5, 61.0, 178.5],
    }
    for tape, volumes in expected_volumes.items():
        row = friday.loc[tape]
        total = sum(volumes)
        exchange_total = sum(volumes[:2])
        assert row.total_shares == pytest.approx(total)
        assert row.total_notional_usd == pytest.approx(total * 100)
        assert row.off_exchange_shares == pytest.approx(volumes[-1])
        assert row.off_exchange_share_fraction == pytest.approx(volumes[-1] / total)
        assert row.reporting_center_volume_hhi == pytest.approx(sum((v / total) ** 2 for v in volumes))
        assert row.exchange_only_volume_hhi == pytest.approx(sum((v / exchange_total) ** 2 for v in volumes[:2]))
        assert 0 <= row.reporting_center_volume_hhi <= 1
        assert 0 <= row.exchange_only_volume_hhi <= 1
        assert (row.reporting_center_count, row.exchange_count, row.trf_count) == (3, 2, 1)
    assert friday.loc["ALL", "off_exchange_share_fraction"] == pytest.approx(0.595)
    assert friday.loc["ALL", "total_trade_count"] == 30


def test_daily_reporting_center_shares_are_complete_and_preserve_scope(venue_context):
    shares = venue_context["shares"]
    assert len(shares) == 2 * 4 * 3
    assert not shares.duplicated(["date", "tape", "market_participant"]).any()
    totals = shares.groupby(["date", "tape"])[
        ["volume_share_fraction", "notional_share_fraction", "trade_share_fraction"]
    ].sum()
    np.testing.assert_allclose(totals, 1.0)
    trf = shares.query("date == @FRIDAY and tape == 'ALL' and market_participant == @TRF").iloc[0]
    assert trf.shares == pytest.approx(178.5)
    assert trf.volume_share_fraction == pytest.approx(178.5 / 300)
    assert trf.reporting_center_type == "trf_reporting_facility"
    snapshot = venue_context["snapshot"]
    assert snapshot.date.unique().tolist() == [MONDAY]
    assert len(snapshot) == 4 * 3


def test_lag_uses_previous_observed_day_across_friday_to_monday(venue_context):
    daily = venue_context["daily"]
    friday = daily.query("date == @FRIDAY")
    monday = daily.query("date == @MONDAY").set_index("tape")
    assert not friday.lag1_available.any()
    assert friday.lag1_date.isna().all()
    assert friday.lag1_total_shares.isna().all()
    assert monday.lag1_available.all()
    assert monday.lag1_date.eq(FRIDAY).all()
    assert monday.lag1_calendar_gap_days.eq(3).all()
    assert monday.loc["ALL", "total_shares"] == pytest.approx(600)
    assert monday.loc["ALL", "lag1_total_shares"] == pytest.approx(300)
    assert monday.loc["ALL", "off_exchange_share_fraction"] == pytest.approx(0.4)
    assert monday.loc["ALL", "lag1_off_exchange_share_fraction"] == pytest.approx(0.595)


def test_absent_target_date_is_rejected_without_silently_selecting_another_day(tmp_path):
    with pytest.raises(ValueError, match="No Cboe.*2026-10-27"):
        build_venue_context(write_history(tmp_path), "2026-10-27", tmp_path / "out")


@pytest.mark.parametrize("value", [-0.125, np.nan, np.inf, -np.inf])
def test_negative_or_nonfinite_cboe_metrics_are_rejected(tmp_path, value):
    frame = history_frame()
    frame.loc[0, "Tape A Shares"] = value
    with pytest.raises(ValueError):
        build_venue_context(write_history(tmp_path, frame), MONDAY, tmp_path / "out")


@pytest.mark.parametrize("column", ["Day", "Market Participant", "Tape C Notional"])
def test_missing_required_cboe_columns_are_rejected(tmp_path, column):
    frame = history_frame().drop(columns=column)
    with pytest.raises(ValueError, match="Missing Cboe columns"):
        build_venue_context(write_history(tmp_path, frame), MONDAY, tmp_path / "out")


def test_duplicate_participant_within_day_is_rejected(tmp_path):
    frame = history_frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate date/participant"):
        build_venue_context(write_history(tmp_path, frame), MONDAY, tmp_path / "out")


@pytest.mark.parametrize("metric", ["Shares", "Notional", "Trade Count"])
def test_tape_totals_must_match_source_total_for_each_participant(tmp_path, metric):
    frame = history_frame()
    frame.loc[0, f"Total {metric}"] += 1
    with pytest.raises(ValueError, match="does not match"):
        build_venue_context(write_history(tmp_path, frame), MONDAY, tmp_path / "out")


def maker_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"Symbol": "AAPL", "Missed Liquidity": 100.0, "Exhausted Liquidity": 50.0,
         "Routed Liquidity": 0.0, "Volume Opportunity": 150.0, "Cboe ADV": 60.0,
         "Liquidity Opportunity": 2.5},
        {"Symbol": "MSFT", "Missed Liquidity": 10.0, "Exhausted Liquidity": 5.0,
         "Routed Liquidity": 0.0, "Volume Opportunity": 15.0, "Cboe ADV": 30.0,
         "Liquidity Opportunity": 0.5},
    ])


def rpi_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"symbol": "AAPL", "side": side, "retail_shares_submitted": submitted,
         "retail_shares_executed": executed, "missed_retail_liquidity": submitted - executed,
         "exhausted_retail_liquidity": 0.0, "shares_exec_div_bats_vol": executed / 100.0,
         "median_order_size": 10.0, "price_improvement": 0.001, "effective_spread": 0.01}
        for side, submitted, executed in [("B", 10.0, 2.0), ("S", 20.0, 3.0), ("T", 5.0, 4.0), ("E", 7.0, 1.0)]
    ])


def write_maker_files(tmp_path, maker=None, rpi=None):
    maker_path, rpi_path = tmp_path / "maker.csv", tmp_path / "rpi_maker_opportunity_20261023.csv"
    (maker_frame() if maker is None else maker).to_csv(maker_path, index=False)
    (rpi_frame() if rpi is None else rpi).to_csv(rpi_path, index=False)
    return maker_path, rpi_path


@pytest.fixture(scope="module")
def maker_context(tmp_path_factory):
    root = tmp_path_factory.mktemp("maker-context")
    output = root / "context"
    maker_path, rpi_path = write_maker_files(root)
    summary = build_maker_context(maker_path, rpi_path, output, symbol=" aapl ")
    return {
        "summary": summary,
        "maker": pd.read_csv(output / "maker_context.csv"),
        "rpi": pd.read_csv(output / "rpi_context.csv"),
        "maker_snapshot": pd.read_csv(output / "maker_snapshot.csv"),
        "rpi_snapshot": pd.read_csv(output / "rpi_snapshot.csv"),
    }


def test_maker_liquidity_opportunity_ratio_2_5_is_250_percent(maker_context):
    row = maker_context["maker"].set_index("symbol").loc["AAPL"]
    assert row.liquidity_opportunity_ratio == pytest.approx(2.5)
    assert row.liquidity_opportunity_percent == pytest.approx(250.0)
    assert row.volume_opportunity_shares_adv == pytest.approx(150.0)
    assert row.cboe_shares_adv == pytest.approx(60.0)
    assert maker_context["maker_snapshot"].symbol.tolist() == ["AAPL"]
    assert maker_context["summary"]["rpi_session_date"] == FRIDAY


def test_rpi_source_side_t_is_preserved_without_assuming_it_totals_buy_and_sell(maker_context):
    source = rpi_frame().set_index("side")
    actual = maker_context["rpi"].set_index("side")
    assert set(actual.index) == {"B", "S", "T", "E"}
    assert len(actual) == len(source)
    for side in source.index:
        assert actual.loc[side, "retail_shares_submitted"] == source.loc[side, "retail_shares_submitted"]
        assert actual.loc[side, "retail_shares_executed"] == source.loc[side, "retail_shares_executed"]
        assert actual.loc[side, "executed_to_submitted_fraction"] == pytest.approx(
            source.loc[side, "retail_shares_executed"] / source.loc[side, "retail_shares_submitted"]
        )
    assert actual.loc["T", "retail_shares_submitted"] == 5.0
    assert actual.loc["T", "retail_shares_submitted"] != actual.loc[["B", "S"], "retail_shares_submitted"].sum()
    assert set(maker_context["rpi_snapshot"].side) == {"B", "S", "T", "E"}


@pytest.mark.parametrize("report", ["maker", "rpi"])
def test_duplicate_maker_symbol_or_rpi_symbol_side_is_rejected(tmp_path, report):
    maker, rpi = maker_frame(), rpi_frame()
    if report == "maker":
        maker = pd.concat([maker, maker.iloc[[0]]], ignore_index=True)
    else:
        rpi = pd.concat([rpi, rpi.iloc[[0]]], ignore_index=True)
    paths = write_maker_files(tmp_path, maker, rpi)
    with pytest.raises(ValueError, match="Duplicate"):
        build_maker_context(*paths, tmp_path / "out")


@pytest.mark.parametrize("report", ["maker", "rpi"])
def test_negative_maker_or_rpi_metrics_are_rejected(tmp_path, report):
    maker, rpi = maker_frame(), rpi_frame()
    if report == "maker":
        maker.loc[0, "Liquidity Opportunity"] = -1.0
    else:
        rpi.loc[0, "retail_shares_executed"] = -1.0
    paths = write_maker_files(tmp_path, maker, rpi)
    with pytest.raises(ValueError, match="nonnegative"):
        build_maker_context(*paths, tmp_path / "out")


def test_na_is_preserved_as_a_real_ticker_in_maker_and_rpi_reports(tmp_path):
    maker, rpi = maker_frame(), rpi_frame()
    maker.loc[0, "Symbol"] = "NA"
    rpi["symbol"] = "NA"
    output = tmp_path / "out"
    summary = build_maker_context(*write_maker_files(tmp_path, maker, rpi), output, symbol="NA")
    assert summary["maker_symbol_found"] is True
    assert summary["rpi_symbol_found"] is True
    maker_snapshot = pd.read_csv(output / "maker_snapshot.csv", keep_default_na=False)
    rpi_snapshot = pd.read_csv(output / "rpi_snapshot.csv", keep_default_na=False)
    assert maker_snapshot.symbol.tolist() == ["NA"]
    assert rpi_snapshot.symbol.tolist() == ["NA"] * 4


@pytest.mark.parametrize("report", ["maker", "rpi"])
def test_actual_empty_ticker_is_rejected_in_maker_and_rpi_reports(tmp_path, report):
    maker, rpi = maker_frame(), rpi_frame()
    if report == "maker":
        maker.loc[0, "Symbol"] = ""
    else:
        rpi.loc[0, "symbol"] = ""
    with pytest.raises(ValueError, match="nonempty"):
        build_maker_context(*write_maker_files(tmp_path, maker, rpi), tmp_path / "out")
