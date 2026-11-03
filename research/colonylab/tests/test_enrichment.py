from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from colonylab.enrichment import attach_context


def input_frames():
    events = pd.DataFrame({
        "event_id": [0, 1, 2, 3],
        "session_date": ["2026-10-22", "2026-10-23", "2026-10-26", "2026-10-23"],
        "symbol": ["AAPL", "AAPL", "AAPL", "NA"],
        "colony": [0, 1, 1, 0], "p_colony_0": [1.0, 0.7, 0.2, 0.8],
    })
    daily = pd.DataFrame({
        "date": ["2026-10-23", "2026-10-23", "2026-10-26"],
        "tape": ["A", "ALL", "ALL"], "total_shares": [100.0, 300.0, 600.0],
    })
    maker = pd.DataFrame({"Symbol": ["AAPL", "NA"], "Liquidity Opportunity": [2.5, 1.2]})
    rpi = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "AAPL", "NA"], "side": ["B", "S", "T", "B"],
        "retail_shares_submitted": [10.0, 20.0, 5.0, 8.0],
        "retail_shares_executed": [2.0, 0.0, 4.0, 2.0],
        "price_improvement": [0.001, 0.0, 0.002, 0.004],
        "effective_spread": [0.01, 0.0, 0.012, 0.02],
    })
    return {"events": events, "daily": daily, "maker": maker, "rpi": rpi}


def write_inputs(tmp_path, frames=None):
    frames = input_frames() if frames is None else frames
    paths = []
    for name in ("events", "daily", "maker", "rpi"):
        path = tmp_path / f"{name}.csv"
        frames[name].to_csv(path, index=False)
        paths.append(path)
    return paths


def attach(paths, output, **changes):
    periods = {"session_date": "2026-10-23", "maker_start": "2026-10-19", "maker_end": "2026-10-23"}
    periods.update(changes)
    return attach_context(*paths, output, **periods)


@pytest.fixture(scope="module")
def enriched(tmp_path_factory):
    root = tmp_path_factory.mktemp("context-enrichment")
    output = root / "enriched.csv"
    summary = attach(write_inputs(root), output)
    return pd.read_csv(output, keep_default_na=False), summary


def test_context_joins_respect_daily_weekly_and_exact_retail_report_dates(enriched):
    output, summary = enriched
    np.testing.assert_allclose(
        pd.to_numeric(output.cboe_daily_total_shares, errors="coerce"),
        [np.nan, 300.0, 600.0, 300.0], equal_nan=True,
    )
    np.testing.assert_allclose(
        pd.to_numeric(output.cboe_maker_bzx_liquidity_opportunity, errors="coerce"),
        [2.5, 2.5, np.nan, 1.2], equal_nan=True,
    )
    np.testing.assert_allclose(
        pd.to_numeric(output.cboe_maker_bzx_liquidity_opportunity_pct, errors="coerce"),
        [250.0, 250.0, np.nan, 120.0], equal_nan=True,
    )
    np.testing.assert_allclose(
        pd.to_numeric(output.cboe_rpi_B_retail_shares_submitted, errors="coerce"),
        [np.nan, 10.0, np.nan, 8.0], equal_nan=True,
    )
    # T is a preserved source category: it is not replaced by B + S.
    assert float(output.loc[1, "cboe_rpi_T_retail_shares_submitted"]) == 5.0
    assert summary["matched_daily_context_rows"] == 3
    assert summary["matched_maker_context_rows"] == 3
    assert summary["matched_retail_context_rows"] == 2


def test_unexecuted_retail_rows_do_not_become_zero_valued_execution_outcomes(enriched):
    output, _ = enriched
    assert float(output.loc[1, "cboe_rpi_S_price_improvement"]) == 0.0
    assert not output.loc[1, "cboe_rpi_S_has_executions"]
    assert output.loc[1, "cboe_rpi_S_price_improvement_when_executed"] == ""
    assert output.loc[1, "cboe_rpi_S_effective_spread_when_executed"] == ""
    assert output.loc[1, "cboe_rpi_B_has_executions"]
    assert float(output.loc[1, "cboe_rpi_B_price_improvement_when_executed"]) == pytest.approx(0.001)


def test_context_attachment_preserves_event_rows_na_ticker_and_model_values(enriched):
    output, summary = enriched
    original = input_frames()["events"]
    pd.testing.assert_frame_equal(output[original.columns], original, check_dtype=False)
    assert len(output) == summary["event_rows"] == len(original)
    assert output.loc[3, "symbol"] == "NA"
    assert summary["model_features_changed"] is False


@pytest.mark.parametrize("report", ["daily", "maker", "rpi"])
def test_duplicate_context_keys_cannot_expand_event_rows(tmp_path, report):
    frames = input_frames()
    if report == "daily":
        duplicate = frames[report].query("tape == 'ALL'").iloc[[0]]
    else:
        duplicate = frames[report].iloc[[0]]
    frames[report] = pd.concat([frames[report], duplicate], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        attach(write_inputs(tmp_path, frames), tmp_path / "out.csv")


def test_reversed_weekly_report_period_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="reversed"):
        attach(write_inputs(tmp_path), tmp_path / "out.csv", maker_start="2026-10-23", maker_end="2026-10-19")
