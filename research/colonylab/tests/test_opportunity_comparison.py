from __future__ import annotations

import pandas as pd
import pytest

from colonylab.context import build_opportunity_comparison


@pytest.fixture(scope="module")
def comparison(tmp_path_factory):
    root = tmp_path_factory.mktemp("opportunity-comparison")
    maker_rows = []
    for symbol, ratio, adv in [("A", 1.0, 100), ("B", 2.0, 100), ("C", 3.0, 100),
                               ("D", 4.0, 100), ("NA", 0.5, 100), ("ZEROADV", 999.0, 0)]:
        opportunity = ratio * adv
        maker_rows.append({
            "Symbol": symbol, "Missed Liquidity": opportunity, "Exhausted Liquidity": 0,
            "Routed Liquidity": 0, "Volume Opportunity": opportunity, "Cboe ADV": adv,
            "Liquidity Opportunity": ratio,
        })
    rpi_rows = []
    observations = [
        ("A", "B", 10, .001), ("B", "B", 10, .002), ("C", "B", 10, .003),
        ("NA", "B", 10, .0005), ("D", "B", 0, 0), ("ZEROADV", "B", 10, 1),
        ("A", "S", 10, .003), ("B", "S", 10, .002), ("C", "S", 10, .001),
        ("A", "T", 10, .005), ("B", "T", 10, .005), ("A", "E", 1, .002),
    ]
    for symbol, side, executed, improvement in observations:
        rpi_rows.append({
            "symbol": symbol, "side": side, "retail_shares_submitted": 100,
            "retail_shares_executed": executed, "missed_retail_liquidity": 100 - executed,
            "exhausted_retail_liquidity": 0, "shares_exec_div_bats_vol": executed / 100,
            "median_order_size": 10, "price_improvement": improvement, "effective_spread": .01 if executed else 0,
        })
    maker_path, rpi_path = root / "maker.csv", root / "rpi_20261023.csv"
    pd.DataFrame(maker_rows).to_csv(maker_path, index=False)
    pd.DataFrame(rpi_rows).to_csv(rpi_path, index=False)
    output = root / "out"
    summary = build_opportunity_comparison(maker_path, rpi_path, output)
    return {
        "summary": summary,
        "joined": pd.read_csv(output / "symbol_opportunity_context.csv", keep_default_na=False),
        "associations": pd.read_csv(output / "opportunity_associations.csv"),
    }


@pytest.mark.parametrize("side, expected_rho, count", [("B", 1.0, 4), ("S", -1.0, 3)])
def test_known_positive_and_negative_rank_associations(comparison, side, expected_rho, count):
    rows = comparison["associations"].query("side == @side")
    assert set(rows.predictor) == {"liquidity_opportunity_ratio", "volume_opportunity_shares_adv"}
    assert rows.n_symbols.eq(count).all()
    assert rows.correlation_defined.all()
    assert rows.outcome.eq("price_improvement").all()
    for rho in rows.spearman_rho:
        assert rho == pytest.approx(expected_rho)


@pytest.mark.parametrize("side, count, distinct_outcomes", [("T", 2, 1), ("E", 1, 1)])
def test_constant_or_single_observation_has_undefined_correlation(comparison, side, count, distinct_outcomes):
    rows = comparison["associations"].query("side == @side")
    assert len(rows) == 2
    assert rows.n_symbols.eq(count).all()
    assert rows.distinct_outcome_values.eq(distinct_outcomes).all()
    assert not rows.correlation_defined.any()
    assert rows.spearman_rho.isna().all()


def test_zero_adv_is_excluded_and_unexecuted_rows_remain_evidence_without_outcomes(comparison):
    joined, summary = comparison["joined"], comparison["summary"]
    assert "ZEROADV" not in set(joined.symbol)
    assert "NA" in set(joined.symbol)
    assert len(joined) == 11
    unexecuted = joined.set_index(["symbol", "side"]).loc[("D", "B")]
    assert unexecuted.price_improvement == 0
    assert not unexecuted.has_executions
    assert not unexecuted.eligible_for_price_improvement_association
    assert unexecuted.price_improvement_usd_per_executed_share == ""
    buy = comparison["associations"].query("side == 'B'")
    assert buy.joined_symbol_side_rows.eq(5).all()
    assert buy.excluded_no_execution_rows.eq(1).all()
    assert summary["excluded_zero_adv_maker_symbols"] == 1
    assert summary["joined_symbols"] == 5
    assert summary["eligible_execution_symbol_side_rows"] == 10
