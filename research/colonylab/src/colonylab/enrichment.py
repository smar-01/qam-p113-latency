"""Retrospective context joins, deliberately separate from model features."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def attach_context(memberships_path: Path, daily_path: Path, maker_path: Path, rpi_path: Path,
                   output_path: Path, *, session_date: str, maker_start: str, maker_end: str) -> dict:
    events = pd.read_csv(memberships_path, keep_default_na=False)
    daily = pd.read_csv(daily_path)
    day = daily.loc[daily["tape"].eq("ALL")].drop(columns="tape")
    if day["date"].duplicated().any():
        raise ValueError("Daily context has duplicate date rows.")
    day = day.rename(columns={name: "cboe_daily_" + name for name in day if name != "date"})
    joined = events.merge(day, left_on="session_date", right_on="date", how="left", validate="many_to_one").drop(columns="date")

    # An undated rolling weekly report cannot be matched without its observed period.
    start, end = pd.Timestamp(maker_start), pd.Timestamp(maker_end)
    if start > end:
        raise ValueError("Maker report period is reversed.")
    maker = pd.read_csv(maker_path, keep_default_na=False).rename(columns={"Symbol": "symbol"})
    if maker["symbol"].duplicated().any():
        raise ValueError("Maker report contains duplicate symbols.")
    maker_columns = [name for name in maker if name != "symbol"]
    maker = maker.rename(columns={name: "cboe_maker_bzx_" + name.lower().replace(" ", "_") for name in maker_columns})
    joined = joined.merge(maker, on="symbol", how="left", validate="many_to_one")
    joined["cboe_maker_bzx_liquidity_opportunity_pct"] = pd.to_numeric(joined["cboe_maker_bzx_liquidity_opportunity"], errors="coerce") * 100
    maker_columns = [name for name in joined if name.startswith("cboe_maker_bzx_")]
    dates = pd.to_datetime(joined["session_date"])
    covered = dates.between(start, end)
    joined.loc[~covered, maker_columns] = float("nan")
    joined["cboe_maker_period_start"] = maker_start
    joined["cboe_maker_period_end"] = maker_end

    retail = pd.read_csv(rpi_path, keep_default_na=False)
    if retail.duplicated(["symbol", "side"]).any():
        raise ValueError("Retail report contains duplicate symbol/side rows.")
    values = [name for name in retail if name not in {"symbol", "side"}]
    wide = retail.pivot(index="symbol", columns="side", values=values)
    wide.columns = [f"cboe_rpi_{side}_{field}" for field, side in wide.columns]
    joined = joined.merge(wide.reset_index(), on="symbol", how="left", validate="many_to_one")
    retail_columns = [name for name in joined if name.startswith("cboe_rpi_")]
    joined.loc[~joined["session_date"].eq(session_date), retail_columns] = float("nan")
    for side in retail["side"].unique():
        prefix = f"cboe_rpi_{side}_"
        executed = pd.to_numeric(joined[prefix + "retail_shares_executed"], errors="coerce")
        joined[prefix + "has_executions"] = executed.gt(0)
        for field in ("price_improvement", "effective_spread"):
            joined[prefix + field + "_when_executed"] = pd.to_numeric(joined[prefix + field], errors="coerce").where(executed.gt(0))
    joined["cboe_rpi_report_date"] = session_date
    joined["external_context_usage"] = "Retrospective only; publication times unverified; not used in colony fit or predictions"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output_path, index=False)
    return {
        "event_rows": len(joined),
        "matched_daily_context_rows": int(joined["cboe_daily_total_shares"].notna().sum()),
        "matched_maker_context_rows": int((covered & joined[maker_columns].notna().any(axis=1)).sum()),
        "matched_retail_context_rows": int(joined[retail_columns].notna().any(axis=1).sum()),
        "output": str(output_path.resolve()),
        "model_features_changed": False,
        "interpretation": "Full-day and weekly context is ex post. A one-symbol one-day sample cannot identify within-day colony effects of these constant covariates.",
    }
