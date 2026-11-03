"""Build a provisional stock/venue volume panel from verified audit downloads."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from .config import write_json


def build_venue_panel(audit_path: Path, output_dir: Path) -> dict:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    rows, excluded = [], []
    success = {"downloaded_nonempty", "cached_nonempty"}
    for dataset, feed in audit["feeds"].items():
        if feed["feed_scope"] != "individual_exchange_feed":
            excluded.append({"dataset": dataset, "reason": feed["feed_scope"]})
            continue
        probes = feed.get("probes", [])
        mbo = next((p for p in probes if p["schema"] == "mbo"), {})
        for probe in probes:
            if probe["schema"] != "ohlcv-1d" or probe["retrieval_status"] not in success:
                continue
            if probe.get("record_limit_reached"):
                excluded.append({"dataset": dataset, "reason": "daily_bar_record_limit_reached"})
                continue
            mappings = {p["publisher_id"]: p for p in probe.get("publisher_mappings", [])}
            for bar in probe.get("daily_bars", []):
                publisher = mappings.get(bar.get("publisher_id"), {})
                if not publisher.get("venue"):
                    excluded.append({"dataset": dataset, "reason": "unmapped_publisher"})
                    continue
                timestamp = pd.Timestamp(bar["ts_event"])
                if timestamp.tzinfo is None:
                    timestamp = timestamp.tz_localize("UTC")
                timestamp = timestamp.tz_convert("UTC")
                volume = float(bar["volume"])
                if not np.isfinite(volume) or volume < 0:
                    raise ValueError("Daily bar has invalid volume.")
                rows.append({
                    "bar_date_utc": timestamp.date().isoformat(),
                    "symbol": str(bar["symbol"]), "venue": publisher["venue"],
                    "dataset": dataset, "publisher_id": bar["publisher_id"],
                    "volume_shares": volume,
                    "mbo_probe_status": mbo.get("retrieval_status", "not_tested"),
                    "mbo_probe_symbols": ",".join(mbo.get("request", {}).get("symbols", [])),
                    "mbo_probe_records": mbo.get("records", 0),
                })
    if not rows:
        raise ValueError("No verified native-venue daily bars are available in the audit.")
    panel = pd.DataFrame(rows)
    key = ["bar_date_utc", "symbol", "venue"]
    if panel.duplicated(key).any():
        raise ValueError("Overlapping feeds describe the same venue; select one source instead of double counting.")
    group = panel.groupby(["bar_date_utc", "symbol"])
    panel["observed_native_volume_shares"] = group["volume_shares"].transform("sum")
    panel["observed_native_venue_count"] = group["venue"].transform("nunique")
    panel["volume_share_among_observed_native_venues"] = panel["volume_shares"] / panel["observed_native_volume_shares"].where(panel["observed_native_volume_shares"] > 0)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "stock_venue_daily_volume.csv"
    panel.sort_values(key).to_csv(path, index=False)
    result = {
        "audit_complete": bool(audit.get("completed_utc")),
        "rows": len(panel), "symbols": sorted(panel["symbol"].unique()),
        "venues": sorted(panel["venue"].unique()), "dates_utc": sorted(panel["bar_date_utc"].unique()),
        "excluded_feeds": excluded, "panel": str(path.resolve()),
        "interpretation": "Provisional share of volume among observed native exchange feeds only; not total US market share.",
        "limitations": [
            "Composite feeds, combined products, TRFs and overnight ATSs are excluded to prevent overlap and scope mixing.",
            "Absent bars are missing coverage or no reported trades; they are not imputed as zero.",
            "Bar dates follow the source UTC-day convention; exchange sessions, auctions and trade-eligibility rules are not reconciled against SIP totals.",
            "The successful MBO probe symbol and record count are recorded separately; a daily bar does not prove MBO coverage for every symbol.",
            "One-second MBO probes are event deltas, not initialized full books or estimates of transmission between venues.",
            "Single-day weights are retrospective. A prospective model should use verified prior-period volumes.",
        ],
    }
    write_json(output_dir / "stock_venue_daily_volume.json", result)
    return result


def check_rpi_denominator(panel_path: Path, rpi_path: Path, output_dir: Path, *, report_date: str) -> dict:
    """Empirically check the reported RPI volume ratio against BYX bars."""
    panel = pd.read_csv(panel_path, keep_default_na=False)
    bars = panel.loc[panel["venue"].eq("BATY") & panel["bar_date_utc"].eq(report_date), ["symbol", "volume_shares"]]
    if bars["symbol"].duplicated().any():
        raise ValueError("BYX volume comparison requires one bar per symbol/date.")
    retail = pd.read_csv(rpi_path, keep_default_na=False)
    compared = retail.merge(bars, on="symbol", how="inner", validate="many_to_one")
    compared = compared.loc[(compared["retail_shares_executed"] > 0) & (compared["shares_exec_div_bats_vol"] > 0) & (compared["volume_shares"] > 0)].copy()
    compared["implied_venue_volume"] = compared["retail_shares_executed"] / compared["shares_exec_div_bats_vol"]
    compared["relative_difference"] = (compared["implied_venue_volume"] - compared["volume_shares"]).abs() / compared["volume_shares"]
    compared["agrees_with_byx_volume"] = np.isclose(compared["implied_venue_volume"], compared["volume_shares"], rtol=1e-6, atol=1e-6)
    path = output_dir / "rpi_volume_crosscheck.csv"
    compared.to_csv(path, index=False)
    result = {"report_date": report_date, "symbols": sorted(compared["symbol"].unique()), "tested_symbol_side_rows": len(compared), "matching_rows": int(compared["agrees_with_byx_volume"].sum()), "all_tested_rows_match": bool(compared["agrees_with_byx_volume"].all()) if len(compared) else None, "output": str(path.resolve()), "interpretation": "Empirical consistency for these symbol-days only: RPI executed shares / reported ratio agrees with BYX daily volume. This is not a general source-dictionary or publication-timing verification."}
    write_json(output_dir / "rpi_volume_crosscheck.json", result)
    return result
