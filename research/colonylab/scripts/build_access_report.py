"""Summarize local access evidence and derive an observed-venue volume panel."""
from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from colonylab.venue_panel import build_venue_panel, check_rpi_denominator
from colonylab.config import write_json


def main() -> None:
    directory = ROOT / "out/access"
    audit = json.loads((directory / "databento_access.json").read_text(encoding="utf-8"))
    wrds = json.loads((directory / "wrds_access.json").read_text(encoding="utf-8"))
    public = json.loads((directory / "public_sources.json").read_text(encoding="utf-8"))
    panel = build_venue_panel(directory / "databento_access.json", directory)
    run_estimates = {"main": float(audit["estimated_budget_reserved_usd"])}
    for name, path in (("initial_sequential", directory / "databento_access.sequential.json"), ("overnight_control", directory / "overnight_control/databento_access.json")):
        if path.exists():
            run_estimates[name] = float(json.loads(path.read_text(encoding="utf-8"))["estimated_budget_reserved_usd"])
    total_estimate = sum(run_estimates.values())
    successes = {"downloaded_nonempty", "cached_nonempty"}
    feeds = []
    for dataset, feed in audit["feeds"].items():
        good = [p for p in feed.get("probes", []) if p["retrieval_status"] in successes]
        feeds.append({"dataset": dataset, "scope": feed["feed_scope"], "verified_schemas": [p["schema"] for p in good], "nonempty_records": sum(p.get("records", 0) for p in good)})
    result = {"databento_audit_complete": bool(audit.get("completed_utc")), "databento_feeds": feeds, "databento_reserved_estimate_usd": audit["estimated_budget_reserved_usd"], "wrds": wrds, "public_sources": public, "venue_panel": panel}
    result["audit_run_reserved_estimates_usd"] = run_estimates
    result["total_audit_reserved_estimate_usd"] = total_estimate
    overnight_path = directory / "overnight_control/databento_access.json"
    if overnight_path.exists():
        result["overnight_control"] = json.loads(overnight_path.read_text(encoding="utf-8"))
    rpi_date = audit["session_date"]
    rpi_path = ROOT / f"data/cboe/rpi_maker_opportunity_{rpi_date.replace('-', '')}.csv"
    if rpi_path.exists():
        result["rpi_volume_crosscheck"] = check_rpi_denominator(directory / "stock_venue_daily_volume.csv", rpi_path, directory, report_date=rpi_date)
    write_json(directory / "access_inventory.json", result)
    lines = ["Historical equity data access check — 2026-10-26", "", "Databento: actual nonempty retrievals by dataset (schema support alone is not counted):"]
    lines.extend(f"  {f['dataset']}: {', '.join(f['verified_schemas']) or 'no nonempty retrieval yet'}" for f in feeds)
    lines.extend(["", f"All audit runs estimated reservations: ${total_estimate:.8f} (not an invoice).", "Blue Ocean overnight MBO/MBP-1 control is recorded separately from the daytime exchange panel.", "WRDS: signed in; Daily TAQ not subscribed; only sample TAQ products verified included.", "FINRA: public live AAPL weekly ATS/non-ATS records retrieved; samples are capped, not complete market totals.", "IEX: historical manifest accessible; large raw captures were not downloaded.", "Nasdaq Trader: monthly XLSX endpoints accessible; academic-use authorization not verified.", "NYSE Volume Summary: public sample accessible; full historical subscription not verified.", "", f"Provisional stock/venue panel: {panel['rows']} rows, {len(panel['venues'])} native venues.", panel["interpretation"], "No cross-venue impact or causal transmission model has been fitted by this access audit."])
    if "rpi_volume_crosscheck" in result:
        check = result["rpi_volume_crosscheck"]
        lines.append(f"RPI/BYX denominator cross-check: {check['matching_rows']}/{check['tested_symbol_side_rows']} executing symbol-side observations match across {', '.join(check['symbols'])}.")
    (directory / "access_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"databento_audit_complete": result["databento_audit_complete"], "verified_nonempty_feeds": sum(bool(f["verified_schemas"]) for f in feeds), "panel_rows": panel["rows"], "venues": panel["venues"], "report": str(directory / "access_report.txt")}, indent=2))


if __name__ == "__main__":
    main()
