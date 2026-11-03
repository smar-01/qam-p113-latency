from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ROOT, safe_error


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Anonymous execution patterns and descriptive response horizons")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Free metadata/account connection checks; no data download")
    doctor.add_argument("--out", type=Path, default=ROOT / "out/doctor.json")
    plan = sub.add_parser("plan", help="Validate a small request and obtain a free cost estimate")
    plan.add_argument("--config", type=Path, default=ROOT / "pilot.toml")
    plan.add_argument("--out", type=Path, default=ROOT / "out/pilot-plan.json")
    fetch = sub.add_parser("fetch", help="Download only if a fresh estimate fits the supplied budget")
    fetch.add_argument("--plan", type=Path, default=ROOT / "out/pilot-plan.json")
    fetch.add_argument("--max-cost-usd", required=True, type=float)
    fetch.add_argument("--out", type=Path, default=ROOT / "data/pilot")
    demo = sub.add_parser("demo", help="Run an explicitly synthetic offline end-to-end test")
    demo.add_argument("--out", type=Path, default=ROOT / "out/demo")
    analyze = sub.add_parser("analyze", help="Analyze an existing local DBN or CSV; no network calls")
    analyze.add_argument("path", type=Path)
    analyze.add_argument("--dataset", default=None, help="Required for CSV, read from DBN metadata otherwise")
    analyze.add_argument("--schema", choices=["trades", "mbp-1"], default=None)
    analyze.add_argument("--out", type=Path, default=ROOT / "out/pilot")
    analyze.add_argument("--window-ms", type=int, default=1000)
    analyze.add_argument("--horizons-ms", default="1,10,100,1000,10000,60000")
    analyze.add_argument("--clusters", type=int, default=3)
    context = sub.add_parser("context", help="Build retrospective Cboe venue/maker/retail context from locally saved public CSVs")
    context.add_argument("--date", default="2026-10-23")
    context.add_argument("--symbol", default="AAPL")
    context.add_argument("--source-dir", type=Path, default=ROOT / "data/cboe")
    context.add_argument("--out", type=Path, default=ROOT / "out/context")
    context.add_argument("--memberships", type=Path, default=ROOT / "out/pilot/event_memberships.csv")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            from .data import doctor as run
            result = run(args.out)
            feeds = {name: {"schemas": info.get("schemas", []), "range_end": info.get("range", {}).get("end"), "error": info.get("error")} for name, info in result["feeds"].items()}
            print(json.dumps({"key_configured": result["key_configured"], "authenticated_account_check": result["authenticated_account_check"], "feeds": feeds, "report": str(args.out)}, indent=2, default=str))
        elif args.command == "plan":
            from .data import estimate
            result = estimate(args.config, args.out)
            print(json.dumps(result, indent=2))
        elif args.command == "fetch":
            from .data import fetch
            print(json.dumps(fetch(args.plan, args.max_cost_usd, args.out), indent=2))
        elif args.command == "demo":
            from .demo import generate_demo
            from .analysis import run_analysis
            result = run_analysis(generate_demo(), args.out, dataset="SYNTHETIC", schema="mbp-1")
            print(json.dumps(result, indent=2, default=str))
        elif args.command == "context":
            from .context import build_venue_context, build_maker_context, build_opportunity_comparison
            from .enrichment import attach_context
            from .config import write_json
            day = args.date.replace("-", "")
            source = args.source_dir
            venue = build_venue_context(source / f"market_history_{args.date[:4]}.csv", args.date, args.out)
            maker = build_maker_context(source / "maker_bzx_latest.csv", source / f"rpi_maker_opportunity_{day}.csv", args.out, symbol=args.symbol)
            comparison = build_opportunity_comparison(source / "maker_bzx_latest.csv", source / f"rpi_maker_opportunity_{day}.csv", args.out)
            result = {"venue": venue, "maker_retail": maker, "opportunity_comparison": comparison}
            if args.memberships.exists():
                periods = json.loads((source / "report_periods.json").read_text(encoding="utf-8"))
                result["event_context"] = attach_context(args.memberships, args.out / "daily_context.csv", source / "maker_bzx_latest.csv", source / f"rpi_maker_opportunity_{day}.csv", args.out / "events_with_context.csv", session_date=args.date, maker_start=periods["maker_week_start"], maker_end=periods["maker_week_end"])
            write_json(args.out / "combined_context.json", result)
            print(json.dumps(result, indent=2, default=str))
        else:
            from .data import load_file
            from .analysis import run_analysis
            frame, detected_dataset, detected_schema = load_file(args.path)
            dataset, schema = args.dataset or detected_dataset, args.schema or detected_schema
            if not dataset or not schema:
                raise ValueError("CSV analysis requires explicit --dataset and --schema.")
            result = run_analysis(frame, args.out, dataset=dataset, schema=schema, window_ms=args.window_ms, horizons_ms=[int(x) for x in args.horizons_ms.split(",")], n_clusters=args.clusters)
            print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as exc:
        print(safe_error(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
