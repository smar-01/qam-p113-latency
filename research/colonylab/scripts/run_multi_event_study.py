"""Design, price, retrieve and analyze the multi-year AAPL event study."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["design", "plan", "fetch", "analyze"])
    parser.add_argument("--config", type=Path, default=ROOT / "aapl_multi_event_protocol.json")
    parser.add_argument("--out", type=Path, default=ROOT / "out/aapl_multi_event")
    parser.add_argument("--workers", type=int, choices=[1, 2, 3, 4], default=3)
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--available-only", action="store_true", help="Checkpoint completed source groups while acquisition continues; publish no partial event results")
    parser.add_argument("--reuse-requests", type=Path, help="Reuse exact previously planned source intervals; retrieve only gaps")
    args = parser.parse_args()
    from colonylab.multi_event_design import build_design
    from colonylab.multi_event_data import plan_requests, fetch_requests
    from colonylab.multi_event_pipeline import analyze
    config = json.loads(args.config.read_text())
    if args.action == "design":
        reuse = json.loads(args.reuse_requests.read_text()) if args.reuse_requests else None
        protocol, requests = build_design(ROOT, args.config, args.out, reuse_requests=reuse)
        print(json.dumps({"event_anchors": sum(row["role"] == "event" for row in protocol["trials"]),
                          "control_trials": sum(row["role"] == "control" for row in protocol["trials"]),
                          "source_windows": len(protocol["source_groups"]), "requests": len(requests)}), flush=True)
    elif args.action == "plan":
        requests = json.loads((args.out / "requests.json").read_text())
        result = plan_requests(requests, ROOT, config["data_dir"], args.out, workers=args.workers)
        print(json.dumps({key: result.get(key) for key in ("ready_to_download", "expected_request_count", "incremental_estimated_cost_usd", "expected_total_records", "estimated_total_billable_bytes")}), flush=True)
    elif args.action == "fetch":
        if args.max_cost_usd is None:
            parser.error("fetch requires the concrete priced study's --max-cost-usd ceiling")
        plan = json.loads((args.out / "download_plan.json").read_text())
        result = fetch_requests(plan, ROOT, config["data_dir"], args.out, args.max_cost_usd, workers=args.workers)
        print(json.dumps({key: result.get(key) for key in ("complete", "acquisition_complete_for_available_requests", "expected_request_count", "incremental_estimated_cost_usd")}), flush=True)
    else:
        result = analyze(ROOT, args.out, workers=args.workers, available_only=args.available_only)
        print(json.dumps({key: result[key] for key in ("status", "completed_source_windows", "event_anchors", "event_bundles", "control_trials", "raw_requests", "raw_records", "matching_audit_cases", "matching_mismatches") if key in result}), flush=True)


if __name__ == "__main__":
    main()
