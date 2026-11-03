"""Download and test a bounded, synchronized three-exchange network pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from colonylab.config import safe_error
from colonylab.network_data import fetch_network, load_network_frames, plan_network


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fetch-only", action="store_true")
    mode.add_argument("--analyze-only", action="store_true")
    mode.add_argument("--plan-only", action="store_true")
    parser.add_argument("--minutes", type=int, default=1, help="Aligned window, from 1 to 15 minutes")
    parser.add_argument("--budget", type=float, default=0.50, help="Maximum incremental metadata cost estimate in USD")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/network_pilot")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out/network_pilot")
    args = parser.parse_args()
    try:
        if args.plan_only:
            result = plan_network(args.data_dir, args.out_dir, minutes=args.minutes,
                                  budget=args.budget, workers=args.workers,
                                  progress=lambda msg: print(msg, flush=True))
            print(json.dumps({"plan": str((args.out_dir / "download_plan.json").resolve()),
                              "estimated_cost_usd": result["incremental_estimated_cost_usd"]}, indent=2))
            return
        if not args.analyze_only:
            result = fetch_network(args.data_dir, args.out_dir, minutes=args.minutes,
                                   budget=args.budget, workers=args.workers,
                                   progress=lambda msg: print(msg, flush=True))
            print(json.dumps({"complete": result["complete"], "rows": result["downloaded_rows"],
                              "estimated_cost_usd": result["incremental_estimated_cost_usd"],
                              "manifest": str((args.out_dir / "data_manifest.json").resolve())}, indent=2))
        if not args.fetch_only:
            from colonylab.network import run_network
            frames = load_network_frames(args.out_dir / "data_manifest.json")
            result = run_network(frames, args.out_dir)
            from colonylab.network_models import run_prediction_test
            result["heldout_prediction"] = run_prediction_test(
                frames, args.out_dir / "shock_events.csv", args.out_dir)
            print(json.dumps(result, indent=2, default=str, allow_nan=False))
    except Exception as exc:
        print(safe_error(exc), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
