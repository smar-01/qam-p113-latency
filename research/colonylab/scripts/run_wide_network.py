"""Acquire or analyze a configured historical native-venue replication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from colonylab.config import safe_error
from colonylab.network_wide_data import (
    fetch_wide_network, load_wide_frames, plan_wide_network, read_wide_config,
    recover_interrupted_wide_downloads,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "wide_network.toml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true", help="Read free metadata; do not retrieve time series")
    mode.add_argument("--fetch-only", action="store_true")
    mode.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--recover-interrupted", action="store_true",
                        help="Preserve failed stream prefixes before retrying the same requests under included access")
    parser.add_argument("--download-workers", type=int,
                        help="Transport concurrency override (1–16); leaves the frozen research configuration unchanged")
    args = parser.parse_args()
    try:
        config = read_wide_config(args.config)
        data_dir = (ROOT / config["data_dir"]).resolve()
        out_dir = (ROOT / config["out_dir"]).resolve()
        progress = lambda msg: print(msg, flush=True)
        if args.recover_interrupted:
            if args.plan_only or args.analyze_only:
                raise ValueError("Interrupted-stream recovery requires a retrieval run.")
            recovered = recover_interrupted_wide_downloads(
                config, data_dir, out_dir / "data_manifest.json")
            progress(f"Preserved {len(recovered['prepared_for_retry'])} interrupted stream prefixes "
                     f"for exact-request retry; {len(recovered['export_recovery'])} completed streams await export.")
        if args.plan_only:
            result = plan_wide_network(config, data_dir, out_dir, config_path=args.config, progress=progress)
            print(json.dumps({"plan": str(out_dir / "download_plan.json"),
                              "estimated_cost_usd": result["incremental_estimated_cost_usd"],
                              "cost_check_skipped": result["cost_check_skipped"]}, indent=2))
            return
        if not args.analyze_only:
            result = fetch_wide_network(config, data_dir, out_dir, config_path=args.config, progress=progress,
                                        download_workers=args.download_workers)
            print(json.dumps({"complete": result["complete"], "rows": result["downloaded_rows"],
                              "estimated_cost_usd": result["incremental_estimated_cost_usd"],
                              "cost_check_skipped": result["cost_check_skipped"],
                              "manifest": str(out_dir / "data_manifest.json")}, indent=2))
        if not args.fetch_only:
            from colonylab.network import run_network
            from colonylab.network_models import run_prediction_test

            frames = load_wide_frames(out_dir / "data_manifest.json", expected_config=config)
            progress("All configured data validated; running frozen training/held-out event study.")
            result = run_network(frames, out_dir, held_out_days=config["held_out_days"])
            progress("Running incremental prediction test with the same fixed chronological split.")
            result["heldout_prediction"] = run_prediction_test(
                frames, out_dir / "shock_events.csv", out_dir,
                held_out_days=config["held_out_days"], save_predictions=False)
            print(json.dumps(result, indent=2, default=str, allow_nan=False))
    except Exception as exc:
        print(safe_error(exc), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
