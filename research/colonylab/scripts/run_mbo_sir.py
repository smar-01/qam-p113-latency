"""Acquire, reconstruct and evaluate a native full-depth MBO SIRS experiment."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from colonylab.config import safe_error, write_json
from colonylab.data import client


def probe() -> dict:
    """Size full-session replay and inspect capped startup records, not model outcomes."""
    import databento as db
    from datetime import datetime, timezone
    datasets = ["XNAS.ITCH", "XNYS.PILLAR", "BATS.PITCH", "ARCX.PILLAR", "EDGX.PITCH",
                "MEMX.MEMOIR", "BATY.PITCH", "EDGA.PITCH"]
    out = ROOT / "out/mbo_sir_probe"
    data = ROOT / "data/mbo_sir_probe"
    out.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    def one(dataset):
        path = data / dataset
        path.mkdir(exist_ok=True)
        receipt = path / "probe.json"
        if receipt.exists():
            return json.loads(receipt.read_text())
        api = client()
        base = dict(dataset=dataset, schema="mbo", stype_in="raw_symbol",
                    start="2026-06-01T00:00:00Z", end="2026-06-01T20:00:00Z")
        row = {"dataset": dataset,
               "three_symbol_records": int(api.metadata.get_record_count(**base, symbols=["AAPL", "MSFT", "SPY"])),
               "aapl_records": int(api.metadata.get_record_count(**base, symbols=["AAPL"])),
               "aapl_billable_bytes": int(api.metadata.get_billable_size(**base, symbols=["AAPL"]))}
        file = path / "startup.dbn.zst"
        if not file.exists():
            api.timeseries.get_range(**base, symbols=["AAPL"], limit=100, path=file)
        store = db.DBNStore.from_file(file)
        records = []
        for r in store:
            if not hasattr(r, "action"):
                continue
            records.append({name: int(getattr(r, name)) if name not in ("action", "side") else str(getattr(r, name))
                            for name in ("ts_recv", "publisher_id", "instrument_id", "order_id", "action", "side", "price", "size", "flags", "sequence")})
        row.update(startup_records=records, startup_probe_capped=True,
                   note="Startup probe deliberately capped at 100 MBO records; excluded from study data.",
                   cost_check_skipped=True)
        write_json(receipt, row)
        return row
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(one, dataset): dataset for dataset in datasets}
        for future in as_completed(jobs):
            dataset = jobs[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {"dataset": dataset, "error": safe_error(exc)}
            rows.append(row)
            write_json(out / "probe.json", {"requests": rows})
            print(json.dumps({k: v for k, v in row.items() if k != "startup_records"}), flush=True)
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "requests": rows}
    write_json(out / "probe.json", result)
    return result


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "mbo_sir.toml")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--repair-failed-only", action="store_true", help="Retry only finished failed downloads recorded in the active manifest; leave the active acquisition manifest untouched")
    parser.add_argument("--conditions-only", action="store_true", help="Save the provider's date-level availability and quality conditions for the frozen study window")
    parser.add_argument("--download-workers", type=int, help="Transport concurrency override (1–32); research requests/settings stay fixed")
    parser.add_argument("--restart-existing-fetch", action="store_true", help="Stop only the verified existing fetch in this workspace, preserve partial streams, then resume")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--replay-available", action="store_true", help="Overlap local book reconstruction with an active acquisition; no network calls")
    args = parser.parse_args()
    try:
        if args.download_workers is not None and not 1 <= args.download_workers <= 32:
            raise ValueError("Runtime download workers must be from one through 32.")
        if (args.download_workers is not None or args.restart_existing_fetch) and not args.fetch_only:
            raise ValueError("Transport controls require --fetch-only.")
        modes = [args.probe_only, args.fetch_only, args.repair_failed_only, args.conditions_only,
                 args.analyze_only, args.replay_available]
        if sum(modes) > 1:
            raise ValueError("Choose a single execution mode.")
        if args.probe_only:
            probe()
            return
        if args.conditions_only:
            from datetime import datetime, timezone
            import time
            from colonylab.mbo_sir_data import read_config
            from colonylab.mbo_sir_pipeline import _json, _digest
            config = read_config(args.config)
            dates = sorted(config["training_days"] + config["held_out_days"])
            datasets = list(dict.fromkeys(config["datasets"] + [config["market_dataset"]]))
            rows = []
            for dataset in datasets:
                for attempt in range(3):
                    try:
                        conditions = client().metadata.get_dataset_condition(
                            dataset=dataset, start_date=dates[0], end_date=dates[-1])
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(1 + attempt)
                rows.append(dict(dataset=dataset, conditions=conditions))
                print(f"Saved provider conditions for {dataset}.", flush=True)
            _json(Path(config["out_dir"]) / "dataset_conditions.json", dict(
                created_utc=datetime.now(timezone.utc).isoformat(),
                config_sha256=_digest(config), intended_dates=dates, datasets=rows,
                note="Provider date-level conditions supplement, and do not replace, the exact-stream and reconstruction audits."))
            return
        if args.repair_failed_only:
            from colonylab.mbo_sir_data import read_config, repair_failed_study
            result = repair_failed_study(read_config(args.config))
            print(json.dumps(result, indent=2), flush=True)
            if not result["complete"]:
                raise SystemExit(1)
            return
        if args.fetch_only:
            from colonylab.mbo_sir_data import read_config, fetch_study
            config = read_config(args.config)
            if args.restart_existing_fetch:
                from colonylab.mbo_sir_control import stop_exact_fetch
                report = stop_exact_fetch(ROOT, config)
                print(json.dumps({"fetch_stopped": report}, indent=2), flush=True)
            result = fetch_study(config, config_path=args.config, download_workers=args.download_workers)
            print(json.dumps({"complete": result["complete"], "requests": len(result["requests"]),
                              "rows": sum(row.get("rows", 0) for row in result["requests"])}, indent=2), flush=True)
            return
        if args.replay_available:
            from colonylab.mbo_sir_data import read_config
            from colonylab.mbo_sir_pipeline import replay_study
            replay_study(read_config(args.config), wait_for_downloads=True)
            return
        from colonylab.mbo_sir_pipeline import run_mbo_sir
        run_mbo_sir(args.config.resolve(), fetch=not args.analyze_only, analyze=not args.fetch_only)
    except Exception as exc:
        print(safe_error(exc), file=sys.stderr, flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
