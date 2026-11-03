"""Price/fetch a small fixed quality probe before the multi-event bulk pull."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "fetch", "quality"])
    parser.add_argument("--max-cost-usd", type=float)
    args = parser.parse_args()
    from colonylab.multi_event_data import plan_requests, fetch_requests
    from colonylab.latency_data import load_entry
    from colonylab.aapl_event_trial import observed_seconds
    from colonylab.event_midpoints import NativeMidpoints
    import pandas as pd
    out = ROOT / "out/aapl_multi_event/pilot"
    data = "data/aapl_multi_event"
    requests = [dict(dataset=venue, schema="mbp-1", symbols=["AAPL"], stype_in="raw_symbol",
                     start=start, end=end, limit=10_000_000)
                for venue in ["XNAS.ITCH", "ARCX.PILLAR", "BATS.PITCH", "EDGX.PITCH"]
                for start, end in [("2022-06-06T16:24:00+00:00", "2022-06-06T17:36:00+00:00"),
                                   ("2025-07-31T20:24:00+00:00", "2025-07-31T21:36:00+00:00")]]
    if args.action == "plan":
        plan = plan_requests(requests, ROOT, data, out, workers=3)
        print(json.dumps({k: plan[k] for k in ["ready_to_download", "incremental_estimated_cost_usd", "unavailable_requests"]}), flush=True)
    elif args.action == "fetch":
        if args.max_cost_usd is None:
            parser.error("fetch requires --max-cost-usd")
        plan = json.loads((out / "download_plan.json").read_text())
        result = fetch_requests(plan, ROOT, data, out, args.max_cost_usd, workers=3)
        print(json.dumps({k: result[k] for k in ["complete", "acquisition_complete_for_available_requests", "downloaded_rows"]}), flush=True)
    else:
        manifest = json.loads((out / "data_manifest.json").read_text())
        rows = []
        for entry in manifest["requests"]:
            if not entry.get("complete") or not entry.get("analyzable"):
                continue
            for age in [250, 1000, 5000]:
                series, _ = load_entry(entry, max_age_ms=age)
                for (_, clock), quotes in series.items():
                    path = NativeMidpoints(quotes.times_ns, quotes.bid + quotes.ask, quotes.valid,
                                           quotes.segments, quotes.start_ns, quotes.end_ns)
                    start = pd.Timestamp(entry["request"]["start"]).value + 6 * 60_000_000_000
                    end = start + 60 * 60_000_000_000
                    rows.append({"day": entry["request"]["start"][:10], "venue": entry["request"]["dataset"],
                                 "clock": clock, "age_ms": age, "coverage": observed_seconds(path, start, end) / 3600})
        frame = pd.DataFrame(rows)
        frame.to_csv(out / "quality_feasibility.csv", index=False)
        print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
