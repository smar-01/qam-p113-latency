"""Read free Databento metadata for the expanded AAPL event study."""
from pathlib import Path
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--costs", action="store_true")
    args = parser.parse_args()
    from colonylab.data import client
    from colonylab.config import safe_error, write_json
    datasets = ["XNAS.ITCH", "XNYS.PILLAR", "ARCX.PILLAR", "BATS.PITCH", "EDGX.PITCH"]
    if args.costs:
        def size(task):
            api = client()
            request = dict(dataset=task[0], schema="mbp-1", symbols=["AAPL"], stype_in="raw_symbol",
                           start=task[1], end=task[2])
            try:
                return {"request": request, "records": api.metadata.get_record_count(**request),
                        "bytes": api.metadata.get_billable_size(**request), "cost_usd": api.metadata.get_cost(**request)}
            except Exception as exc:
                return {"request": request, "error": safe_error(exc)}
        tasks = [(name, start, end) for name in datasets if name != "XNYS.PILLAR" for start, end in
                 [("2022-06-06T16:25:00Z", "2022-06-06T17:35:00Z"),
                  ("2025-07-31T20:25:00Z", "2025-07-31T21:35:00Z")]]
        with ThreadPoolExecutor(max_workers=3) as pool:
            rows = list(pool.map(size, tasks))
        write_json(ROOT / "out/aapl_multi_event/representative_costs.json", {"requests": rows, "market_data_downloaded": False})
        for row in rows:
            print(json.dumps(row, default=str), flush=True)
        return int(any("error" in row for row in rows))
    def probe(dataset):
        api = client()
        try:
            return {"dataset": dataset, "schemas": api.metadata.list_schemas(dataset=dataset),
                    "range": api.metadata.get_dataset_range(dataset=dataset)}
        except Exception as exc:
            return {"dataset": dataset, "error": safe_error(exc)}
    rows = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(probe, name) for name in datasets]):
            row = future.result()
            rows.append(row)
            print(json.dumps(row, default=str), flush=True)
    write_json(ROOT / "out/aapl_multi_event/provider_coverage.json", {"feeds": rows, "market_data_downloaded": False})
    if any("error" in row for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
