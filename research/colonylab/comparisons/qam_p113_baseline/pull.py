"""Pull per-venue top-of-book (MBP-1) from Databento.

The API key is read ONLY from the env var DATABENTO_API_KEY. Without it this script
prints the plan and stops; nothing is downloaded and nothing is written.

Examples
    python pull.py --cost-only                         # price the default pull, download nothing
    python pull.py --start 2026-10-15 --end 2026-10-16 --symbols AAPL SPY MSFT --cost-only
    python pull.py --start 2026-10-15 --end 2026-10-16 --max-cost 0 --yes
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from p113.venues import BY_DATASET, DEFAULT_DATASETS, VENUES, check_against_package

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+", default=["AAPL", "SPY", "MSFT"])
    p.add_argument("--start", default="2026-10-15", help="inclusive, YYYY-MM-DD or ISO time (UTC)")
    p.add_argument("--end", default="2026-10-16", help="exclusive, YYYY-MM-DD or ISO time (UTC)")
    p.add_argument("--schema", default="mbp-1", help="mbp-1 (default), bbo-1s, mbo, ...")
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS),
                   help="dataset codes, or 'all' for every venue in p113/venues.py")
    p.add_argument("--cost-only", action="store_true", help="price the pull with metadata.get_cost and stop")
    p.add_argument("--max-cost", type=float, default=0.0,
                   help="refuse to download if the quoted total (USD) is above this (default 0)")
    p.add_argument("--yes", action="store_true", help="download without the interactive confirm")
    return p.parse_args(argv)


def resolve_datasets(names: list[str]) -> list[str]:
    if names == ["all"]:
        return [v.dataset for v in VENUES]
    unknown = [n for n in names if n not in BY_DATASET]
    if unknown:
        sys.exit(f"unknown dataset(s) {unknown}; known: {sorted(BY_DATASET)}")
    return names


def out_path(dataset: str, schema: str, start: str, end: str) -> Path:
    tag = f"{schema}_{start[:10]}_{end[:10]}".replace(":", "")
    return DATA / dataset / f"{tag}.dbn.zst"


def main(argv=None) -> int:
    a = parse_args(argv)
    missing = check_against_package()
    if missing:
        sys.exit(f"installed databento does not know {missing}; update p113/venues.py")
    datasets = resolve_datasets(a.datasets)

    print(f"plan: {a.schema} for {a.symbols}, {a.start} -> {a.end}, {len(datasets)} datasets")
    for d in datasets:
        print(f"  {d:<13} {BY_DATASET[d].name:<18} -> {out_path(d, a.schema, a.start, a.end)}")

    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        print("\nDATABENTO_API_KEY is not set: stopping before any API call. Nothing was downloaded.")
        return 2

    import databento as db

    client = db.Historical(key)  # key stays in memory only

    # 1) check each dataset serves the schema, 2) price it. Nothing is downloaded here.
    rows, total = [], 0.0
    for d in datasets:
        schemas = client.metadata.list_schemas(dataset=d)
        if a.schema not in schemas:
            print(f"  skip {d}: schema {a.schema} not offered (has {schemas})")
            continue
        kw = dict(dataset=d, symbols=a.symbols, schema=a.schema, start=a.start, end=a.end)
        cost = client.metadata.get_cost(**kw)
        size = client.metadata.get_billable_size(**kw)
        rows.append((d, cost, size))
        total += cost
    print(f"\n{'dataset':<13} {'cost USD':>10} {'billable GB':>12}")
    for d, c, s in rows:
        print(f"{d:<13} {c:>10.4f} {s / 1e9:>12.3f}")
    print(f"{'TOTAL':<13} {total:>10.4f} {sum(r[2] for r in rows) / 1e9:>12.3f}")

    if a.cost_only:
        print("\n--cost-only: nothing downloaded.")
        return 0
    if total > a.max_cost:
        print(f"\nquoted {total:.4f} USD is above --max-cost {a.max_cost}; not downloading.")
        return 3
    if not a.yes and input("download now? [y/N] ").strip().lower() != "y":
        print("cancelled.")
        return 0

    for d, _, _ in rows:
        path = out_path(d, a.schema, a.start, a.end)
        if path.exists():
            print(f"  have {path}, skipping")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        client.timeseries.get_range(dataset=d, symbols=a.symbols, schema=a.schema,
                                    start=a.start, end=a.end, path=tmp)
        tmp.replace(path)
        print(f"  saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
