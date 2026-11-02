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
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

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


def request_identity(dataset: str, schema: str, start: str, end: str,
                     symbols: list[str] | tuple[str, ...], *, stype_in: str = "raw_symbol",
                     limit: int | None = None) -> dict:
    """Canonical complete query identity; calendar dates alone are insufficient."""
    def timestamp(value):
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.astimezone(timezone.utc).isoformat()
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("Use explicit, nonempty, unique symbols.")
    result = {"dataset": dataset, "schema": schema, "symbols": sorted(symbols),
              "start": timestamp(start), "end": timestamp(end), "stype_in": stype_in}
    if result["start"] >= result["end"]:
        raise ValueError("Request end must be strictly after start.")
    if limit is not None:
        if type(limit) is not int or limit <= 0:
            raise ValueError("Record limit must be a positive integer.")
        result["limit"] = limit
    return result


def _request_hash(request: dict) -> str:
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def out_path(dataset: str, schema: str, start: str, end: str,
             symbols: list[str] | tuple[str, ...] | None = None, *,
             stype_in: str = "raw_symbol", limit: int | None = None) -> Path:
    """Content-bound query path. Main always supplies the requested symbol set."""
    identity = request_identity(dataset, schema, start, end,
                                list(symbols) if symbols is not None else ["AAPL", "SPY", "MSFT"],
                                stype_in=stype_in, limit=limit)
    tag = f"{schema}_{start[:10]}_{end[:10]}".replace(":", "")
    return DATA / dataset / f"{tag}_{_request_hash(identity)[:20]}.dbn.zst"


def cache_reusable(path: Path, request: dict) -> bool:
    """Never infer completeness or request identity from a filename alone."""
    if not path.exists():
        return False
    marker = path.with_name(path.name + ".json")
    if not marker.is_file():
        raise ValueError(f"Existing cache has no verified completion receipt: {path}")
    receipt = json.loads(marker.read_text(encoding="utf-8"))
    if (receipt.get("request") != request or receipt.get("complete") is not True
            or receipt.get("file_size_bytes") != path.stat().st_size
            or receipt.get("content_sha256") != _file_hash(path)):
        raise ValueError(f"Existing cache request, bytes or completion differs: {path}")
    return True


def _record_complete(path: Path, request: dict) -> None:
    receipt = {"request": request, "complete": True, "file_size_bytes": path.stat().st_size,
               "content_sha256": _file_hash(path)}
    marker = path.with_name(path.name + ".json")
    temporary = marker.with_name(marker.name + "." + uuid4().hex + ".partial")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(marker)


def main(argv=None) -> int:
    a = parse_args(argv)
    missing = check_against_package()
    if missing:
        sys.exit(f"installed databento does not know {missing}; update p113/venues.py")
    datasets = resolve_datasets(a.datasets)

    print(f"plan: {a.schema} for {a.symbols}, {a.start} -> {a.end}, {len(datasets)} datasets")
    for d in datasets:
        print(f"  {d:<13} {BY_DATASET[d].name:<18} -> {out_path(d, a.schema, a.start, a.end, a.symbols)}")

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
        kw = request_identity(d, a.schema, a.start, a.end, a.symbols)
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
        request = request_identity(d, a.schema, a.start, a.end, a.symbols)
        path = out_path(d, a.schema, a.start, a.end, a.symbols)
        if cache_reusable(path, request):
            print(f"  have {path}, skipping")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + "." + uuid4().hex + ".partial")
        client.timeseries.get_range(**request, path=tmp)
        tmp.replace(path)
        _record_complete(path, request)
        print(f"  saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
