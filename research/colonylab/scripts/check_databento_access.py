"""Check historical US-equity feed retrieval with a maximum $0.10 estimate."""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from colonylab.access import audit_access
from colonylab.config import safe_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, default=date(2026, 10, 23))
    parser.add_argument("--budget", type=float, default=0.10)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/access_audit")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out/access")
    parser.add_argument("--datasets", nargs="+", help="Optional subset of catalogued US-equity datasets")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "SPY"],
                        help="One to three explicit symbols; only the first is used for intraday probes")
    parser.add_argument("--intraday-time", default="14:00:00", help="UTC whole-second clock for one-second event probes")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent feed checks, from 1 to 4")
    args = parser.parse_args()
    try:
        result = audit_access(day=args.date, budget=args.budget, data_dir=args.data_dir,
                              out_dir=args.out_dir, datasets=args.datasets, symbols=args.symbols,
                              intraday_time=args.intraday_time,
                              workers=args.workers,
                              progress=lambda msg: print(msg, flush=True))
    except Exception as exc:
        print(safe_error(exc), file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps({
        "catalog_datasets": len(result["catalog_datasets"]),
        "selected_equity_datasets": len(result["selected_datasets"]),
        "estimated_budget_reserved_usd": result["estimated_budget_reserved_usd"],
        "retrieval_status_counts": result["retrieval_status_counts"],
        "report": str((args.out_dir / "databento_access.json").resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
