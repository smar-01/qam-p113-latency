"""Independently inspect capped startup probes; never treat them as study data."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from colonylab.mbo_book import Book, F_BAD_TS_RECV, F_LAST, F_SNAPSHOT, F_MAYBE_BAD_BOOK, UNDEF_PRICE


def audit_startup(probe_path: Path) -> dict:
    import databento as db
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    rows = []
    for feed in sorted(probe["requests"], key=lambda row: row["dataset"]):
        records = feed.get("startup_records", [])
        raw_path = ROOT / "data/mbo_sir_probe" / feed["dataset"] / "startup.dbn.zst"
        metadata = db.DBNStore.from_file(raw_path).metadata if raw_path.exists() else None
        midnight_start = bool(metadata is not None and str(metadata.schema) == "mbo"
                              and int(metadata.start) % (86_400 * 1_000_000_000) == 0)
        book = Book(allow_legacy_startup_clear=midnight_start)
        initialized_at = None
        authoritative_clears = []
        synthetic_clear_candidates = []
        for position, record in enumerate(records):
            flags = record["flags"]
            if (record["action"] == "R" and flags == F_BAD_TS_RECV
                    and record["sequence"] == 0 and record["order_id"] == 0
                    and record["price"] == UNDEF_PRICE and record["size"] == 0
                    and record["side"] == "N"):
                synthetic_clear_candidates.append({"position": position, "time_ns": record["ts_recv"],
                                                   "flags": flags})
            if record["action"] == "R" and not flags & F_MAYBE_BAD_BOOK and (
                    not flags & F_BAD_TS_RECV or flags & F_SNAPSHOT):
                authoritative_clears.append({"position": position, "time_ns": record["ts_recv"],
                                             "flags": flags, "final": bool(flags & F_LAST)})
            book.apply(record, emit=False)
            if book.initialized and initialized_at is None:
                initialized_at = position
        rows.append({
            "dataset": feed["dataset"], "probe_error": feed.get("error"),
            "verified_midnight_request_start": midnight_start,
            "probe_records": len(records), "probe_is_capped": feed.get("startup_probe_capped"),
            "action_counts": dict(Counter(record["action"] for record in records)),
            "authoritative_clears": authoritative_clears,
            "synthetic_clear_candidates": synthetic_clear_candidates,
            "first_initialization_position": initialized_at,
            "initialized_by_probe_end": book.initialized,
            "healthy_by_probe_end": book._healthy,
            "two_sided_valid_by_probe_end": book.valid,
            "orders_by_probe_end": len(book.orders),
            "book_diagnostics": dict(book.diagnostics),
            "first_record": records[0] if records else None,
        })
    return {
        "status": "startup_clears_observed" if rows and all(
            row["initialized_by_probe_end"] and not row["probe_error"] for row in rows) else "startup_requires_investigation",
        "purpose": "Initialization evidence from deliberately capped startup probes only; no model outcome inspection.",
        "limitations": [
            "A capped startup probe is not a complete historical file or a full-session reconstruction audit.",
            "An early one-sided book can be initialized and healthy without yet having a usable two-sided quote.",
            "Each uncapped venue/date replay still requires its own clear and exact MBP-10 price, size and count validation.",
        ],
        "requests": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=ROOT / "out/mbo_sir_probe/probe.json")
    parser.add_argument("--out", type=Path, default=ROOT / "out/mbo_sir_probe/startup_audit.json")
    args = parser.parse_args()
    result = audit_startup(args.probe)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "feeds": len(result["requests"]),
                      "initialized": sum(row["initialized_by_probe_end"] for row in result["requests"]),
                      "output": str(args.out)}, allow_nan=False))


if __name__ == "__main__":
    main()
