"""Download the public Cboe context reports; no Databento key or charges."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-10-23", help="Session date; latest maker report is a separate weekly source")
    parser.add_argument("--out", type=Path, default=ROOT / "data/cboe")
    args = parser.parse_args()
    day = date.fromisoformat(args.date)
    args.out.mkdir(parents=True, exist_ok=True)
    sources = {
        f"market_history_{day.year}.csv": f"https://cdn.cboe.com/resources/us/equities/market-statistics/historical-market-volume/market_history_{day.year}.csv",
        "maker_bzx_latest.csv": "https://www.cboe.com/us/equities/market_statistics/maker_report/csv/?book=all&mkt=bzx",
        f"rpi_maker_opportunity_{day:%Y%m%d}.csv": f"https://cdn.cboe.com/resources/us/equities/market-statistics/retail-price-improvement/daily/rpt_rpi_maker_opportunity_{day:%Y%m%d}.csv",
    }

    def download(item):
        name, url = item
        path = args.out / name
        if path.exists():
            return name, {"url": url, "status": "cached", "bytes": path.stat().st_size}
        try:
            with urlopen(Request(url, headers={"User-Agent": "ColonyLab research data check"}), timeout=45) as response:
                payload = response.read(16_000_001)
            if len(payload) > 16_000_000 or b"<html" in payload[:500].lower():
                raise ValueError("Source did not return a bounded CSV file")
            path.write_bytes(payload)
            return name, {"url": url, "status": "downloaded", "bytes": len(payload), "fetched_utc": datetime.now(timezone.utc).isoformat()}
        except Exception as exc:
            return name, {"url": url, "status": "error", "error": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = dict(pool.map(download, sources.items()))
    manifest = args.out / "sources.json"
    previous = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    # Preserve the original retrieval time when reusing local source files.
    for name, result in results.items():
        if result["status"] == "cached" and name in previous:
            result["original_retrieval"] = previous[name]
    manifest.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    if any(result["status"] == "error" for result in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
