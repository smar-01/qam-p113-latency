"""Read publisher-declared times from the sourced Apple earnings pages."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--businesswire", action="store_true")
    args = parser.parse_args()
    from colonylab.aapl_event_trial import dump
    calendar = json.loads((ROOT / "aapl_multi_event_calendar.json").read_text())
    events = [row for row in calendar["events"] if row["event_class"] == "earnings_release"]
    wire = {"2025-07-31": "20250731999776", "2025-05-01": "20250501689914",
            "2024-08-01": "20240801799244", "2024-05-02": "20240502965412", "2023-08-03": "20230803893875"}
    if args.businesswire:
        events = [row for row in events if row["date"] in wire]
    def read(row):
        url = next(url for url in row["source_urls"] if url.startswith("https://www.apple.com/newsroom/"))
        if args.businesswire:
            quarter = "second" if "q2" in row["id"] else "third"
            url = "https://www.businesswire.com/news/home/" + wire[row["date"]] + "/en/Apple-reports-" + quarter + "-quarter-results"
        response = requests.get(url, timeout=25, headers={"User-Agent": "AAPL event-time research (public publisher metadata)"})
        response.raise_for_status()
        source = response.text
        found = []
        for block in re.findall(r'<script[^>]*type=[\"\x27]application/ld\+json[\"\x27][^>]*>(.*?)</script>', source, re.I | re.S):
            try:
                obj = json.loads(block)
            except ValueError:
                continue
            def walk(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key in ("datePublished", "dateModified"):
                            found.append({"field": key, "value": item})
                        if isinstance(item, (dict, list)):
                            walk(item)
                elif isinstance(value, list):
                    for item in value:
                        walk(item)
            walk(obj)
        return {"event_id": row["id"], "url": url, "http_status": response.status_code,
                "source_sha256": hashlib.sha256(response.content).hexdigest(), "publisher_metadata": found}
    records = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for row in pool.map(read, events):
            records.append(row)
            print(json.dumps(row), flush=True)
    dump(ROOT / ("out/aapl_multi_event/businesswire_publication_metadata.json" if args.businesswire else "out/aapl_multi_event/apple_publication_metadata.json"),
         {"retrieved_utc": datetime.now(timezone.utc).isoformat(), "records": records,
          "interpretation": "Publisher-declared publication time, not proof of first external dissemination."})


if __name__ == "__main__":
    main()
