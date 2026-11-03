"""Bounded, unauthenticated checks of public venue-volume source access.

Never reads credentials, accepts an agreement, downloads Nasdaq research data,
or downloads IEX packet captures. Public reachability is not account entitlement.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 2_000_000
USER_AGENT = "ColonyLab/0.1 public source access check"


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict[str, str]] = []
        self.active: dict[str, str] | None = None

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "a" and values.get("href"):
            self.active = {"href": values["href"], "text": ""}

    def handle_data(self, data):
        if self.active is not None:
            self.active["text"] += data.strip()

    def handle_endtag(self, tag):
        if tag == "a" and self.active is not None:
            self.links.append(self.active)
            self.active = None


def probe(url: str, method: str = "GET", payload=None, max_bytes=MAX_BYTES):
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json,text/html,text/plain,*/*"}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json"
    result = {"url": url, "method": method, "authentication": "none"}
    try:
        with urlopen(Request(url, data=body, headers=headers, method=method), timeout=15) as response:
            result.update(status=response.status, final_url=response.url,
                          content_type=response.headers.get("Content-Type"),
                          content_length=response.headers.get("Content-Length"))
            content = response.read(max_bytes + 1) if method != "HEAD" else b""
        result["bytes_read"] = len(content)
        result["truncated"] = len(content) > max_bytes
        return result, content[:max_bytes]
    except HTTPError as exc:
        result.update(status=exc.code, error=str(exc), content_type=exc.headers.get("Content-Type"))
        return result, exc.read(min(max_bytes, 1500))
    except Exception as exc:
        result.update(status=None, error=f"{type(exc).__name__}: {exc}")
        return result, b""


def get_links(content: bytes, base: str):
    parser = Links()
    parser.feed(content.decode("utf-8", errors="replace"))
    return [{**link, "url": urljoin(base, link["href"])} for link in parser.links]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "out/access/public_sources.json")
    parser.add_argument("--week", default="2026-09-21")
    args = parser.parse_args()
    pages = {
        "nasdaq": "https://nasdaqtrader.com/Trader.aspx?id=marketsharedaily",
        "nyse": "https://www.nyse.com/data-products/catalog/volume-summary",
        "nyse_samples": "https://ftp.nyse.com/Historical%20Data%20Samples/VOLUME%20SUMMARY/",
        "iex": "https://iextrading.com/trading/market-data/",
        "iex_manifest": "https://iextrading.com/api/1.0/hist?date=20261023",
        "finra": "https://www.finra.org/filing-reporting/otc-transparency",
        "finra_live": "https://api.finra.org/data/group/otcMarket/name/weeklySummary?limit=1",
        "finra_metadata": "https://api.finra.org/metadata/group/otcMarket/name/weeklySummary",
    }
    with ThreadPoolExecutor(max_workers=5) as pool:
        fetched = dict(zip(pages, pool.map(probe, pages.values())))
    report = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Unauthenticated public access only; no user account entitlement inferred.",
        "sources": {},
    }
    nasdaq_page, nasdaq_body = fetched["nasdaq"]
    nasdaq_links = [link for link in get_links(nasdaq_body, pages["nasdaq"])
                    if link["text"] == "September 2026"]
    endpoint_probes = [probe(link["url"], method="HEAD")[0] for link in nasdaq_links[:2]]
    report["sources"]["nasdaq_trader"] = {
        "page": nasdaq_page,
        "latest_complete_month_links": nasdaq_links[:2],
        "download_endpoint_head_only": endpoint_probes,
        "permission_status": "Prior written academic/commercial authorization not verified; no report downloaded.",
        "permission_clause_present": b"prior, written authorization" in nasdaq_body,
        "data_downloaded": False,
    }
    nyse_page, _ = fetched["nyse"]
    sample_page, sample_body = fetched["nyse_samples"]
    sample_links = [link for link in get_links(sample_body, pages["nyse_samples"])
                    if link["text"].startswith("EQY_US_ALL_SUM_DOLVOL_") and link["href"].endswith(".txt")]
    if not sample_links:
        sample_links = [link for link in get_links(sample_body, pages["nyse_samples"])
                        if not link["href"].endswith("/")][:3]
    sample_metadata = [probe(link["url"], method="HEAD")[0] for link in sample_links[:2]]
    sample_header = {}
    if sample_links:
        sample_check, sample_bytes = probe(sample_links[0]["url"], max_bytes=4096)
        sample_header = {"probe": sample_check, "header_lines": sample_bytes.decode(errors="replace").splitlines()[:3]}
    report["sources"]["nyse_volume_summary"] = {
        "page": nyse_page, "public_sample_directory": sample_page,
        "sample_links": sample_links[:3], "sample_metadata": sample_metadata,
        "public_sample_header": sample_header,
        "account_entitlement": "Not checked; product page links Purchase Now. Public samples do not establish subscription.",
        "full_sample_downloaded": False, "sample_prefix_read": bool(sample_header),
    }
    iex_page, _ = fetched["iex"]
    manifest_probe, manifest_body = fetched["iex_manifest"]
    manifest_summary = {}
    try:
        manifest = json.loads(manifest_body)
        day_keys = sorted(manifest) if isinstance(manifest, dict) else []
        entries = [entry for entries in manifest.values() for entry in entries] if isinstance(manifest, dict) else manifest
        if isinstance(entries, list):
            manifest_summary = {"day_keys": day_keys, "entry_count": len(entries),
                                "feeds": sorted({str(entry.get("feed", "")) for entry in entries}),
                                "entry_metadata": [{key: entry.get(key) for key in ("date", "feed", "version", "protocol", "size")}
                                                   for entry in entries[:5]]}
    except (json.JSONDecodeError, TypeError, AttributeError):
        manifest_summary = {"parse_error": "Response was not a HIST JSON manifest."}
    report["sources"]["iex_hist"] = {
        "page": iex_page, "manifest": manifest_probe, "manifest_summary": manifest_summary,
        "raw_capture_downloaded": False,
        "terms_url": "https://www.iex.io/legal/hist-data-terms",
        "account_entitlement": "Public HIST manifest check only; no real-time feed entitlement inferred.",
    }
    finra_page, _ = fetched["finra"]
    finra_live, finra_body = fetched["finra_live"]
    finra_metadata, finra_metadata_body = fetched["finra_metadata"]
    finra_result = {"page": finra_page, "live_one_row_get": finra_live, "metadata": finra_metadata}
    try:
        metadata = json.loads(finra_metadata_body)
        finra_result["metadata_field_names"] = [field.get("name") for field in metadata.get("fields", [])]
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    try:
        tiny_sample = json.loads(finra_body)
        if isinstance(tiny_sample, list):
            finra_result["one_row_sample"] = tiny_sample[:1]
    except json.JSONDecodeError:
        pass
    if finra_live.get("status") == 200:
        filtered, filtered_body = probe(
            "https://api.finra.org/data/group/otcMarket/name/weeklySummary",
            method="POST", payload={
                "limit": 50,
                "fields": ["issueSymbolIdentifier", "weekStartDate", "marketParticipantName", "MPID", "summaryTypeCode", "totalWeeklyShareQuantity", "totalWeeklyTradeCount", "initialPublishedDate"],
                "compareFilters": [
                    {"compareType": "EQUAL", "fieldName": "issueSymbolIdentifier", "fieldValue": "AAPL"},
                    {"compareType": "EQUAL", "fieldName": "weekStartDate", "fieldValue": args.week},
                ],
            }, max_bytes=100000,
        )
        finra_result["filtered_aapl_week"] = filtered
        finra_result["filtered_request_limit"] = 50
        finra_result["complete_stock_week_panel"] = False
        try:
            rows = json.loads(filtered_body)
            if isinstance(rows, list):
                finra_result["filtered_row_count"] = len(rows)
                finra_result["filtered_sample"] = rows[:3]
            else:
                finra_result["filtered_error"] = rows
        except json.JSONDecodeError:
            finra_result["filtered_error"] = filtered_body.decode(errors="replace")[:500]
        ats_probe, ats_body = probe(
            "https://api.finra.org/data/group/otcMarket/name/weeklySummary", method="POST",
            payload={"limit": 5,
                     "fields": ["issueSymbolIdentifier", "weekStartDate", "marketParticipantName", "MPID", "summaryTypeCode", "totalWeeklyShareQuantity", "totalWeeklyTradeCount", "initialPublishedDate"],
                     "compareFilters": [
                         {"compareType": "EQUAL", "fieldName": "issueSymbolIdentifier", "fieldValue": "AAPL"},
                         {"compareType": "EQUAL", "fieldName": "weekStartDate", "fieldValue": args.week},
                         {"compareType": "EQUAL", "fieldName": "summaryTypeCode", "fieldValue": "ATS_W_SMBL_FIRM"},
                     ]}, max_bytes=10000)
        finra_result["filtered_ats_aapl_week"] = ats_probe
        try:
            finra_result["ats_sample"] = json.loads(ats_body)
        except json.JSONDecodeError:
            finra_result["ats_error"] = ats_body.decode(errors="replace")[:500]
    finra_result["account_entitlement"] = "Unauthenticated endpoint check only; no FINRA API account or credential created."
    report["sources"]["finra_otc"] = finra_result
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
