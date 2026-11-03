"""Render the independently audited, saved-data AAPL event case study."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"XNAS.ITCH": "Nasdaq", "XNYS.PILLAR": "NYSE", "ARCX.PILLAR": "Arca",
          "BATS.PITCH": "BZX", "EDGX.PITCH": "EDGX", "MEMX.MEMOIR": "MEMX",
          "BATY.PITCH": "BYX", "EDGA.PITCH": "EDGA"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out).resolve()
    audit = json.loads((out / "result_audit.json").read_text())
    if audit["status"] != "passed" or not audit.get("independent_statistics_reconstructed"):
        raise ValueError("The independent event-trial audit must pass before publishing")
    for name, digest in audit["output_sha256"].items():
        if sha(out / name) != digest:
            raise ValueError(f"Result changed after audit: {name}")
    summary = json.loads((out / "summary.json").read_text())
    protocol = json.loads((out / "protocol.json").read_text())
    calendar = json.loads((out / "event_calendar.json").read_text())
    results = summary["results"]
    contrasts = pd.DataFrame(results["contrasts"])
    changes = pd.DataFrame(results["daily_changes"])
    latency = pd.DataFrame(results["latency_daily"])
    shocks = pd.DataFrame(results["shock_daily"])
    leadership = pd.DataFrame(results["leadership_daily"])
    profile = pd.read_csv(out / "price_profile.csv")
    event = protocol["event_day"]
    primary = contrasts.loc[contrasts.universe.eq("common5") & contrasts.contrast.eq("primary30")
                            & contrasts.reference_pool.eq("primary_weekday")]
    specifications = [
        ("Mean pair-median latency (µs)", "mean_pair_median_us", "latency", "horizon_ms", 10.),
        ("Match rate (%; changes in pp)", "match_rate", "latency", "horizon_ms", 10.),
        ("Clustered shocks / venue-minute", "mean_venue_clustered_shocks_per_minute", "shock_intensity", "shock_threshold_bp", 1.),
        ("Raw shocks / venue-minute", "mean_venue_shocks_per_minute", "shock_intensity", "shock_threshold_bp", 1.),
    ]
    selected, table = [], []
    for label, metric, kind, dimension, value in specifications:
        row = next(row for row in results["contrasts"] if row["universe"] == "common5"
                   and row["contrast"] == "primary30" and row["reference_pool"] == "primary_weekday"
                   and row["metric"] == metric and row["kind"] == kind and row.get(dimension) == value)
        selected.append(row)
        scale = 100 if metric == "match_rate" else 1
        def f(key):
            return "Unsupported" if row[key] is None else f"{row[key] * scale:,.3f}"
        table.append({"Measure": label, "Before": f("event_pre_value"), "After": f("event_post_value"),
                      "Change": f("event_change"), "Monday median change": f("reference_change_median"),
                      "Event minus reference median": f("event_minus_reference_median"),
                      "References": f'{row["reference_n_valid"]}/{row["reference_n_expected"]}'})

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), constrained_layout=True)
    for ax, row, spec in zip(axes.flat, selected, specifications):
        label, metric, kind, dim, value = spec
        group = changes.loc[changes.universe.eq("common5") & changes.contrast.eq("primary30")
                            & changes.kind.eq(kind) & changes.metric.eq(metric) & changes[dim].eq(value)
                            & changes.day.isin(protocol["primary_reference_days"]) & changes.complete]
        scale = 100 if metric == "match_rate" else 1
        x = np.arange(len(group))
        ax.scatter(x, group.change.to_numpy(float) * scale, color="#627d98", s=45, label="Reference Monday")
        if row["event_change"] is not None:
            ax.axhline(row["event_change"] * scale, color="#b33b24", linewidth=2, label="WWDC change")
        ax.axhline(0, color="#c9cdd1", linewidth=.8)
        ax.set_xticks(x, [date[5:] for date in group.day], rotation=25)
        ax.set_title(label + ": after minus before")
        ax.set_xlabel("Same-clock reference dates")
        if metric == "match_rate":
            ax.set_ylabel("Percentage points")
        ax.legend(fontsize=8)
    fig.suptitle("One WWDC event compared with six Monday date changes\n30 minutes before/after 17:00 UTC; common five venues")
    fig.savefig(out / "event_contrasts.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    primary_path = profile.loc[profile.minute_from_anchor.between(-30, 30)]
    references = primary_path.loc[primary_path.day.isin(protocol["primary_reference_days"])].pivot(
        index="minute_from_anchor", columns="day", values="return_from_anchor_bp")
    if len(references):
        ax.fill_between(references.index, references.quantile(.25, axis=1), references.quantile(.75, axis=1),
                        color="#bccad5", alpha=.7, label="Pointwise Monday IQR (descriptive)")
        ax.plot(references.index, references.median(axis=1), color="#627d98", label="Monday median")
    path = primary_path.loc[primary_path.day.eq(event)]
    ax.plot(path.minute_from_anchor, path.return_from_anchor_bp, color="#b33b24", linewidth=1.2, label="WWDC June 8")
    ax.axvline(0, color="#666", linestyle="--", linewidth=.8)
    ax.axhline(0, color="#aaa", linewidth=.6)
    ax.set(xlabel="Minutes from scheduled keynote start", ylabel="Log return from anchor (basis points)",
           title="AAPL common-five midpoint path: primary 30-minute windows; 1-second descriptive grid")
    ax.legend(fontsize=9)
    fig.savefig(out / "event_price_path.png", dpi=170)
    plt.close(fig)

    def htable(frame):
        return pd.DataFrame(frame).to_html(index=False, escape=True, border=0, na_rep="Unsupported", float_format=lambda x: f"{x:,.3f}")
    sensitivities = contrasts.loc[contrasts.kind.eq("latency") & contrasts.metric.eq("mean_pair_median_us")
                                  & contrasts.contrast.isin(["primary30", "immediate5"])
                                  & contrasts.reference_pool.eq("primary_weekday"),
                                  ["universe", "contrast", "horizon_ms", "event_pre_value", "event_post_value",
                                   "event_change", "reference_change_median", "reference_n_valid", "status"]]
    anchor_tests = contrasts.loc[contrasts.universe.eq("common5") & contrasts.horizon_ms.eq(10)
                                 & contrasts.metric.eq("mean_pair_median_us") & contrasts.reference_pool.eq("primary_weekday"),
                                 ["contrast", "event_pre_value", "event_post_value", "event_change", "reference_change_median", "status"]]
    shock_sensitivity = contrasts.loc[contrasts.contrast.eq("primary30")
                                      & contrasts.reference_pool.eq("primary_weekday")
                                      & contrasts.metric.eq("mean_venue_clustered_shocks_per_minute"),
                                      ["universe", "shock_threshold_bp", "event_pre_value", "event_post_value",
                                       "event_relative_change_percent", "reference_change_median", "status"]]
    lead = leadership.loc[leadership.universe.eq("common5") & leadership.day.eq(event)
                           & leadership.window.isin(["pre30", "post30"])]
    lead = lead.pivot(index="venue", columns="window", values="leadership_share").reset_index()
    lead["change_percentage_points"] = 100 * (lead.post30 - lead.pre30)
    lead["venue"] = lead.venue.map(LABELS)
    lead["pre30"] *= 100
    lead["post30"] *= 100
    counts = latency.loc[latency.universe.eq("common5") & latency.day.eq(event) & latency.horizon_ms.eq(10)
                          & latency.window.isin(["pre30", "post30"]),
                         ["window", "expected_pairs", "supported_pairs", "minimum_venue_coverage", "pair_events",
                          "eligible_pair_events", "matched_pair_events", "quality_truncated_pair_events", "right_truncated_pair_events"]]
    exclusions = latency.loc[latency.horizon_ms.eq(10) & latency.window.isin(["pre30", "post30"])
                              & latency.day.isin(protocol["reference_days"]) & ~latency.complete,
                              ["day", "window", "universe", "reason"]]
    broader = contrasts.loc[contrasts.universe.eq("common5") & contrasts.contrast.eq("primary30")
                             & contrasts.reference_pool.eq("all_reference_days")
                             & ((contrasts.metric.eq("mean_pair_median_us") & contrasts.horizon_ms.eq(10))
                                | (contrasts.metric.eq("mean_venue_clustered_shocks_per_minute") & contrasts.shock_threshold_bp.eq(1))),
                             ["metric", "event_change", "reference_change_median", "reference_change_q25",
                              "reference_change_q75", "event_minus_reference_median", "reference_n_valid", "reference_n_expected", "status"]]
    cover_rows = []
    for row in calendar["events"]:
        links = " ".join(f'<a href="{html.escape(url, quote=True)}">Source {i+1}</a>' for i, url in enumerate(row["source_urls"]))
        cover_rows.append(f'<tr><td>{html.escape(row["label"])}</td><td>{html.escape(row["date"])}</td>'
                          f'<td>{html.escape(str(row.get("event_time_utc") or "Date only"))}</td>'
                          f'<td>{html.escape(row["coverage_status"].replace("_", " "))}</td>'
                          f'<td>{html.escape(row.get("notes", [""])[0])}</td><td>{links}</td></tr>')
    latency_row, rate_row, shock_row, _ = selected
    if latency_row["event_change"] is not None and shock_row["event_change"] is not None:
        latency_direction = "slower" if latency_row["event_change"] > 0 else "faster"
        shock_direction = "higher" if shock_row["event_change"] > 0 else "lower"
        interpretation = (f"At the primary settings, the covered WWDC window has {shock_direction} shock activity and "
                          f"{latency_direction} conditional midpoint matching after the scheduled start. "
                          f"The latency change is {latency_row['event_change']:.2f} microseconds, versus a "
                          f"{latency_row['reference_change_median']:.2f}-microsecond median change on the six reference Mondays. "
                          "This is a timing association in one event; differences in liquidity, event composition and feed transport remain possible explanations.")
    else:
        interpretation = "The primary event contrast lacks sufficient observation or matching support."
    findings = {"status": "completed", "event_day": event, "measurement": summary["measurement"],
                "primary_latency": latency_row, "primary_match_rate": rate_row, "primary_clustered_shocks": shock_row,
                "single_event_only": True, "event_type_comparison_supported": False,
                "market_downloads": 0, "audit_status": audit["status"], "interpretation": interpretation,
                "figures": ["event_contrasts.png", "event_price_path.png"]}
    (out / "findings.json").write_text(json.dumps(findings, indent=2, allow_nan=False), encoding="utf-8")
    limits = "".join(f"<li>{html.escape(text)}</li>" for text in protocol["limitations"])
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>AAPL saved-data event trial</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;color:#23333f;max-width:1180px;margin:40px auto;padding:0 24px}}h1,h2{{line-height:1.2}}h2{{margin-top:40px}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #d7dfe5;vertical-align:top}}th{{background:#edf2f5}}img{{width:100%;height:auto}}.note{{padding:18px;background:#edf2f5;border-left:4px solid #627d98}}a{{color:#186b9b}}.scroll{{overflow-x:auto}}</style>
<h1>Does AAPL look different around WWDC?</h1>
<p>Saved-data exploratory case study, June 8, 2026. The scheduled keynote starts at 17:00 UTC (1 p.m. New York).
This report measures native receive-clock midpoint first passage, shock activity and descriptive counting leadership.</p>
<p class="note"><b>One covered event.</b> These data support a WWDC case study. They do not support a direct comparison
of product events with earnings releases, calls or SEC filings. All calculations use existing local market data; no market-data downloads were made.</p>
<h2>Primary comparison</h2><p>Thirty minutes before versus thirty minutes after the scheduled start; fixed common five venues,
20 directed pairs, 10-ms complete observation horizon. Latency is the equal-pair mean of conditional pair medians,
not a pooled median or causal transmission time. Match rates use complete eligible behind-events.
Shocks are native midpoint changes of at least 1 basis point; the primary rate thins bursts with a causal 100-ms separation per venue.</p>
<div class="scroll">{htable(table)}</div>
<p>{html.escape(interpretation)}</p>
<p>The reference comparison uses six other saved Mondays at the same time. Subtracting their median pre/post change
describes how unusual this date looks; it is not a causal estimate. The broader reference set contains 25 dates and retains explicit coverage exclusions.</p>
<img src="event_contrasts.png" alt="WWDC pre/post changes compared with six Monday changes">
<h2>Price path</h2><img src="event_price_path.png" alt="AAPL WWDC price path and reference Monday paths">
<p>The 1-second path is descriptive. All microsecond latency measurements use original native event times.
The scheduled start does not identify the precise release time of individual announcements. The shaded interval is a pointwise reference IQR, not a confidence interval.</p>
<h2>Broader reference dates</h2><p>The same comparison against all 25 planned reference dates. Missing reference coverage remains explicit.</p>
<div class="scroll">{htable(broader)}</div>
<h2>Horizon and venue sensitivity</h2><div class="scroll">{htable(sensitivities)}</div>
<p>The eight-venue event sample has healthy book coverage, but several directed pairs have fewer than 20 matches before the event.
Those latency comparisons remain unsupported. The two broader-reference exclusions instead arise from missing Arca book coverage.</p>
<h2>Shock-definition sensitivity</h2><p>Rates and percentage changes depend on the venue universe and shock threshold.
A large relative change from a tiny baseline is not evidence of a proportionately large economic effect.</p>
<div class="scroll">{htable(shock_sensitivity)}</div>
<h2>Window sensitivity</h2><p>These comparisons retain the same horizon and common five venues while changing the macro-time windows.
The minus/plus-five-minute variants shift the planned anchor, rather than claiming a known feature-reveal timestamp.</p>{htable(anchor_tests)}
<h2>Counting leadership</h2><p>Percentage of outgoing matches among outgoing plus incoming matches, using the same common-five eligible pair grid.
This is a descriptive count ratio, not Hasbrouck information share or an identified source of information.</p>{htable(lead)}
<h2>Quality and observation counts</h2>{htable(counts)}
<p>All {summary["source_cache_tasks"]} venue/date replay caches and their raw-source identities are verified.
The indexed matcher passed {summary["independent_matching_cases"]:,} bounded direct-scan audit cases with
{summary["independent_matching_mismatches"]} mismatches. Complete-follow-up cohorts retain end and quality truncation counts.</p>
<h3>Unsupported reference rows at the primary horizon</h3>{htable(exclusions)}
<p>Initialized quiet MBO books remain observed. No arbitrary MBP quote-age filter is applied to a fully initialized native order book.
Native integer midpoint reconstruction is checked against the existing validated BBO grid, which is never substituted for microsecond events.</p>
<h2>Event coverage and gaps</h2><p>Official Apple and SEC sources identify scheduled events and filing acceptance times.
Date-only releases are not assigned an invented intraday timestamp. Acceptance time is not a verified public dissemination time.
Coverage labels describe the saved study, rather than the existence of a corporate event.</p>
<p>Calendar evidence is fixed as of {html.escape(calendar["as_of_utc"])}.</p>
<table><tr><th>Event</th><th>Date</th><th>Anchor UTC</th><th>Saved coverage</th><th>Gap or interpretation</th><th>Evidence</th></tr>{''.join(cover_rows)}</table>
<h2>Limits on conclusions</h2><ul>{limits}</ul>
<p>Reference dates are not certified news-free, and a single corporate event remains one observational unit regardless of the number of quotes.
Latency is measured for all eligible native midpoint changes; the shock-rate calculation is a separate activity measure, not a shock-conditioned causal latency estimate.</p>
<h2>Reproduce and inspect</h2><p><a href="protocol.json">Fixed design</a> · <a href="event_calendar.json">Sourced event calendar</a> ·
<a href="result_audit.json">Independent audit</a> · <a href="summary.json">All results</a> · <a href="contrasts.csv">Reference contrasts</a> ·
<a href="pair_metrics.csv">Pair metrics</a> · <a href="venue_metrics.csv">Venue metrics</a> · <a href="provenance.json">Provenance</a></p>
<pre>.venv\\Scripts\\python.exe scripts/run_aapl_event_trial.py --workers 2
.venv\\Scripts\\python.exe scripts/audit_aapl_event_trial.py
.venv\\Scripts\\python.exe scripts/report_aapl_event_trial.py</pre></html>'''
    (out / "findings.html").write_text(document, encoding="utf-8")
    artifacts = ["findings.html", "findings.json", "event_contrasts.png", "event_price_path.png"]
    (out / "readout_manifest.json").write_text(json.dumps({"report_code_sha256": sha(__file__),
        "result_audit_sha256": sha(out / "result_audit.json"), "artifacts": {name: sha(out / name) for name in artifacts}}, indent=2), encoding="utf-8")
    return findings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "out/aapl_event_trial")
    args = parser.parse_args()
    result = build(args.out)
    print(json.dumps({key: result[key] for key in ("status", "event_day", "audit_status", "event_type_comparison_supported")}, indent=2))
