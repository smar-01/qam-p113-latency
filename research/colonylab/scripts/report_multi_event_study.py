"""Render independently audited AAPL event/control results and support gaps."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"wwdc": "WWDC", "product_launch": "Product keynote", "earnings_release": "Earnings release",
          "earnings_call": "Earnings call", "sec_10q": "10-Q acceptance proxy", "sec_10k": "10-K acceptance proxy"}
METRICS = {"latency_us": ("Conditional midpoint latency", "µs", 1),
           "shock_rate": ("Clustered shocks / venue-minute", "shocks / venue-minute", 1),
           "match_rate": ("Matched within horizon", "percentage points", 100)}
GENERATED = {"findings.html", "findings.json", "wwdc_excess.png", "event_support.png",
             "event_freshness_sensitivity.png", "readout_manifest.json"}
SUPPLEMENTAL = ("pilot/tie_diagnostics.json", "apple_publication_metadata.json",
                "businesswire_browser_evidence.json")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def read_json(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value {value} in {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def finite(value):
    return value is not None and isinstance(value, (int, float, np.number)) and np.isfinite(value)


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def number(value, scale=1, digits=3):
    return f"{value * scale:,.{digits}f}" if finite(value) else "Unsupported"


def table(rows):
    if not rows:
        return '<p class="muted">No rows are planned for this view.</p>'
    return '<div class="scroll">' + pd.DataFrame(rows).to_html(index=False, escape=True, border=0,
        na_rep="Unsupported", float_format=lambda value: number(value)) + '</div>'


def select(rows, **identity):
    return [row for row in rows if all(row.get(key) == value for key, value in identity.items())]


def metric_name(metric):
    return METRICS.get(metric, (metric, "", 1))[0]


def cohort(row):
    return "2026 discovery" if row.get("analysis_scope") == "discovery" else "Historical primary"


def control_reasons(row):
    omitted = row.get("excluded_controls", [])
    return "; ".join(f"{item['day']}: {item.get('reason') or 'insufficient support'}" for item in omitted)


def pilot_section(evidence):
    diagnostic = evidence.get("pilot/tie_diagnostics.json")
    if diagnostic is None:
        return '<p class="muted">No pilot diagnostic supplement is supplied with this result.</p>'
    require(diagnostic.get("status") == "completed" and diagnostic.get("policy_changed") is False,
            "Pilot diagnostic is incomplete or records a policy change")
    rows = []
    for venue in diagnostic["venues"]:
        by_age = {row["age_ms"]: row for row in venue["analysis_window_exposure_60_minutes"]}
        strict, sensitivity = by_age.get(250, {}), by_age.get(5000, {})
        counts = venue["source_window_counts_72_minutes"]
        rows.append({"Venue": venue["venue"], "250-ms age: 60-minute coverage (%)": number(strict.get("coverage_fraction"), 100),
            "5000-ms age: 60-minute coverage (%)": number(sensitivity.get("coverage_fraction"), 100),
            "5000-ms age: lost exposure from tied conflicts (%)":
                number(sensitivity.get("lost_exposure_share_from_sameclock_conflicts"), 100),
            "Invalid / locked / crossed completed books":
                f"{counts['invalid_completed_book_rows']} / {counts['locked_completed_book_rows']} / {counts['crossed_completed_book_rows']}"})
    start, end = diagnostic["analysis_window"]["start_utc"], diagnostic["analysis_window"]["end_utc"]
    return f'''<p>The July 31, 2025 earnings-call pilot diagnosed receive-clock exposure over
{html.escape(start)} to {html.escape(end)}. These are coverage fractions over the whole 60-minute pilot span;
the final study still checks each planned window separately.</p>{table(rows)}
<p>At 250 ms, quote-age expiry removes substantial after-hours exposure. At 5000 ms, unresolved tied observations
explain most excluded exposure on Arca, BZX and EDGX. The observed trade/cancel pairs share receive time,
exchange time and sequence; the frozen policy treats conflicting completed states as unknown until another
unambiguous update. Quiet intervals make that lost support last longer. The pilot's completed-book checks above
separately report invalid, locked and crossed states.</p>
<p>The four-venue coverage requirement remained unchanged. Unsupported contrasts are unestimated;
they do not mean zero price response. This is a diagnosis of these pilot streams, not an assumption about every
event or control in the full study. <a href="pilot/tie_diagnostics.json">Inspect the pilot's methods, source hashes and exposure accounting</a>.</p>'''


def class_rows(rows):
    output = []
    for row in rows:
        scale = METRICS[row["metric"]][2]
        output.append({"Cohort": cohort(row), "Class": LABELS.get(row["event_type"], row["event_type"]),
            "Session": row["session"], "Measure": metric_name(row["metric"]), "Clock": row["clock"],
            "Max quote age (ms)": row["age_ms"], "Horizon (ms)": row["horizon_ms"],
            "Events supported / planned": f"{row['n_events']}/{row['n_events_expected']}",
            "Bundles supported / planned": f"{row['n_bundles']}/{row['n_bundles_expected']}",
            "Years supported / planned": f"{row['n_years']}/{row['n_years_expected']}",
            "Median excess change": number(row.get("median_excess_change"), scale),
            "Bundle excess quartiles (Q25, Q75)":
                f"{number(row.get('excess_change_q25'), scale)}, {number(row.get('excess_change_q75'), scale)}",
            "Leave-one-year-out median range":
                f"{number(row.get('leave_one_year_out_min'), scale)}, {number(row.get('leave_one_year_out_max'), scale)}",
            "Status": row["status"]})
    return output


def event_rows(rows):
    output = []
    for row in rows:
        scale = METRICS[row["metric"]][2]
        output.append({"Date": row["day"], "Class": LABELS.get(row["event_type"], row["event_type"]),
            "Cohort": cohort(row), "Measure": metric_name(row["metric"]),
            "Before": number(row.get("event_pre_value"), scale),
            "After": number(row.get("event_post_value"), scale),
            "Event change": number(row.get("event_change"), scale),
            "Control median change": number(row.get("control_change_median"), scale),
            "Excess change": number(row.get("excess_change"), scale),
            "Controls supported / required": f"{row['controls_valid']}/{row['controls_expected']}",
            "Status": row["status"], "Event/control exclusion": row.get("reason") or control_reasons(row),
            "Bundle": row["bundle_id"]})
    return output


def clock_sign_reversals(class_rows_input, excess):
    """Flag supported opposite-sign class medians without merging clock cohorts."""
    groups = {}
    for row in class_rows_input:
        key = (row["analysis_scope"], row["event_type"], row["session"], row["age_ms"],
               row["horizon_ms"], row["contrast"], row["metric"])
        groups.setdefault(key, {})[row["clock"]] = row
    output = []
    for key, clocks in sorted(groups.items()):
        receive, exchange = clocks.get("ts_recv"), clocks.get("ts_event")
        if receive is None or exchange is None:
            continue
        a, b = receive.get("median_excess_change"), exchange.get("median_excess_change")
        if not (receive["n_bundles"] > 0 and exchange["n_bundles"] > 0 and finite(a) and finite(b)
                and np.sign(a) != np.sign(b) and a != 0 and b != 0):
            continue
        identity = dict(zip(("analysis_scope", "event_type", "session", "age_ms", "horizon_ms", "contrast", "metric"), key))
        supported_ids = {}
        for clock in ("ts_recv", "ts_event"):
            supported_ids[clock] = sorted(row["event_id"] for row in select(excess, **identity, clock=clock)
                                          if row["complete"])
        output.append({**identity, "ts_recv_median_excess_change": a, "ts_event_median_excess_change": b,
            "ts_recv_events_supported": receive["n_events"], "ts_event_events_supported": exchange["n_events"],
            "ts_recv_bundles_supported": receive["n_bundles"], "ts_event_bundles_supported": exchange["n_bundles"],
            "ts_recv_status": receive["status"], "ts_event_status": exchange["status"],
            "same_supported_event_ids": supported_ids["ts_recv"] == supported_ids["ts_event"],
            "supported_event_ids_by_clock": supported_ids,
            "interpretation": "Supported class medians have opposite signs across receive and exchange clocks; this sensitivity does not establish a clock-independent direction."})
    return output


def figures(out, excess, summaries, protocol, event_trials):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    clock = protocol["primary_clock"]
    age = protocol["primary_age_ms"]
    horizon = protocol["primary_horizon_ms"]
    primary = select(excess, clock=clock, age_ms=age, horizon_ms=horizon, contrast="primary30")
    wwdc = select(primary, event_type="wwdc")
    years = sorted({row["year"] for row in wwdc})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, metric in zip(axes, ("latency_us", "shock_rate")):
        rows = select(wwdc, metric=metric)
        supported = [row for row in rows if row["complete"] and finite(row.get("excess_change"))]
        for scope, color, marker, label in (("historical_primary", "#285e83", "o", "Historical primary"),
                                            ("discovery", "#bd5532", "D", "2026 discovery")):
            group = [row for row in supported if row["analysis_scope"] == scope]
            if group:
                ax.scatter([row["year"] for row in group], [row["excess_change"] for row in group],
                           s=65, color=color, marker=marker, label=label, zorder=3)
        absent = [str(row["year"]) for row in rows if row not in supported]
        if not supported:
            ax.text(.5, .56, "No supported excess estimates", transform=ax.transAxes, ha="center", va="center")
            ax.set_yticks([])
        if absent:
            ax.text(.02, .02, "Unsupported: " + ", ".join(absent), transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=8, color="#5c6973")
        ax.axhline(0, color="#aeb9c1", linewidth=.8)
        ax.set_xticks(years)
        if years:
            ax.set_xlim(min(years) - .5, max(years) + .5)
        ax.set(xlabel="WWDC year", ylabel=METRICS[metric][1], title=metric_name(metric))
        if supported:
            ax.legend(fontsize=9)
    fig.suptitle(f"WWDC pre/post change minus its five-control median change\n{clock}; max quote age {age:g} ms; horizon {horizon:g} ms; ±30 minutes")
    fig.savefig(out / "wwdc_excess.png", dpi=170)
    plt.close(fig)

    ages = sorted(set(protocol["quote_age_ms"]))
    historical = select(summaries, analysis_scope="historical_primary", clock=clock,
                        horizon_ms=horizon, contrast="primary30")
    strata = sorted({(row["event_type"], row["session"]) for row in historical})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    for ax, metric in zip(axes, ("latency_us", "shock_rate")):
        if not strata:
            ax.text(.5, .5, "No historical class rows", transform=ax.transAxes, ha="center")
            ax.set_axis_off()
            continue
        width = .8 / max(1, len(ages))
        for index, quote_age in enumerate(ages):
            values, labels = [], []
            for event_type, session in strata:
                found = select(historical, event_type=event_type, session=session, age_ms=quote_age, metric=metric)
                require(len(found) == 1, "Support plot lacks a unique class/freshness row")
                row = found[0]
                expected = row["n_events_expected"]
                values.append(100 * row["n_events"] / expected if expected else 0)
                labels.append(f"{row['n_events']}/{expected}")
            x = np.arange(len(strata)) - .4 + width / 2 + index * width
            bars = ax.bar(x, values, width, label=f"{quote_age:g} ms")
            ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
        ax.set_xticks(np.arange(len(strata)), [LABELS.get(kind, kind) + "\n" + session for kind, session in strata])
        ax.set(ylim=(0, 115), ylabel="Supported event anchors (%)", title=metric_name(metric))
        ax.legend(title="Max quote age", fontsize=8, title_fontsize=9)
    fig.suptitle("Support is an outcome of the fixed quality rules\nHistorical primary cohort; same five frozen controls required for every event excess")
    fig.savefig(out / "event_support.png", dpi=170)
    plt.close(fig)

    sensitivity = [row for row in excess if row["clock"] == clock and row["age_ms"] == 5000
                   and row["horizon_ms"] == horizon and row["contrast"] == "primary30"
                   and row["event_type"] in ("wwdc", "product_launch")]
    primary_rth = [row for row in primary if row["event_type"] in ("wwdc", "product_launch")
                   and row["metric"] in ("latency_us", "shock_rate")]
    primary_label = (f"primary {age:g}-ms freshness unsupported" if primary_rth and not any(row["complete"] for row in primary_rth)
                     else f"primary {age:g}-ms support is reported separately")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), constrained_layout=True)
    sensitivity_years = sorted({row["year"] for row in sensitivity})
    for ax, metric in zip(axes, ("latency_us", "shock_rate")):
        selected = select(sensitivity, metric=metric)
        supported = [row for row in selected if row["complete"] and finite(row.get("excess_change"))]
        styles = (("wwdc", "historical_primary", "#285e83", "o", -.07, "WWDC historical"),
                  ("product_launch", "historical_primary", "#398064", "s", .07, "Product keynote"),
                  ("wwdc", "discovery", "#bd5532", "D", -.07, "WWDC 2026 discovery"),
                  ("product_launch", "discovery", "#bd5532", "D", .07, "Product discovery"))
        for event_type, scope, color, marker, offset, label in styles:
            group = select(supported, event_type=event_type, analysis_scope=scope)
            if group:
                ax.scatter([row["year"] + offset for row in group], [row["excess_change"] for row in group],
                           s=65, color=color, marker=marker, label=label, zorder=3)
        if not supported:
            ax.text(.5, .55, "No supported sensitivity estimates", transform=ax.transAxes, ha="center")
        unsupported = [f"{LABELS.get(row['event_type'], row['event_type'])} {row['year']}"
                       for row in selected if row not in supported]
        if unsupported:
            ax.text(.02, .02, "Unsupported: " + ", ".join(unsupported), transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=8, color="#5c6973", wrap=True)
        ax.axhline(0, color="#aeb9c1", linewidth=.8)
        ax.set_xticks(sensitivity_years)
        ax.set(xlabel="Event year (markers offset slightly for readability)", ylabel=METRICS[metric][1],
               title=metric_name(metric) + f" ({len(supported)}/{len(selected)} supported)")
        if len(sensitivity_years) == 1:
            ax.set_xlim(sensitivity_years[0] - .5, sensitivity_years[0] + .5)
        if supported:
            ax.legend(fontsize=8)
    fig.suptitle(f"Predeclared 5000-ms freshness sensitivity; {primary_label}\nEvent change minus five-control median change; {clock}; {horizon:g}-ms horizon; ±30 minutes")
    fig.savefig(out / "event_freshness_sensitivity.png", dpi=170)
    plt.close(fig)


def build(out):
    out = Path(out).resolve()
    audit_path = out / "result_audit.json"
    audit = read_json(audit_path)
    require(audit.get("passed") is True and audit.get("status") == "passed"
            and audit.get("independent_statistics_reconstructed") is True,
            "Independent statistics audit must pass before rendering a findings report")
    bindings = audit.get("output_sha256", {})
    require(isinstance(bindings, dict) and bindings, "Audit lacks output hash bindings")
    for name, digest in bindings.items():
        path = (out / name).resolve()
        require(path.is_relative_to(out), "Audit contains an output path outside its directory")
        require(path.is_file() and sha(path) == digest, f"Output changed after audit: {name}")
    calendar_name = "event_calendar.json" if (out / "event_calendar.json").is_file() else "calendar.json"
    inputs = ["summary.json", "protocol.json", calendar_name, "metrics.csv"]
    require(all(name in bindings for name in inputs), "Report inputs are not all bound by the independent audit")
    summary, protocol, calendar = (read_json(out / name) for name in inputs[:3])
    supplemental_hashes = {name: sha(out / name) for name in SUPPLEMENTAL if (out / name).is_file()}
    supplemental = {name: read_json(out / name) for name in supplemental_hashes}
    require(summary.get("status") == "completed", "Study has no completed results")
    results = summary["results"]
    require(all(isinstance(results.get(name), list) for name in
                ("event_changes", "event_excess", "class_summary", "class_comparisons", "limitations")),
            "Study summary lacks required result tables")
    event_trials = [row for row in protocol["trials"] if row["role"] == "event"]
    controls = [row for row in protocol["trials"] if row["role"] == "control"]
    ids = {row["event_id"] for row in event_trials}
    require(len(ids) == len(event_trials), "Report registry has duplicate event anchors")
    counts = Counter(row["event_type"] for row in event_trials)
    discovery = set(protocol.get("discovery_event_ids", []))
    bundles = {row["bundle_id"] for row in event_trials}
    clock, age, horizon = protocol["primary_clock"], protocol["primary_age_ms"], protocol["primary_horizon_ms"]
    primary = select(results["event_excess"], clock=clock, age_ms=age, horizon_ms=horizon, contrast="primary30")
    primary_classes = select(results["class_summary"], clock=clock, age_ms=age,
                             horizon_ms=horizon, contrast="primary30")
    require(all(row["event_id"] in ids for row in primary), "Result includes an event absent from the frozen registry")
    require(len(primary) == len(event_trials) * len(METRICS), "Primary report grid lacks an event/metric row")
    for row in primary:
        require(not row["complete"] or (row["controls_valid"] == row["controls_expected"] == protocol["controls_per_event"]
                and finite(row["excess_change"])), "Supported event excess lacks complete frozen controls")
    figures(out, results["event_excess"], results["class_summary"], protocol, event_trials)

    support = []
    for metric in METRICS:
        for scope in ("historical_primary", "discovery"):
            rows = select(primary, metric=metric, analysis_scope=scope)
            if rows:
                support.append({"Cohort": "2026 discovery" if scope == "discovery" else "Historical primary",
                                "Measure": metric_name(metric), "Supported excesses / planned":
                                f"{sum(row['complete'] for row in rows)}/{len(rows)}"})
    primary_history = select(primary_classes, analysis_scope="historical_primary")
    freshness = select(results["class_summary"], analysis_scope="historical_primary", clock=clock,
                       horizon_ms=horizon, contrast="primary30")
    horizon_rows = [row for row in results["class_summary"] if row["analysis_scope"] == "historical_primary"
                    and row["clock"] == clock and row["age_ms"] == age and row["contrast"] == "primary30"
                    and row["metric"] in ("latency_us", "match_rate")]
    clocks = select(results["class_summary"], analysis_scope="historical_primary", age_ms=age,
                    horizon_ms=horizon, contrast="primary30")
    sensitivity_clocks = select(results["class_summary"], analysis_scope="historical_primary", age_ms=5000,
                               horizon_ms=horizon, contrast="primary30", metric="latency_us")
    sign_reversal = clock_sign_reversals(sensitivity_clocks, results["event_excess"])
    if sign_reversal:
        reversed_classes = ", ".join(LABELS.get(row["event_type"], row["event_type"]) for row in sign_reversal)
        clock_conclusion = (f"At the predeclared 5000-ms freshness sensitivity, {reversed_classes} class medians have opposite signs "
                            "on receive and exchange clocks. Their direction is therefore sensitive to the clock definition; "
                            "these estimates do not establish a stable clock-independent event-class finding.")
        if any(not row["same_supported_event_ids"] for row in sign_reversal):
            clock_conclusion += " The supported event cohorts also differ between clocks for at least one reversal."
    else:
        clock_conclusion = ("No opposite-sign supported class medians were identified between the receive and exchange clocks "
                            "in the 5000-ms latency sensitivity. This does not establish clock invariance, especially where support is absent.")
    if primary_history and not any(row["n_events"] > 0 for row in primary_history):
        clock_conclusion += f" The primary {age:g}-ms controlled comparisons remain unsupported."
    comparison_rows = []
    comparisons = select(results["class_comparisons"], analysis_scope="historical_primary", clock=clock,
                         age_ms=age, horizon_ms=horizon, contrast="primary30")
    for row in comparisons:
        scale = METRICS[row["metric"]][2]
        comparison_rows.append({"Class A": LABELS.get(row["event_type_a"], row["event_type_a"]),
            "Class B": LABELS.get(row["event_type_b"], row["event_type_b"]), "Measure": metric_name(row["metric"]),
            "Sessions A / B": row["session_a"] + " / " + row["session_b"],
            "Paired years supported / planned": f"{row['n_paired_years']}/{row['n_paired_years_expected']}",
            "Median excess difference A minus B": number(row.get("median_difference_a_minus_b"), scale)
                if row["n_paired_years"] > 0 else "Unsupported",
            "Session confounded": "Yes" if row["session_confounded"] else "No",
            "Status": row["status"], "Excluded years": ", ".join(str(item["year"]) for item in row.get("excluded_years", []))})
    supported_comparisons = [row for row in comparisons if row["n_paired_years"] > 0
                            and finite(row.get("median_difference_a_minus_b"))]
    comparison_note = ("The table describes supported paired years only; counts and excluded years retain the frozen design."
                       if supported_comparisons else
                       "No common fully supported years are available at the primary settings. No numeric event-class comparison is reported.")
    wwdc_rows = select(primary, event_type="wwdc")
    metrics = pd.read_csv(out / "metrics.csv")
    exposure = metrics.loc[metrics.clock.eq(clock) & metrics.horizon_ms.eq(horizon)
                           & metrics.metric.eq("shock_rate") & metrics.window.isin(["pre30", "post30"])]
    quality_rows = []
    for trial in event_trials:
        for quote_age in sorted(set(protocol["quote_age_ms"])):
            chosen = exposure.loc[exposure.event_id.eq(trial["event_id"]) & exposure.age_ms.eq(quote_age)]
            event_book = chosen.loc[chosen.role.eq("event")]
            control_book = chosen.loc[chosen.role.eq("control")]
            quality_rows.append({"Date": trial["day"], "Class": LABELS.get(trial["event_type"], trial["event_type"]),
                "Session": trial["session"], "Max quote age (ms)": quote_age,
                "Lowest event venue/window coverage (%)": number(event_book.minimum_venue_coverage.min(), 100),
                "Lowest control venue/window coverage (%)": number(control_book.minimum_venue_coverage.min(), 100),
                "Event windows supported / planned": f"{int(event_book.complete.sum())}/{len(event_book)}",
                "Control windows supported / planned": f"{int(control_book.complete.sum())}/{len(control_book)}"})
    window_rows = [row for row in results["class_summary"] if row["analysis_scope"] == "historical_primary"
                   and row["clock"] == clock and row["age_ms"] == age and row["horizon_ms"] == horizon
                   and row["metric"] in ("latency_us", "shock_rate")]
    window_display = []
    for row in window_rows:
        base = class_rows([row])[0]
        window_display.append({"Window contrast": row["contrast"], **base})

    coverage_rows = []
    for row in calendar["events"]:
        selected_anchor = row["id"] in ids
        coverage_rows.append("<tr><td>" + html.escape(LABELS.get(row["event_class"], row["event_class"]))
            + "</td><td>" + html.escape(row["date"]) + "</td><td>"
            + html.escape(row.get("event_time_utc") or "Date only; no intraday anchor")
            + "</td><td>" + ("Selected anchor" if selected_anchor else html.escape(row["analysis_role"]))
            + "</td><td>" + " ".join(f'<a href="{html.escape(url, quote=True)}">Source {index + 1}</a>'
                for index, url in enumerate(row["source_urls"])) + "</td><td>"
            + html.escape(" ".join(row.get("notes", []))) + "</td></tr>")
    gaps = Counter(row["analysis_role"] for row in calendar["events"] if row["id"] not in ids)
    date_only = sum(row.get("event_time_utc") is None for row in calendar["events"])
    limitations = list(dict.fromkeys([*protocol.get("limitations", []), *results["limitations"]]))
    findings = clean({"status": "completed", "audit_status": audit["status"],
        "measurement": summary.get("measurement", protocol.get("measurement")),
        "event_anchors": len(event_trials), "event_bundles": len(bundles), "control_trials": len(controls),
        "event_class_anchor_counts": dict(counts), "discovery_event_ids": sorted(discovery),
        "primary_clock": clock, "primary_age_ms": age, "primary_horizon_ms": horizon,
        "primary_event_excess": primary, "primary_class_summary": primary_classes,
        "primary_class_comparisons": comparisons, "primary_supported_class_comparisons": len(supported_comparisons),
        "calendar_date_only_entries": date_only, "calendar_unselected_roles": dict(gaps),
        "publisher_timed_earnings_release_anchors": sum(row["event_class"] == "earnings_release"
            and row["id"] in ids and row["timestamp_nature"] == "publisher_declared_publication" for row in calendar["events"]),
        "supplemental_evidence_sha256": supplemental_hashes,
        "primary_support": support, "limitations": limitations,
        "sensitivity_latency_class_clocks": sensitivity_clocks, "sign_reversal": sign_reversal,
        "clock_sensitivity_conclusion": clock_conclusion,
        "interpretation": "Descriptive matched-control excess changes by event, reporting bundle and year; no causal effect, confidence interval or p-value.",
        "figures": ["wwdc_excess.png", "event_support.png", "event_freshness_sensitivity.png"]})
    (out / "findings.json").write_text(json.dumps(findings, indent=2, allow_nan=False), encoding="utf-8")
    event_classes = "; ".join(f"{LABELS.get(name, name)}: {count}" for name, count in sorted(counts.items()))
    observed_years = sorted({row["year"] for row in event_trials})
    datasets = ", ".join(protocol["datasets"])
    limit_html = "".join(f"<li>{html.escape(item)}</li>" for item in limitations)
    audit_cases = audit.get("independent_matching_cases", summary.get("matching_audit_cases"))
    quality_threshold = 100 * protocol["minimum_coverage_fraction"]
    supplemental_links = " · ".join(f'<a href="{html.escape(name, quote=True)}">{html.escape(label)}</a>'
        for name, label in (("pilot/tie_diagnostics.json", "Pilot tie/exposure diagnosis"),
                            ("apple_publication_metadata.json", "Apple publication metadata"),
                            ("businesswire_browser_evidence.json", "Original Business Wire publisher-header verification"))
        if name in supplemental_hashes)
    release_note = ("Selected earnings releases use verified publisher-declared publication minutes on the original Apple-issued Business Wire pages; calls use Apple's scheduled starts. "
                    "Apple Newsroom publication metadata alone is date-only, and modification/migration timestamps are not used as release clocks."
                    if counts.get("earnings_release", 0) else
                    "No earnings release has a verified intraday anchor in this registry; call anchors remain distinct from release timing.")
    report = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AAPL corporate-event study</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;color:#23333f;max-width:1220px;margin:40px auto;padding:0 24px}}h1,h2{{line-height:1.2}}h2{{margin-top:38px}}table{{border-collapse:collapse;font-size:12px;min-width:100%;margin:15px 0}}th,td{{text-align:left;vertical-align:top;padding:9px;border-bottom:1px solid #d7dfe5}}th{{background:#edf2f5}}img{{width:100%;height:auto}}.note{{padding:18px;background:#edf2f5;border-left:4px solid #627d98}}.scroll{{overflow-x:auto}}.muted{{color:#586976}}a{{color:#186b9b}}summary{{cursor:pointer;font-weight:600;margin:14px 0}}code{{font-size:.9em}}</style></head><body>
<h1>How does AAPL behave around corporate events?</h1>
<p>Audited exploratory study across {', '.join(map(str, observed_years))}: {len(event_trials)} event anchors,
{len(bundles)} reporting/keynote bundles and {len(controls)} matched-control trials. {html.escape(event_classes)}.</p>
<p class="note"><b>Support comes before interpretation.</b> A supported event excess requires supported before/after measurements
for that event and every one of its {protocol['controls_per_event']} frozen same-year, weekday and local-clock controls.
Unsupported rows remain visible. Larger quote-age sensitivities do not replace an unsupported primary estimate.</p>
<h2>What the primary data support</h2>{table(support)}
<p>Primary settings: {html.escape(clock)}, {age:g}-ms maximum quote age, {horizon:g}-ms matching horizon,
30 minutes before/after the sourced anchor. Fixed venues: {html.escape(datasets)}.
Each venue/window requires at least {quality_threshold:g}% valid observation; each intended directed latency pair requires
at least {protocol['minimum_matched_events']} complete-follow-up matches. Quiet MBP-1 quotes expire under the frozen age rule.</p>
<h2>WWDC replication and the 2026 discovery case</h2>
<p>Historical WWDC occurrences are kept separate from the previously inspected 2026 discovery event.
Each plotted point is an event's after-minus-before change minus its own five-control median change.
Only supported excess estimates are plotted; missing years are labelled.</p>
<img src="wwdc_excess.png" alt="Supported WWDC excess latency and clustered shock changes, separating historical events and 2026 discovery">
{table(event_rows([row for row in wwdc_rows if row['metric'] in ('latency_us', 'shock_rate')]))}
<h2>Historical class summaries at the primary settings</h2>
<p>Latency is the equal-pair mean of conditional midpoint first-passage medians in microseconds.
The matching-rate table entries use percentages for levels and percentage points for changes.
Shock intensity is the equal-venue rate of midpoint changes at the fixed {protocol['primary_shock_threshold_bp']:g}-bp threshold,
with causal {protocol['shock_cluster_separation_ms']:g}-ms burst thinning. These are separate measures; latency is not conditioned on a shock.</p>
{table(class_rows(primary_history))}
<p>Class medians give equal weight to supported corporate-event bundles. Any incomplete summary describes its supported subset,
with planned events, bundles and years retained. Quartiles and leave-one-year-out ranges describe dispersion and sensitivity;
they are not confidence intervals.</p>
<h2>Class comparisons</h2><p>{html.escape(comparison_note)}</p>{table(comparison_rows)}
<p>Comparisons use within-year differences in the classes' matched-control excess changes. Every planned bundle in each
class/year stratum must be supported. WWDC/product events generally occur in regular hours, while earnings anchors occur after hours:
event class and session remain confounded even with same-clock controls.</p>
<h2>Quote-age sensitivity and support</h2>
<p>The frozen {', '.join(f'{value:g}' for value in protocol['quote_age_ms'])}-ms quote-age settings retain their own observable cohorts.
Higher after-hours support at a longer age threshold cannot establish that the stricter primary measurement succeeded.
Bars show supported/planned anchors, not effect estimates.</p>
<img src="event_support.png" alt="Supported historical event counts by class and maximum quote age for latency and shock activity">
<h3>Predeclared 5000-ms freshness sensitivity</h3>
<p>This separate figure shows supported WWDC and product-keynote excess changes under the predeclared 5000-ms age setting.
It displays event-level heterogeneity and retains the 2026 WWDC discovery marker. It does not turn an unsupported 250-ms
primary contrast into a supported result, and its observable cohort can differ from the primary cohort.</p>
<img src="event_freshness_sensitivity.png" alt="Predeclared 5000-ms freshness sensitivity for supported WWDC and product-keynote excess latency and shocks, with 2026 discovery separate">
<h3>Clock dependence of the 5000-ms latency sensitivity</h3>
<p>{html.escape(clock_conclusion)}</p>{table(class_rows(sensitivity_clocks))}
<p>Both clock definitions are shown at the same freshness threshold and horizon. Supported/planned counts remain explicit;
different clocks can still yield different observable cohorts. These descriptive sensitivity medians do not replace the primary result.</p>
{table(class_rows(freshness))}
<details><summary>Matching-horizon sensitivity</summary>{table(class_rows(horizon_rows))}</details>
<details><summary>Receive-clock and exchange-clock summaries</summary>{table(class_rows(clocks))}</details>
<details><summary>Macro-time window sensitivity</summary>{table(window_display)}</details>
<h2>Event-level latency, shock activity and matching rates</h2>
<p>Every selected anchor appears below at the primary settings. Unsupported event changes or control comparisons remain undefined;
their support status and exclusion reasons explain the gap. A supported event's measured pre/post change can coexist with
an unsupported controlled excess when its five controls fail support. An unestimated controlled comparison does not imply
missing raw files or zero market response.</p>{table(event_rows(primary))}
<details><summary>5000-ms event measurements and their five-control support</summary>
{table(event_rows(select(results['event_excess'], clock=clock, age_ms=5000, horizon_ms=horizon, contrast='primary30')))}</details>
<h2>Venue observation coverage</h2>
<p>Coverage below is the lowest venue/window fraction among the event's two primary windows or among all corresponding control windows.
The threshold is {quality_threshold:g}%. After-hours quote gaps can prevent a valid comparison even when many quote records exist.
Minimum pair-match counts and complete follow-up are additional latency checks.</p>{table(quality_rows)}
<p>The independent audit passed and verified the input/output bindings before this report was rendered.
Bounded native first-passage checks reported {number(audit_cases, digits=0)} sampled cases.
Source-byte, stream, quote-quality and arithmetic checks are documented in the audit; they do not rule out every upstream feed issue.</p>
<h2>After-hours pilot limitation</h2>{pilot_section(supplemental)}
<h2>Release/call bundles and source coverage</h2>
<p>{html.escape(release_note)}</p>
<p>Earnings releases and calls on the same reporting occasion share one bundle. Their anchors are distinct, but they do not create independent
earnings replications. A call's prewindow may already contain the results release, and overlapping anchor windows share observations.
Scheduled starts and publisher-declared times are not nanosecond first-information timestamps.</p>
<p>The sourced catalog contains {date_only} date-only entries. Date-only releases are not assigned invented intraday times;
SEC acceptance proxies are distinct from verified public dissemination. Provider availability, valid venue exposure and sampled controls
determine usable coverage separately. The calendar is not an exhaustive list of Apple or macroeconomic news.</p>
<p>{supplemental_links}</p>
<details><summary>Sourced calendar, selected anchors and excluded/context entries</summary><div class="scroll"><table>
<tr><th>Class</th><th>Date</th><th>Anchor UTC</th><th>Registry role</th><th>Sources</th><th>Timing/coverage notes</th></tr>
{''.join(coverage_rows)}</table></div></details>
<h2>Limits on conclusions</h2><ul>{limit_html}</ul>
<h2>Reproduce and inspect</h2><p><a href="protocol.json">Frozen protocol and registry</a> ·
<a href="{calendar_name}">Sourced calendar</a> · <a href="result_audit.json">Independent audit</a> ·
<a href="summary.json">All results</a> · <a href="event_excess.csv">Event excesses</a> ·
<a href="class_summary.csv">Class summaries</a> · <a href="class_comparisons.csv">Class comparisons</a> ·
<a href="metrics.csv">Measured values and support</a> · <a href="venue_metrics.csv">Venue coverage and shocks</a></p>
<pre>python scripts/audit_multi_event_study.py --out "{html.escape(str(out))}"
python scripts/report_multi_event_study.py --out "{html.escape(str(out))}"</pre>
</body></html>'''
    (out / "findings.html").write_text(report, encoding="utf-8")
    artifacts = sorted(GENERATED - {"readout_manifest.json"})
    require(all(sha(out / name) == digest for name, digest in supplemental_hashes.items()),
            "Supplemental evidence changed while rendering the report")
    manifest = {"report_code_sha256": sha(Path(__file__)), "result_audit_sha256": sha(audit_path),
                "inputs": {name: bindings[name] for name in inputs},
                "supplemental_evidence_sha256": supplemental_hashes,
                "artifacts": {name: sha(out / name) for name in artifacts}}
    (out / "readout_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", "--out-dir", type=Path, default=ROOT / "out/aapl_multi_event")
    args = parser.parse_args()
    result = build(args.out)
    print(json.dumps({key: result[key] for key in ("status", "audit_status", "event_anchors", "event_bundles", "control_trials")}), flush=True)


if __name__ == "__main__":
    main()
