"""Build the final plain-language latency readout from audited local results.

This script does no matching, fetching or new statistical fitting.  It requires
both science audits to have passed, publishes explicitly labelled descriptive
aggregates, and restores the team's counting-style leadership output.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VENUE_LABELS = {"XNAS.ITCH": "Nasdaq", "XNYS.PILLAR": "NYSE", "BATS.PITCH": "BZX",
                "ARCX.PILLAR": "Arca", "EDGX.PITCH": "EDGX", "MEMX.MEMOIR": "MEMX",
                "BATY.PITCH": "BYX", "EDGA.PITCH": "EDGA"}


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finite_number(value):
    return float(value) if value is not None and np.isfinite(value) else None


def _leadership(daily):
    """Outgoing matched counts / all matched counts involving each venue."""
    all_rows = daily.loc[daily.regime.eq("all")]
    keys = ["study", "phase", "clock", "symbol", "horizon_ms"]
    records = []
    for key, group in all_rows.groupby(keys, sort=True):
        require(not group.duplicated(["date", "leader", "follower"]).any(),
                "Duplicate all-regime date/pair rows in leadership calculation.")
        values = group.n_matched.to_numpy(float)
        require(np.isfinite(values).all() and (values >= 0).all() and (values == np.floor(values)).all(),
                "Invalid matched counts for leadership calculation.")
        led, followed = group.groupby("leader").n_matched.sum(), group.groupby("follower").n_matched.sum()
        venues = sorted(set(led.index) | set(followed.index))
        outgoing = led.reindex(venues, fill_value=0).to_numpy(np.int64)
        incoming = followed.reindex(venues, fill_value=0).to_numpy(np.int64)
        total = int(group.n_matched.sum())
        require(int(outgoing.sum()) == total == int(incoming.sum()), "Leadership matched-count conservation fails.")
        denominator = outgoing + incoming
        require(int(denominator.sum()) == 2 * total, "Leadership denominator does not count every match twice.")
        share = np.divide(outgoing, denominator, out=np.full(len(venues), np.nan), where=denominator > 0)
        require(((share[np.isfinite(share)] >= 0) & (share[np.isfinite(share)] <= 1)).all(),
                "Leadership shares fall outside [0,1].")
        for position, venue in enumerate(venues):
            records.append({**dict(zip(keys, key)), "venue": venue, "dates": int(group.date.nunique()),
                            "matched_outgoing": int(outgoing[position]), "matched_incoming": int(incoming[position]),
                            "matched_involving_venue": int(denominator[position]),
                            "leadership_share": share[position],
                            "measure": "outgoing_matched_count_over_outgoing_plus_incoming; not_Hasbrouck_or_causal"})
    return pd.DataFrame(records)


def _primary_tables(daily, inference, protocol):
    study, phase = protocol["primary_study"], protocol["primary_phase"]
    horizon, minimum_dates = protocol["primary_horizon_ms"], protocol["minimum_inference_dates"]
    keep = (inference.study.eq(study) & inference.phase.eq(phase) & inference.horizon_ms.eq(horizon))
    primary = inference.loc[keep]
    raw = daily.loc[daily.study.eq(study) & daily.phase.eq(phase)
                    & daily.horizon_ms.eq(horizon) & daily.regime.eq("all")]
    latency_rows, regime_rows, supported_tables = [], [], []
    for (clock, symbol), group in raw.groupby(["clock", "symbol"], sort=True):
        pairs = group[["leader", "follower"]].drop_duplicates()
        points = primary.loc[primary.clock.eq(clock) & primary.symbol.eq(symbol)
                             & primary.contrast.eq("latency")]
        supported = points.loc[points.interval_status.eq("ok") & points.n_dates.eq(minimum_dates)]
        require(not supported.duplicated(["leader", "follower"]).any(), "Duplicate supported latency pair estimates.")
        supported_tables.append((clock, symbol, supported.copy(), sorted(set(group.leader) | set(group.follower))))
        eligible, matched = int(group.n_eligible.sum()), int(group.n_matched.sum())
        require(0 <= matched <= eligible, "Primary complete matched counts are inconsistent.")
        subset = group.merge(supported[["leader", "follower"]], on=["leader", "follower"],
                             how="inner", validate="many_to_one")
        subset_eligible, subset_matched = int(subset.n_eligible.sum()), int(subset.n_matched.sum())
        estimates = supported.estimate_us.to_numpy(float)
        require(np.isfinite(estimates).all(), "Supported latency estimates are not finite.")
        fastest = supported.loc[supported.estimate_us.idxmin()] if len(supported) else None
        slowest = supported.loc[supported.estimate_us.idxmax()] if len(supported) else None
        latency_rows.append({
            "clock": clock, "symbol": symbol, "horizon_ms": horizon,
            "intended_dates": minimum_dates, "total_directed_pairs": len(pairs),
            "full_date_supported_pairs": len(supported),
            "median_across_supported_pair_equal_date_means_us": finite_number(np.median(estimates)) if len(estimates) else None,
            "minimum_supported_pair_equal_date_mean_us": finite_number(np.min(estimates)) if len(estimates) else None,
            "maximum_supported_pair_equal_date_mean_us": finite_number(np.max(estimates)) if len(estimates) else None,
            "minimum_supported_pair": f"{fastest.leader} -> {fastest.follower}" if fastest is not None else None,
            "maximum_supported_pair": f"{slowest.leader} -> {slowest.follower}" if slowest is not None else None,
            "all_pairs_eligible_pair_events": eligible, "all_pairs_matched_pair_events": matched,
            "all_pairs_complete_match_rate_percent": 100 * matched / eligible if eligible else None,
            "supported_pairs_eligible_pair_events": subset_eligible, "supported_pairs_matched_pair_events": subset_matched,
            "supported_pairs_complete_match_rate_percent": 100 * subset_matched / subset_eligible if subset_eligible else None,
        })
        regimes = primary.loc[primary.clock.eq(clock) & primary.symbol.eq(symbol)
                              & primary.contrast.eq("volatile_vs_calm")]
        paired = regimes.loc[regimes.interval_status.eq("ok") & regimes.n_dates.eq(minimum_dates)]
        effects = paired.relative_change_percent.to_numpy(float)
        finite = np.isfinite(effects)
        require(np.isfinite(paired.ci95_low_us).all() and np.isfinite(paired.ci95_high_us).all(),
                "Supported regime intervals are missing.")
        positive_ci = paired.ci95_low_us.gt(0)
        negative_ci = paired.ci95_high_us.lt(0)
        regime_rows.append({
            "clock": clock, "symbol": symbol, "horizon_ms": horizon,
            "total_directed_pairs": len(pairs), "full_date_paired_supported_pairs": len(paired),
            "pairs_with_defined_relative_effect": int(finite.sum()),
            "median_paired_volatile_minus_calm_percent": finite_number(np.median(effects[finite])) if finite.any() else None,
            "positive_point_estimates": int(paired.estimate_us.gt(0).sum()),
            "negative_point_estimates": int(paired.estimate_us.lt(0).sum()),
            "positive_marginal_95pct_intervals": int(positive_ci.sum()),
            "negative_marginal_95pct_intervals": int(negative_ci.sum()),
            "marginal_intervals_including_zero": int((~positive_ci & ~negative_ci).sum()),
            "interpretation": "Exploratory marginal date-block intervals; no multiplicity adjustment or familywise discovery claim.",
        })
    require(len(latency_rows) == len(protocol["clocks"]) * raw.symbol.nunique(),
            "Primary clock/symbol result grid is incomplete.")
    return latency_rows, regime_rows, supported_tables


def _html_table(rows, columns, labels, *, signed=()):
    header = "".join(f"<th>{html.escape(labels.get(column, column))}</th>" for column in columns)
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column)
            if value is None:
                label = "—"
            elif isinstance(value, (float, np.floating)):
                label = f"{value:+.2f}" if column in signed else f"{value:,.2f}"
            elif isinstance(value, (int, np.integer)):
                label = f"{value:,}"
            else:
                label = str(value)
            cells.append(f"<td>{html.escape(label)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return '<div class="table"><table><thead><tr>' + header + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"


def build_readout(out_dir="out/latency", *, root=ROOT):
    root = Path(root).resolve()
    out = Path(out_dir)
    if not out.is_absolute():
        out = root / out
    required_json = ["summary.json", "protocol.json", "result_audit.json", "raw_audit.json", "regression_test_summary.json"]
    inputs = {name: read_json(out / name) for name in required_json}
    summary, protocol = inputs["summary.json"], inputs["protocol.json"]
    final_audit, raw_audit = inputs["result_audit.json"], inputs["raw_audit.json"]
    regression = inputs["regression_test_summary.json"]
    require(summary.get("status") == "completed", "Readout requires a completed main analysis.")
    require(final_audit.get("status") == "passed" and raw_audit.get("status") == "passed",
            "Readout requires both independent science audits to pass.")
    require(final_audit.get("raw_hashes_freshly_verified") is True,
            "Final readout requires the full fresh-source-hash audit, not arithmetic-only mode.")
    if "summary_sha256" in raw_audit:
        require(raw_audit["summary_sha256"] == sha256_file(out / "summary.json"),
                "Raw audit is bound to a different summary; verify it again before building the readout.")
    if "protocol_sha256" in raw_audit:
        require(raw_audit["protocol_sha256"] == sha256_file(out / "protocol.json"),
                "Raw audit is bound to a different protocol.")
    require(final_audit.get("raw_source_requests") == summary["source_requests"]
            and final_audit.get("raw_records_reconciled") == summary["raw_records"],
            "Main summary and all-source audit disagree on source coverage.")
    daily = pd.read_csv(out / "daily_latency.csv")
    inference = pd.read_csv(out / "inference.csv")
    require(len(daily) == final_audit["daily_rows"] and len(inference) == final_audit["inference_rows"],
            "Result table row counts differ from the passed audit.")
    leadership = _leadership(daily)
    primary, regimes, matrices = _primary_tables(daily, inference, protocol)
    regime_available = any(row["full_date_paired_supported_pairs"] for row in regimes)
    regime_interpretation = ("The primary calm-versus-volatile comparison is unavailable: no directed pair has supported matched calm and volatile medians on all twenty dates. The team's volatile-period finding is neither replicated nor rejected by these data under the quality rules."
                             if not regime_available else
                             "Only fully paired supported dates enter the primary regime comparison; marginal intervals remain exploratory and unadjusted for multiple comparisons.")
    volatility_coverage = []
    volatility_path = out / "realized_volatility.csv"
    if volatility_path.is_file():
        volatility = pd.read_csv(volatility_path, usecols=["study", "phase", "clock", "symbol", "regime"])
        volatility = volatility.loc[volatility.study.eq(protocol["primary_study"])
                                    & volatility.phase.eq(protocol["primary_phase"])]
        for (clock, symbol), group in volatility.groupby(["clock", "symbol"], sort=True):
            counts = group.regime.value_counts()
            calm, volatile, unknown = (int(counts.get(name, 0)) for name in ("calm", "volatile", "unknown"))
            require(calm + volatile + unknown == len(group), "Unexpected realized-volatility regime label.")
            volatility_coverage.append({"clock": clock, "symbol": symbol, "sampled_seconds": len(group),
                                        "calm_seconds": calm, "volatile_seconds": volatile, "unknown_seconds": unknown,
                                        "classified_percent": 100 * (calm + volatile) / len(group) if len(group) else None})
    leadership.to_csv(out / "leadership_share.csv", index=False)
    matrix_files = []
    for clock, symbol, supported, venues in matrices:
        matrix = supported.pivot(index="leader", columns="follower", values="estimate_us").reindex(index=venues, columns=venues)
        require(np.isnan(np.diag(matrix.to_numpy(float))).all(), "Latency matrix diagonal must be blank.")
        filename = f"latency_matrix_{clock}_{symbol}.csv"
        matrix.to_csv(out / filename)
        matrix_files.append(filename)
    sensitivity = read_json(out / "freshness_sensitivity.json") if (out / "freshness_sensitivity.json").is_file() else None
    if sensitivity is not None:
        require(sensitivity.get("status") == "completed", "Saved freshness sensitivity is incomplete.")
    figures = [name for name in summary["figures"] if (out / name).is_file()]
    require(len(figures) == len(summary["figures"]), "A reported figure is missing.")
    input_paths = required_json + ["daily_latency.csv", "inference.csv"]
    if sensitivity is not None:
        input_paths.append("freshness_sensitivity.json")
    if volatility_path.is_file():
        input_paths.append("realized_volatility.csv")
    findings = {
        "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "How quickly does a venue's best bid or ask reach another venue's new quote price?",
        "primary_scope": {"study": protocol["primary_study"], "phase": protocol["primary_phase"],
                          "dates": protocol["minimum_inference_dates"], "horizon_ms": protocol["primary_horizon_ms"],
                          "quote_maximum_age_ms": protocol["max_quote_age_ms"]},
        "scope": {"source_requests": summary["source_requests"], "raw_records": summary["raw_records"],
                  "dates": summary["dates"], "studies": summary["studies"],
                  "analysis_status": summary["analysis_status"]},
        "primary_latency_by_clock_symbol": primary,
        "primary_regime_by_clock_symbol": regimes,
        "primary_regime_comparison_available": bool(regime_available),
        "primary_regime_interpretation": regime_interpretation,
        "primary_volatility_coverage": volatility_coverage,
        "latency_estimate_definition": "For each fully supported directed pair: equal-date mean of twenty daily conditional matched-event medians. The displayed typical value is the median across these pair estimates, never a pooled-event median.",
        "match_rate_definition": "Event-count-weighted matched / fully observed behind pair-events, pooled across all directed pairs and dates within each stock/clock. Supported-pair-only rates are also retained in JSON.",
        "leadership_measure": {"file": "leadership_share.csv", "rows": len(leadership),
                               "definition": "For each venue, sum outgoing matched events divided by outgoing plus incoming matched events, within study/phase/stock/clock/horizon.",
                               "limitations": "Descriptive counting measure sensitive to event rates, horizons and coverage; not Hasbrouck information share, causal price discovery, or shares summing to one across venues."},
        "matrix_files": matrix_files, "figures": figures,
        "validation": {"all_source_and_result_audit": final_audit, "selected_date_independent_raw_audit": raw_audit,
                       "regression_tests": regression},
        "freshness_sensitivity": sensitivity,
        "limitations": protocol["limitations"],
        "input_fingerprints": {name: sha256_file(out / name) for name in input_paths},
        "readout_generator_sha256": sha256_file(Path(__file__)),
    }
    (out / "findings.json").write_text(json.dumps(findings, indent=2, allow_nan=False), encoding="utf-8")
    primary_columns = ["clock", "symbol", "full_date_supported_pairs", "median_across_supported_pair_equal_date_means_us",
                       "minimum_supported_pair_equal_date_mean_us", "maximum_supported_pair_equal_date_mean_us",
                       "all_pairs_complete_match_rate_percent"]
    primary_labels = {"clock": "Clock", "symbol": "Stock", "full_date_supported_pairs": "Full-date supported pairs",
                      "median_across_supported_pair_equal_date_means_us": "Median across pair estimates, µs",
                      "minimum_supported_pair_equal_date_mean_us": "Smallest pair estimate, µs",
                      "maximum_supported_pair_equal_date_mean_us": "Largest pair estimate, µs",
                      "all_pairs_complete_match_rate_percent": "Matched share, all pairs (%)"}
    primary_html = _html_table(primary, primary_columns, primary_labels)
    regime_columns = ["clock", "symbol", "full_date_paired_supported_pairs", "median_paired_volatile_minus_calm_percent",
                      "positive_point_estimates", "positive_marginal_95pct_intervals", "negative_marginal_95pct_intervals"]
    regime_labels = {"clock": "Clock", "symbol": "Stock", "full_date_paired_supported_pairs": "Paired full-date pairs",
                     "median_paired_volatile_minus_calm_percent": "Median volatile−calm effect (%)",
                     "positive_point_estimates": "Positive estimates", "positive_marginal_95pct_intervals": "95% intervals above zero",
                     "negative_marginal_95pct_intervals": "95% intervals below zero"}
    regime_html = _html_table(regimes, regime_columns, regime_labels, signed=("median_paired_volatile_minus_calm_percent",))
    regime_note = (f"<p class=\"note\"><strong>{html.escape(regime_interpretation)}</strong> Points from shorter supported date subsets in the figure are descriptive only.</p>"
                   if not regime_available else
                   "<p>Only the full-date paired tasks counted below enter the primary regime summaries; unsupported pairs remain missing.</p>")
    volatility_html = _html_table(volatility_coverage, ["clock", "symbol", "sampled_seconds", "calm_seconds", "volatile_seconds", "unknown_seconds", "classified_percent"],
                                   {"clock": "Clock", "symbol": "Stock", "sampled_seconds": "Sampled seconds", "calm_seconds": "Calm",
                                    "volatile_seconds": "Volatile", "unknown_seconds": "Unknown", "classified_percent": "Classified (%)"}) if volatility_coverage else ""
    scopes = [{"study": study["name"], "dates": study["dates"], "venues": study["venues"],
               "stocks": ", ".join(study["symbols"]), "minutes_per_date": study["minutes_per_date"]}
              for study in summary["studies"]]
    scope_html = _html_table(scopes, ["study", "dates", "venues", "stocks", "minutes_per_date"],
                             {"study": "Study", "dates": "Dates", "venues": "Venues", "stocks": "Stocks", "minutes_per_date": "Minutes per date"})
    audit_items = [
        ("Raw requests freshly checked", final_audit["raw_source_requests"]),
        ("Raw records reconciled to completed receipts", final_audit["raw_records_reconciled"]),
        ("Daily latency / regime / horizon rows checked", final_audit["daily_rows"]),
        ("Date-level estimates and intervals independently rebuilt", final_audit["inference_rows"]),
        ("Pair/side matching tasks sampled", final_audit["independent_matching_tasks"]),
        ("Sampled first-passage cases checked", final_audit["independent_event_cases"]),
        ("Selected-date raw requests independently decoded", raw_audit.get("raw_requests_checked")),
        ("Selected-date raw records independently decoded", raw_audit.get("raw_records_checked")),
        ("Quote series independently rebuilt", raw_audit.get("quote_series_independently_rebuilt")),
        ("Complete pair/horizon raw summaries checked", raw_audit.get("full_pair_horizon_summaries_checked")),
        ("Raw/replayed mismatches", raw_audit.get("mismatches")),
    ]
    audit_html = _html_table([{"check": label, "count": value} for label, value in audit_items], ["check", "count"],
                            {"check": "Validation evidence", "count": "Count"})
    regression_text = html.escape(regression.get("interpretation", "See the linked regression summary for exact test outcomes."))
    sensitivity_html = "<p>No quote-age sensitivity file was included.</p>"
    if sensitivity is not None:
        selected = [{**row, "primary_match_percent": 100 * row["primary_match_rate"],
                     "reference_match_percent": 100 * row["reference_match_rate"]}
                    for row in sensitivity["aggregates"]
                    if row["clock"] == "ts_recv" and row["horizon_ms"] == protocol["primary_horizon_ms"]
                    and row["primary_match_rate"] is not None and row["reference_match_rate"] is not None]
        sensitivity_html = (f"<p>The separate 1000 ms comparison covers only {sensitivity['common_completed_study_dates']} common training dates. It is a limited descriptive check: changing freshness changes event cohorts. No p-values or confidence intervals are computed.</p>"
                            + _html_table(selected, ["symbol", "eligible_change_percent", "primary_match_percent", "reference_match_percent",
                                                    "median_paired_conditional_median_change_us"],
                                          {"symbol": "Stock", "eligible_change_percent": "Eligible change (%)",
                                           "primary_match_percent": "250 ms matched (%)", "reference_match_percent": "1000 ms matched (%)",
                                           "median_paired_conditional_median_change_us": "Median paired daily/pair change, µs"})
                            + '<p><a href="freshness_sensitivity.json">Exact dates, identity checks and driver-change disclosure</a> · <a href="freshness_sensitivity.csv">All clocks and horizons</a></p>')
    figure_html = "".join(f'<figure><img src="{html.escape(name)}" alt="{html.escape(name.replace("_", " "))}"><figcaption>{html.escape(name)}</figcaption></figure>' for name in figures)
    matrix_links = " · ".join(f'<a href="{html.escape(name)}">{html.escape(name.removeprefix("latency_matrix_").removesuffix(".csv"))}</a>' for name in matrix_files)
    limitation_html = "".join(f"<li>{html.escape(value)}</li>" for value in protocol["limitations"])
    typical = [row["median_across_supported_pair_equal_date_means_us"] for row in primary
               if row["median_across_supported_pair_equal_date_means_us"] is not None]
    rates = [row["all_pairs_complete_match_rate_percent"] for row in primary
             if row["all_pairs_complete_match_rate_percent"] is not None]
    result_sentence = (f"Across the stock/clock groups, the typical supported pair estimate ranges from {min(typical):,.1f} to {max(typical):,.1f} µs. "
                       if typical else "No directed pair meets the complete twenty-date support rule. ")
    if rates:
        result_sentence += f"Only {min(rates):.1f}–{max(rates):.1f}% of fully observed behind pair-events reach the new price within 10 ms."
    body = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ColonyLab latency findings</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;color:#172c3b;max-width:1180px;margin:40px auto;padding:0 24px}}h1,h2{{line-height:1.2}}h2{{margin-top:42px}}.note{{background:#edf4f7;border-left:4px solid #176b87;padding:15px 20px}}.table{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:13px;margin:16px 0}}th,td{{padding:9px 10px;border-bottom:1px solid #d3dee4;text-align:right;vertical-align:top}}th:first-child,td:first-child{{text-align:left}}th{{background:#edf4f7}}a{{color:#126782}}img{{width:100%;height:auto}}figure{{margin:30px 0}}figcaption{{font-size:12px;color:#526875}}.small{{font-size:14px;color:#445f70}}li{{margin:8px 0}}code{{background:#edf4f7;padding:2px 5px}}</style></head><body>
<h1>How quickly do venues follow a quote change?</h1>
<p>This adds the team's direct reaction-latency analysis to ColonyLab using existing native MBP-1 event records, both timestamp clocks, quote-quality barriers, and independent validation.</p>
<p class="note"><strong>{summary['source_requests']:,} verified requests · {summary['raw_records']:,} records · {summary['dates']} saved dates.</strong> These are partial trading-day windows. The primary tables use twenty inherited held-out dates from the eight-venue study. This is new exploratory analysis of previously studied data, not a new untouched latency holdout.</p>
<h2>Primary 10 ms result</h2><p><strong>{html.escape(result_sentence)}</strong></p><p>For each directed pair, average its daily conditional matched-event median equally across all twenty supported dates. The table's typical value is the median across those pair estimates. It is <strong>not a pooled-event median</strong>. Sparse pairs are excluded from these pair-estimate summaries.</p>
{primary_html}
<p class="small">Matched shares pool fully observed behind pair-events across all directed pairs and dates within each stock/clock. They include fully observed nonmatches in the denominator. Already-at-price, no-state, quality-truncated and request-end-truncated cases are separate in the source table. The supported-pair-only rates are also available in <a href="findings.json">findings.json</a>.</p>
<p>Read matched latency together with matching probability: a small conditional median does not mean every venue caught up within the horizon. Receive timestamps include feed transport; exchange timestamps compare different clocks. Neither is a pure causal propagation-time measurement.</p>
<h2>Calm versus volatile periods</h2>{regime_note}<p>When supported, each comparison pairs calm and volatile daily medians on the same twenty dates. A positive effect means the volatile conditional median was higher. The displayed effect is the median across supported pair effects. Intervals resample whole five-session date blocks and are <strong>exploratory marginal intervals</strong>; the interval counts below are not adjusted for many comparisons.</p>
{regime_html}
<p class="small">Volatility uses a complete trailing five-minute window and a retrospective within-date/symbol quantile. Warm-up and unsupported seconds remain unknown. This is a descriptive regime split, not a forecasting signal available in advance.</p>
{volatility_html}
<h2>What data are covered?</h2>{scope_html}
<p>The primary quote-age convention is {protocol['max_quote_age_ms']:g} ms. Matching requires valid two-sided quotes, completed events, conservative timestamp ties, and uninterrupted observation in both feeds over the entire matching horizon. The existing coarse MBO model grid is not used as a substitute for native bid/ask event times.</p>
<h2>Quote-age sensitivity</h2>{sensitivity_html}
<h2>Validation</h2><p>Both science audits passed. The all-source audit verifies source identities, request grids, count conservation, restricted waiting times and independently reconstructed inference. The separate raw audit independently rebuilds selected-date quote states and complete first-passage summaries.</p>{audit_html}
<p>{regression_text}</p><p><a href="result_audit.json">All-source/result audit</a> · <a href="raw_audit.json">Independent selected-date raw audit</a> · <a href="raw_audit_cases.csv">Raw matching cases</a> · <a href="regression_test_summary.json">Regression test summary</a> · <a href="regression_tests.xml">Full test report</a></p>
<h2>Figures</h2>{figure_html}
<p class="small">Event-study shares are signed ratios to each leader midpoint change. They are not bounded between zero and one: small leader denominators and wide local quotes can produce large positive or negative ratios. These curves do not establish a literal fraction of causal price absorption.</p>
<h2>Tables and reproducibility</h2><p><a href="leadership_share.csv">Counting-style leadership share</a> measures outgoing matched events divided by outgoing plus incoming matched events. It is not Hasbrouck information share or causal price discovery, and venue shares need not sum to one.</p>
<p>Primary full-support latency matrices: {matrix_links}.</p>
<p><a href="index.html">Original figure index</a> · <a href="findings.json">Machine-readable findings</a> · <a href="daily_latency.csv">Daily latency, censoring and restricted waits</a> · <a href="daily_latency_by_side.csv">Bid/ask sensitivity</a> · <a href="inference.csv">Date estimates and intervals</a> · <a href="quote_quality.csv">Quote-quality counts</a> · <a href="leadlag_summary.csv">Lead-lag diagnostics</a> · <a href="event_study.csv">Event studies</a> · <a href="latency_histograms.csv">Latency histograms</a> · <a href="provenance.json">Source/code provenance</a> · <a href="protocol.json">Protocol</a></p>
<h2>Limits on interpretation</h2><ul>{limitation_html}</ul>
<p class="small">Measurement ideas adapted independently from <a href="https://github.com/smar-01/qam-p113-latency/tree/62d6dcd8af9ecd7136096dbcce3eb40371987422">smar-01/qam-p113-latency</a>. All results here are generated from ColonyLab's saved data.</p>
</body></html>'''
    (out / "findings.html").write_text(body, encoding="utf-8")
    return findings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("out/latency"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = build_readout(args.out_dir, root=args.root)
    print(json.dumps({"status": result["status"], "primary_latency": result["primary_latency_by_clock_symbol"],
                      "primary_regimes": result["primary_regime_by_clock_symbol"]}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
