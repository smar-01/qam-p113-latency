"""Render the separately audited, offline ADF/KPSS diagnostic readout.

Classifications are nominal exploratory diagnostics on regular price paths.
This script never transforms inputs or changes the original latency study.
"""
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
CLASSES = ("stationarity_compatible", "unit_root_compatible", "conflicting", "inconclusive", "not_tested")
LABELS = {"stationarity_compatible": "Stationarity compatible", "unit_root_compatible": "Unit-root compatible",
          "conflicting": "Conflicting rejections", "inconclusive": "Neither null rejected", "not_tested": "Not tested"}
SPECS = (("log_level", "c"), ("log_level", "ct"), ("log_return", "c"))
GENERATED = ("findings.html", "findings.json", "stationarity_classifications.png", "stationarity_coverage.png")
ADF_DOC = "https://www.statsmodels.org/v0.14.4/generated/statsmodels.tsa.stattools.adfuller.html"
KPSS_DOC = "https://www.statsmodels.org/v0.14.4/generated/statsmodels.tsa.stattools.kpss.html"


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value {value}: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def finite(value):
    return value is not None and isinstance(value, (int, float, np.number)) and np.isfinite(value)


def number(value, scale=1, digits=3):
    return f"{float(value) * scale:,.{digits}f}" if finite(value) else "Unavailable"


def text(value):
    return "" if value is None or (isinstance(value, (float, np.floating)) and pd.isna(value)) else str(value)


def json_cell(value):
    if value is None or (isinstance(value, (float, np.floating)) and pd.isna(value)):
        return ""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return value
    else:
        parsed = value
    if isinstance(parsed, list):
        return "; ".join(str(item) for item in parsed)
    return json.dumps(parsed, ensure_ascii=False) if isinstance(parsed, dict) else str(parsed)


def table(rows):
    if not rows:
        return '<p class="muted">No rows apply.</p>'
    return '<div class="scroll">' + pd.DataFrame(rows).to_html(index=False, escape=True, border=0,
        na_rep="Unavailable", float_format=lambda value: number(value)) + '</div>'


def specification(transformation, regression):
    label = "Log midpoint level" if transformation == "log_level" else "One-second log return"
    return label + ("; constant + trend (ct)" if regression == "ct" else "; constant (c)")


def pvalue(value, relation=None):
    if not finite(value):
        return "Not tested"
    qualifier = text(relation)
    if qualifier in ("", "=", "equal", "exact", "interpolated"):
        return f"{value:.6g}"
    return f"{qualifier} {value:.6g}"


def verified_inputs(out):
    required = ("results.json", "protocol.json", "requests.csv", "runs.csv", "tests.csv", "analysis_provenance.json")
    audit_path = out / "result_audit.json"
    audit = read_json(audit_path)
    require(audit.get("status") == "passed" and audit.get("passed", True) is True,
            "The separate stationarity audit must pass before rendering")
    provenance = read_json(out / "analysis_provenance.json")
    bindings = {}
    for manifest in (provenance.get("outputs", {}), audit.get("output_sha256", {})):
        require(isinstance(manifest, dict), "Invalid diagnostic output hash manifest")
        for name, record in manifest.items():
            digest = record.get("sha256") if isinstance(record, dict) else record
            require(isinstance(digest, str) and len(digest) == 64, f"Invalid artifact hash: {name}")
            path = (out / name).resolve()
            require(path.is_relative_to(out) and path.is_file() and sha(path) == digest,
                    f"Diagnostic output changed after provenance/audit: {name}")
            bindings[name] = digest
    require(all(name in bindings for name in required), "Not all report inputs are bound by provenance or audit")
    return audit, {name: bindings[name] for name in required}


def summarized_tests(requests, tests):
    # The same exact price path may serve several request identities. Within
    # a clock/age stratum, count that path once for each test specification.
    identities = requests[["series_id", "clock", "age_ms"]].drop_duplicates()
    joined = identities.merge(tests, on="series_id", how="left", validate="many_to_many")
    require(joined.classification.notna().all(), "A requested path has no diagnostic test rows")
    rows = []
    keys = ["clock", "age_ms", "transformation", "regression"]
    for key, group in joined.groupby(keys, sort=True, dropna=False):
        require(not group.duplicated("series_id").any(), "Duplicate diagnostic pair within a display stratum")
        counts = Counter(group.classification)
        rows.append({**dict(zip(keys, key)), "unique_series": len(group),
                     "tested": int(group.classification.ne("not_tested").sum()),
                     **{name: counts.get(name, 0) for name in CLASSES}})
    return rows


def coverage_summary(requests):
    rows = []
    for (clock, age), group in requests.groupby(["clock", "age_ms"], sort=True):
        points = group.longest_points.to_numpy(float)
        rows.append({"clock": clock, "age_ms": age, "requests": len(group),
            "requests_300_points": int(np.sum(points >= 300)), "requests_600_points": int(np.sum(points >= 600)),
            "median_longest_points": float(np.median(points)),
            "median_valid_fraction": float(group.valid_fraction_window.median()),
            "median_longest_fraction": float(group.fraction_window.median()),
            "unique_series": int(group.series_id.nunique())})
    return rows


def global_tests_summary(tests, alpha):
    """Count every exact series/specification test once across all requests."""
    rows = []
    for transformation, regression in SPECS:
        group = tests.loc[tests.transformation.eq(transformation) & tests.regression.eq(regression)]
        counts = Counter(group.classification)
        rows.append({"transformation": transformation, "regression": regression,
            "unique_series": int(group.series_id.nunique()),
            "tested": int(group.classification.ne("not_tested").sum()),
            **{name: counts.get(name, 0) for name in CLASSES},
            "adf_nominal_rejections": int(group.adf_pvalue.lt(alpha).sum()),
            "adf_holm_rejections": int(group.adf_pvalue_holm.lt(alpha).sum())})
    return rows


def figures(out, groups, coverage):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    clocks = sorted({row["clock"] for row in groups})
    colors = {"stationarity_compatible": "#387d91", "unit_root_compatible": "#cb8e44", "conflicting": "#b85751",
              "inconclusive": "#c8b95c", "not_tested": "#cbd2d7"}
    fig, axes = plt.subplots(max(1, len(clocks)), 1, figsize=(12, max(4.4, 3.8 * len(clocks))),
                             constrained_layout=True, squeeze=False)
    for ax, clock in zip(axes[:, 0], clocks):
        selected = [row for row in groups if row["clock"] == clock]
        selected.sort(key=lambda row: (row["age_ms"], SPECS.index((row["transformation"], row["regression"]))))
        x = np.arange(len(selected))
        bottom = np.zeros(len(selected))
        for name in CLASSES:
            values = np.asarray([row[name] for row in selected])
            ax.bar(x, values, bottom=bottom, color=colors[name], label=LABELS[name])
            bottom += values
        ax.set_xticks(x, [f"{row['age_ms']:g} ms\n" + ("level" if row["transformation"] == "log_level" else "return")
                         + f" / {row['regression']}" for row in selected])
        ax.set(ylabel="Unique price paths in stratum", title=clock)
        if not selected:
            ax.text(.5, .5, "No diagnostic groups", transform=ax.transAxes, ha="center")
    if clocks:
        axes[0, 0].legend(ncol=3, fontsize=8, loc="upper right")
    else:
        axes[0, 0].text(.5, .5, "No diagnostic groups", transform=axes[0, 0].transAxes, ha="center")
    fig.suptitle("Nominal exploratory ADF/KPSS classifications\nUnique paths within each clock / quote-age / test specification; not independent event counts")
    fig.savefig(out / "stationarity_classifications.png", dpi=170)
    plt.close(fig)

    clocks = sorted({row["clock"] for row in coverage})
    fig, axes = plt.subplots(1, max(1, len(clocks)), figsize=(12, 4.9), constrained_layout=True, squeeze=False)
    for ax, clock in zip(axes[0], clocks):
        selected = sorted([row for row in coverage if row["clock"] == clock], key=lambda row: row["age_ms"])
        x = np.arange(len(selected))
        for offset, key, label, color in ((-.18, "requests_300_points", "Longest price run ≥300 points", "#387d91"),
                                           (.18, "requests_600_points", "Longest price run ≥600 points", "#cb8e44")):
            rates = [100 * row[key] / row["requests"] for row in selected]
            bars = ax.bar(x + offset, rates, .34, label=label, color=color)
            ax.bar_label(bars, labels=[f"{row[key]}/{row['requests']}" for row in selected], fontsize=8, padding=3)
        ax.set_xticks(x, [f"{row['age_ms']:g} ms" for row in selected])
        ax.set(ylim=(0, 115), xlabel="Maximum quote age", ylabel="Planned diagnostic-window requests (%)", title=clock)
        ax.legend(fontsize=8)
    if not clocks:
        axes[0, 0].text(.5, .5, "No coverage rows", transform=axes[0, 0].transAxes, ha="center")
    fig.suptitle("Strict contiguous support before stationarity testing\nRequest counts can share identical paths; 600-point flags are descriptive, not a stronger stationarity guarantee")
    fig.savefig(out / "stationarity_coverage.png", dpi=170)
    plt.close(fig)


def build(out):
    out = Path(out).resolve()
    audit, bindings = verified_inputs(out)
    results, protocol = read_json(out / "results.json"), read_json(out / "protocol.json")
    require(results.get("status") == "completed", "Stationarity results are not completed")
    require(results.get("protocol") == protocol, "Diagnostic protocol differs from results snapshot")
    requests, runs, tests = (pd.read_csv(out / name) for name in ("requests.csv", "runs.csv", "tests.csv"))
    required_requests = {"request_id", "series_id", "clock", "age_ms", "longest_points", "fraction_window", "valid_fraction_window"}
    required_tests = {"series_id", "transformation", "regression", "classification", "status", "skip_reason", "n_observations",
                      "adf_stat", "adf_pvalue", "adf_pvalue_holm", "adf_lags", "adf_maxlag", "adf_nobs",
                      "kpss_stat", "kpss_pvalue", "kpss_pvalue_relation", "kpss_lags", "warnings"}
    require(required_requests <= set(requests) and required_tests <= set(tests)
            and {"series_id", "n_price_points", "duration_seconds"} <= set(runs), "Diagnostic report schema is incomplete")
    require(not requests.request_id.duplicated().any() and not runs.series_id.duplicated().any()
            and not tests.duplicated(["series_id", "transformation", "regression"]).any(), "Duplicate report input identities")
    require(set(tests.classification) <= set(CLASSES), "Unknown diagnostic classification")
    require(set(tests[["transformation", "regression"]].itertuples(index=False, name=None)) <= set(SPECS),
            "Unplanned transformation/regression combination")
    counts = results["counts"]
    require(counts["diagnostic_windows"] == len(requests) and counts["unique_test_pairs"] == len(tests),
            "Result counters differ from request/test tables")
    tested = tests.classification.ne("not_tested")
    require(counts["tested_pairs"] == int(tested.sum()) and counts["skipped_pairs"] == int((~tested).sum()),
            "Result tested/skipped counts differ")
    groups, coverage = summarized_tests(requests, tests), coverage_summary(requests)
    figures(out, groups, coverage)
    alpha = float(protocol.get("alpha", protocol.get("significance_level", .05)))
    require(0 < alpha < 1, "Invalid diagnostic significance threshold")
    global_summary = global_tests_summary(tests, alpha)
    global_rows = [{"Specification": specification(row["transformation"], row["regression"]),
        "Unique paths": row["unique_series"], "Tested": row["tested"],
        **{LABELS[name]: row[name] for name in CLASSES},
        "ADF nominal rejections": row["adf_nominal_rejections"],
        "ADF Holm rejections": row["adf_holm_rejections"]} for row in global_summary]
    return_row = next(row for row in global_summary if row["transformation"] == "log_return" and row["regression"] == "c")
    level_rows = [row for row in global_summary if row["transformation"] == "log_level"]
    tested_paths = int(tests.loc[tested, "series_id"].nunique())
    if return_row["tested"]:
        more_often = all(row["tested"] and return_row["stationarity_compatible"] / return_row["tested"]
                         > row["stationarity_compatible"] / row["tested"] for row in level_rows)
        result_paragraph = ("Log returns fit the nominal stationarity pattern more often than log price levels on these selected segments. "
                            if more_often else "The nominal classifications differ across level and return specifications. ")
        result_paragraph += (f"Of {return_row['tested']:,} testable log-return paths, {return_row['stationarity_compatible']:,} were stationarity compatible "
                             f"and {return_row['conflicting']:,} produced conflicting rejections. "
                             f"Their ADF unit-root null was rejected after global Holm correction in {return_row['adf_holm_rejections']:,} cases. ")
    else:
        result_paragraph = "No log-return path supplied a tested stationarity pair; no comparison of level and return classifications is available. "
    primary_age = float(protocol.get("primary_age_ms", 250))
    primary_coverage = [row for row in coverage if row["age_ms"] == primary_age]
    if {row["clock"] for row in primary_coverage} == {"ts_recv", "ts_event"} and not any(row["requests_300_points"] for row in primary_coverage):
        result_paragraph += (f"Neither clock supplied a 300-point contiguous price run at the primary {primary_age:g}-ms freshness setting; "
                             "the tested evidence comes from longer-age sensitivities. ")
    result_paragraph += (f"The {tested_paths:,} testable paths share market and event contexts and are not {tested_paths:,} independent experiments. "
                         "No transformation is applied automatically, and the native level-based latency matcher remains unchanged.")
    good_adf = np.isfinite(tests.adf_pvalue.to_numpy(float))
    require(np.isfinite(tests.loc[good_adf, "adf_pvalue_holm"]).all(), "A computed ADF test lacks its Holm adjustment")
    family = tests.loc[good_adf]
    correction = {"unique_adf_tests_in_family": len(family), "alpha": alpha,
                  "nominal_adf_rejections": int(family.adf_pvalue.lt(alpha).sum()),
                  "holm_adf_rejections": int(family.adf_pvalue_holm.lt(alpha).sum()),
                  "scope": "ADF correction across unique computed series/transformation/regression tests only; no request-row replication and no adjusted joint ADF/KPSS classification."}
    summary_rows = []
    for row in groups:
        summary_rows.append({"Clock": row["clock"], "Max quote age (ms)": row["age_ms"],
            "Specification": specification(row["transformation"], row["regression"]), "Unique paths": row["unique_series"],
            "Tested": row["tested"], **{LABELS[name]: row[name] for name in CLASSES}})
    coverage_rows = [{"Clock": row["clock"], "Max quote age (ms)": row["age_ms"], "Window requests": row["requests"],
        "Longest price run ≥300 points": row["requests_300_points"], "Longest price run ≥600 points": row["requests_600_points"],
        "Median longest price-run points": number(row["median_longest_points"], digits=0),
        "Median valid grid coverage (%)": number(row["median_valid_fraction"], 100),
        "Median fraction in selected longest run (%)": number(row["median_longest_fraction"], 100)} for row in coverage]
    skip_counts = Counter(tests.loc[~tested, "skip_reason"].fillna("unspecified"))
    skip_rows = [{"Skip reason": reason, "Unique test pairs": count} for reason, count in sorted(skip_counts.items())]
    details = []
    for row in tests.sort_values(["series_id", "transformation", "regression"]).head(100).to_dict("records"):
        details.append({"Series ID": row["series_id"], "Specification": specification(row["transformation"], row["regression"]),
            "Input observations": number(row["n_observations"], digits=0), "Class (nominal)": LABELS[row["classification"]],
            "ADF statistic": number(row["adf_stat"]), "ADF p": pvalue(row["adf_pvalue"]),
            "ADF Holm p": pvalue(row["adf_pvalue_holm"]), "ADF lags / maximum":
                f"{number(row['adf_lags'], digits=0)} / {number(row['adf_maxlag'], digits=0)}",
            "ADF regression observations": number(row["adf_nobs"], digits=0),
            "KPSS statistic": number(row["kpss_stat"]), "KPSS p (relation preserved)": pvalue(row["kpss_pvalue"], row["kpss_pvalue_relation"]),
            "KPSS lags": number(row["kpss_lags"], digits=0), "Status / skip": text(row["status"]) + "; " + text(row["skip_reason"]),
            "Warnings": json_cell(row["warnings"])})
    relation_counts = Counter(text(value) or "not supplied" for value in tests.loc[tested, "kpss_pvalue_relation"])
    assumptions = [
        "The original latency/event study is unchanged; this diagnostic does not test irregular latency observations or automatically transform main-study inputs.",
        "Within each planned one-second price window, select the longest strict no-gap supported run using quality alone; ties use the earliest run. Invalid intervals are not interpolated or stitched together.",
        "Three hundred one-second observations are approximately five minutes (299 seconds between endpoints); 600 observations are approximately ten minutes. The 600-point flag is descriptive, not validation of long-span stationarity.",
        "Log-level c and log-return c are separate diagnostics. The log-level ct specification is a deterministic-trend sensitivity, paired with ct in both ADF and KPSS.",
        "Nominal joint classifications use opposite null hypotheses at the stated alpha; failing to reject is not proof of a null hypothesis.",
        "ADF Holm adjustment uses unique computed tests only. It does not convert the displayed nominal ADF/KPSS matrix into a jointly adjusted classification.",
        "KPSS p-values are table-bounded near 0.01 and 0.10; preserve bound relations and warnings rather than interpreting endpoints as precise probabilities.",
        "A large number of one-second samples does not replace a long time span. Event-related breaks and market microstructure can affect test behavior.",
        "The longest supported segment can represent only a fraction of a scheduled window; conclusions concern that selected segment, not the whole session.",
        "Requests, identical price paths and related event bundles are dependent; grouped classifications do not establish event-type causal effects or independent replication counts.",
        "Stationarity diagnostics alone do not select a model transformation. Differencing a price path can change the native first-passage price-target estimand.",
    ]
    limitations = list(dict.fromkeys([*assumptions, *results.get("limitations", [])]))
    findings = clean({"status": "completed", "audit_status": audit["status"], "counts": counts,
        "nominal_classification_summary": groups, "global_unique_test_summary": global_summary,
        "testable_unique_price_paths": tested_paths, "readout_conclusion": result_paragraph,
        "coverage_summary": coverage, "adf_holm": correction,
        "kpss_relation_counts": dict(relation_counts), "skip_reason_counts": dict(skip_counts),
        "automatic_transformation_applied": False, "main_latency_study_changed": False,
        "interpretation": "Exploratory segment-specific diagnostics; nominal classifications are not a transformation rule or causal event-class comparison.",
        "limitations": limitations, "figures": ["stationarity_classifications.png", "stationarity_coverage.png"]})
    (out / "findings.json").write_text(json.dumps(findings, indent=2, allow_nan=False), encoding="utf-8")
    matrix = [
        {"ADF rejects unit root": "Yes", "KPSS rejects stationarity": "No", "Nominal interpretation": "Stationarity compatible under the selected deterministic specification"},
        {"ADF rejects unit root": "No", "KPSS rejects stationarity": "Yes", "Nominal interpretation": "Unit-root / nonstationarity compatible; not proof that differencing is required"},
        {"ADF rejects unit root": "Yes", "KPSS rejects stationarity": "Yes", "Nominal interpretation": "Conflicting rejections: inspect specification, breaks, lags and sampling"},
        {"ADF rejects unit root": "No", "KPSS rejects stationarity": "No", "Nominal interpretation": "Inconclusive: neither null rejected"},
    ]
    settings = json.dumps(protocol, indent=2, ensure_ascii=False)
    limit_html = "".join(f"<li>{html.escape(item)}</li>" for item in limitations)
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>AAPL stationarity diagnostics</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;color:#23333f;max-width:1220px;margin:40px auto;padding:0 24px}}h1,h2{{line-height:1.2}}h2{{margin-top:38px}}table{{border-collapse:collapse;min-width:100%;font-size:12px;margin:15px 0}}th,td{{text-align:left;vertical-align:top;padding:9px;border-bottom:1px solid #d7dfe5}}th{{background:#edf2f5}}img{{width:100%;height:auto}}.scroll{{overflow-x:auto}}.note{{background:#edf2f5;padding:18px;border-left:4px solid #627d98}}.muted{{color:#586976}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}a{{color:#186b9b}}summary{{cursor:pointer;font-weight:600;margin:14px 0}}</style></head><body>
<h1>Do these short AAPL price segments look stationary?</h1>
<p>Separate offline diagnostic: {counts['diagnostic_windows']:,} planned window requests,
{counts['unique_series']:,} unique price paths and {counts['unique_test_pairs']:,} unique ADF/KPSS test pairs.
{counts['tested_pairs']:,} pairs were tested; {counts['skipped_pairs']:,} were skipped.</p>
<p class="note"><b>The original latency study is unchanged.</b> These tests examine regular one-second log midpoint levels
and log returns on strictly supported segments. They do not test irregular latency observations, apply automatic transformations,
or repair missing quote support in the event study.</p>
<p>{html.escape(result_paragraph)}</p>
<h2>Global unique-test results by specification</h2>
<p>This table counts each exact path/specification pair once across the full diagnostic. It avoids recounting a path
when the same data occur in multiple request, clock or quote-age strata.</p>{table(global_rows)}
<h2>What was selected and tested</h2>
<p>Each planned before/after window uses its longest strict no-gap supported price run. Selection uses quote quality and run length,
with the earliest run breaking a length tie; invalid gaps are neither interpolated nor stitched together.
The minimum of 300 one-second observations is only about five minutes. The separate 600-point flag is approximately ten minutes
and remains descriptive. Differenced log returns contain fewer observations than their underlying level path, and the ADF lag regression
uses fewer observations again.</p>
<p>Level tests with a constant (<code>c</code>) and return tests with a constant are reported separately.
Level tests with a constant plus linear trend (<code>ct</code>) are a sensitivity check. Match the deterministic specification
across ADF and KPSS; a trend-stationary alternative differs from stationarity around a fixed mean.</p>
<h2>Coverage and finite observation span</h2><img src="stationarity_coverage.png" alt="Planned diagnostic requests with strict longest price runs of at least 300 or 600 one-second points">
{table(coverage_rows)}
<p>Coverage request counts describe planned exposures and can share identical paths. Passing a price-run length threshold does not guarantee
that every transformation or lag regression can be tested. A short selected segment does not establish whole-window or whole-session stationarity.</p>
<h2>Nominal exploratory classifications</h2><p>The tests have opposite nulls: ADF tests for a unit root;
KPSS tests stationarity around a constant or deterministic trend. The following matrix uses nominal α={alpha:g}.</p>{table(matrix)}
<img src="stationarity_classifications.png" alt="Nominal ADF/KPSS classifications and not-tested counts by clock, maximum quote age and specification">
{table(summary_rows)}
<p>Each display stratum counts an exact price path once per specification. The same path can still appear in different clock/age strata;
adding these bars does not produce an independent-sample count. The classifications are diagnostic evidence, not a transformation prescription.</p>
<h2>ADF multiplicity and KPSS bounds</h2>{table([correction])}
<p>Holm correction is applied across {len(family):,} unique computed ADF tests, rather than duplicating tests for every request row.
The stacked classifications above remain nominal ADF/KPSS classifications; the ADF-only adjustment is reported separately.
KPSS interpolates a limited critical-value table. A boundary return near 0.01 or 0.10 is retained with its bound relation and warning,
not interpreted as an exact tail probability.</p>{table([{'KPSS p-value relation':key,'Computed unique test pairs':value} for key,value in sorted(relation_counts.items())])}
<h2>Skipped tests</h2>{table(skip_rows)}
<p>A skipped, constant, too-short or otherwise unsupported series has no stationarity conclusion. Warnings and selected lag counts
are retained in the detailed outputs.</p>
<details><summary>First 100 unique test pairs in stable identity order</summary><p>This preview is ordered by series identifier,
transformation and regression, not by p-value. <a href="tests.csv">The complete unique-test table</a> contains all pairs and critical values.</p>{table(details)}</details>
<h2>Limits on conclusions</h2><ul>{limit_html}</ul>
<p>The finite span matters even with many observations: see <a href="https://www.sciencedirect.com/science/article/pii/0165176585900588">Shiller and Perron on sampling frequency and power</a>.
Event-related trend/level breaks can affect conventional unit-root conclusions: see <a href="https://www.princeton.edu/~erp/ERParchives/archivepdfs/M338.pdf">Perron's structural-break analysis</a>.
High-frequency prices also contain microstructure noise: see <a href="https://www.princeton.edu/~yacine/sampling.pdf">Aït-Sahalia, Mykland and Zhang</a>.</p>
<h2>Inspect and reproduce</h2><p><a href="results.json">Complete diagnostic results</a> · <a href="protocol.json">Frozen diagnostic protocol</a> ·
<a href="requests.csv">Planned requests and run selection</a> · <a href="runs.csv">Unique selected price runs</a> ·
<a href="tests.csv">Unique tests, p-value bounds and warnings</a> · <a href="series.npz">Saved regular price paths</a> ·
<a href="analysis_provenance.json">Input/output provenance</a> · <a href="result_audit.json">Separate diagnostic audit</a></p>
<p>Official documentation: <a href="{ADF_DOC}">statsmodels ADF</a> · <a href="{KPSS_DOC}">statsmodels KPSS</a>.</p>
<details><summary>Exact frozen diagnostic settings</summary><pre>{html.escape(settings)}</pre></details>
<pre>python scripts/report_stationarity.py --out-dir "{html.escape(str(out))}"</pre></body></html>'''
    (out / "findings.html").write_text(document, encoding="utf-8")
    require(all(sha(out / name) == digest for name, digest in bindings.items()), "Diagnostic inputs changed while rendering")
    manifest = {"report_code_sha256": sha(Path(__file__)), "result_audit_sha256": sha(out / "result_audit.json"),
                "inputs": bindings, "artifacts": {name: sha(out / name) for name in GENERATED}}
    (out / "readout_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out/aapl_stationarity")
    args = parser.parse_args()
    result = build(args.out_dir)
    print(json.dumps({"status": result["status"], "audit_status": result["audit_status"],
                      "tested_pairs": result["counts"]["tested_pairs"], "skipped_pairs": result["counts"]["skipped_pairs"]}), flush=True)


if __name__ == "__main__":
    main()
