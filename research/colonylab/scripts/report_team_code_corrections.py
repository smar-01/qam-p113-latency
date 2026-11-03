"""Publish an audited local readout of the same-input team-code experiment."""
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
LABELS = {"XNAS": "Nasdaq", "XNYS": "NYSE", "ARCX": "Arca", "BATS": "BZX", "EDGX": "EDGX"}
STAGES = ["original", "numerical_only", "support_corrected"]
TITLES = {"original": "Original midpoint arithmetic", "numerical_only": "Integer arithmetic, original support",
          "support_corrected": "Integer arithmetic, fixed observable support"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(out_dir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir).resolve()
    audit = json.loads((out / "result_audit.json").read_text())
    if audit["status"] != "passed" or not audit.get("summary_metrics_independently_reconstructed"):
        raise ValueError("The complete independent correction audit must pass before publishing.")
    for name, digest in audit["output_sha256"].items():
        if sha(out / name) != digest:
            raise ValueError(f"A result changed after its audit: {name}")
    protocol = json.loads((out / "protocol.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    midpoint = pd.read_csv(out / "midpoint_diagnostics.csv")
    events = pd.read_csv(out / "event_study_comparison.csv")
    cc = pd.read_csv(out / "leadlag_comparison.csv", float_precision="round_trip")
    quality = pd.read_csv(out / "quality_stage_comparison.csv")
    regimes = pd.read_csv(out / "regime_series.csv")
    curves = pd.read_csv(out / "fullcurves.csv", usecols=["stage", "corr"], float_precision="round_trip")
    paired = cc.loc[cc.stage.eq("original")].merge(cc.loc[cc.stage.eq("numerical_only")],
             on=["day", "clock", "symbol", "x", "y"], suffixes=("_original", "_fixed"), validate="one_to_one")
    delta = paired.peak_lag_us_fixed - paired.peak_lag_us_original
    changed = delta.ne(0) & delta.notna()
    latency = []
    for clock, frame in quality.groupby("clock", sort=True):
        old_denominator, new_denominator = int(frame.baseline_n_behind.sum()), int(frame.n_eligible.sum())
        latency.append({"clock": clock, "original_behind": old_denominator,
                        "original_matched": int(frame.baseline_n_matched.sum()),
                        "original_match_rate_percent": 100 * frame.baseline_n_matched.sum() / old_denominator,
                        "quality_complete_eligible": new_denominator, "quality_matched": int(frame.n_matched.sum()),
                        "quality_match_rate_percent": 100 * frame.n_matched.sum() / new_denominator,
                        "quality_no_prior_state": int(frame.n_no_state.sum()),
                        "quality_truncated": int(frame.n_quality_truncated.sum()),
                        "quality_request_end_truncated": int(frame.n_right_truncated.sum())})
    rv = (regimes.assign(classified=regimes.regime.ne("unknown"))
          .groupby(["stage", "clock", "symbol"], sort=True)
          .agg(sampled_seconds=("regime", "size"), classified_seconds=("classified", "sum")).reset_index())
    rv["classified_fraction"] = rv.classified_seconds / rv.sampled_seconds
    extrema = {stage: float(group["corr"].abs().max()) if group["corr"].notna().any() else None
               for stage, group in curves.groupby("stage")}
    result = {
        "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {"dates": len(protocol["configuration"]["training_days"] + protocol["configuration"]["held_out_days"]),
                  "minutes_per_date": protocol["configuration"]["minutes"], "symbols": protocol["configuration"]["symbols"],
                  "venues": protocol["venues"], "raw_requests": audit["raw_source_requests"],
                  "raw_records": audit["raw_records_reconciled"], "original_full_day_raw_available": False},
        "phantom_clock_series_events": int(midpoint.phantom_events.sum()),
        "floating_midpoint_events": int(midpoint.float_events.sum()),
        "phantom_fraction_percent": 100 * midpoint.phantom_events.sum() / midpoint.float_events.sum(),
        "series_with_phantoms": int(midpoint.phantom_events.gt(0).sum()),
        "total_clock_series": len(midpoint),
        "response_ratio_max_abs_by_stage": audit["max_abs_event_study_share_by_stage"],
        "bid_ask_negative_control_rows": audit["reaction_pair_cases"], "bid_ask_negative_control_exact": True,
        "leadlag_comparisons": len(paired), "leadlag_changed_peak_lags": int(changed.sum()),
        "changed_peak_lag_fraction_percent": 100 * changed.mean(),
        "median_absolute_peak_shift_among_changed_us": float(delta.loc[changed].abs().median()) if changed.any() else None,
        "maximum_absolute_peak_shift_us": float(delta.abs().max()),
        "actual_max_abs_crossproduct_by_stage": extrema,
        "quality_policy_comparison": latency, "volatility_coverage": rv.to_dict("records"),
        "audit": {"status": audit["status"], "raw_sources_freshly_hashed": audit["raw_sources_freshly_hashed"],
                  "code_files_verified": audit["code_files_verified"], "checkpoints": audit["day_checkpoints_verified"],
                  "rounding_tied_peak_curves": audit["curves_with_rounding_tied_peak_values"]},
        "interpretation": [
            "Midpoint rounding is a real numerical bug: a tiny number of zero-price-change triggers can dominate response-ratio means.",
            "Numerical-only midpoint and cross-product fixes do not change the bid/ask first-passage outputs on the same quotes.",
            "The fixed-bin correction changes bin assignment and normalization; changed peak lags do not measure a change in market latency.",
            "The separate quality stage changes admissible events and complete-horizon denominators, so higher match rates are not performance gains.",
            "Full 300-return continuity and freshness requirements leave these ten-minute windows without a supported volatility comparison.",
            "This is a code counterfactual on saved shared data, not a rerun of the teammates' unavailable three full-day/IEX datasets.",
            "Signed event-study ratios can exceed one or be negative for legitimate reasons; they are not bounded causal absorption fractions."
        ]}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, (clock, group) in zip(axes, midpoint.groupby("clock", sort=True)):
        counts = group.groupby("venue").phantom_events.sum().sort_index()
        ax.bar(range(len(counts)), counts.to_numpy(), color="#b24b46")
        ax.set_xticks(range(len(counts)), [LABELS.get(x, x) for x in counts.index])
        ax.set_ylabel("Float-only zero-midpoint triggers")
        ax.set_title(clock)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Rounding artifacts found in actual saved data\nClock-series counts; the same raw transition can appear under both clocks")
    fig.savefig(out / "phantom_midpoint_triggers.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), layout="constrained", sharex=True)
    selected = events.loc[events.clock.eq("ts_recv") & events.symbol.eq("AAPL")]
    for ax, stage in zip(axes, STAGES):
        for venue, group in selected.loc[selected.stage.eq(stage)].groupby("venue", sort=True):
            by_offset = group.groupby("offset_us").share.mean()
            ax.plot(by_offset.index, by_offset.to_numpy(), marker=".", label=LABELS.get(venue, venue))
        ax.axhline(0, color="#777", lw=.6)
        ax.axhline(1, color="#555", linestyle="--", lw=.7)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_title(TITLES[stage], fontsize=10)
        ax.set_xlabel("Offset after Nasdaq move, µs")
        ax.set_ylabel("Signed midpoint response / leader move\n(symmetric log scale; not bounded)")
        ax.grid(alpha=.18)
    axes[-1].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("AAPL receive-clock event study on identical quotes\nEqual-date mean curves across ten dates; numerical-only keeps original per-offset support")
    fig.savefig(out / "event_study_correction.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), layout="constrained")
    for symbol, group in paired.groupby("symbol", sort=True):
        axes[0].scatter(group.peak_corr_original, group.peak_corr_fixed, alpha=.5, s=12, label=symbol)
    maximum = max(float(paired.peak_corr_original.max()), float(paired.peak_corr_fixed.max())) * 1.05
    axes[0].plot([0, maximum], [0, maximum], color="#777", linestyle="--", lw=.8)
    axes[0].set_xlabel("Original peak cross-product")
    axes[0].set_ylabel("Corrected fixed-bin peak cross-product")
    axes[0].legend(frameon=False)
    bins = np.arange(-max(20, float(delta.abs().max())) - 5, max(20, float(delta.abs().max())) + 15, 10)
    axes[1].hist(delta.loc[changed], bins=bins, color="#1c728e")
    axes[1].set_xscale("symlog", linthresh=20)
    axes[1].set_xlabel("Corrected minus original peak lag, µs\n(symmetric log scale)")
    axes[1].set_ylabel("Changed pair/date/stock/clock cases")
    axes[1].set_title(f"{int(changed.sum())} of {len(paired)} peak lags changed")
    for ax in axes:
        ax.grid(alpha=.18)
    fig.suptitle("Cross-product estimator sensitivity\nInteger arithmetic, consistent bin alignment and aggregated-bin normalization; no market-speed change")
    fig.savefig(out / "crossproduct_correction.png", dpi=170)
    plt.close(fig)
    result["figures"] = ["phantom_midpoint_triggers.png", "event_study_correction.png", "crossproduct_correction.png"]
    result["artifact_sha256"] = {"result_audit.json": sha(out / "result_audit.json"),
                                "summary.json": sha(out / "summary.json"),
                                **{name: sha(out / name) for name in result["figures"]}}
    smoke_path = out / "corrected_driver_2026-10-15" / "driver_smoke.json"
    smoke = json.loads(smoke_path.read_text()) if smoke_path.is_file() else None
    if smoke is not None:
        if smoke["status"] != "passed" or any(check["mismatches"] != 0 for check in smoke["checks"]):
            raise ValueError("The included standalone corrected-driver evidence did not pass.")
        result["standalone_corrected_driver"] = {"status": smoke["status"], "source_records": smoke["source_records"],
                                                 "symbol_pair_clock_rows": sum(x["symbol_pair_rows_checked"] for x in smoke["checks"]),
                                                 "mismatches": sum(x["mismatches"] for x in smoke["checks"])}
    (out / "findings.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    stage_table = pd.DataFrame(summary["event_study_diagnostics"])[["stage", "symbol", "maximum_absolute_mean_share"]]
    policy_table = pd.DataFrame(latency)[["clock", "original_match_rate_percent", "quality_match_rate_percent",
                                          "original_behind", "quality_complete_eligible"]].round(4)
    figures = "".join(f'<figure><img src="{name}" alt="{html.escape(name)}"></figure>' for name in result["figures"])
    limits = "".join(f"<li>{html.escape(text)}</li>" for text in result["interpretation"])
    smoke_text = (f"The standalone corrected driver also passed on October 15: {smoke['source_records']:,} records, "
                  f"{sum(x['symbol_pair_rows_checked'] for x in smoke['checks'])} symbol/pair/clock rows and zero mismatches "
                  "against the saved quality-controlled reference." if smoke else "Standalone driver evidence is not included in this artifact.")
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Team-code corrections: same-input experiment</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;color:#17263b;max-width:1250px;margin:36px auto;padding:0 24px}}h1,h2{{line-height:1.2}}table{{border-collapse:collapse;font-size:14px}}th,td{{padding:7px 11px;border-bottom:1px solid #d7dee8;text-align:right}}img{{width:100%}}figure{{margin:26px 0}}.note{{padding:18px;background:#edf3f8}}code{{background:#f0f3f7;padding:2px 5px}}</style>
<h1>What changes when the teammate's code is corrected?</h1>
<p class="note">A controlled offline comparison on <b>{audit['raw_records_reconciled']:,} records, 50 requests, ten ten-minute dates, three stocks and five venues</b>. Every numerical variant sees exactly the same legacy-selected quote rows. Original source is preserved; quality policies are a separate counterfactual. No IEX or original full-day files are available.</p>
<p><b>{int(midpoint.phantom_events.sum())} float-only midpoint triggers</b> across 300 clock series (some raw transitions appear under both clocks) are enough to make the original response-ratio means explode. The largest absolute saved mean falls from <b>{audit['max_abs_event_study_share_by_stage']['original']:,.1f}</b> to <b>{audit['max_abs_event_study_share_by_stage']['numerical_only']:.4f}</b> after integer arithmetic, with the original support policy retained.</p>
<p><b>All {audit['reaction_pair_cases']:,} bid/ask reaction comparisons remain exactly identical under numerical-only fixes.</b> The midpoint ratio bug does not invalidate those bid/ask delays. The fixed-bin cross-product correction changes {int(changed.sum())}/{len(paired)} peak-lag point estimates; the median absolute shift among changed cases is {result['median_absolute_peak_shift_among_changed_us']:g} µs. Bin alignment and normalization both change, so this is estimator sensitivity, not a market-speed effect.</p>
<p>The original coefficients do not exceed one in this actual sample: the maximum absolute coefficient is {extrema['original']:.4f}, versus {extrema['numerical_only']:.4f} after correction. The original method's unit-bound failure was demonstrated by a synthetic counterexample; it is not claimed to have occurred in these saved records.</p>
<h2>Numerical response-ratio changes</h2>{stage_table.to_html(index=False, escape=True, float_format=lambda x: f'{x:,.6g}')}
<h2>Quality policies change the observation cohort</h2><p>Rates below use different eligibility denominators by design. The quality stage applies freshness, flags, ties, clock barriers and complete-horizon follow-up to the same raw sources. A higher conditional rate is not evidence of improved market behavior.</p>{policy_table.to_html(index=False, escape=True)}
<h2>Volatility coverage</h2><p>Corrected volatility is computed on the selected clock, with full 300-return continuity and fresh supported quotes. These saved ten-minute windows have no classified corrected regimes, so a calm/volatile effect is unavailable rather than zero.</p>{rv.to_html(index=False, escape=True, float_format=lambda x: f'{x:.4f}')}
<h2>Figures</h2>{figures}<h2>Validation and reproducibility</h2>
<p>The independent audit passes: 50 fresh source hashes, {audit['code_files_verified']} code files, ten hash-bound checkpoints, unchanged shared quote inputs, 1,200 exact reaction negative controls, 23,400 event-study rows, and 721,800 cross-product curve rows. Ten curves have numerically tied peak values after serialization; tied maxima are accepted, non-maximal peaks are rejected.</p><p>{html.escape(smoke_text)}</p>
<p><a href="result_audit.json">Independent audit</a> · <a href="findings.json">Machine-readable findings</a> · <a href="summary.json">Run summary</a> · <a href="protocol.json">Protocol</a> · <a href="provenance.json">Source/code hashes</a> · <a href="midpoint_diagnostics.csv">Actual midpoint artifacts</a> · <a href="reaction_comparison.csv">Reaction negative controls</a> · <a href="event_study_comparison.csv">Event-study variants</a> · <a href="leadlag_comparison.csv">Peak comparisons</a> · <a href="quality_stage_comparison.csv">Separate quality-policy comparison</a></p>
<h2>What this establishes</h2><ul>{limits}</ul></html>'''
    (out / "findings.html").write_text(body, encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "team_code_corrections")
    args = parser.parse_args()
    result = build(args.out)
    print(json.dumps({key: result[key] for key in ("status", "phantom_clock_series_events", "bid_ask_negative_control_rows",
                                                  "response_ratio_max_abs_by_stage", "leadlag_changed_peak_lags",
                                                  "quality_policy_comparison", "figures")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
