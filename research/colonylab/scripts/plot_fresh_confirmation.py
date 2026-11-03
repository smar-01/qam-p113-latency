"""Plot the saved fresh-date paired network/volume prediction comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VENUES = {"XNAS.ITCH": "Nasdaq", "XNYS.PILLAR": "NYSE", "BATS.PITCH": "BZX",
          "ARCX.PILLAR": "Arca", "EDGX.PITCH": "EDGX", "MEMX.MEMOIR": "MEMX",
          "BATY.PITCH": "BYX", "EDGA.PITCH": "EDGA"}


def gains(frame: pd.DataFrame) -> np.ndarray:
    return (100 * (frame.reference_log_loss_sum - frame.volume_log_loss_sum)
            / frame.reference_log_loss_sum).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out/fresh_confirmation")
    args = parser.parse_args()
    result = json.loads((args.out_dir / "confirmation_summary.json").read_text())
    daily = pd.read_csv(args.out_dir / "daily_scores.csv").sort_values("day")
    value = gains(daily)
    pooled = result["pooled"]["log_loss"]["relative_gain_percent"]
    interval = result["five_session_block_bootstrap"]["log_loss"]["relative_gain_percent_ci95"]
    retention = 100 * result["coverage"]["paired_bins"] / result["coverage"]["candidate_bins"]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.subplots_adjust(left=.1, right=.97, bottom=.3, top=.73)
    ax.bar(np.arange(len(daily)), value, color=np.where(value >= 0, "#22749b", "#ba624c"))
    ax.axhline(0, color="#444444", linewidth=1)
    ax.axhline(pooled, color="#364d64", linestyle="--", linewidth=1.3,
               label=f"Pooled: {pooled:+.3f}%")
    ax.set_xticks(np.arange(len(daily)), daily.day.str.slice(5), rotation=45, ha="right")
    ax.set_xlabel("Fresh held-out session (2026)")
    ax.set_ylabel("Relative log-loss reduction (%)")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(loc="best", framealpha=.9)
    fig.suptitle("Volume-candle features added to the venue network", fontsize=17, y=.96)
    fig.text(.1, .805,
             f"8 venues · AAPL, MSFT, SPY · 20 fresh test sessions · {result['pooled']['rows']:,} paired forecasts\n"
             f"Volume gain: {pooled:+.3f}% | 95% five-session block interval: [{interval[0]:+.3f}%, {interval[1]:+.3f}%]\n"
             f"Coverage: {retention:.1f}% of candidate forecasts meet the original quote and candle freshness rules",
             fontsize=10.5, linespacing=1.5)
    fig.text(.06, .075,
             "Positive bars favor adding volume candles. Both fixed models use identical rows; calibration and fitting use 20 earlier sessions.\n"
             "Forecasts every 100 ms predict the next 10 ms, during 10:00–10:30 New York time. Intervals resample whole sessions in blocks;\n"
             "20 dates provide only about four non-overlapping week-length blocks. Results are predictive associations, not causal or profit estimates.",
             fontsize=9, linespacing=1.55)
    fig.savefig(args.out_dir / "confirmation_by_day.png", dpi=160)
    plt.close(fig)

    tasks = pd.read_csv(args.out_dir / "task_daily_scores.csv")
    pooled_tasks = tasks.groupby(["dataset", "symbol"], as_index=False).sum(numeric_only=True)
    pooled_tasks["gain_percent"] = gains(pooled_tasks)
    matrix = pooled_tasks.pivot(index="dataset", columns="symbol", values="gain_percent")
    matrix = matrix.reindex([v for v in VENUES if v in matrix.index]).reindex(columns=["AAPL", "MSFT", "SPY"])
    values = matrix.to_numpy()
    span = max(float(np.nanmax(np.abs(values))), .01)
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.subplots_adjust(left=.17, right=.87, bottom=.24, top=.8)
    picture = ax.imshow(values, cmap="RdBu", vmin=-span, vmax=span, aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)), [VENUES[v] for v in matrix.index])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i,j]:+.3f}%", ha="center", va="center",
                    color="white" if abs(values[i,j]) > .6 * span else "#222222", fontsize=11)
    fig.colorbar(picture, ax=ax, pad=.04, label="Log-loss reduction (%)")
    fig.suptitle("Volume's added value by stock and target venue", fontsize=16, y=.95)
    fig.text(.17, .85, f"Positive in {int((values > 0).sum())}/{values.size} tasks | {retention:.1f}% of candidate forecasts retained", fontsize=10)
    fig.text(.07, .08,
             "Each cell compares the existing eight-venue network with the same network plus volume candles.\n"
             "Every task uses the same frozen settings and common paired rows. Cells are dependent;\n"
             "uncertainty is computed across whole dates, retaining all stocks and venues together.",
             fontsize=9, linespacing=1.6)
    fig.savefig(args.out_dir / "confirmation_by_venue.png", dpi=160)
    plt.close(fig)

    coverage = pd.read_csv(args.out_dir / "coverage.csv")
    coverage = coverage.loc[coverage.split.eq("held_out")].groupby("day", as_index=False).sum(numeric_only=True)
    paired = 100 * coverage.common_bins / coverage.candidate_bins
    candle_excluded = 100 * (coverage.quote_eligible_bins - coverage.common_bins) / coverage.candidate_bins
    quote_excluded = 100 * (coverage.candidate_bins - coverage.quote_eligible_bins) / coverage.candidate_bins
    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.subplots_adjust(left=.09, right=.98, bottom=.3, top=.81)
    x = np.arange(len(coverage))
    ax.bar(x, paired, color="#22749b", label="Scored by both models")
    ax.bar(x, candle_excluded, bottom=paired, color="#c48452", label="Excluded by candle freshness")
    ax.bar(x, quote_excluded, bottom=paired + candle_excluded, color="#d7dce0", label="Excluded by quote coverage")
    ax.set_xticks(x, coverage.day.str.slice(5), rotation=45, ha="right")
    ax.set(ylim=(0, 100), ylabel="Candidate forecasts (%)", xlabel="Fresh held-out session (2026)")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.21), ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Coverage of the eight-venue confirmation", fontsize=16, y=.97)
    fig.text(.06, .065,
             "Freshness requirements select a jointly observable, active cohort. The reference and volume models use exactly the blue rows.\n"
             "The original quote-history rules and ten-second completed-candle age limit remain fixed across all dates and venues.\n"
             "Counts are task forecasts; multiple target venues can share a market timestamp and are not independent observations.",
             fontsize=9, linespacing=1.5)
    fig.savefig(args.out_dir / "confirmation_coverage.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
