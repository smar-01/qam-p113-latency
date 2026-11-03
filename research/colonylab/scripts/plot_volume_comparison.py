"""Render saved paired candle-model scores and the realized volume clock."""
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
VENUES = {"ARCX.PILLAR": "Arca", "BATS.PITCH": "BZX", "EDGX.PITCH": "EDGX",
          "XNAS.ITCH": "Nasdaq", "XNYS.PILLAR": "NYSE"}


def relative_gain(record):
    return 100 * record["delta_time_vs_volume_log_loss"] / record["time"]["log_loss"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out/volume_comparison")
    args = parser.parse_args()
    result = json.loads((args.out_dir / "volume_comparison_summary.json").read_text())
    tasks = sorted((r for r in result["tasks"] if r["status"] == "ok"), key=lambda r: (r["symbol"], r["dataset"]))
    values = np.array([relative_gain(r) for r in tasks])
    fig, ax = plt.subplots(figsize=(10, max(6.4, len(tasks) * .38 + 2.5)))
    fig.subplots_adjust(left=.22, right=.95, top=.83, bottom=.22)
    y = np.arange(len(tasks))
    ax.barh(y, values, color=np.where(values >= 0, "#236c98", "#b95d46"))
    ax.set_yticks(y, [f"{r['symbol']} / {VENUES.get(r['dataset'], r['dataset'])}" for r in tasks])
    ax.invert_yaxis()
    ax.axvline(0, color="#444444", linewidth=1)
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    span = max(float(np.max(np.abs(values))), .05)
    ax.set_xlim(min(float(values.min() - span * .25), -span * .25), max(float(values.max() + span * .3), span * .3))
    for i, value in enumerate(values):
        ax.text(value + span * .025 if value >= 0 else value - span * .025, i, f"{value:+.2f}%",
                ha="left" if value >= 0 else "right", va="center", fontsize=9)
    ax.set_xlabel("Relative log-loss reduction versus time candles (positive favors volume)")
    pooled = result["pooled"]
    gain = relative_gain(pooled)
    reference_gain = 100 * pooled["delta_reference_vs_volume_log_loss"] / pooled["reference"]["log_loss"]
    fig.suptitle("Volume candles versus one-second candles", fontsize=16, y=.96)
    fig.text(.22, .87, f"{pooled['test_rows']:,} identical test bins | volume log-loss benefit: {gain:+.2f}% vs time; {reference_gain:+.2f}% vs reference", fontsize=9.5)
    fig.text(.05, .075,
             "Both models add matched candle features to the same network reference. The next-10-ms outcome is unchanged.\n"
             "Share thresholds and all scalers/models are fitted on training dates only; whole prints may overshoot the threshold.\n"
             "Exploratory reuse of five test dates. Dependent observations; no causal, significance or trading-profit claim.",
             fontsize=9, linespacing=1.5)
    fig.savefig(args.out_dir / "volume_vs_time.png", dpi=160)
    plt.close(fig)

    days = result["per_day"]
    daily = np.array([relative_gain(r) for r in days])
    bars = pd.read_csv(args.out_dir / "volume_candles.csv")
    bars = bars.loc[bars.day.isin(result["held_out_days"]) & (bars.duration_ms > 0)]
    symbols = sorted(bars.symbol.unique())
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.subplots_adjust(left=.13, right=.96, top=.89, bottom=.19, hspace=.45)
    axes[0].bar(np.arange(len(days)), daily, color=np.where(daily >= 0, "#236c98", "#b95d46"), width=.6)
    axes[0].axhline(0, color="#444444", linewidth=1)
    axes[0].set_xticks(np.arange(len(days)), [r["day"] for r in days])
    axes[0].set_ylabel("Volume benefit vs time (%)")
    axes[0].set_title("Paired prediction loss by test date")
    axes[0].grid(axis="y", alpha=.2)
    axes[0].set_axisbelow(True)
    axes[1].boxplot([bars.loc[bars.symbol == symbol, "duration_ms"].to_numpy() / 1000 for symbol in symbols],
                    tick_labels=symbols, whis=(5, 95), showfliers=False)
    axes[1].axhline(1, color="#b95d46", linestyle="--", label="One-second time candle")
    axes[1].set(yscale="log", ylabel="Volume-candle duration (seconds)", title="The volume clock accelerates and slows with trading")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=.2)
    fig.suptitle("Prediction comparison and realized candle timing", fontsize=16, y=.96)
    fig.text(.05, .07,
             "Positive prediction bars favor volume candles. Boxplots pool the five test dates and venues; whiskers are 5th/95th percentiles.\n"
             "Volume durations include waiting since the previous close. Thresholds target one second of average training shares,\n"
             "with whole-print overshoot; they do not force equal realized durations or exactly equal shares.",
             fontsize=9, linespacing=1.5)
    fig.savefig(args.out_dir / "volume_clock_by_day.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
