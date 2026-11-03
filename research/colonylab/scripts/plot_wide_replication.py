"""Plot fixed held-out dates from saved network and prediction summaries."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out/wide_network")
    args = parser.parse_args()
    prediction = json.loads((args.out_dir / "prediction_summary.json").read_text())
    days = prediction["held_out_days"]
    by_day = {row["day"]: row for row in prediction["per_day_scores"]}
    gain = np.array([100 * (by_day[day]["baseline"]["log_loss"] - by_day[day]["network"]["log_loss"])
                     / by_day[day]["baseline"]["log_loss"] for day in days])
    edges = pd.read_csv(args.out_dir / "edge_response.csv")
    held_edges = edges.loc[(edges.split == "held_out") & (edges.horizon_ms == 100)]
    edge = held_edges.groupby("day").agg(
        matched=("matched_count", "sum"), observed=("observed_hits", "sum"),
        placebo=("placebo_mean_hits", "sum"))
    excess = np.array([100 * (edge.loc[day, "observed"] - edge.loc[day, "placebo"])
                       / edge.loc[day, "matched"] if edge.loc[day, "matched"] else np.nan for day in days])
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(top=.88, bottom=.18, hspace=.35, left=.12, right=.97)
    x = np.arange(len(days))
    for ax, values in zip(axes, (gain, excess)):
        ax.bar(x, values, color=np.where(values >= 0, "#236c98", "#b95d46"), width=.6)
        ax.axhline(0, color="#444444", linewidth=1)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        scale = max(float(np.nanmax(np.abs(values))), .1)
        ax.set_ylim(min(float(np.nanmin(values)) - scale * .35, -scale * .2),
                    max(float(np.nanmax(values)) + scale * .35, scale * .2))
        for i, value in enumerate(values):
            if np.isfinite(value):
                ax.annotate(f"{value:+.2f}", (i, value), xytext=(0, 5 if value >= 0 else -5),
                            textcoords="offset points", va="bottom" if value >= 0 else "top", ha="center")
    axes[0].set_ylabel("Relative log-loss improvement (%)")
    axes[0].set_title("Does other-venue history improve next-10-ms shock prediction?", fontsize=12)
    axes[1].set_ylabel("Observed minus placebo (pp)")
    axes[1].set_title("100-ms same-direction response, with the original event filters", fontsize=12)
    axes[1].set_xticks(x, days)
    axes[1].set_xlabel("Held-out date (each uses the same frozen training model)")
    venue_count = len(set(held_edges.source) | set(held_edges.target))
    stock_support = held_edges.groupby("symbol").matched_count.sum()
    support = "; ".join(f"{symbol} {100 * count / stock_support.sum():.1f}%" for symbol, count in stock_support.items())
    fig.suptitle(f"Replication across {venue_count} venues and {len(days)} held-out dates", fontsize=16, y=.96)
    fig.text(.06, .045,
             "Positive values favor added venue history (top) or exceed shifted timing controls (bottom).\n"
             "Different tasks and events share market information; bars are descriptive, with no causal or significance claim.\n"
             "The event study excludes the first 2 ms and separates shocks by 100 ms; negative excess does not imply inhibition.\n"
             f"100 ms matched event support: {support}.",
             fontsize=9, linespacing=1.5)
    fig.savefig(args.out_dir / "replication_by_day.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
