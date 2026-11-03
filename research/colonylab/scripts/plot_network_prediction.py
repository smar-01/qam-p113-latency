"""Plot saved held-out prediction scores without network calls or model refits."""
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
VENUES = {"BATS.PITCH": "Cboe BZX", "XNAS.ITCH": "Nasdaq", "XNYS.PILLAR": "NYSE",
          "ARCX.PILLAR": "NYSE Arca", "EDGX.PITCH": "Cboe EDGX"}


def date_label(days: list[str]) -> str:
    return days[0] if len(days) == 1 else f"{min(days)} to {max(days)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out/network_pilot")
    args = parser.parse_args()
    summary = json.loads((args.out_dir / "prediction_summary.json").read_text())
    tasks = pd.DataFrame(summary["tasks"])
    tasks = tasks.loc[tasks.status == "ok"].sort_values(["symbol", "dataset"])
    if tasks.empty:
        raise ValueError("No fitted task scores are available to plot.")
    values = tasks.delta_log_loss.to_numpy()
    labels = [f"{row.symbol} / {VENUES.get(row.dataset, row.dataset)}" for row in tasks.itertuples()]
    fig, ax = plt.subplots(figsize=(10, max(6.4, len(tasks) * .38 + 2.5)))
    fig.subplots_adjust(left=.23, right=.94, top=.83, bottom=.23)
    y = np.arange(len(tasks))
    ax.barh(y, values, color=np.where(values >= 0, "#236c98", "#b95d46"))
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.axvline(0, color="#444444", linewidth=1)
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    ax.set_xlabel("Baseline log loss minus network log loss (positive favors network)")
    span = max(np.max(np.abs(values)), .001)
    ax.set_xlim(min(-.0015, values.min() - span * .12), values.max() + span * .34)
    for i, value in enumerate(values):
        ax.text(value + span * .025 if value >= 0 else value - span * .025, i,
                f"{value:+.5f}", ha="left" if value >= 0 else "right", va="center", fontsize=9)
    pooled = summary["pooled_scores"]
    relative = 100 * pooled["delta_log_loss"] / pooled["baseline"]["log_loss"]
    held_out = summary.get("held_out_days", [summary["held_out_day"]])
    direction = "improves" if relative >= 0 else "worsens"
    fig.suptitle("Cross-venue prediction on held-out dates", fontsize=15, y=.96)
    fig.text(.23, .87, f"{date_label(held_out)} | {summary['pooled_test_bins']:,} bins | pooled log loss {direction} {abs(relative):.2f}%",
             fontsize=10)
    fig.text(.05, .08,
             f"Train: {date_label(summary['training_days'])}; test: {len(held_out)} held-out date(s). Next-10-ms quote shocks.\n"
             "Baseline: own history + past market-proxy activity + elapsed time. Network adds other-venue history.\n"
             "Dependent bins; no significance, causal effect or trading-profit claim. MBO and maker/RPI fields are not predictors.",
             fontsize=9, linespacing=1.5)
    fig.savefig(args.out_dir / "prediction_comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
