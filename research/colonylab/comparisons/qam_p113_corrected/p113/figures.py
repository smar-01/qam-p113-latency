"""Figures: latency heatmap, latency distributions, lead-lag curves, regime split, event study."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def heatmap(matrix: pd.DataFrame, path: Path, title: str = "median reaction latency (µs)"):
    n = max(len(matrix.columns), 1)
    fig, ax = plt.subplots(figsize=(max(6.0, 2 + 0.7 * n), max(5.0, 1.5 + 0.55 * n)))
    vals = matrix.to_numpy(dtype=float)
    im = ax.imshow(vals, cmap="viridis_r")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xlabel("follower (reacts)")
    ax.set_ylabel("leader (moves first)")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                light = im.norm(vals[i, j]) < 0.5  # viridis_r: low values are light
                ax.text(j, i, f"{vals[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="k" if light else "w")
    fig.colorbar(im, ax=ax, label="µs")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def distributions(lat: pd.DataFrame, leader: str, path: Path):
    m = lat[(lat["leader"] == leader) & (lat["status"] == "matched")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not m.empty:
        lo = max(m["latency_ns"].min() / 1e3, 0.1)
        bins = np.logspace(np.log10(lo), np.log10(m["latency_ns"].max() / 1e3 + 1), 80)
        for f, g in m.groupby("follower"):
            ax.hist(g["latency_ns"] / 1e3, bins=bins, histtype="step", lw=1.4,
                    label=f"{f} (n={len(g)}, med {g['latency_ns'].median() / 1e3:.0f} µs)")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
    ax.set_xlabel("time for follower to match the leader's new price (µs, log)")
    ax.set_ylabel("events")
    ax.set_title(f"reaction latency after a {leader} quote change")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def leadlag(curves: pd.DataFrame, x: str, path: Path):
    c = curves[(curves["x"] == x) | (curves["y"] == x)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (a, b, sym), g in c.groupby(["x", "y", "symbol"]):
        # orient every curve as "x leads other" for positive lag
        other, sign = (b, 1) if a == x else (a, -1)
        ax.plot(sign * g["lag_us"], g["corr"], lw=1.2, label=f"{x} vs {other} ({sym})")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel(f"lag (µs); positive = other venue moves after {x}")
    ax.set_ylabel("normalized binned-increment cross-product")
    ax.set_title(f"descriptive lead-lag vs {x}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def regime_bars(summary_by_regime: pd.DataFrame, leader: str, path: Path):
    s = summary_by_regime[(summary_by_regime["leader"] == leader) &
                          summary_by_regime["regime"].isin(["calm", "volatile"])]
    p = s.pivot(index="follower", columns="regime", values="median_us")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not p.empty:
        p.plot.bar(ax=ax)
    ax.set_ylabel("median reaction latency (µs)")
    ax.set_title(f"after a {leader} quote change: calm vs volatile")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def event_study(es: pd.DataFrame, leader: str, symbol: str, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for v, g in es.groupby("venue"):
        ax.plot(g["offset_us"], g["share"], marker=".", lw=1.2, label=v, zorder=3 if v == leader else 2)
    ax.axhline(1, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel(f"time since the {leader} mid change (µs)")
    ax.set_ylabel("signed midpoint response / leader midpoint change")
    ax.set_title(f"midpoint response after {leader} moves, {symbol}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
