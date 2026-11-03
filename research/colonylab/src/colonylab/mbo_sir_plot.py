"""Standalone scientific figures from completed MBO/SIRS result tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE = "independent_mbo_hazard"
CANDIDATE = "sirs_network"
COMPARATORS = (REFERENCE, "logistic_network", "past_shifted_sirs")
MODEL_LABELS = {REFERENCE: "Independent MBO hazard", "logistic_network": "Logistic network",
                "past_shifted_sirs": "SIRS with sources 30 s in the past"}
VENUE_LABELS = {"XNAS.ITCH": "Nasdaq", "XNYS.PILLAR": "NYSE", "BATS.PITCH": "BZX",
                "ARCX.PILLAR": "NYSE Arca", "EDGX.PITCH": "EDGX", "MEMX.MEMOIR": "MEMX",
                "BATY.PITCH": "BYX", "EDGA.PITCH": "EDGA"}


def _paired_daily(daily: pd.DataFrame, reference: str, candidate: str = CANDIDATE,
                  *, expected_days=None) -> pd.DataFrame:
    """Validate exact paired date coverage and compute gains from loss totals."""
    required = {"day", "model", "rows", "log_loss_sum"}
    if not required.issubset(daily.columns):
        raise ValueError("Daily scores lack required model/date loss totals.")
    selected = daily.loc[daily.model.isin([reference, candidate])].copy()
    if selected.duplicated(["day", "model"]).any():
        raise ValueError("Duplicate model/date score rows.")
    records = []
    dates = sorted(expected_days if expected_days is not None else selected.day.unique())
    if not dates or set(selected.day) != set(dates):
        raise ValueError("Daily scores do not contain the exact intended dates.")
    for day in dates:
        scores = selected.loc[selected.day.eq(day)].set_index("model")
        if set(scores.index) != {reference, candidate}:
            raise ValueError(f"Missing paired model/date scores: {day}")
        r, c = scores.loc[reference], scores.loc[candidate]
        values = np.asarray([r.rows, c.rows, r.log_loss_sum, c.log_loss_sum], dtype=float)
        if (not np.isfinite(values).all() or r.rows <= 0 or r.rows != c.rows
                or r.rows != int(r.rows) or r.log_loss_sum <= 0 or c.log_loss_sum < 0):
            raise ValueError(f"Invalid or unequal paired score denominators/losses: {day}")
        records.append({"day": day, "rows": int(r.rows), "reference_log_loss_sum": float(r.log_loss_sum),
                        "candidate_log_loss_sum": float(c.log_loss_sum),
                        "gain_percent": 100 * (r.log_loss_sum - c.log_loss_sum) / r.log_loss_sum})
    return pd.DataFrame(records)


def _pooled_gain(paired: pd.DataFrame) -> float:
    return float(100 * (paired.reference_log_loss_sum.sum() - paired.candidate_log_loss_sum.sum())
                 / paired.reference_log_loss_sum.sum())


def _coverage_totals(coverage: pd.DataFrame, *, days, datasets) -> pd.DataFrame:
    needed = {"day", "dataset", "candidate_bins", "eligible_bins"}
    if not needed.issubset(coverage):
        raise ValueError("Coverage table lacks candidate/eligible counts.")
    selected = coverage.loc[coverage.day.isin(days)].copy()
    expected = {(day, dataset) for day in days for dataset in datasets}
    if (selected.duplicated(["day", "dataset"]).any()
            or set(zip(selected.day, selected.dataset)) != expected):
        raise ValueError("Held-out venue/date coverage is incomplete or duplicated.")
    values = selected[["candidate_bins", "eligible_bins"]].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or np.any(values < 0) or np.any(values != np.floor(values))
            or np.any(values[:, 0] == 0) or np.any(values[:, 1] > values[:, 0])):
        raise ValueError("Invalid forecast coverage counts.")
    grouped = selected.groupby("dataset")[["candidate_bins", "eligible_bins"]].sum().reindex(datasets)
    grouped["retention_percent"] = 100 * grouped.eligible_bins / grouped.candidate_bins
    return grouped.reset_index()


def plot_results(out_dir: str | Path) -> dict[str, str]:
    """Write four figures from actual completed outputs without fetching data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    out = Path(out_dir)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    if not summary.get("complete"):
        raise ValueError("Plots require a completed experiment summary.")
    days, datasets = summary["held_out_days"], summary["datasets"]
    daily = pd.read_csv(out / "daily_scores.csv")
    tasks = pd.read_csv(out / "task_daily_scores.csv")
    coverage = pd.read_csv(out / "coverage.csv")
    edges = pd.read_csv(out / "edges.csv")
    pairs = {reference: _paired_daily(daily, reference, expected_days=days) for reference in COMPARATORS}
    for reference, paired in pairs.items():
        stated = summary["comparisons"][reference]["pooled"]["log_loss"]["relative_gain_percent"]
        if not np.isclose(stated, _pooled_gain(paired), atol=1e-9, rtol=1e-9):
            raise ValueError(f"Summary and daily score tables disagree: {reference}")
    totals = _coverage_totals(coverage, days=days, datasets=datasets)
    if not {"source", "target", "beta_per_second"}.issubset(edges):
        raise ValueError("Fitted edge table lacks source/target/rate columns.")
    expected_edges = {(source, target) for source in datasets for target in datasets if source != target}
    if (edges.duplicated(["source", "target"]).any()
            or set(zip(edges.source, edges.target)) != expected_edges
            or not np.isfinite(edges.beta_per_second).all() or (edges.beta_per_second < 0).any()):
        raise ValueError("Directed edge table is incomplete, duplicated or invalid.")
    if "cohort_sha256" in tasks and tasks.groupby(["day", "dataset"]).cohort_sha256.nunique().max() != 1:
        raise ValueError("Task model cohorts differ.")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white"})
    files = {}

    # Training edge rates represent conditional predictive associations. No
    # significance threshold or causal effect is inferred from arrow width.
    fig, ax = plt.subplots(figsize=(10, 9))
    angle = np.pi / 2 - np.arange(len(datasets)) * 2 * np.pi / len(datasets)
    positions = {dataset: np.array([np.cos(a), np.sin(a)]) for dataset, a in zip(datasets, angle)}
    positive = edges.loc[edges.beta_per_second > 1e-8].sort_values("beta_per_second")
    maximum = float(positive.beta_per_second.max()) if len(positive) else 0.
    for edge in positive.itertuples():
        proportion = edge.beta_per_second / maximum
        ax.add_patch(FancyArrowPatch(positions[edge.source], positions[edge.target],
                     connectionstyle="arc3,rad=0.15", arrowstyle="-|>", mutation_scale=10 + 5 * proportion,
                     shrinkA=24, shrinkB=24, linewidth=.45 + 4.5 * proportion,
                     color="#2563a6", alpha=.20 + .55 * proportion, zorder=1))
    for dataset, point in positions.items():
        ax.scatter(*point, s=1300, color="#0f766e", edgecolor="white", linewidth=2, zorder=3)
        label = VENUE_LABELS.get(dataset, dataset).replace("NYSE Arca", "NYSE\nArca")
        ax.text(*point, label, ha="center", va="center", color="white", fontsize=9, weight="bold", zorder=4)
    if maximum:
        rates = sorted(set([float(positive.beta_per_second.min()), float(positive.beta_per_second.median()), maximum]))
        handles = [Line2D([0], [0], color="#2563a6", linewidth=.45 + 4.5 * rate / maximum,
                          label=f"β = {rate:.3g} / second") for rate in rates]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, -.04), ncol=len(handles), frameon=False)
    else:
        ax.text(0, 0, "No positive fitted rates above 10⁻⁸ / second", ha="center", color="#475569")
    ax.set(xlim=(-1.35, 1.35), ylim=(-1.35, 1.35), aspect="equal")
    ax.axis("off")
    fig.suptitle("Fitted venue association network", y=.98, fontsize=17, weight="bold")
    fig.text(.5, .94, "Training SIRS rates · source predictor → target · wider arrows represent larger β", ha="center", color="#475569")
    fig.text(.5, .025, "Conditional predictive associations. Held-out benefit is evaluated separately; arrows do not establish causal transmission.",
             ha="center", fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, .045, 1, .91))
    path = out / "network_beta.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    files["network"] = str(path.resolve())

    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), gridspec_kw={"height_ratios": [1.3, 1]})
    primary = pairs[REFERENCE]
    gains = primary.gain_percent.to_numpy()
    axes[0].bar(np.arange(len(primary)), gains, color=np.where(gains >= 0, "#0f766e", "#b45354"), width=.75)
    axes[0].axhline(0, color="#64748b", linewidth=.9)
    axes[0].set_xticks(np.arange(len(primary)), [day[5:] for day in primary.day], rotation=45, ha="right")
    axes[0].set(ylabel="Relative log-loss reduction (%)", title="Daily held-out SIRS gain versus independent MBO hazard")
    axes[0].grid(axis="y", alpha=.15)
    pooled = [_pooled_gain(pairs[reference]) for reference in COMPARATORS]
    axes[1].barh(np.arange(3), pooled, color=["#0f766e" if value >= 0 else "#b45354" for value in pooled], height=.5)
    for i, reference in enumerate(COMPARATORS):
        interval = summary["comparisons"][reference]["five_session_block_bootstrap"]["log_loss"]["relative_gain_percent_ci95"]
        if interval is not None:
            axes[1].plot(interval, [i, i], color="#1e293b", linewidth=1.5)
            axes[1].plot(interval, [i, i], "|", color="#1e293b", markersize=9)
        axes[1].scatter([pooled[i]], [i], color="#1e293b", s=17, zorder=4)
    axes[1].axvline(0, color="#64748b", linewidth=.9)
    axes[1].set_yticks(np.arange(3), ["vs " + MODEL_LABELS[reference] for reference in COMPARATORS])
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Pooled relative log-loss reduction from SIRS (%)", title="Same held-out cohorts · lines show paired five-session-block 95% intervals")
    axes[1].grid(axis="x", alpha=.15)
    fig.text(.5, .015, "Positive gains favor SIRS. All probability models share timestamps and labels. Secondary comparisons are exploratory; no profit estimate is shown.",
             ha="center", fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, .035, 1, 1), h_pad=2.8)
    path = out / "sirs_model_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    files["model_comparison"] = str(path.resolve())

    fig, ax = plt.subplots(figsize=(11, 5.5))
    values = totals.retention_percent.to_numpy()
    ax.barh(np.arange(len(totals)), values, color="#2563a6", height=.6)
    for i, record in enumerate(totals.itertuples()):
        label = f"{record.retention_percent:.1f}%  ({int(record.eligible_bins):,} forecasts)"
        ax.text(record.retention_percent - 1.2, i, label, ha="right", va="center", fontsize=9, color="white")
    ax.set_yticks(np.arange(len(totals)), [VENUE_LABELS.get(dataset, dataset) for dataset in totals.dataset])
    ax.invert_yaxis()
    ax.set(xlim=(0, 100), xlabel="Held-out scheduled anchors retained (%)",
           title="Forecast coverage after common quality rules and target susceptibility")
    ax.grid(axis="x", alpha=.15)
    fig.text(.5, .015, "Denominator: every scheduled held-out 100 ms anchor. Exclusions include warmup, book/market support and the target's infectious/refractory states.",
             ha="center", fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, .04, 1, 1))
    path = out / "sirs_coverage.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    files["coverage"] = str(path.resolve())

    venue_gains = []
    for dataset in datasets:
        pair = _paired_daily(tasks.loc[tasks.dataset.eq(dataset)], REFERENCE, expected_days=days)
        venue_gains.append(_pooled_gain(pair))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(np.arange(len(datasets)), venue_gains,
            color=["#0f766e" if value >= 0 else "#b45354" for value in venue_gains], height=.6)
    ax.axvline(0, color="#64748b", linewidth=.9)
    ax.set_yticks(np.arange(len(datasets)), [VENUE_LABELS.get(dataset, dataset) for dataset in datasets])
    ax.invert_yaxis()
    ax.set(xlabel="Relative held-out log-loss reduction (%)", title="SIRS versus independent MBO hazard by target venue")
    ax.grid(axis="x", alpha=.15)
    fig.text(.5, .015, "Each venue uses its own common eligible cohort. These descriptive target breakdowns are not independent experiments.",
             ha="center", fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, .04, 1, 1))
    path = out / "sirs_task_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    files["task_comparison"] = str(path.resolve())
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("out/mbo_sir"))
    args = parser.parse_args()
    print(json.dumps(plot_results(args.out_dir), indent=2))


if __name__ == "__main__":
    main()
