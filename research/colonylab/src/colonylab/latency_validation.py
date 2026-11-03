"""Date-level inference and figures for the offline venue-latency extension.

The estimand is an equal-date mean of daily conditional matched-event medians,
not a pooled event median.  Calm/volatile comparisons use the exact same dates.
Five-session moving blocks resample the entire calendar with every stock,
directed pair and clock together.  Intervals are marginal and exploratory: the
latency question was added after these data existed, and the source study's
train/held-out labels do not make this a preregistered latency confirmation.

The descriptive measurements extend smar-01/qam-p113-latency at commit
62d6dcd8af9ecd7136096dbcce3eb40371987422.  This module does not reuse that
project's matches as independent observations or assume they are causal.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


TASK_COLUMNS = ["clock", "symbol", "leader", "follower", "horizon_ms"]
REQUIRED_COLUMNS = {"study", "date", *TASK_COLUMNS, "regime", "n_matched", "median_us"}
OUTPUT_COLUMNS = [
    "study", "phase", *TASK_COLUMNS, "contrast", "inference_role",
    "n_dates", "n_calendar_dates", "n_observed_calendar_dates", "n_missing_or_sparse_dates",
    "estimate_us", "calm_us", "volatile_us", "relative_change_percent",
    "ci95_low_us", "ci95_high_us", "relative_ci95_low_percent", "relative_ci95_high_percent",
    "interval_status", "bootstrap_replicates", "block_sessions", "seed",
]

VENUE_LABELS = {
    "XNAS.ITCH": "Nasdaq", "XNYS.PILLAR": "NYSE", "BATS.PITCH": "BZX",
    "ARCX.PILLAR": "Arca", "EDGX.PITCH": "EDGX", "MEMX.MEMOIR": "MEMX",
    "BATY.PITCH": "BYX", "EDGA.PITCH": "EDGA", "IEXG.TOPS": "IEX",
}


def moving_block_counts(n_dates: int, *, replicates: int, block_sessions: int,
                        seed: int) -> np.ndarray:
    """Counts of whole-date draws, without wrapping the observed date range.

    One result matrix is shared by all tasks in a study/phase.  Blocks consist
    of consecutive observed trading-session positions, not calendar days.
    """
    for name, value in (("n_dates", n_dates), ("replicates", replicates),
                        ("block_sessions", block_sessions)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    length = min(n_dates, block_sessions)
    n_blocks = (n_dates + length - 1) // length
    starts = np.random.default_rng(seed).integers(0, n_dates - length + 1,
                                                size=(replicates, n_blocks))
    indices = (starts[..., None] + np.arange(length)).reshape(replicates, -1)[:, :n_dates]
    counts = np.zeros((replicates, n_dates), dtype=np.int32)
    np.add.at(counts, (np.arange(replicates)[:, None], indices), 1)
    return counts


def _validate_daily(daily: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(daily)
    if missing:
        raise ValueError("Latency daily summaries lack: " + ", ".join(sorted(missing)))
    frame = daily.copy()
    if "phase" not in frame:
        frame["phase"] = "all"
    for key in ["study", "phase", "date", "clock", "symbol", "leader", "follower", "regime"]:
        if frame[key].isna().any() or not frame[key].map(lambda x: isinstance(x, str) and bool(x)).all():
            raise ValueError(f"{key} must contain explicit nonempty strings.")
    for day in frame.date.unique():
        try:
            stamp = pd.Timestamp(day)
            if stamp.strftime("%Y-%m-%d") != day or stamp.weekday() > 4:
                raise ValueError
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("Latency dates must be explicit ISO weekdays.") from exc
    identity = ["study", "phase", "date", *TASK_COLUMNS, "regime"]
    if frame.duplicated(identity).any():
        raise ValueError("Duplicate latency task/date/regime summaries.")
    if frame.leader.eq(frame.follower).any():
        raise ValueError("Latency tasks must use different leader/follower venues.")
    for field in ("n_matched", "horizon_ms", "median_us"):
        try:
            frame[field] = pd.to_numeric(frame[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be numeric.") from exc
    counts = frame.n_matched.to_numpy(float)
    if (not np.isfinite(counts).all() or (counts < 0).any()
            or np.any(counts != np.floor(counts))):
        raise ValueError("n_matched must be a finite nonnegative integer count.")
    horizons = frame.horizon_ms.to_numpy(float)
    if not np.isfinite(horizons).all() or (horizons <= 0).any():
        raise ValueError("horizon_ms must be finite and positive.")
    medians = frame.median_us.to_numpy(float)
    if np.isinf(medians).any() or (medians[np.isfinite(medians)] < 0).any():
        raise ValueError("Daily medians must be nonnegative or missing.")
    if np.any((counts > 0) & ~np.isfinite(medians)):
        raise ValueError("A positive matched count requires a finite daily median.")
    if np.any((counts == 0) & np.isfinite(medians)):
        raise ValueError("Zero matched events cannot have a finite daily median.")
    if np.any(medians > horizons * 1000 + 1e-8):
        raise ValueError("A matched daily median cannot exceed its horizon.")
    return frame


def _calendar(scope: pd.DataFrame, expected: Sequence[str] | None) -> list[str]:
    observed = sorted(scope.date.unique())
    if expected is None:
        return observed
    dates = list(expected)
    if not dates or len(set(dates)) != len(dates) or dates != sorted(dates):
        raise ValueError("Expected latency dates must be nonempty, unique and chronological.")
    for day in dates:
        try:
            stamp = pd.Timestamp(day)
            if not isinstance(day, str) or stamp.strftime("%Y-%m-%d") != day or stamp.weekday() > 4:
                raise ValueError
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("Expected latency dates must be explicit ISO weekdays.") from exc
    if not set(observed) <= set(dates):
        raise ValueError("Latency summaries contain dates outside the expected calendar.")
    return dates


def _finite_mean(values: np.ndarray) -> np.ndarray:
    present = np.isfinite(values)
    denominator = present.sum(axis=0)
    return np.divide(np.where(present, values, 0.).sum(axis=0), denominator,
                     out=np.full(values.shape[1], np.nan), where=denominator > 0)


def _sample_means(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    present = np.isfinite(values)
    denominator = counts @ present.astype(float)
    numerator = counts @ np.where(present, values, 0.)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
                     where=denominator > 0)


def infer_latency(
    daily: pd.DataFrame, *, min_matches: int = 20, bootstrap_replicates: int = 2000,
    block_sessions: int = 5, seed: int = 20261102,
    primary_study: str = "fresh_confirmation", primary_phase: str = "held_out",
    min_inference_dates: int = 20,
    expected_dates: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Summarize daily latency and paired regime differences with date blocks.

    ``expected_dates`` maps ``"study:phase"`` to the intended session dates.
    An absent intended date prevents all intervals for that study/phase.
    A task/date/regime with fewer than ``min_matches`` is absent from that
    task's descriptive estimand; the resulting support loss remains explicit.
    Only ``primary_study``/``primary_phase`` receives intervals, and only when
    every intended session is present and at least ``min_inference_dates``
    task dates meet the support rule.  All other scopes remain descriptive.

    ``contrast='latency'`` estimates the mean daily all-regime median.
    ``contrast='volatile_vs_calm'`` estimates the mean paired volatile-minus-
    calm median.  Relative change divides this difference by the mean calm
    median on exactly those paired dates.  It is not a mean of daily ratios.
    """
    for name, value in (("min_matches", min_matches), ("bootstrap_replicates", bootstrap_replicates),
                        ("block_sessions", block_sessions), ("min_inference_dates", min_inference_dates)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    frame = _validate_daily(daily)
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    output: list[dict] = []
    for (study, phase), scope in frame.groupby(["study", "phase"], sort=True):
        intended = None if expected_dates is None else expected_dates.get(f"{study}:{phase}")
        dates = _calendar(scope, intended)
        n_observed = scope.date.nunique()
        complete_calendar = n_observed == len(dates)
        is_primary = study == primary_study and phase == primary_phase
        role = "primary_post_hoc" if is_primary else "descriptive"
        # Build exactly the same task columns for every regime; absent regime
        # rows must become missing rather than changing the task universe.
        tasks = pd.MultiIndex.from_frame(scope[TASK_COLUMNS].drop_duplicates().sort_values(TASK_COLUMNS))
        matrices = {}
        for regime in ("all", "calm", "volatile"):
            selected = scope.loc[scope.regime.eq(regime)].copy()
            selected.loc[selected.n_matched < min_matches, "median_us"] = np.nan
            matrix = selected.pivot(index="date", columns=TASK_COLUMNS, values="median_us")
            matrices[regime] = matrix.reindex(index=dates, columns=tasks).to_numpy(float)
        counts = (moving_block_counts(len(dates), replicates=bootstrap_replicates,
                                      block_sessions=block_sessions, seed=seed)
                  if is_primary and complete_calendar and len(dates) >= min_inference_dates else None)
        paired = np.isfinite(matrices["calm"]) & np.isfinite(matrices["volatile"])
        calm = np.where(paired, matrices["calm"], np.nan)
        volatile = np.where(paired, matrices["volatile"], np.nan)
        for contrast, values, reference, candidate in (
            ("latency", matrices["all"], None, None),
            ("volatile_vs_calm", volatile - calm, calm, volatile),
        ):
            support = np.isfinite(values).sum(axis=0)
            estimates = _finite_mean(values)
            calm_means = _finite_mean(reference) if reference is not None else np.full(len(tasks), np.nan)
            volatile_means = _finite_mean(candidate) if candidate is not None else np.full(len(tasks), np.nan)
            relative = np.divide(100 * estimates, calm_means, out=np.full(len(tasks), np.nan),
                                 where=calm_means > 0)
            intervals = np.full((2, len(tasks)), np.nan)
            relative_intervals = np.full((2, len(tasks)), np.nan)
            defined = np.zeros(len(tasks), dtype=bool)
            relative_defined = np.zeros(len(tasks), dtype=bool)
            if counts is not None:
                # Keep matrix multiplies bounded even for large task grids.
                for first in range(0, len(tasks), 256):
                    stop = min(first + 256, len(tasks))
                    draws = _sample_means(values[:, first:stop], counts)
                    ok = np.isfinite(draws).all(axis=0)
                    defined[first:stop] = ok
                    if ok.any():
                        intervals[:, first + np.flatnonzero(ok)] = np.quantile(draws[:, ok], [.025, .975], axis=0)
                    if reference is not None:
                        sampled_calm = _sample_means(reference[:, first:stop], counts)
                        relative_draws = np.divide(100 * draws, sampled_calm,
                                                   out=np.full_like(draws, np.nan), where=sampled_calm > 0)
                        rel_ok = np.isfinite(relative_draws).all(axis=0)
                        relative_defined[first:stop] = rel_ok
                        if rel_ok.any():
                            relative_intervals[:, first + np.flatnonzero(rel_ok)] = np.quantile(
                                relative_draws[:, rel_ok], [.025, .975], axis=0)
            for position, task in enumerate(tasks):
                if not is_primary:
                    status = "descriptive"
                elif not complete_calendar:
                    status = "missing_intended_dates"
                elif support[position] < min_inference_dates:
                    status = "insufficient_supported_dates"
                elif not defined[position]:
                    status = "undefined_resample"
                else:
                    status = "ok"
                ci = intervals[:, position] if status == "ok" else [np.nan, np.nan]
                rel_ci = (relative_intervals[:, position]
                          if status == "ok" and relative_defined[position] else [np.nan, np.nan])
                output.append({
                    "study": study, "phase": phase, **dict(zip(TASK_COLUMNS, task)),
                    "contrast": contrast, "inference_role": role,
                    "n_dates": int(support[position]), "n_calendar_dates": len(dates),
                    "n_observed_calendar_dates": int(n_observed),
                    "n_missing_or_sparse_dates": int(len(dates) - support[position]),
                    "estimate_us": estimates[position], "calm_us": calm_means[position],
                    "volatile_us": volatile_means[position], "relative_change_percent": relative[position],
                    "ci95_low_us": ci[0], "ci95_high_us": ci[1],
                    "relative_ci95_low_percent": rel_ci[0], "relative_ci95_high_percent": rel_ci[1],
                    "interval_status": status,
                    "bootstrap_replicates": bootstrap_replicates if status == "ok" else 0,
                    "block_sessions": min(block_sessions, len(dates)), "seed": seed,
                })
    return pd.DataFrame(output, columns=OUTPUT_COLUMNS).sort_values(
        ["study", "phase", *TASK_COLUMNS, "contrast"], kind="stable").reset_index(drop=True)


def _plot_scope(daily: pd.DataFrame, inference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if daily.empty or inference.empty:
        raise ValueError("Latency figures require nonempty daily and inference tables.")
    primary = inference.loc[inference.inference_role.eq("primary_post_hoc")]
    choice = primary if len(primary) else inference
    first = choice.sort_values(["study", "phase"]).iloc[0]
    study, phase = first.study, first.phase
    selected = daily.loc[daily.study.eq(study)].copy()
    if "phase" not in selected:
        selected["phase"] = "all"
    selected = selected.loc[selected.phase.eq(phase)]
    inferred = inference.loc[inference.study.eq(study) & inference.phase.eq(phase)]
    return selected, inferred, f"{study}, {phase.replace('_', ' ')}"


def render_latency_figures(daily: pd.DataFrame, inference: pd.DataFrame,
                           out_dir: str | Path) -> list[Path]:
    """Write static figures from completed result tables, offline.

    The selected scope is the primary post-hoc study/phase if present.  At the
    default 10 ms horizon, each heatmap cell is an equal-date mean conditional
    median.  Horizon denominators exclude right truncation and quality exits;
    therefore the horizon figure is sensitivity analysis, not a survival curve
    on a fixed event cohort.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    selected, inferred, label = _plot_scope(daily, inference)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    horizons = sorted(inferred.horizon_ms.unique())
    horizon = 10. if 10. in horizons else float(horizons[0])
    clocks = set(inferred.clock)
    clock = "ts_recv" if "ts_recv" in clocks else sorted(clocks)[0]
    full_scope = inferred.inference_role.eq("primary_post_hoc").any()
    all_base = inferred.loc[inferred.contrast.eq("latency") & inferred.clock.eq(clock)
                            & inferred.horizon_ms.eq(horizon)]
    symbols = sorted(all_base.symbol.unique())
    base = all_base.loc[all_base.interval_status.eq("ok")] if full_scope else all_base
    if not symbols:
        raise ValueError("No all-regime latency estimates to plot.")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.titleweight": "bold"})
    figure_width = max(6, 5.2 * len(symbols))
    finite = base.estimate_us[np.isfinite(base.estimate_us) & base.estimate_us.gt(0)]
    color_low = max(.001, float(finite.min())) if len(finite) else .001
    color_high = max(color_low * 1.001, float(finite.max())) if len(finite) else 1.
    norm = LogNorm(vmin=color_low, vmax=color_high) if len(finite) else None
    fig, axes = plt.subplots(1, len(symbols), figsize=(figure_width, 5.5), squeeze=False,
                             layout="constrained")
    image = None
    for ax, symbol in zip(axes[0], symbols):
        group = base.loc[base.symbol.eq(symbol)]
        universe = all_base.loc[all_base.symbol.eq(symbol)]
        venues = sorted(set(universe.leader) | set(universe.follower))
        matrix = group.pivot(index="leader", columns="follower", values="estimate_us").reindex(
            index=venues, columns=venues)
        masked = np.ma.masked_invalid(matrix.to_numpy(float))
        image = ax.imshow(masked, cmap="viridis", norm=norm)
        ax.set_xticks(range(len(venues)), [VENUE_LABELS.get(x, x) for x in venues], rotation=55, ha="right")
        ax.set_yticks(range(len(venues)), [VENUE_LABELS.get(x, x) for x in venues])
        ax.set_xlabel("Follower")
        ax.set_ylabel("Leader")
        ax.set_title(symbol)
        for row in range(len(venues)):
            for column in range(len(venues)):
                value = matrix.iloc[row, column]
                if np.isfinite(value):
                    ax.text(column, row, f"{value:.0f}", ha="center", va="center", fontsize=6,
                            color="white" if norm is not None and norm(value) < .6 else "black")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.75, label="Microseconds (log color scale)")
    support_note = "full 20-date support only; unsupported cells blank" if full_scope else "descriptive; supported date counts may vary"
    fig.suptitle(f"Conditional matched latency: equal-date mean of daily medians\n{label}; {clock}; {horizon:g} ms horizon; {support_note}")
    path = out / "latency_heatmap.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    files.append(path)

    regime = inferred.loc[inferred.contrast.eq("volatile_vs_calm") & inferred.clock.eq(clock)
                          & inferred.horizon_ms.eq(horizon)]
    fig, axes = plt.subplots(1, len(symbols), figsize=(figure_width, 4.7), squeeze=False,
                             layout="constrained")
    for ax, symbol in zip(axes[0], symbols):
        group = regime.loc[regime.symbol.eq(symbol) & np.isfinite(regime.relative_change_percent)].sort_values(
            "relative_change_percent")
        x = np.arange(len(group))
        y = group.relative_change_percent.to_numpy(float)
        ax.scatter(x, y, s=12, color="#176b87", zorder=3)
        usable = group.interval_status.eq("ok") & np.isfinite(group.relative_ci95_low_percent)
        if usable.any():
            positions = np.flatnonzero(usable.to_numpy())
            ax.vlines(positions, group.loc[usable, "relative_ci95_low_percent"],
                      group.loc[usable, "relative_ci95_high_percent"], color="#78909c", alpha=.6, linewidth=.8)
        if group.empty:
            ax.text(.5, .55, "No supported calm/volatile dates\nunder quality rules", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="#526875")
        elif not usable.any():
            low, high = int(group.n_dates.min()), int(group.n_dates.max())
            ax.text(.02, .98, f"Descriptive only: {low}–{high} paired dates\nNo complete 20-date intervals",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#526875")
        ax.axhline(0, color="#555", linewidth=.8)
        ax.set_title(symbol)
        ax.set_xlabel("Directed pairs, sorted by effect")
        if len(y) and np.nanmax(np.abs(y)) > 300:
            ax.set_yscale("symlog", linthresh=10)
            ax.set_ylabel("Volatile minus calm (%; symmetric log scale)")
        else:
            ax.set_ylabel("Volatile minus calm (%)")
        ax.grid(axis="y", alpha=.18)
    has_intervals = regime.interval_status.eq("ok").any()
    interval_note = "bars: marginal 95% date-block intervals" if has_intervals else "descriptive points only; no complete 20-date intervals"
    fig.suptitle(f"Paired-date difference in conditional medians\n{label}; {clock}; {horizon:g} ms; {interval_note}")
    path = out / "latency_regimes.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    files.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout="constrained")
    all_rows = selected.loc[selected.regime.eq("all")]
    for this_clock, group in all_rows.groupby("clock", sort=True):
        if {"n_matched", "n_censored"} <= set(group):
            counts = group.groupby("horizon_ms")[["n_matched", "n_censored"]].sum()
            denominator = counts.n_matched + counts.n_censored
            rate = np.divide(100 * counts.n_matched, denominator,
                             out=np.full(len(counts), np.nan), where=denominator > 0)
            axes[0].plot(counts.index, rate, marker="o", label=this_clock)
        values = inferred.loc[inferred.clock.eq(this_clock) & inferred.contrast.eq("latency")]
        if full_scope:
            values = values.loc[values.interval_status.eq("ok")]
        medians = values.groupby("horizon_ms").estimate_us.mean()
        axes[1].plot(medians.index, medians, marker="o", label=this_clock)
    axes[0].set_ylabel("Matched / (matched + censored), %")
    axes[1].set_ylabel("Mean across pair estimates, µs")
    for ax in axes:
        ax.set_xlabel("Matching horizon, ms")
        ax.set_xscale("log")
        ax.grid(alpha=.18)
        ax.legend(frameon=False)
    mean_support = "pair means require full 20-date support" if full_scope else "descriptive pair means may use different dates"
    fig.suptitle(f"Matching-horizon sensitivity: {label}\nQuality exits/right truncation excluded; {mean_support}; event cohorts can differ")
    path = out / "latency_horizons.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    files.append(path)

    fig, ax = plt.subplots(figsize=(7.3, 6.5), layout="constrained")
    clock_rows = inferred.loc[inferred.contrast.eq("latency") & inferred.horizon_ms.eq(horizon)]
    if full_scope:
        clock_rows = clock_rows.loc[clock_rows.interval_status.eq("ok")]
    if {"ts_recv", "ts_event"} <= set(clock_rows.clock):
        paired_clocks = clock_rows.pivot(index=["symbol", "leader", "follower"], columns="clock",
                                         values="estimate_us").dropna(subset=["ts_recv", "ts_event"])
        paired_clocks = paired_clocks.loc[paired_clocks.ts_recv.gt(0) & paired_clocks.ts_event.gt(0)]
        for symbol, group in paired_clocks.groupby(level="symbol"):
            ax.scatter(group.ts_recv, group.ts_event, s=22, alpha=.7, label=symbol)
        if len(paired_clocks):
            low = float(paired_clocks[["ts_recv", "ts_event"]].min().min()) * .8
            high = float(paired_clocks[["ts_recv", "ts_event"]].max().max()) * 1.2
            ax.plot([low, high], [low, high], color="#777", linestyle="--", linewidth=.9)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend(frameon=False)
    else:
        ax.text(.5, .5, "Both clocks are required for this comparison", transform=ax.transAxes,
                ha="center", va="center")
    ax.set_xlabel("Receive-clock conditional latency, µs")
    ax.set_ylabel("Exchange-clock conditional latency, µs")
    ax.grid(alpha=.18)
    clock_support = "full 20-date support on both clocks" if full_scope else "descriptive; date support may vary"
    ax.set_title(f"Clock sensitivity of directed-pair estimates\n{label}; {horizon:g} ms horizon; {clock_support}")
    path = out / "latency_clock_comparison.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    files.append(path)
    files.extend(_render_secondary_figures(selected, inferred, out, label=label,
                                           clock=clock, horizon=horizon))
    return files


def _render_secondary_figures(selected: pd.DataFrame, inferred: pd.DataFrame, out: Path,
                              *, label: str, clock: str, horizon: float) -> list[Path]:
    """Plot optional native-event diagnostics with their actual weighting.

    The root pipeline's per-day exports are optional so small standalone uses
    of ``render_latency_figures`` need only the daily/inference tables.  Each
    line label reports its supporting date and/or event count.  The event
    study keeps one date cohort over every offset for each follower.
    """
    import matplotlib.pyplot as plt

    files = []
    study, phase = inferred.iloc[0][["study", "phase"]]
    symbol = "AAPL" if "AAPL" in set(selected.symbol) else sorted(selected.symbol.unique())[0]
    leader = "XNAS.ITCH" if "XNAS.ITCH" in set(selected.leader) else sorted(selected.leader.unique())[0]
    leader_label = VENUE_LABELS.get(leader, leader)

    def scope(frame):
        keep = (frame.study.eq(study) & frame.phase.eq(phase) & frame.clock.eq(clock)
                & frame.symbol.eq(symbol))
        return frame.loc[keep].copy()

    path = out / "leadlag_curves.csv"
    if path.is_file():
        curves = scope(pd.read_csv(path, usecols=["study", "phase", "date", "clock", "symbol",
                                                  "leader", "follower", "lag_us", "corr"]))
        curves = curves.loc[curves.leader.eq(leader) | curves.follower.eq(leader)]
        if len(curves):
            fig, ax = plt.subplots(figsize=(11.5, 6.2), layout="constrained")
            for (first, second), group in curves.groupby(["leader", "follower"], sort=True):
                pivot = group.pivot(index="date", columns="lag_us", values="corr").sort_index(axis=1)
                # A date contributes to the whole curve or none of it.
                common = pivot.loc[np.isfinite(pivot.to_numpy(float)).all(axis=1)]
                if common.empty:
                    continue
                first_label, second_label = VENUE_LABELS.get(first, first), VENUE_LABELS.get(second, second)
                ax.plot(common.columns.to_numpy(float), common.mean(axis=0).to_numpy(float),
                        linewidth=1.2, label=f"{first_label} → {second_label} ({len(common)} dates)")
            ax.axvline(0, color="#666", linestyle="--", linewidth=.8)
            ax.axhline(0, color="#777", linewidth=.7)
            ax.set_xlabel("Lag, µs; positive = right-hand venue follows left-hand venue in legend")
            ax.set_ylabel("Uncentered normalized binned-increment cross-product")
            ax.set_title(f"Lead-lag diagnostics: equal-date mean curves\n{label}; {symbol}; {clock}; pairs involving {leader_label}")
            ax.grid(alpha=.18)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
            target = out / "latency_leadlag.png"
            fig.savefig(target, dpi=170)
            plt.close(fig)
            files.append(target)

    path = out / "event_study.csv"
    if path.is_file():
        events = scope(pd.read_csv(path, usecols=["study", "phase", "date", "clock", "symbol",
                                                  "leader", "follower", "offset_us", "share", "n"]))
        events = events.loc[events.leader.eq(leader)]
        if len(events):
            fig, ax = plt.subplots(figsize=(11.5, 6.2), layout="constrained")
            for follower, group in events.groupby("follower", sort=True):
                pivot = group.pivot(index="date", columns="offset_us", values="share").sort_index(axis=1)
                counts = group.pivot(index="date", columns="offset_us", values="n").reindex(
                    index=pivot.index, columns=pivot.columns)
                supported = np.isfinite(pivot.to_numpy(float)).all(axis=1) & (counts.to_numpy(float) > 0).all(axis=1)
                common, count_table = pivot.loc[supported], counts.loc[supported]
                if common.empty:
                    continue
                if (count_table.nunique(axis=1) != 1).any():
                    raise ValueError("Event-study event cohorts change across offsets within a date/pair.")
                count = int(count_table.iloc[:, 0].sum())
                follower_label = VENUE_LABELS.get(follower, follower)
                ax.plot(common.columns.to_numpy(float), common.mean(axis=0).to_numpy(float), marker=".",
                        linewidth=1.2, label=f"{follower_label} ({len(common)} dates; {count:,} events)")
            ax.axhline(1, color="#555", linestyle="--", linewidth=.9)
            ax.axhline(0, color="#777", linewidth=.7)
            ax.axvline(0, color="#777", linewidth=.7)
            ax.set_xscale("symlog", linthresh=10)
            ax.set_xlabel(f"Offset after {leader_label} midpoint change, µs (symmetric log scale)")
            ax.set_ylabel("Follower midpoint response / leader midpoint change")
            ax.set_title(f"Event study: equal-date mean shares, fixed offset cohort per pair\n{label}; {symbol}; {clock}; leader {leader_label}")
            ax.grid(alpha=.18)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
            target = out / "latency_event_study.png"
            fig.savefig(target, dpi=170)
            plt.close(fig)
            files.append(target)

    path = out / "latency_histograms.csv"
    if path.is_file():
        histograms = scope(pd.read_csv(path, usecols=["study", "phase", "date", "clock", "symbol",
                                                      "leader", "follower", "horizon_ms", "bin_left_us",
                                                      "bin_right_us", "count"]))
        histograms = histograms.loc[histograms.leader.eq(leader) & histograms.horizon_ms.eq(horizon)]
        if len(histograms):
            fig, ax = plt.subplots(figsize=(11.5, 6.2), layout="constrained")
            for follower, group in histograms.groupby("follower", sort=True):
                counts = group.groupby(["bin_left_us", "bin_right_us"], sort=True)["count"].sum().reset_index()
                total = int(counts["count"].sum())
                if not total:
                    continue
                centers = np.sqrt(counts.bin_left_us * counts.bin_right_us)
                percent = 100 * counts["count"].to_numpy(float) / total
                follower_label = VENUE_LABELS.get(follower, follower)
                ax.step(centers, percent, where="mid", linewidth=1.2,
                        label=f"{follower_label} ({group.date.nunique()} dates; {total:,} matches)")
            ax.set_xscale("log")
            ax.set_xlim(right=horizon * 1000)
            ax.set_xlabel("Matched latency, µs (log scale)")
            ax.set_ylabel("Matched events per log-spaced bin (%)")
            ax.set_title(f"Event-weighted matched-latency distributions across dates\n{label}; {symbol}; {clock}; leader {leader_label}; {horizon:g} ms horizon")
            ax.grid(alpha=.18)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
            target = out / "latency_distributions.png"
            fig.savefig(target, dpi=170)
            plt.close(fig)
            files.append(target)

    fields = ["n_already", "n_no_state", "n_quality_truncated", "n_right_truncated", "n_matched", "n_censored"]
    if set(fields + ["n_events"]) <= set(selected):
        counts = selected.loc[selected.regime.eq("all") & selected.clock.eq(clock)
                              & selected.horizon_ms.eq(horizon)].groupby("symbol")[fields + ["n_events"]].sum()
        if len(counts):
            fig, ax = plt.subplots(figsize=(10.5, 5.8), layout="constrained")
            base = np.zeros(len(counts))
            labels = ["Already at price", "No valid prior state", "Quality-truncated horizon",
                      "Request-end truncation", "Complete horizon, matched", "Complete horizon, unmatched"]
            colors = ["#78909c", "#b0bec5", "#c98256", "#e9c46a", "#176b87", "#64a6bd"]
            total = counts.n_events.to_numpy(float)
            for field, category, color in zip(fields, labels, colors):
                height = np.divide(100 * counts[field].to_numpy(float), total,
                                   out=np.zeros(len(total)), where=total > 0)
                ax.bar(np.arange(len(counts)), height, bottom=base, label=category, color=color)
                base += height
            ax.set_xticks(np.arange(len(counts)), [f"{name}\n{int(number):,} pair-events"
                                                  for name, number in zip(counts.index, total)])
            ax.set_ylabel("Share of all leader/follower pair-events (%)")
            ax.set_ylim(0, 100)
            ax.set_title(f"Observation eligibility and price matching: event-weighted counts\n{label}; {clock}; {horizon:g} ms horizon; all directed pairs")
            ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.13))
            target = out / "latency_observation_coverage.png"
            fig.savefig(target, dpi=170)
            plt.close(fig)
            files.append(target)
    return files
