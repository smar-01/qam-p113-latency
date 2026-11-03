"""A descriptive, receive-clock venue quote-shock propagation pilot.

This module measures temporal association, not who submitted orders or a causal
contagion rate. MBP-1 supplies authoritative top-of-book states. MBO is used only
for observed order-event counts; no book is reconstructed from a partial window.
"""

from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd


HORIZONS_MS = (10, 100, 1000)
PLACEBO_SHIFTS_MS = (5000, 15000, 30000)
MIN_LEAD_MS = 2
MAX_AGE_MS = 250
MIN_SEPARATION_MS = 100
F_LAST, F_SNAPSHOT, F_BAD_TS_RECV, F_MAYBE_BAD_BOOK = 128, 32, 8, 4
VENUE_LABELS = {"XNAS.ITCH": "Nasdaq", "XNYS.PILLAR": "NYSE", "BATS.PITCH": "Cboe BZX",
                "ARCX.PILLAR": "NYSE Arca", "EDGX.PITCH": "Cboe EDGX"}
REFRACTORY_CAVEAT = (
    "Source and target shocks are each de-duplicated for 100 ms. A synchronous target shock in the "
    "excluded first 2 ms can consume its refractory period and suppress a later qualifying hit; "
    "negative excess response is not evidence that a venue inhibits another venue."
)


def _number(frame: pd.DataFrame, name: str, default: float = np.nan) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _text(values: pd.Series) -> pd.Series:
    return values.map(lambda value: value.decode() if isinstance(value, bytes) else str(value))


def _clock(frame: pd.DataFrame) -> tuple[np.ndarray, str]:
    if "ts_recv" in frame:
        value, name = frame["ts_recv"], "ts_recv"
    elif isinstance(frame.index, pd.DatetimeIndex) and frame.index.name == "ts_recv":
        value, name = frame.index, "ts_recv"
    elif "ts_event" in frame:
        value, name = frame["ts_event"], "ts_event"
    elif isinstance(frame.index, pd.DatetimeIndex) and frame.index.name == "ts_event":
        value, name = frame.index, "ts_event"
    else:
        raise ValueError("A ts_recv or ts_event timestamp column/named DatetimeIndex is required.")
    dates = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(dates).any():
        raise ValueError("The selected network analysis clock contains missing timestamps.")
    # Explicit ns conversion avoids datetime64[us] integer-unit surprises.
    dates = dates.astype("datetime64[ns, UTC]")
    return np.asarray(dates.astype("int64"), dtype=np.int64), name


def normalize_quotes(frame: pd.DataFrame, dataset: str, date: str) -> tuple[pd.DataFrame, dict]:
    """Normalize MBP-1 states, retaining invalid/tied states as carry barriers.

    Non-F_LAST intermediate book updates are omitted. Snapshot, inaccurate
    receive-time and maybe-bad-book records are barriers. A missing flags field
    is supported for synthetic inputs, and disclosed in diagnostics.
    """
    required = {"bid_px_00", "ask_px_00"}
    if not required.issubset(frame.columns):
        raise ValueError("Venue analysis requires MBP-1 bid_px_00 and ask_px_00.")
    times, clock = _clock(frame)
    symbols = _text(frame["symbol"]).to_numpy() if "symbol" in frame else np.repeat("UNKNOWN", len(frame))
    flags = _number(frame, "flags", F_LAST)
    finite_flags = np.isfinite(flags)
    bits = np.where(finite_flags, flags, 0).astype(np.uint64)
    last = (bits & F_LAST) != 0
    bad = (~finite_flags) | ((bits & (F_SNAPSHOT | F_BAD_TS_RECV | F_MAYBE_BAD_BOOK)) != 0)
    # A known bad row remains a barrier even if it lacks F_LAST.
    included = last | bad
    bid, ask = _number(frame, "bid_px_00"), _number(frame, "ask_px_00")
    bsize, asize = _number(frame, "bid_sz_00"), _number(frame, "ask_sz_00")
    valid = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask >= bid) & (ask < 1e8) & ~bad
    if "bid_sz_00" in frame:
        valid &= np.isfinite(bsize) & (bsize > 0)
    if "ask_sz_00" in frame:
        valid &= np.isfinite(asize) & (asize > 0)
    mid = (bid + ask) / 2
    total = bsize + asize
    depth = np.divide(bsize - asize, total, out=np.full(len(frame), np.nan), where=np.isfinite(total) & (total > 0))
    quotes = pd.DataFrame({
        "time_ns": times, "symbol": symbols, "bid": bid, "ask": ask,
        "bid_size": bsize, "ask_size": asize, "mid": np.where(valid, mid, np.nan),
        "spread_bps": np.where(valid, (ask - bid) / mid * 1e4, np.nan),
        "depth_imbalance": np.where(valid, depth, np.nan), "valid": valid,
    }).loc[included].sort_values(["symbol", "time_ns"], kind="stable")
    keys = ["symbol", "time_ns"]
    if len(quotes):
        state = ["bid", "ask", "bid_size", "ask_size"]
        grouped = quotes.groupby(keys, sort=False)
        different = grouped[state].nunique(dropna=False).gt(1).any(axis=1)
        invalid = ~grouped["valid"].all()
        barrier_keys = different.index[different | invalid]
        quotes = quotes.drop_duplicates(keys, keep="first").set_index(keys)
        quotes.loc[quotes.index.isin(barrier_keys), "valid"] = False
        quotes.loc[~quotes["valid"], ["bid", "ask", "bid_size", "ask_size", "mid", "spread_bps", "depth_imbalance"]] = np.nan
        quotes = quotes.reset_index()
        quotes["segment"] = (~quotes["valid"]).groupby(quotes["symbol"]).cumsum().astype(np.int64)
        ambiguous = int(different.sum())
    else:
        quotes["segment"] = pd.Series(dtype=np.int64)
        ambiguous = 0
    quotes["dataset"], quotes["day"] = dataset, date
    diagnostics = {
        "dataset": dataset, "day": date, "clock": clock, "raw_records": len(frame),
        "flags_present": "flags" in frame, "intermediate_records_omitted": int((~included).sum()),
        "snapshot_records": int(((bits & F_SNAPSHOT) != 0).sum()),
        "bad_receive_time_records": int(((bits & F_BAD_TS_RECV) != 0).sum()),
        "maybe_bad_book_records": int(((bits & F_MAYBE_BAD_BOOK) != 0).sum()),
        "ambiguous_timestamps": ambiguous, "state_records": len(quotes),
        "invalid_state_records": int((~quotes["valid"]).sum()),
    }
    return quotes.reset_index(drop=True), diagnostics


def prepare_quotes(frame: pd.DataFrame, dataset: str, day: str) -> pd.DataFrame:
    return normalize_quotes(frame, dataset, day)[0]


def chronological_split(days: list[str], held_out_days: list[str] | None = None) -> tuple[list[str], list[str]]:
    """Select whole held-out dates after every training date, without tuning."""
    ordered = list(days)
    if len(ordered) < 2 or len(set(ordered)) != len(ordered):
        raise ValueError("At least two distinct ordered days are required for training and held-out evaluation.")
    if ordered != sorted(ordered):
        raise ValueError("Analysis days must be in increasing chronological order.")
    if held_out_days is None:
        return ordered[:-1], ordered[-1:]
    if isinstance(held_out_days, str) or not held_out_days:
        raise ValueError("held_out_days must be a nonempty list of observed dates.")
    held = list(held_out_days)
    if len(set(held)) != len(held) or not set(held).issubset(ordered):
        raise ValueError("Held-out dates must be unique and present in the observed dates.")
    training = [day for day in ordered if day not in held]
    if not training:
        raise ValueError("At least one training date must remain before the held-out dates.")
    held = [day for day in ordered if day in held]
    if max(training) >= min(held):
        raise ValueError("Every held-out date must be chronologically after every training date.")
    return training, held


def _jump_candidates(quotes: pd.DataFrame, max_age_ms: float) -> tuple[np.ndarray, np.ndarray]:
    if quotes.empty:
        return np.array([], dtype=int), np.array([], dtype=float)
    mid = quotes["mid"].to_numpy(dtype=float)
    time = quotes["time_ns"].to_numpy(dtype=np.int64)
    valid = quotes["valid"].to_numpy(dtype=bool) if "valid" in quotes else np.isfinite(mid)
    same = np.ones(max(0, len(quotes) - 1), dtype=bool)
    for column in ("symbol", "day", "dataset", "segment"):
        if column in quotes:
            value = quotes[column].to_numpy()
            same &= value[1:] == value[:-1]
    delta = np.diff(mid)
    mask = valid[1:] & valid[:-1] & same & (np.diff(time) > 0) & (np.diff(time) <= max_age_ms * 1e6)
    mask &= np.isfinite(delta) & (np.abs(delta) > 1e-9)
    positions = np.flatnonzero(mask) + 1
    return positions, delta[mask]


def fit_threshold(quotes: pd.DataFrame) -> float:
    """Use training states only: max($0.005, 75th percentile nonzero jump)."""
    _, delta = _jump_candidates(quotes, MAX_AGE_MS)
    return float(max(0.005, np.quantile(np.abs(delta), 0.75))) if len(delta) else 0.005


def extract_shocks(
    quotes: pd.DataFrame, threshold: float, *, min_separation_ms: float = MIN_SEPARATION_MS,
    max_age_ms: float = MAX_AGE_MS,
) -> pd.DataFrame:
    """Detect new signed jumps; greedily retain events at least 100 ms apart."""
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("A shock threshold must be finite and positive.")
    positions, delta = _jump_candidates(quotes, max_age_ms)
    large = np.abs(delta) + 1e-9 >= threshold
    positions, delta = positions[large], delta[large]
    selected, selected_delta = [], []
    last_by_group: dict[tuple, int] = {}
    group_columns = [column for column in ("dataset", "symbol", "day") if column in quotes]
    group_arrays = [quotes[column].to_numpy() for column in group_columns]
    timestamps = quotes["time_ns"].to_numpy(dtype=np.int64)
    for position, jump in zip(positions, delta):
        key = tuple(values[position] for values in group_arrays)
        timestamp = int(timestamps[position])
        if timestamp - last_by_group.get(key, -10**30) >= min_separation_ms * 1e6:
            selected.append(position)
            selected_delta.append(jump)
            last_by_group[key] = timestamp
    columns = [column for column in ("time_ns", "mid", "segment", "symbol", "dataset", "day") if column in quotes]
    result = quotes.iloc[selected][columns].copy().reset_index(drop=True)
    result["jump"] = selected_delta
    result["direction"] = np.sign(result["jump"]).astype(int)
    result["sign"] = result["direction"]
    return result


def _edge_arrays(
    source_shocks: pd.DataFrame, target_quotes: pd.DataFrame, target_shocks: pd.DataFrame, *,
    horizon_ms: float, shift_ms: float = 0, min_lead_ms: float = MIN_LEAD_MS,
    max_age_ms: float = MAX_AGE_MS,
) -> dict[str, np.ndarray]:
    if horizon_ms <= min_lead_ms or min_lead_ms < 0:
        raise ValueError("The response horizon must exceed a nonnegative minimum lead.")
    count = len(source_shocks)
    out = {"eligible": np.zeros(count, dtype=bool), "hit": np.zeros(count, dtype=bool),
           "markout": np.full(count, np.nan), "delay": np.full(count, np.nan)}
    if not count or target_quotes.empty:
        return out
    times = source_shocks["time_ns"].to_numpy(dtype=np.int64) + int(shift_ms * 1e6)
    directions = source_shocks["direction"].to_numpy(dtype=int)
    qt = target_quotes["time_ns"].to_numpy(dtype=np.int64)
    mid = target_quotes["mid"].to_numpy(dtype=float)
    segment = target_quotes["segment"].to_numpy(dtype=int) if "segment" in target_quotes else np.zeros(len(qt), dtype=int)
    ends = times + int(horizon_ms * 1e6)
    before = np.searchsorted(qt, times, side="left") - 1
    after = np.searchsorted(qt, ends, side="right") - 1
    b, a = np.maximum(before, 0), np.maximum(after, 0)
    eligible = (before >= 0) & (after >= 0) & (ends <= qt[-1])
    eligible &= np.isfinite(mid[b]) & np.isfinite(mid[a]) & (mid[b] > 0)
    eligible &= (times - qt[b] <= max_age_ms * 1e6) & (ends - qt[a] <= max_age_ms * 1e6)
    # A recovered valid quote cannot bridge an intervening invalid state.
    eligible &= segment[b] == segment[a]
    if "valid" in target_quotes:
        valid = target_quotes["valid"].to_numpy(dtype=bool)
        eligible &= valid[b] & valid[a]
    else:
        valid = np.isfinite(mid)
    invalid_prefix = np.r_[0, np.cumsum(~valid | ~np.isfinite(mid))]
    eligible &= (invalid_prefix[a + 1] - invalid_prefix[b]) == 0
    out["eligible"] = eligible
    out["markout"][eligible] = directions[eligible] * (mid[a[eligible]] / mid[b[eligible]] - 1) * 1e4
    if not target_shocks.empty:
        for sign in (-1, 1):
            st = target_shocks.loc[target_shocks["direction"] == sign, "time_ns"].to_numpy(dtype=np.int64)
            if not len(st):
                continue
            # Exclude tied/immediate events and the full clock uncertainty band.
            index = np.searchsorted(st, times + int(min_lead_ms * 1e6), side="right")
            safe = np.minimum(index, len(st) - 1)
            hit = eligible & (directions == sign) & (index < len(st)) & (st[safe] <= ends)
            out["hit"] |= hit
            out["delay"][hit] = (st[safe[hit]] - times[hit]) / 1e6
    return out


def _summarize_arrays(arrays: dict[str, np.ndarray], *, matched: np.ndarray | None = None) -> dict:
    eligible = arrays["eligible"].copy()
    if matched is not None:
        eligible &= matched
    hits = arrays["hit"] & eligible
    count, nhit = int(eligible.sum()), int(hits.sum())
    return {
        "source_count": len(eligible), "eligible_count": count, "hit_count": nhit,
        "response_probability": nhit / count if count else np.nan,
        "signed_markout_bps": float(np.mean(arrays["markout"][eligible])) if count else np.nan,
        "median_reaction_delay_ms": float(np.median(arrays["delay"][hits])) if nhit else np.nan,
    }


def edge_response(
    source_shocks: pd.DataFrame, target_quotes: pd.DataFrame, target_shocks: pd.DataFrame, *,
    horizon_ms: float, shift_ms: float = 0, min_lead_ms: float = MIN_LEAD_MS,
    max_age_ms: float = MAX_AGE_MS,
) -> dict:
    """New same-direction shock probability and target midpoint markout.

    Positive shift_ms evaluates target activity later in the same session;
    equivalently it shifts target timestamps backward. No circular wrap occurs.
    """
    return _summarize_arrays(_edge_arrays(
        source_shocks, target_quotes, target_shocks, horizon_ms=horizon_ms,
        shift_ms=shift_ms, min_lead_ms=min_lead_ms, max_age_ms=max_age_ms,
    ))


def _mbo_activity(frame: pd.DataFrame, dataset: str, day: str) -> list[dict]:
    data = frame.copy()
    if "symbol" not in data:
        data["symbol"] = "UNKNOWN"
    data["symbol"] = _text(data["symbol"])
    flags = _number(data, "flags", 0)
    bits = np.where(np.isfinite(flags), flags, 0).astype(np.uint64)
    valid = np.isfinite(flags) & ((bits & (F_SNAPSHOT | F_BAD_TS_RECV | F_MAYBE_BAD_BOOK)) == 0)
    data = data.loc[valid].copy()
    data["action"] = _text(data["action"]) if "action" in data else "UNKNOWN"
    rows = []
    for (symbol, action), group in data.groupby(["symbol", "action"], sort=True):
        rows.append({"dataset": dataset, "day": day, "symbol": symbol,
                     "action": action, "order_event_records": len(group),
                     "note": "Observed non-snapshot MBO records; no full-depth book reconstruction."})
    return rows


def _held_out_support(edges: pd.DataFrame) -> list[dict]:
    held = edges.loc[edges["split"] == "held_out"]
    result = []
    for (horizon, symbol), group in held.groupby(["horizon_ms", "symbol"], sort=True):
        total = int(held.loc[held["horizon_ms"] == horizon, "matched_count"].sum())
        count = int(group["matched_count"].sum())
        result.append({"symbol": symbol, "horizon_ms": int(horizon),
                       "matched_edge_event_count": count,
                       "matched_support_share": count / total if total else None,
                       "observed_hits": int(group["observed_hits"].sum()),
                       "placebo_mean_hits": float(group["placebo_mean_hits"].sum())})
    return result


def _response_totals(held: pd.DataFrame) -> dict:
    total = int(held["matched_count"].sum())
    hits = int(held["observed_hits"].sum())
    placebo_hits = float(held["placebo_mean_hits"].sum())
    return {"matched_edge_event_count": total, "hits": hits,
            "placebo_mean_hits": placebo_hits,
            "response_probability": hits / total if total else None,
            "placebo_response_probability": placebo_hits / total if total else None,
            "excess_probability_pp": (hits - placebo_hits) / total * 100 if total else None}


def _plot(edges: pd.DataFrame, directory: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    held = edges.loc[edges["split"] == "held_out"]
    if held.empty:
        return
    support = [row for row in _held_out_support(edges) if row["horizon_ms"] == 100]
    support.sort(key=lambda row: row["matched_edge_event_count"], reverse=True)
    counts = "; ".join(
        f"{row['symbol']} {row['matched_support_share'] * 100:.1f}% (N={row['matched_edge_event_count']:,})"
        if row["matched_support_share"] is not None else f"{row['symbol']} N=0" for row in support
    )
    caption = ("100 ms matched support: " + counts + ".\n"
               "N counts edge-events. Excludes first 2 ms; 100 ms shock de-duplication. Negative excess does not imply inhibition.")
    views = held.groupby(["source", "target", "horizon_ms"], sort=True).agg(
        hits=("observed_hits", "sum"), eligible=("matched_count", "sum"),
        placebo_hits=("placebo_mean_hits", "sum"),
    ).reset_index()
    views["excess_pp"] = (views["hits"] - views["placebo_hits"]) / views["eligible"].replace(0, np.nan) * 100
    wider = len(views[["source", "target"]].drop_duplicates()) > 6
    fig, ax = plt.subplots(figsize=(13 if wider else 10, 7.2 if wider else 6.2), constrained_layout=True)
    for (source, target), group in views.groupby(["source", "target"]):
        source_label, target_label = VENUE_LABELS.get(source, source), VENUE_LABELS.get(target, target)
        ax.plot(group["horizon_ms"], group["excess_pp"], marker="o", label=f"{source_label} → {target_label}")
    ax.axhline(0, color="#777777", linewidth=1)
    ax.set(xscale="log", xlabel="Response horizon (ms; lead must exceed 2 ms)",
           ylabel="Observed minus shifted response probability (percentage points)",
           title="Held-out quote-shock response, weighted by matched support")
    if wider:
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, .5))
    else:
        ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.2)
    fig.supxlabel(caption, fontsize=8)
    fig.savefig(directory / "response_curve.png", dpi=160)
    plt.close(fig)
    h = views.loc[views["horizon_ms"] == 100]
    venues = sorted(set(edges["source"]) | set(edges["target"]))
    values = np.full((len(venues), len(venues)), np.nan)
    sample_sizes = np.zeros_like(values, dtype=int)
    for row in h.itertuples():
        values[venues.index(row.source), venues.index(row.target)] = row.excess_pp
        sample_sizes[venues.index(row.source), venues.index(row.target)] = row.eligible
    scale = max(.1, np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1
    fig, ax = plt.subplots(figsize=(10.5 if len(venues) > 3 else 9.5,
                                    7.4 if len(venues) > 3 else 6.2), constrained_layout=True)
    plot = ax.imshow(values, cmap="RdBu_r", vmin=-scale, vmax=scale)
    labels = [VENUE_LABELS.get(venue, venue) for venue in venues]
    ax.set_xticks(range(len(venues)), labels)
    ax.set_yticks(range(len(venues)), labels)
    ax.set(xlabel="Responding venue", ylabel="Source venue", title="100 ms held-out association, weighted by matched support")
    for i in range(len(venues)):
        for j in range(len(venues)):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:+.2f} pp\nN={sample_sizes[i, j]:,}", ha="center", va="center",
                        color="white" if abs(values[i, j]) > scale * .55 else "black")
            elif i != j:
                ax.text(j, i, "No support\nN=0", ha="center", va="center")
    fig.colorbar(plot, ax=ax, label="Response probability excess over timing placebos (pp)")
    fig.supxlabel(caption, fontsize=8)
    fig.savefig(directory / "venue_network.png", dpi=160)
    plt.close(fig)


def refresh_presentation(outdir: Path) -> dict:
    """Refresh plots and support disclosure from existing outputs without refitting."""
    outdir = Path(outdir)
    edges = pd.read_csv(outdir / "edge_response.csv")
    _plot(edges, outdir)
    path = outdir / "network_summary.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result["held_out_support_by_symbol"] = _held_out_support(edges)
    result["held_out_support_by_day_symbol"] = [
        {"day": day, **row} for day, group in edges.loc[edges["split"] == "held_out"].groupby("day", sort=True)
        for row in _held_out_support(group)]
    result["held_out_100ms_by_day"] = [
        {"day": day, **_response_totals(group)}
        for day, group in edges.loc[(edges["split"] == "held_out") & (edges["horizon_ms"] == 100)].groupby("day", sort=True)]
    if REFRACTORY_CAVEAT not in result["limitations"]:
        result["limitations"].append(REFRACTORY_CAVEAT)
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def run_network(
    frames: dict[tuple[str, str, str], pd.DataFrame], outdir: Path, *, day_order: list[str] | None = None,
    held_out_days: list[str] | None = None,
) -> dict:
    """Fit thresholds on earlier days; measure associations on later whole dates."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    quote_groups: dict[tuple[str, str, str], pd.DataFrame] = {}
    quality, activity, clocks = [], [], set()
    for (dataset, schema, day), frame in sorted(frames.items()):
        if schema == "mbo":
            activity.extend(_mbo_activity(frame, dataset, day))
        if schema != "mbp-1":
            continue
        quotes, info = normalize_quotes(frame, dataset, day)
        clocks.add(info["clock"])
        for symbol, group in quotes.groupby("symbol", sort=True):
            group = group.reset_index(drop=True)
            quote_groups[(dataset, str(symbol), day)] = group
            quality.append({**info, "symbol": symbol, "symbol_state_records": len(group),
                            "symbol_invalid_state_records": int((~group["valid"]).sum()),
                            "coverage_start_ns": int(group["time_ns"].min()),
                            "coverage_end_ns": int(group["time_ns"].max())})
    if len(clocks) > 1:
        raise ValueError("Mixed analysis clocks across feeds are not permitted.")
    days = sorted({key[2] for key in quote_groups}) if day_order is None else list(day_order)
    if set(days) != {key[2] for key in quote_groups}:
        raise ValueError("day_order must contain exactly the observed quote dates.")
    training_days, held_out_days = chronological_split(days, held_out_days)
    held_out_day = held_out_days[0] if len(held_out_days) == 1 else None
    thresholds, shock_groups, event_frames = [], {}, []
    for dataset, symbol in sorted({key[:2] for key in quote_groups}):
        training = [quote_groups[(dataset, symbol, day)] for day in training_days if (dataset, symbol, day) in quote_groups]
        if not training:
            raise ValueError(f"No training quote coverage for {dataset}/{symbol}.")
        train = pd.concat(training, ignore_index=True)
        threshold = fit_threshold(train)
        thresholds.append({"dataset": dataset, "symbol": symbol, "threshold_usd": threshold,
                           "training_days": ",".join(training_days), "held_out_day": held_out_day,
                           "held_out_days": ",".join(held_out_days)})
        for day in days:
            key = dataset, symbol, day
            if key not in quote_groups:
                continue
            shocks = extract_shocks(quote_groups[key], threshold)
            shocks["split"] = "held_out" if day in held_out_days else "training"
            shocks["threshold"] = threshold
            shock_groups[key] = shocks
            event_frames.append(shocks)
    edge_rows, shifted_rows, sensitivity_rows = [], [], []
    for symbol in sorted({key[1] for key in quote_groups}):
        for day in days:
            venues = sorted(key[0] for key in quote_groups if key[1:] == (symbol, day))
            for source, target in permutations(venues, 2):
                source_events = shock_groups[(source, symbol, day)]
                target_events = shock_groups[(target, symbol, day)]
                target_quotes = quote_groups[(target, symbol, day)]
                identity = {"symbol": symbol, "day": day, "split": "held_out" if day in held_out_days else "training",
                            "source": source, "target": target}
                for horizon in HORIZONS_MS:
                    arrays = [_edge_arrays(source_events, target_quotes, target_events, horizon_ms=horizon, shift_ms=shift)
                              for shift in (0, *PLACEBO_SHIFTS_MS)]
                    matched = np.logical_and.reduce([array["eligible"] for array in arrays])
                    summaries = [_summarize_arrays(array, matched=matched) for array in arrays]
                    observed, placebos = summaries[0], summaries[1:]
                    placebo_hits = float(np.mean([row["hit_count"] for row in placebos]))
                    count = observed["eligible_count"]
                    probability = observed["response_probability"]
                    placebo_probability = placebo_hits / count if count else None
                    edge_rows.append({**identity, "horizon_ms": horizon, "min_lead_ms": MIN_LEAD_MS,
                        "source_count": len(source_events), "matched_count": count,
                        "observed_eligible_before_matching": int(arrays[0]["eligible"].sum()),
                        "observed_hits": observed["hit_count"], "response_probability": probability,
                        "placebo_mean_hits": placebo_hits, "placebo_response_probability": placebo_probability,
                        "excess_response_probability": probability - placebo_probability if count else None,
                        "signed_markout_bps": observed["signed_markout_bps"],
                        "placebo_signed_markout_bps": float(np.mean([row["signed_markout_bps"] for row in placebos])) if count else None,
                        "median_reaction_delay_ms": observed["median_reaction_delay_ms"],
                    })
                    for shift, result in zip((0, *PLACEBO_SHIFTS_MS), summaries):
                        shifted_rows.append({**identity, "horizon_ms": horizon, "source_trigger_shift_ms": shift,
                                             "matched_denominator": True, **result})
                    # Excluding the first 10 ms probes sensitivity to receive-feed transport delays.
                    if horizon > 10:
                        arrays_10 = [_edge_arrays(source_events, target_quotes, target_events, horizon_ms=horizon,
                                                 shift_ms=shift, min_lead_ms=10) for shift in (0, *PLACEBO_SHIFTS_MS)]
                        results = [_summarize_arrays(array, matched=matched) for array in arrays_10]
                        mean_hits = float(np.mean([row["hit_count"] for row in results[1:]]))
                        sensitivity_rows.append({**identity, "horizon_ms": horizon, "min_lead_ms": 10,
                            "matched_count": count, "observed_hits": results[0]["hit_count"],
                            "placebo_mean_hits": mean_hits,
                            "excess_response_probability": (results[0]["hit_count"] - mean_hits) / count if count else None})
    edges = pd.DataFrame(edge_rows)
    if edges.empty:
        raise ValueError("No symbol/date has overlapping quote coverage from at least two venues.")
    pd.DataFrame(quality).to_csv(outdir / "quote_quality.csv", index=False)
    pd.DataFrame(thresholds).to_csv(outdir / "shock_thresholds.csv", index=False)
    pd.concat(event_frames, ignore_index=True).to_csv(outdir / "shock_events.csv", index=False)
    pd.DataFrame(activity, columns=["dataset", "day", "symbol", "action", "order_event_records", "note"]).to_csv(outdir / "mbo_activity.csv", index=False)
    edges.to_csv(outdir / "edge_response.csv", index=False)
    pd.DataFrame(shifted_rows).to_csv(outdir / "timing_placebos.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(outdir / "clock_sensitivity.csv", index=False)
    _plot(edges, outdir)
    held = edges.loc[(edges["split"] == "held_out") & (edges["horizon_ms"] == 100)]
    result = {
        "experiment": "receive-clock signed quote-shock venue association pilot",
        "training_days": training_days, "held_out_day": held_out_day, "held_out_days": held_out_days,
        "analysis_clock": next(iter(clocks)),
        "venues": sorted({key[0] for key in quote_groups}), "symbols": sorted({key[1] for key in quote_groups}),
        "quote_groups": len(quote_groups), "shock_events": int(sum(len(value) for value in shock_groups.values())),
        "horizons_ms": list(HORIZONS_MS), "min_lead_ms": MIN_LEAD_MS, "max_quote_age_ms": MAX_AGE_MS,
        "source_min_separation_ms": MIN_SEPARATION_MS, "placebo_source_trigger_shifts_ms": list(PLACEBO_SHIFTS_MS),
        "held_out_100ms": _response_totals(held),
        "held_out_100ms_by_day": [{"day": day, **_response_totals(held.loc[held["day"] == day])}
                                  for day in held_out_days],
        "held_out_support_by_symbol": _held_out_support(edges),
        "held_out_support_by_day_symbol": [
            {"day": day, **row} for day in held_out_days
            for row in _held_out_support(edges.loc[edges["day"] == day])],
        "threshold_rule": "Per venue/symbol max($0.005, training-only 75th percentile absolute nonzero adjacent midpoint jump).",
        "matching_rule": "Each edge/horizon uses the same source events eligible at observed and all three shifted trigger times; no circular wrapping.",
        "limitations": [
            "Temporal association is not causal influence, participant identity, an SIR rate, or an estimated Hawkes model.",
            "Receive-clock ordering includes venue/feed transport delay; reported reaction delays are observed co-occurrence delays, not infrastructure latencies.",
            "Shared market information, correlated order flow and omitted venues can explain associations.",
            "A limited collection of intraday windows does not establish generalization across market regimes.",
            "Source de-duplication reduces overlap but events remain dependent; no independent-event significance tests or causal p-values are reported.",
            REFRACTORY_CAVEAT,
            "Markouts describe target midpoint movement from strictly before the source trigger; early movements inside the excluded clock band remain in markouts.",
            "MBO action counts describe observed deltas only; no queues or full-depth liquidity have been reconstructed.",
            "Public volume, maker opportunity and retail improvement are not predictors in this pilot; later integration requires publication-aware alignment.",
        ],
        "outputs": {name: str((outdir / name).resolve()) for name in (
            "edge_response.csv", "timing_placebos.csv", "clock_sensitivity.csv", "shock_events.csv",
            "shock_thresholds.csv", "quote_quality.csv", "mbo_activity.csv", "venue_network.png", "response_curve.png")},
    }
    (outdir / "network_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


run_network_pilot = run_network
