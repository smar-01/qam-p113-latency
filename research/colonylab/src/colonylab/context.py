"""Offline descriptive context from Cboe's public aggregate reports.

Reporting centers are exchanges or FINRA trade-reporting facilities, never
trader identities. No report in this module identifies an EQUS.MINI print.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .config import write_json


HISTORY_URL = "https://cdn.cboe.com/resources/us/equities/market-statistics/historical-market-volume/market_history_{year}.csv"
HISTORY_PAGE = "https://www.cboe.com/markets/us/equities/market-statistics/historical-market-volume"
MAKER_URL = "https://www.cboe.com/us/equities/market_statistics/maker_report/csv/?book=all&mkt=bzx"
MAKER_PAGE = "https://www.cboe.com/us/equities/market_statistics/maker_report/"
RPI_PAGE = "https://www.cboe.com/markets/us/equities/market-statistics/retail-price-improvement/daily"
TAPES = ("A", "B", "C", "ALL")
VENUE_METRICS = (
    "total_shares", "total_notional_usd", "total_trade_count",
    "reporting_center_count", "exchange_count", "trf_count", "off_exchange_shares",
    "off_exchange_share_fraction", "reporting_center_volume_hhi", "exchange_only_volume_hhi",
)
MAKER_FIELDS = {
    "Missed Liquidity": "missed_liquidity_shares_adv",
    "Exhausted Liquidity": "exhausted_liquidity_shares_adv",
    "Routed Liquidity": "routed_liquidity_shares_adv",
    "Volume Opportunity": "volume_opportunity_shares_adv",
    "Cboe ADV": "cboe_shares_adv",
    "Liquidity Opportunity": "liquidity_opportunity_ratio",
}
RPI_FIELDS = (
    "retail_shares_submitted", "retail_shares_executed", "missed_retail_liquidity",
    "exhausted_retail_liquidity", "shares_exec_div_bats_vol", "median_order_size",
    "price_improvement", "effective_spread",
)


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    # NA is a legitimate equity ticker, not a missing-value sentinel.
    frame = pd.read_csv(path, keep_default_na=False)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing Cboe columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("Cboe report contains no rows.")
    return frame


def _nonnegative(frame: pd.DataFrame, columns) -> None:
    for name in columns:
        values = pd.to_numeric(frame[name], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all() or (values < 0).any():
            raise ValueError(f"Cboe field {name!r} must be finite and nonnegative.")
        frame[name] = values


def _identifiers(frame: pd.DataFrame, columns) -> None:
    for name in columns:
        if frame[name].isna().any() or frame[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"Cboe identifier {name!r} must be nonempty.")
        frame[name] = frame[name].astype(str).str.strip()


def _records(frame: pd.DataFrame) -> list[dict]:
    # pandas' JSON conversion safely converts NaN/NaT to JSON null.
    return json.loads(frame.to_json(orient="records", double_precision=15))


def _provenance(path: Path, fallback_url: str | None) -> dict:
    source = {"local_path": str(path.resolve()), "url": fallback_url}
    manifest = path.parent / "sources.json"
    if manifest.exists():
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        if isinstance(entries, dict) and isinstance(entries.get(path.name), dict):
            source.update(entries[path.name])
    source["source_publication_timestamp_known"] = False
    source["point_in_time_archive"] = False
    return source


def _fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def build_venue_context(csv_path: Path, session_date: str, output_dir: Path) -> dict:
    """Validate an annual Cboe CSV and produce descriptive daily context.

    HHI values use fractions and range from 0 to 1. ``lag1`` means the previous
    date *observed in this file*, not a verified publication-available predictor.
    """
    csv_path, output_dir = Path(csv_path), Path(output_dir)
    try:
        target = pd.Timestamp(session_date)
        if pd.isna(target):
            raise ValueError
        target_date = target.date().isoformat()
    except (ValueError, TypeError) as exc:
        raise ValueError("session_date must identify a valid calendar date.") from exc
    columns = [f"Tape {tape} {metric}" for tape in "ABC" for metric in ("Shares", "Notional", "Trade Count")]
    columns += [f"Total {metric}" for metric in ("Shares", "Notional", "Trade Count")]
    source = _read_csv(csv_path, {"Day", "Market Participant", *columns})
    _identifiers(source, ("Day", "Market Participant"))
    dates = pd.to_datetime(source["Day"], format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        raise ValueError("Cboe Day contains invalid dates.")
    source["Day"] = dates.dt.strftime("%Y-%m-%d")
    if source.duplicated(["Day", "Market Participant"]).any():
        raise ValueError("Duplicate date/participant in Cboe historical report.")
    _nonnegative(source, columns)
    for metric in ("Shares", "Notional", "Trade Count"):
        total = source[f"Total {metric}"].to_numpy(dtype=float)
        parts = source[[f"Tape {tape} {metric}" for tape in "ABC"]].sum(axis=1).to_numpy(dtype=float)
        tolerance = (0, 0) if metric == "Trade Count" else (1e-9, 1e-4)
        if not np.isclose(total, parts, rtol=tolerance[0], atol=tolerance[1]).all():
            raise ValueError(f"Cboe Total {metric} does not match the sum of its tapes.")
    count_columns = [column for column in columns if "Trade Count" in column]
    if any(not np.equal(source[column], np.floor(source[column])).all() for column in count_columns):
        raise ValueError("Cboe trade counts must be integers.")
    # FINRA TRFs report off-exchange prints; these are not execution venues.
    trf = source["Market Participant"].str.contains(r"\bTRF\b|\bFINRA\b", case=False, regex=True)
    source["reporting_center_type"] = np.where(trf, "trf_reporting_facility", "exchange")
    blocks = []
    for tape in TAPES:
        prefix = "Total" if tape == "ALL" else f"Tape {tape}"
        block = source[["Day", "Market Participant", "reporting_center_type"]].rename(columns={"Day": "date", "Market Participant": "market_participant"}).copy()
        block["tape"] = tape
        for source_metric, output_metric in (("Shares", "shares"), ("Notional", "notional_usd"), ("Trade Count", "trade_count")):
            block[output_metric] = source[f"{prefix} {source_metric}"].to_numpy()
        blocks.append(block)
    shares = pd.concat(blocks, ignore_index=True)
    for value, name in (("shares", "volume_share_fraction"), ("notional_usd", "notional_share_fraction"), ("trade_count", "trade_share_fraction")):
        denominator = shares.groupby(["date", "tape"])[value].transform("sum")
        shares[name] = shares[value] / denominator.where(denominator > 0)
    rows = []
    for (date, tape), group in shares.groupby(["date", "tape"], sort=True):
        exchange = group[group["reporting_center_type"] == "exchange"]
        facilities = group[group["reporting_center_type"] == "trf_reporting_facility"]
        volume = float(group["shares"].sum())
        exchange_volume = float(exchange["shares"].sum())
        rows.append({
            "date": date, "tape": tape,
            "total_shares": volume,
            "total_notional_usd": float(group["notional_usd"].sum()),
            "total_trade_count": int(group["trade_count"].sum()),
            "reporting_center_count": len(group), "exchange_count": len(exchange), "trf_count": len(facilities),
            "off_exchange_shares": float(facilities["shares"].sum()),
            "off_exchange_share_fraction": _fraction(float(facilities["shares"].sum()), volume),
            "reporting_center_volume_hhi": float(np.square(group["shares"] / volume).sum()) if volume else np.nan,
            "exchange_only_volume_hhi": float(np.square(exchange["shares"] / exchange_volume).sum()) if exchange_volume else np.nan,
        })
    daily = pd.DataFrame(rows).sort_values(["tape", "date"]).reset_index(drop=True)
    by_tape = daily.groupby("tape", sort=False)
    daily["lag1_date"] = by_tape["date"].shift(1)
    daily["lag1_available"] = daily["lag1_date"].notna()
    for metric in VENUE_METRICS:
        daily[f"lag1_{metric}"] = by_tape[metric].shift(1)
    daily["lag1_calendar_gap_days"] = (pd.to_datetime(daily["date"]) - pd.to_datetime(daily["lag1_date"])).dt.days
    daily = daily.sort_values(["date", "tape"]).reset_index(drop=True)
    snapshot = shares[shares["date"] == target_date].sort_values(["tape", "volume_share_fraction"], ascending=[True, False])
    if snapshot.empty:
        raise ValueError(f"No Cboe historical-volume observations for {target_date}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "daily_context": output_dir / "daily_context.csv",
        "daily_market_shares": output_dir / "daily_market_shares.csv",
        "venue_snapshot": output_dir / "venue_snapshot.csv",
        "market_share_plot": output_dir / "venue_market_share.png",
        "summary": output_dir / "venue_context.json",
    }
    daily.to_csv(paths["daily_context"], index=False)
    shares.sort_values(["date", "tape", "market_participant"]).to_csv(paths["daily_market_shares"], index=False)
    snapshot.to_csv(paths["venue_snapshot"], index=False)
    _plot_venues(snapshot[snapshot["tape"] == "ALL"], target_date, paths["market_share_plot"])
    years = sorted(set(source["Day"].str[:4]))
    result = {
        "session_date": target_date, "created_utc": datetime.now(timezone.utc).isoformat(),
        "date_start": source["Day"].min(), "date_end": source["Day"].max(),
        "observed_dates": int(source["Day"].nunique()), "source_rows": len(source),
        "source": _provenance(csv_path, HISTORY_URL.format(year=years[0]) if len(years) == 1 else HISTORY_PAGE),
        "context_snapshot": _records(daily[daily["date"] == target_date]),
        "output_files": {name: str(path.resolve()) for name, path in paths.items()},
        "definitions": {
            "shares": "Shares traded, including fractional shares where present; not thousands or millions.",
            "notional_usd": "Sum of execution price times executed shares, in US dollars.",
            "hhi": "Sum of squared share-volume fractions, on the 0–1 scale.",
            "reporting_center_volume_hhi": "Concentration across exchanges and TRFs; a TRF may combine many execution firms.",
            "exchange_only_volume_hhi": "Concentration among exchange-reported share volume, renormalized to exchange-only volume.",
            "off_exchange_share_fraction": "Share volume reported through FINRA facilities divided by all reported shares; not a retail fraction.",
            "lag1": "Previous observed date per tape in this file; first date has null lags. Calendar gaps are reported.",
        },
        "limitations": [
            "Daily market-wide totals by reporting center and listing tape, without security or intraday timestamps.",
            "Market Participant labels designate reporting centers, not investors, brokers, funds or market makers.",
            "Tape A/B/C are listing-network categories, not the venue on which an execution occurred.",
            "The EQUS.MINI pilot is an anonymized component-feed subset over ten minutes; these daily consolidated totals do not validate its venue attribution or completeness.",
            "Same-session context is retrospective descriptive context only; it must not be an intraday model predictor.",
            "Lagged values establish date ordering only. Publication timestamps and historical revisions are unknown, so point-in-time availability is not established.",
            "Missing dates are not imputed. The previous observed date can span missing reports; inspect lag1_calendar_gap_days.",
        ],
    }
    write_json(paths["summary"], result)
    return result


def _plot_venues(snapshot: pd.DataFrame, session_date: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = snapshot.sort_values("volume_share_fraction")
    colors = np.where(data["reporting_center_type"] == "exchange", "#267a9c", "#d89436")
    fig, ax = plt.subplots(figsize=(11, max(5, len(data) * .29)))
    ax.barh(data["market_participant"], data["volume_share_fraction"] * 100, color=colors)
    ax.set(xlabel="Share of consolidated daily share volume (%)", title=f"Reporting-center participation · {session_date} · all tapes")
    ax.text(0, -.13, "Blue: exchanges. Orange: FINRA reporting facilities (off-exchange, not retail identities).\nSource: Cboe, UTDF/CTS. Daily market-wide context; no trade-level attribution.", transform=ax.transAxes, fontsize=8, va="top")
    fig.subplots_adjust(left=.32, bottom=.18, right=.97, top=.92)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _load_maker_report(maker_csv: Path) -> pd.DataFrame:
    maker = _read_csv(maker_csv, {"Symbol", *MAKER_FIELDS})
    _identifiers(maker, ("Symbol",))
    if maker["Symbol"].duplicated().any():
        raise ValueError("Duplicate symbol in Cboe maker report.")
    _nonnegative(maker, [column for column in MAKER_FIELDS if column != "Liquidity Opportunity"])
    ratio = pd.to_numeric(maker["Liquidity Opportunity"], errors="coerce")
    undefined_ratio = maker["Liquidity Opportunity"].astype(str).str.strip().eq("") & maker["Cboe ADV"].eq(0)
    invalid_ratio = (~np.isfinite(ratio.to_numpy(dtype=float)) | (ratio < 0)) & ~undefined_ratio
    if invalid_ratio.any():
        raise ValueError("Cboe Liquidity Opportunity must be finite and nonnegative, or blank with zero ADV.")
    maker["Liquidity Opportunity"] = ratio
    maker = maker.rename(columns={"Symbol": "symbol", **MAKER_FIELDS})
    maker["liquidity_opportunity_percent"] = maker["liquidity_opportunity_ratio"] * 100
    maker["zero_adv"] = maker["cboe_shares_adv"] == 0
    # The raw ratio is meaningless where its ADV denominator is zero. Preserve
    # it as source evidence but exclude those rows from cross-symbol rankings.
    maker["liquidity_opportunity_percentile"] = maker["liquidity_opportunity_ratio"].where(~maker["zero_adv"]).rank(method="average", pct=True) * 100
    return maker


def _load_rpi_report(rpi_csv: Path) -> pd.DataFrame:
    rpi = _read_csv(rpi_csv, {"symbol", "side", *RPI_FIELDS})
    _identifiers(rpi, ("symbol", "side"))
    if rpi.duplicated(["symbol", "side"]).any():
        raise ValueError("Duplicate symbol/side in Cboe RPI report.")
    _nonnegative(rpi, RPI_FIELDS)
    rpi["executed_to_submitted_fraction"] = rpi["retail_shares_executed"] / rpi["retail_shares_submitted"].where(rpi["retail_shares_submitted"] > 0)
    rpi["has_executions"] = rpi["retail_shares_executed"] > 0
    rpi["price_improvement_usd_per_executed_share"] = rpi["price_improvement"].where(rpi["has_executions"])
    rpi["effective_spread_reported_when_executed"] = rpi["effective_spread"].where(rpi["has_executions"])
    return rpi


def _rpi_report_date(rpi_csv: Path) -> str | None:
    date_match = re.search(r"(?<!\d)(\d{8})(?!\d)", rpi_csv.name)
    report_date = None
    if date_match:
        date = pd.to_datetime(date_match[1], format="%Y%m%d", errors="coerce")
        if not pd.isna(date):
            report_date = date.date().isoformat()
    return report_date


def _maker_report_period(maker_csv: Path) -> dict | None:
    periods_path = maker_csv.parent / "report_periods.json"
    if periods_path.exists():
        periods = json.loads(periods_path.read_text(encoding="utf-8"))
        if isinstance(periods, dict) and isinstance(periods.get(maker_csv.name), dict):
            entry = periods[maker_csv.name]
            if entry.get("period_start") and entry.get("period_end"):
                start = pd.Timestamp(entry["period_start"])
                end = pd.Timestamp(entry["period_end"])
                if pd.isna(start) or pd.isna(end) or end < start:
                    raise ValueError("Invalid explicit maker reporting period.")
                return {"period_start": start.date().isoformat(), "period_end": end.date().isoformat(), "source_page": entry.get("source_page"), "origin": "explicit local report_periods.json; not inferred from CSV"}
    return None


def build_maker_context(maker_csv: Path, rpi_csv: Path, output_dir: Path, symbol: str = "AAPL") -> dict:
    """Preserve Cboe maker-opportunity and RPI report evidence by symbol.

    Maker figures are weekly average daily shares. RPI rows retain their source
    side labels. Neither report labels the investor behind an anonymous print.
    """
    maker_csv, rpi_csv, output_dir = Path(maker_csv), Path(rpi_csv), Path(output_dir)
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("A nonempty symbol is required for maker context.")
    symbol = symbol.strip().upper()
    maker, rpi = _load_maker_report(maker_csv), _load_rpi_report(rpi_csv)
    report_date = _rpi_report_date(rpi_csv)
    maker_snapshot = maker[maker["symbol"].str.upper() == symbol].copy()
    rpi_snapshot = rpi[rpi["symbol"].str.upper() == symbol].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "maker_context": output_dir / "maker_context.csv", "rpi_context": output_dir / "rpi_context.csv",
        "maker_snapshot": output_dir / "maker_snapshot.csv", "rpi_snapshot": output_dir / "rpi_snapshot.csv",
        "maker_retail_plot": output_dir / "maker_retail_context.png", "summary": output_dir / "maker_context.json",
    }
    for name, frame in (("maker_context", maker), ("rpi_context", rpi), ("maker_snapshot", maker_snapshot), ("rpi_snapshot", rpi_snapshot)):
        frame.to_csv(paths[name], index=False)
    _plot_maker(maker_snapshot, rpi_snapshot, symbol, paths["maker_retail_plot"])
    rpi_url = None if report_date is None else "https://cdn.cboe.com/resources/us/equities/market-statistics/retail-price-improvement/daily/rpt_rpi_maker_opportunity_" + report_date.replace("-", "") + ".csv"
    maker_period = _maker_report_period(maker_csv)
    result = {
        "symbol": symbol, "rpi_session_date": report_date,
        "maker_source_rows": len(maker), "rpi_source_rows": len(rpi),
        "maker_symbol_found": not maker_snapshot.empty, "rpi_symbol_found": not rpi_snapshot.empty,
        "maker_snapshot": _records(maker_snapshot), "rpi_snapshot": _records(rpi_snapshot),
        "selected_maker": _records(maker_snapshot)[0] if not maker_snapshot.empty else None,
        "maker_period": maker_period,
        "maker_percentile_population": {"valid_positive_adv_symbols": int((~maker["zero_adv"]).sum()), "excluded_zero_adv_symbols": int(maker["zero_adv"].sum()), "method": "Average-rank empirical percentile of raw opportunity ratio among symbols with positive ADV; 0–100 scale."},
        "sources": {"maker": _provenance(maker_csv, MAKER_URL), "rpi": _provenance(rpi_csv, rpi_url or RPI_PAGE)},
        "output_files": {name: str(path.resolve()) for name, path in paths.items()},
        "definitions": {
            "maker_share_units": "Average daily shares over the reporting week, not observed ten-minute or single-day volume.",
            "liquidity_opportunity_ratio": "Raw CSV ratio; the web page displays this value multiplied by 100 as a percentage.",
            "executed_to_submitted_fraction": "Executed divided by submitted shares within the same reported symbol/side, not a market-wide retail participation fraction.",
            "rpi_side": "Preserved source labels B/S/T/E; no source-side aggregation is performed.",
            "rpi_price_improvement": "Raw value retained. Derived dollars-per-executed-share field is null without executions; source: Cboe's May 2024 RPI article, footnote 7.",
            "rpi_effective_spread": "Raw value retained; derived field is null without executions. Report-specific units and weighting were not verified.",
        },
        "limitations": [
            "Maker opportunity measures liquidity demand at a Cboe venue and potential liquidity-provider opportunity; it does not identify market makers.",
            "Maker report period is not encoded in its CSV. Use separately archived source-page metadata to establish its week; latest-file URLs can change.",
            "RPI opportunity is specific to the reporting program and includes submitted/unexecuted demand; it is not all retail trading.",
            "Submitted minus executed shares need not equal missed plus exhausted liquidity; no accounting identity or all-side totals are imposed.",
            "Retail flow observations do not reveal the identity or investment style of individual investors.",
            "Daily RPI and weekly maker observations may include activity after the pilot window. They are retrospective descriptive context only, never intraday predictors.",
            "Report publication timestamps and historical revisions are unknown; point-in-time availability is not established.",
            "Reports can attach symbol/date context to event rows but do not match or identify individual executions and are not validated colony labels.",
        ],
    }
    write_json(paths["summary"], result)
    return result


def _plot_maker(maker: pd.DataFrame, rpi: pd.DataFrame, symbol: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if maker.empty:
        axes[0].text(.5, .5, "No symbol in maker report", ha="center", transform=axes[0].transAxes)
    else:
        row = maker.iloc[0]
        names = ["Missed", "Exhausted", "Routed", "Cboe matched ADV"]
        values = [row["missed_liquidity_shares_adv"], row["exhausted_liquidity_shares_adv"], row["routed_liquidity_shares_adv"], row["cboe_shares_adv"]]
        axes[0].barh(names, values, color=["#d89436", "#d89436", "#d89436", "#267a9c"])
    axes[0].set(title=f"{symbol} · BZX maker opportunity", xlabel="Weekly average daily shares")
    if rpi.empty:
        axes[1].text(.5, .5, "No symbol in RPI report", ha="center", transform=axes[1].transAxes)
    else:
        positions = np.arange(len(rpi))
        axes[1].bar(positions - .18, rpi["retail_shares_submitted"], width=.36, label="Submitted", color="#d89436")
        axes[1].bar(positions + .18, rpi["retail_shares_executed"], width=.36, label="Executed", color="#267a9c")
        axes[1].set_xticks(positions, rpi["side"])
        axes[1].legend(frameon=False)
    axes[1].set(title=f"{symbol} · reported RPI activity", xlabel="Source side labels (preserved separately)", ylabel="Daily shares")
    fig.text(.03, .025, "Different report scopes and time periods. Opportunity/submitted flow is not executed market share or a participant identity. Source: Cboe.", fontsize=8)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_opportunity_comparison(maker_csv: Path, rpi_csv: Path, output_dir: Path) -> dict:
    """Describe cross-symbol maker/RPI associations without participant labels.

    This deliberately compares different market and time scopes: weekly BZX
    opportunity with daily BYX RPI outcomes. It is exploratory context, not a
    validation of execution colonies or a causal/predictive estimate.
    """
    maker_csv, rpi_csv, output_dir = Path(maker_csv), Path(rpi_csv), Path(output_dir)
    maker, rpi = _load_maker_report(maker_csv), _load_rpi_report(rpi_csv)
    eligible_maker = maker[(maker["cboe_shares_adv"] > 0) & np.isfinite(maker["liquidity_opportunity_ratio"])].copy()
    joined = rpi.merge(eligible_maker, on="symbol", how="inner", validate="many_to_one")
    joined["retrospective_context_only"] = True
    joined["verified_point_in_time_available"] = False
    joined["eligible_for_price_improvement_association"] = joined["has_executions"] & np.isfinite(joined["price_improvement"])
    rows = []
    for side in sorted(rpi["side"].unique()):
        group = joined[(joined["side"] == side) & joined["eligible_for_price_improvement_association"]]
        for predictor in ("liquidity_opportunity_ratio", "volume_opportunity_shares_adv"):
            x, y = group[predictor], group["price_improvement"]
            defined = len(group) >= 2 and x.nunique() > 1 and y.nunique() > 1
            # Spearman's rho is Pearson correlation of average ranks. This
            # handles tied values without adding a scipy runtime dependency.
            rho = float(x.rank(method="average").corr(y.rank(method="average"))) if defined else np.nan
            rows.append({
                "side": side, "predictor": predictor, "outcome": "price_improvement",
                "n_symbols": len(group), "joined_symbol_side_rows": int((joined["side"] == side).sum()),
                "excluded_no_execution_rows": int(((joined["side"] == side) & ~joined["has_executions"]).sum()),
                "spearman_rho": rho, "distinct_predictor_values": int(x.nunique()), "distinct_outcome_values": int(y.nunique()),
                "correlation_defined": defined,
            })
    associations = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "symbol_opportunity_context": output_dir / "symbol_opportunity_context.csv",
        "opportunity_associations": output_dir / "opportunity_associations.csv",
        "opportunity_comparison_plot": output_dir / "opportunity_comparison.png",
        "summary": output_dir / "opportunity_comparison.json",
    }
    joined.sort_values(["symbol", "side"]).to_csv(paths["symbol_opportunity_context"], index=False)
    associations.to_csv(paths["opportunity_associations"], index=False)
    _plot_opportunity_comparison(joined, associations, paths["opportunity_comparison_plot"])
    report_date = _rpi_report_date(rpi_csv)
    result = {
        "maker_period": _maker_report_period(maker_csv), "rpi_session_date": report_date,
        "maker_source_symbols": len(maker), "positive_adv_maker_symbols": len(eligible_maker),
        "excluded_zero_adv_maker_symbols": int(maker["zero_adv"].sum()),
        "joined_symbols": int(joined["symbol"].nunique()), "joined_symbol_side_rows": len(joined),
        "eligible_execution_symbol_side_rows": int(joined["eligible_for_price_improvement_association"].sum()),
        "retrospective_context_only": True, "verified_point_in_time_available": False,
        "associations": _records(associations),
        "sources": {"maker": _provenance(maker_csv, MAKER_URL), "rpi": _provenance(rpi_csv, RPI_PAGE)},
        "output_files": {name: str(path.resolve()) for name, path in paths.items()},
        "method": "Unweighted cross-security Spearman correlation of average ranks within each raw RPI side; positive BZX ADV, finite opportunity ratio, and positive reported RPI executed shares required. No side aggregation, p-values, confidence intervals, or causal/predictive inference.",
        "limitations": [
            "Weekly BZX maker opportunity and daily BYX RPI observations have different venue and period coverage. Their association is only an exploratory cross-security description.",
            "Dollar-per-share price improvement depends on price level, quoted spread, tick size, liquidity and security composition. Rank correlation does not control these confounders.",
            "A security with minimal executed RPI shares receives the same rank weight as one with substantial executions. Execution support is retained for follow-up sensitivity work.",
            "Zero-execution rows remain in the joined evidence table but are excluded from price-improvement associations; raw zero quality fields in those rows are not observed execution outcomes.",
            "Raw RPI side labels are retained separately without asserting unverified report-specific meanings for T/E.",
            "No investor identities, market-maker identities, execution-colony labels, trade-level joins or exchange-reaction measurements follow from these reports.",
            "The reports are retrospective, may contain revised data, and have unverified publication timestamps; they are not model predictors or out-of-sample evidence.",
            "Correlations use one available report pair. Repeated dates, pre-specified controls and dependence-aware uncertainty are needed for a research conclusion.",
        ],
    }
    write_json(paths["summary"], result)
    return result


def _plot_opportunity_comparison(joined: pd.DataFrame, associations: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sides = sorted(associations["side"].unique())
    fig, axes = plt.subplots(max(1, (len(sides) + 1) // 2), 2, figsize=(11, max(4, 3.7 * ((len(sides) + 1) // 2))), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, side in zip(axes.flat, sides):
        ax.set_visible(True)
        group = joined[(joined["side"] == side) & joined["eligible_for_price_improvement_association"]]
        stat = associations[(associations["side"] == side) & (associations["predictor"] == "liquidity_opportunity_ratio")].iloc[0]
        ax.scatter(group["liquidity_opportunity_ratio"].rank(pct=True) * 100, group["price_improvement"].rank(pct=True) * 100, s=4, alpha=.15, color="#267a9c", rasterized=True)
        rho = f"{stat['spearman_rho']:.3f}" if stat["correlation_defined"] else "undefined"
        ax.set(title=f"Raw side {side} · N={len(group):,} · rho={rho}", xlabel="BZX weekly opportunity/ADV percentile", ylabel="BYX daily price-improvement percentile", xlim=(0, 100), ylim=(0, 100))
    fig.suptitle("Cross-security opportunity and RPI outcomes · retrospective description", fontsize=13)
    fig.text(.04, .025, "Positive ADV and executed RPI shares only. Different venues/periods; uncontrolled price, spread and liquidity. No participant identities or causal inference.", fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, .95))
    fig.savefig(path, dpi=160)
    plt.close(fig)
