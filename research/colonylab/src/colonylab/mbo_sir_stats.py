"""Paired session inference for MBO/SIRS model comparisons.

Daily totals must already use identical eligible forecasts for each model.
The wrapper checks their denominators, then uses the existing independently
tested whole-session bootstrap. It does not treat venues or bins as replicates.
"""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .confirmation_stats import summarize_confirmation


REQUIRED_COLUMNS = ("day", "model", "rows", "log_loss_sum", "brier_sum")


def _candidate_names(value):
    """Remove candle-specific names from the shared numeric inference output."""
    if isinstance(value, dict):
        return {key.replace("volume", "candidate"): _candidate_names(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_candidate_names(item) for item in value]
    if isinstance(value, str):
        return value.replace("volume candles", "the candidate model").replace(
            "reference-minus-volume", "reference-minus-candidate")
    return value


def summarize_sir_comparison(
    daily: pd.DataFrame, *, reference_model: str, candidate_model: str,
    expected_days: Sequence[str], bootstrap_replicates: int = 10_000,
    seed: int = 20261026, block_length: int = 5,
) -> dict:
    """Compare two long-form daily score tables without dropping missing days.

    Each supplied model has exactly one aggregate row per intended test date.
    Extra models may be present but are ignored explicitly. Model losses and
    observation denominators stay paired in every bootstrap draw. Fewer than
    twenty complete test dates produce descriptive intervals only; they do not
    support the confirmation evidence flag.
    """
    if not reference_model or not candidate_model or reference_model == candidate_model:
        raise ValueError("Distinct nonempty reference and candidate model names are required.")
    identity = {"reference_model": reference_model, "candidate_model": candidate_model}
    failure = {
        **identity, "status": "insufficient_coverage", "coverage_errors": [],
        "supports_positive_incremental_prediction": False,
    }
    missing = sorted(set(REQUIRED_COLUMNS) - set(daily.columns))
    if missing:
        failure["coverage_errors"].append("Missing daily score columns: " + ", ".join(missing))
        return failure
    selected = daily.loc[daily.model.isin([reference_model, candidate_model])].copy()
    if selected.duplicated(["day", "model"]).any():
        failure["coverage_errors"].append(
            "Duplicate model/date score rows; aggregate all stocks and venues before inference.")
        return failure
    paired = None
    for model, prefix in ((reference_model, "reference"), (candidate_model, "volume")):
        part = selected.loc[selected.model.eq(model),
                            ["day", "rows", "log_loss_sum", "brier_sum"]].copy()
        part = part.rename(columns={"rows": prefix + "_rows",
                                    "log_loss_sum": prefix + "_log_loss_sum",
                                    "brier_sum": prefix + "_brier_sum"})
        paired = part if paired is None else paired.merge(part, on="day", how="outer", validate="one_to_one")
    if paired is None or paired.empty:
        failure["coverage_errors"].append("No paired held-out model/date scores supplied.")
        return failure
    unequal = ~paired.reference_rows.eq(paired.volume_rows)
    if unequal.any():
        failure["coverage_errors"].append(
            "Both models require identical paired row counts on every date: "
            + ", ".join(paired.loc[unequal, "day"].astype(str)))
        return failure
    paired["rows"] = paired.reference_rows
    result = _candidate_names(summarize_confirmation(
        paired, expected_days=expected_days, bootstrap_replicates=bootstrap_replicates,
        seed=seed, block_length=block_length))
    result.update(identity)
    result["paired_cohort_requirement"] = (
        "Identical forecast timestamps, labels and eligible task rows for both models; "
        "equal denominators alone cannot establish timestamp-level identity.")
    # Keep absent or corrupt dates as coverage failures. A small but complete
    # holdout has useful descriptive arithmetic without meeting confirmation.
    short_message = (f"Only {result.get('observed_day_count', 0)} paired dates; "
                     "at least 20 were predeclared for confirmation.")
    if result.get("coverage_errors") == [short_message]:
        result["status"] = "descriptive_short_holdout"
        count = result["observed_day_count"]
        result["limitations"] = [
            (f"Only {count} held-out dates provide about {count / 5:g} non-overlapping "
             "five-session blocks; intervals are descriptive and may be unstable.")
            if item.startswith("Twenty dates contain only about four") else item
            for item in result.get("limitations", [])
        ]
    result.setdefault("limitations", []).extend([
        "SIRS infection and refractory durations are predeclared state definitions, not fitted recovery or immunity rates.",
        "Directed nonnegative transmission coefficients measure conditional predictive associations, not causal venue influence.",
    ])
    return result
