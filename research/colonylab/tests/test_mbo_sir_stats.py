"""Independent checks of the generic paired MBO/SIRS daily score interface."""
import json

import numpy as np
import pandas as pd
import pytest

from colonylab.mbo_sir_stats import summarize_sir_comparison


def scores(count=20, gain=.01):
    days = pd.bdate_range("2026-06-01", periods=count).strftime("%Y-%m-%d").tolist()
    records = []
    for index, day in enumerate(days):
        rows = 100 + index * 10
        for model, loss, brier in (("own_mbo", .2, .05),
                                   ("sirs", .2 - gain, .05 - gain / 10)):
            records.append({"day": day, "model": model, "rows": rows,
                            "log_loss_sum": rows * loss, "brier_sum": rows * brier})
    return pd.DataFrame(records), days


def summarize(frame, days, **kwargs):
    return summarize_sir_comparison(frame, reference_model="own_mbo", candidate_model="sirs",
                                    expected_days=days, bootstrap_replicates=200, **kwargs)


def test_pooled_loss_and_generic_names_match_paired_totals():
    frame, days = scores()
    result = summarize(frame, days)
    assert result["status"] == "ok"
    assert result["pooled"]["rows"] == int(frame.loc[frame.model.eq("sirs"), "rows"].sum())
    assert result["pooled"]["log_loss"]["reference"] == pytest.approx(.2)
    assert result["pooled"]["log_loss"]["candidate"] == pytest.approx(.19)
    assert result["pooled"]["log_loss"]["relative_gain_percent"] == pytest.approx(5.)
    assert result["supports_positive_incremental_prediction"]
    assert "volume" not in json.dumps(result, allow_nan=False)


def test_short_complete_holdout_is_descriptive_and_not_confirmation():
    frame, days = scores(10)
    result = summarize(frame, days)
    assert result["status"] == "descriptive_short_holdout"
    assert result["pooled"]["log_loss"]["absolute_gain"] == pytest.approx(.01)
    assert not result["supports_positive_incremental_prediction"]


def test_entire_missing_date_invalidates_inference_instead_of_shortening_holdout():
    frame, days = scores()
    result = summarize(frame.loc[frame.day.ne(days[-1])], days)
    assert result["status"] == "insufficient_coverage"
    assert any(days[-1] in error for error in result["coverage_errors"])
    assert "pooled" not in result


def test_only_one_model_missing_a_date_is_explicit_failure():
    frame, days = scores()
    frame = frame.loc[~(frame.day.eq(days[3]) & frame.model.eq("sirs"))]
    result = summarize(frame, days)
    assert result["status"] == "insufficient_coverage"
    assert any(days[3] in error for error in result["coverage_errors"])
    assert "pooled" not in result


def test_unequal_model_denominators_cannot_be_bootstrapped_together():
    frame, days = scores()
    frame.loc[frame.model.eq("sirs") & frame.day.eq(days[2]), "rows"] += 1
    result = summarize(frame, days)
    assert result["status"] == "insufficient_coverage"
    assert "identical paired row counts" in result["coverage_errors"][0]


def test_duplicate_model_day_cannot_inflate_sample_size():
    frame, days = scores()
    result = summarize(pd.concat([frame, frame.iloc[:1]], ignore_index=True), days)
    assert "Duplicate model/date" in result["coverage_errors"][0]


def test_other_models_do_not_change_requested_pair():
    frame, days = scores()
    other = frame.loc[frame.model.eq("sirs")].copy()
    other["model"] = "legacy_logistic"
    other["rows"] = 1
    other["log_loss_sum"] = 9999.
    assert summarize(frame, days) == summarize(pd.concat([frame, other], ignore_index=True), days)


def test_identical_models_have_zero_paired_intervals():
    frame, days = scores(gain=0)
    result = summarize(frame, days)
    assert not result["supports_positive_incremental_prediction"]
    assert result["paired_day_bootstrap"]["log_loss"]["absolute_gain_ci95"] == [0., 0.]


def test_daily_order_does_not_change_fixed_bootstrap_result():
    frame, days = scores()
    assert summarize(frame, days) == summarize(frame.sample(frac=1, random_state=123), days)


def test_zero_or_bad_daily_totals_fail_without_partial_inference():
    frame, days = scores()
    frame.loc[frame.day.eq(days[1]), "rows"] = 0
    assert "pooled" not in summarize(frame, days)
    frame, days = scores()
    frame.loc[0, "log_loss_sum"] = np.nan
    assert "pooled" not in summarize(frame, days)


@pytest.mark.parametrize("reference,candidate", [("", "sirs"), ("same", "same")])
def test_empty_or_same_model_names_rejected(reference, candidate):
    frame, days = scores()
    with pytest.raises(ValueError):
        summarize_sir_comparison(frame, reference_model=reference, candidate_model=candidate,
                                 expected_days=days)
