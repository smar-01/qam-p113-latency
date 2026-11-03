import json

import pandas as pd
import pytest

from colonylab.venue_panel import build_venue_panel, check_rpi_denominator


def feed(venue, volume, *, scope="individual_exchange_feed", capped=False):
    return {"feed_scope": scope, "probes": [{
        "schema": "ohlcv-1d", "retrieval_status": "downloaded_nonempty",
        "record_limit_reached": capped,
        "publisher_mappings": [{"publisher_id": 1, "venue": venue}],
        "daily_bars": [{"ts_event": "2026-10-23T00:00:00Z", "symbol": "NA", "publisher_id": 1, "volume": volume}],
    }]}


def audit_file(tmp_path, feeds):
    path = tmp_path / "audit.json"
    path.write_text(json.dumps({"completed_utc": "2026-10-26T16:00:00Z", "feeds": feeds}))
    return path


def test_native_shares_exclude_overlapping_composites_and_preserve_na_symbol(tmp_path):
    path = audit_file(tmp_path, {"XNAS.ITCH": feed("XNAS", 25), "ARCX.PILLAR": feed("ARCX", 75), "EQUS.MINI": feed("EQUS", 900, scope="composite_feed_not_an_execution_venue")})
    summary = build_venue_panel(path, tmp_path / "out")
    frame = pd.read_csv(summary["panel"], keep_default_na=False).set_index("venue")
    assert summary["venues"] == ["ARCX", "XNAS"]
    assert frame.symbol.eq("NA").all()
    assert frame.observed_native_volume_shares.eq(100).all()
    assert frame.loc["XNAS", "volume_share_among_observed_native_venues"] == pytest.approx(.25)
    assert frame.loc["ARCX", "volume_share_among_observed_native_venues"] == pytest.approx(.75)
    assert frame.mbo_probe_status.eq("not_tested").all()


def test_overlapping_native_aliases_cannot_double_count_the_same_venue(tmp_path):
    path = audit_file(tmp_path, {"FEED.A": feed("XNAS", 25), "FEED.B": feed("XNAS", 25)})
    with pytest.raises(ValueError, match="Overlapping feeds"):
        build_venue_panel(path, tmp_path / "out")


def test_capped_bars_do_not_produce_false_complete_volume_weights(tmp_path):
    path = audit_file(tmp_path, {"XNAS.ITCH": feed("XNAS", 25, capped=True)})
    with pytest.raises(ValueError, match="No verified native"):
        build_venue_panel(path, tmp_path / "out")


@pytest.mark.parametrize("volume", [-1, float("inf"), float("nan")])
def test_invalid_volume_is_rejected(tmp_path, volume):
    path = audit_file(tmp_path, {"XNAS.ITCH": feed("XNAS", volume)})
    with pytest.raises(ValueError, match="invalid volume"):
        build_venue_panel(path, tmp_path / "out")


def test_rpi_ratio_crosscheck_masks_zero_executions_and_uses_matching_byx_date(tmp_path):
    panel = tmp_path / "panel.csv"
    pd.DataFrame([
        {"symbol": "NA", "venue": "BATY", "bar_date_utc": "2026-10-23", "volume_shares": 1000},
        {"symbol": "NA", "venue": "BATY", "bar_date_utc": "2026-10-22", "volume_shares": 9999},
        {"symbol": "NA", "venue": "XNAS", "bar_date_utc": "2026-10-23", "volume_shares": 5555},
    ]).to_csv(panel, index=False)
    retail = tmp_path / "retail.csv"
    pd.DataFrame([
        {"symbol": "NA", "side": "B", "retail_shares_executed": 10, "shares_exec_div_bats_vol": .01},
        {"symbol": "NA", "side": "E", "retail_shares_executed": 0, "shares_exec_div_bats_vol": 0},
    ]).to_csv(retail, index=False)
    result = check_rpi_denominator(panel, retail, tmp_path, report_date="2026-10-23")
    assert result["symbols"] == ["NA"]
    assert result["tested_symbol_side_rows"] == 1
    assert result["matching_rows"] == 1


def test_rpi_volume_disagreement_is_reported_instead_of_claiming_validation(tmp_path):
    panel, retail = tmp_path / "panel.csv", tmp_path / "retail.csv"
    pd.DataFrame([{"symbol": "TEST", "venue": "BATY", "bar_date_utc": "2026-10-23", "volume_shares": 1000}]).to_csv(panel, index=False)
    pd.DataFrame([{"symbol": "TEST", "side": "B", "retail_shares_executed": 10, "shares_exec_div_bats_vol": .02}]).to_csv(retail, index=False)
    result = check_rpi_denominator(panel, retail, tmp_path, report_date="2026-10-23")
    assert result["tested_symbol_side_rows"] == 1
    assert result["all_tested_rows_match"] is False
