from pathlib import Path

import databento_dbn as dbn
import numpy as np
import pandas as pd
import pytest

import colonylab.mbo_replay as replay
from colonylab.mbo_book import F_LAST, F_BAD_TS_RECV, F_MAYBE_BAD_BOOK, PRICE_SCALE, UNDEF_PRICE


START = 1_000_000_000
END = 3_000_000_000


def mbo(action, side="N", price=UNDEF_PRICE, size=0, oid=0, ts=1, flags=F_LAST, sequence=0, publisher=2, instrument=1):
    return dbn.MBOMsg(publisher_id=publisher, instrument_id=instrument,
                      ts_event=ts, ts_recv=ts, action=dbn.Action(action), side=dbn.Side(side),
                      price=price, size=size, order_id=oid, flags=flags,
                      channel_id=0, sequence=sequence)


def level(bid=100, ask=101, bid_size=100, ask_size=200, bid_count=1, ask_count=1):
    return dbn.BidAskPair(bid_px=int(bid * PRICE_SCALE), ask_px=int(ask * PRICE_SCALE),
                          bid_sz=bid_size, ask_sz=ask_size, bid_ct=bid_count, ask_ct=ask_count)


def mbp(ts, sequence, *, levels=None, flags=F_LAST, publisher=2, instrument=1):
    return dbn.MBP10Msg(publisher_id=publisher, instrument_id=instrument,
                        ts_event=ts, ts_recv=ts, price=UNDEF_PRICE, size=0,
                        action=dbn.Action.NONE, side=dbn.Side.NONE, depth=0,
                        sequence=sequence, flags=flags,
                        levels=levels or [level()] + [dbn.BidAskPair() for _ in range(9)])


def write_dbn(path: Path, schema, records, *, start=0, end=END, limit=None):
    metadata = dbn.Metadata(dataset="XNAS.ITCH", start=start, end=end,
                             stype_in=dbn.SType.RAW_SYMBOL, stype_out=dbn.SType.INSTRUMENT_ID,
                             schema=dbn.Schema(schema), symbols=["AAPL"], limit=limit)
    path.write_bytes(metadata.encode() + b"".join(bytes(record) for record in records))
    return path


def initial():
    return [mbo("R", ts=100, flags=F_BAD_TS_RECV),
            mbo("A", "B", 100 * PRICE_SCALE, 100, 1, ts=START - 500_000_000, sequence=1),
            mbo("A", "A", 101 * PRICE_SCALE, 200, 2, ts=START - 400_000_000, sequence=2)]


@pytest.fixture
def files(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "_day_bounds", lambda day: (0, START, END))

    def run(records=None, references=None, **source_options):
        source = write_dbn(tmp_path / "mbo.dbn", "mbo", records or initial(), **source_options)
        reference = write_dbn(tmp_path / "mbp10.dbn", "mbp-10", references or [], start=START)
        return replay.replay_file(source, reference, day="2026-06-01")

    return run


def test_day_bounds_full_regular_session():
    midnight, start, end = replay._day_bounds("2026-06-01")
    assert pd.Timestamp(midnight, tz="UTC") == pd.Timestamp("2026-06-01T00:00Z")
    assert pd.Timestamp(start, tz="UTC") == pd.Timestamp("2026-06-01T13:30Z")
    assert pd.Timestamp(end, tz="UTC") == pd.Timestamp("2026-06-01T20:00Z")
    assert len(np.arange(start, end, 100_000_000)) == 234_000


def test_real_dbn_replay_strict_anchor_and_exact_validation(files):
    ts = START + 100_000_000
    records = initial() + [mbo("M", "B", 100_200_000_000, 75, 1, ts=ts, sequence=8)]
    expected = [level(bid=100.2, bid_size=75)] + [dbn.BidAskPair() for _ in range(9)]
    result = files(records, [mbp(ts, 8, levels=expected)])
    grid = result["grid"].set_index("time_ns")
    assert grid.loc[ts, "bid"] == 100 * PRICE_SCALE  # equal-time record is not a predictor
    assert grid.loc[ts + 100_000_000, "bid"] == 100_200_000_000
    assert result["validation"]["passed"]
    assert result["validation"]["matched_keys"] == 1
    assert len(result["jumps"]) == 1
    assert result["jumps"].iloc[0]["jump"] == pytest.approx(0.1)
    assert result["jumps"].iloc[0]["time_ns"] == ts
    assert result["diagnostics"]["legacy_startup_clears"] == 1
    assert result["grid"]["bid"].dtype == np.int64


def test_quiet_complete_books_remain_valid_without_age_cutoff(files):
    result = files()
    assert result["grid"]["valid"].all()
    assert result["grid"].iloc[-1]["last_complete_age_ms"] == 2300
    assert not len(result["jumps"])
    assert result["diagnostics"]["records_replayed"] == 3
    assert not result["validation"]["passed"]  # no references cannot pass validation


def test_unfinished_atomic_event_excludes_grid_and_future_boundary(files):
    t0, t1 = START + 50_000_000, START + 150_000_000
    records = initial() + [mbo("F", "B", 100 * PRICE_SCALE, 25, 1, ts=t0, flags=0, sequence=3),
                           mbo("C", "B", 100 * PRICE_SCALE, 25, 1, ts=t1, sequence=3)]
    result = files(records)
    grid = result["grid"].set_index("time_ns")
    assert not grid.loc[START + 100_000_000, "valid"]
    assert grid.loc[START + 200_000_000, "valid"]
    assert grid.loc[START + 200_000_000, "cum_execution_removed_shares_B"] == 25
    assert grid.loc[START + 200_000_000, "cum_cancel_shares_B"] == 0
    assert result["atomic_intervals_ns"].tolist() == [[t0, t1]]
    assert t0 in result["invalid_times_ns"]


@pytest.mark.parametrize("failure", ["crossed", "one_sided", "bad_flag"])
def test_invalid_barriers_reset_analytic_segments_and_do_not_bridge_jumps(files, failure):
    t0, t1, t2 = START + 50_000_000, START + 150_000_000, START + 250_000_000
    if failure == "crossed":
        changes = [mbo("M", "B", 102 * PRICE_SCALE, 100, 1, ts=t0, sequence=3),
                   mbo("M", "B", 100 * PRICE_SCALE, 100, 1, ts=t1, sequence=4)]
    elif failure == "one_sided":
        changes = [mbo("C", "A", 101 * PRICE_SCALE, 200, 2, ts=t0, sequence=3),
                   mbo("A", "A", 101 * PRICE_SCALE, 200, 3, ts=t1, sequence=4)]
    else:
        changes = [mbo("N", ts=t0, flags=F_MAYBE_BAD_BOOK | F_LAST, sequence=3),
                   mbo("R", ts=t1, sequence=4),
                   mbo("A", "B", 100 * PRICE_SCALE, 100, 1, ts=t1, sequence=5),
                   mbo("A", "A", 101 * PRICE_SCALE, 200, 2, ts=t1, sequence=6)]
    records = initial() + changes + [mbo("M", "B", 100_200_000_000, 100, 1, ts=t2, sequence=7)]
    result = files(records)
    grid = result["grid"].set_index("time_ns")
    assert not grid.loc[START + 100_000_000, "valid"]
    assert grid.loc[START + 200_000_000, "valid"]
    assert grid.loc[START + 200_000_000, "segment"] > grid.loc[START, "segment"]
    assert t0 in result["invalid_times_ns"]
    assert result["jumps"]["time_ns"].tolist() == [t2]


def test_full_session_metadata_required_and_capped_sources_rejected(files):
    with pytest.raises(ValueError, match="midnight"):
        files(start=1)
    with pytest.raises(ValueError, match="RTH close"):
        files(end=END - 1)
    assert files(limit=100)["diagnostics"]["records_replayed"] == 3
    with pytest.raises(ValueError, match="reached request limit"):
        files(limit=3)


def test_arbitrary_midday_legacy_clear_cannot_be_trusted(files):
    startup = mbo("R", ts=START, flags=F_BAD_TS_RECV)
    with pytest.raises(ValueError, match="precede"):
        files([startup])


def test_validation_missing_keys_and_raw_count_and_price_mismatches(files):
    ts = START + 100_000_000
    records = initial() + [mbo("N", ts=ts, sequence=8)]
    changed = [level(bid_size=99, ask_count=3)] + [dbn.BidAskPair() for _ in range(9)]
    result = files(records, [mbp(ts, 8, levels=changed), mbp(ts + 1, 9)])
    validation = result["validation"]
    assert not validation["passed"]
    assert validation["matched_keys"] == 1 and validation["missing_keys"] == 1
    assert validation["price_size_mismatch_keys"] == 1
    assert validation["count_mismatch_keys"] == 1
    assert validation["mismatch_keys"] == 1
    assert validation["first_differences"][0]["fields"][0]["field"] == "bid_sz"


def test_validation_uses_last_final_state_for_duplicate_same_key(files):
    ts = START + 100_000_000
    records = initial() + [mbo("M", "B", 100 * PRICE_SCALE, 80, 1, ts=ts, sequence=8),
                           mbo("M", "B", 100 * PRICE_SCALE, 60, 1, ts=ts, sequence=8)]
    expected = [level(bid_size=60)] + [dbn.BidAskPair() for _ in range(9)]
    result = files(records, [mbp(ts, 8), mbp(ts, 8, levels=expected)])
    assert result["validation"]["passed"]
    assert result["validation"]["duplicate_reference_keys"] == 1


def test_validation_quality_flags_are_explicitly_counted(files):
    ts = START + 100_000_000
    result = files(initial() + [mbo("N", ts=ts, sequence=8)],
                   [mbp(ts, 8), mbp(ts + 1, 9, flags=0), mbp(ts + 2, 10, flags=F_LAST | F_BAD_TS_RECV)])
    assert result["validation"]["passed"]
    assert result["validation"]["excluded_non_last_records"] == 1
    assert result["validation"]["excluded_quality_flag_records"] == 1


def test_no_clear_means_invalid_entire_grid_and_namespace_is_checked(files):
    result = files(initial()[1:])
    assert not result["grid"]["valid"].any()
    assert not result["diagnostics"]["initialized"]
    with pytest.raises(ValueError, match="namespaces"):
        files(initial(), [mbp(START, 2, publisher=5)])


def test_empty_book_levels_and_order_counts_match_exactly(files):
    ts = START + 100_000_000
    records = initial() + [mbo("C", "B", 100 * PRICE_SCALE, 100, 1, ts=ts, sequence=8)]
    expected = [dbn.BidAskPair(ask_px=101 * PRICE_SCALE, ask_sz=200, ask_ct=1)] + [dbn.BidAskPair() for _ in range(9)]
    result = files(records, [mbp(ts, 8, levels=expected)])
    assert result["validation"]["passed"]
    assert not result["grid"].iloc[-1]["valid"]
    assert result["grid"].iloc[-1]["bid"] == UNDEF_PRICE


def test_replay_prefix_feature_values_do_not_depend_on_future_records(files):
    ts = START + 900_000_000
    prefix = files(initial())["grid"]
    full = files(initial() + [mbo("M", "B", 100_400_000_000, 50, 1, ts=ts, sequence=8)])["grid"]
    cols = [col for col in prefix.columns if col != "time_ns"]
    pd.testing.assert_frame_equal(prefix.loc[prefix.time_ns <= ts, cols], full.loc[full.time_ns <= ts, cols])


def test_invalid_anchor_resolution_rejected(tmp_path):
    for step in [0, -1, 0.1, 17]:
        with pytest.raises(ValueError, match="anchor_step_ms"):
            replay.replay_file(tmp_path / "unused", tmp_path / "unused", day="2026-06-01", anchor_step_ms=step)


def test_unfinished_final_atomic_event_is_a_future_label_barrier(files):
    ts = START + 50_000_000
    result = files(initial() + [mbo("F", "B", 100 * PRICE_SCALE, 25, 1, ts=ts, flags=0, sequence=3)])
    assert result["grid"].iloc[0]["valid"]
    assert not result["grid"].iloc[1:]["valid"].any()
    assert result["invalid_times_ns"].tolist() == [ts]
    assert result["reset_times_ns"].tolist() == [ts]
    assert result["atomic_intervals_ns"].tolist() == [[ts, END]]
    assert result["diagnostics"]["unfinished_atomic_event_at_eof"]


def test_same_timestamp_atomic_fill_cancel_preserves_causal_grid(files):
    ts = START + 100_000_000
    records = initial() + [mbo("F", "B", 100 * PRICE_SCALE, 25, 1, ts=ts, flags=0, sequence=3),
                           mbo("C", "B", 100 * PRICE_SCALE, 25, 1, ts=ts, sequence=3)]
    result = files(records)
    grid = result["grid"].set_index("time_ns")
    assert grid.loc[ts, "bid_size"] == 100
    assert grid.loc[ts + 100_000_000, "bid_size"] == 75
    assert grid["valid"].all()
    assert not len(result["invalid_times_ns"])
    assert not len(result["atomic_intervals_ns"])


def test_batched_quiet_grid_is_exactly_equal_to_individual_snapshots():
    from colonylab.mbo_book import Book

    anchors = np.arange(START, END, 100_000_000, dtype=np.int64)
    for healthy in (False, True):
        book = Book()
        if healthy:
            for message in initial():
                book.apply(message, emit=False)
        single, batch = replay._grid_arrays(anchors.copy()), replay._grid_arrays(anchors.copy())
        for position in range(len(anchors)):
            replay._write_grid(single, position, book, 17)
        replay._write_grid_span(batch, 0, len(anchors), book, 17)
        pd.testing.assert_frame_equal(pd.DataFrame(single), pd.DataFrame(batch), check_exact=True)


def test_zero_size_modify_terminal_execution_validates_against_empty_level(files):
    ts = START + 100_000_000
    records = initial() + [mbo("T", "N", 100_900_000_000, 200, 0, ts=ts, flags=0, sequence=8),
                           mbo("F", "N", 100_900_000_000, 200, 2, ts=ts, flags=0, sequence=8),
                           mbo("M", "A", 101 * PRICE_SCALE, 0, 2, ts=ts, sequence=8)]
    expected = [dbn.BidAskPair(bid_px=100 * PRICE_SCALE, bid_sz=100, bid_ct=1)] + [dbn.BidAskPair() for _ in range(9)]
    result = files(records, [mbp(ts, 8, levels=expected)])
    assert result["validation"]["passed"]
    assert result["grid"].iloc[-1]["cum_execution_removed_shares_A"] == 200
    assert result["grid"].iloc[-1]["cum_removed_shares_A"] == 200
