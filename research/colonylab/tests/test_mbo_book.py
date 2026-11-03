from copy import deepcopy
from types import SimpleNamespace

import pytest

from colonylab.mbo_book import (
    Book, MarketBooks, PRICE_SCALE, UNDEF_PRICE, UNDEF_TIMESTAMP,
    F_LAST, F_SNAPSHOT, F_BAD_TS_RECV, F_MAYBE_BAD_BOOK, F_TOB, F_MBP,
)


def record(action="A", side="B", price=100, size=100, order_id=1, ts=1, flags=F_LAST, **extra):
    return dict(action=action, side=side, price=int(price * PRICE_SCALE), size=size,
                order_id=order_id, ts_recv=ts, flags=flags, **extra)


def cleared():
    book = Book()
    book.apply(record("R", "N", 0, 0, 0, ts=0))
    return book


def two_sided():
    book = cleared()
    book.apply(record("A", "B", 100, 100, 1, ts=1))
    book.apply(record("A", "A", 101, 200, 2, ts=2))
    return book


def test_no_assumed_midday_empty_initialization():
    book = Book()
    state = book.apply(record())
    assert not state["valid"] and not state["initialized"]
    assert not book.orders
    assert book.diagnostics["ignored_uninitialized_or_invalid"] == 1
    book.apply(record("R", "N", 0, 0, 0, ts=2))
    assert book.initialized and not book.valid
    assert book.initialization_source == "observed_clear"


def test_explicit_trusted_empty_option_is_auditable():
    book = Book(trusted_empty_start=True)
    book.apply(record())
    assert book.snapshot()["initialization_source"] == "trusted_empty_start"
    assert book.initialized


def test_full_book_known_answers_and_top_depth():
    book = two_sided()
    book.apply(record("A", "B", 100, 50, 3, ts=3))
    book.apply(record("A", "B", 99, 300, 4, ts=4))
    state = book.snapshot()
    assert state["valid"]
    assert state["bid"] == 100 * PRICE_SCALE
    assert state["ask"] == 101 * PRICE_SCALE
    assert state["bid_size"] == 150 and state["bid_count"] == 2
    assert state["ask_size"] == 200
    assert state["bid_depth_5"] == 450
    assert state["ask_depth_5"] == 200
    assert state["depth_imbalance_1"] == pytest.approx(-50 / 350)
    assert state["depth_imbalance_10"] == pytest.approx(250 / 650)
    assert state["mid"] == 100.5
    assert state["spread_bps"] == pytest.approx(10000 / 100.5)
    assert state["order_count"] == 4


def test_partial_full_cancel_aggregates_and_lifetimes():
    book = two_sided()
    book.apply(record("C", "B", 100, 40, 1, ts=11))
    assert book.orders[1].size == 60
    assert book.snapshot()["cum_cancel_shares_B"] == 40
    assert book.snapshot()["cum_cancel_lifetime_ns"] == 10
    book.apply(record("C", "B", 100, 60, 1, ts=21))
    state = book.snapshot()
    assert 1 not in book.orders and 100 * PRICE_SCALE not in book.bid_levels
    assert state["cum_cancel_lifetime_count"] == 2
    assert state["cum_cancel_lifetime_ns"] == 30
    assert state["cum_cancel_age_share_ns"] == 1600
    assert not state["valid"]  # One-sided but fully initialized.


def test_fill_trade_do_not_decrement_and_execute_cancel_not_double_counted():
    book = two_sided()
    assert book.apply(record("T", "A", 100, 25, 0, ts=3, flags=0)) is None
    assert book.orders[1].size == 100
    book.apply(record("F", "B", 100, 25, 1, ts=3, flags=0))
    assert book.orders[1].size == 100
    state = book.apply(record("C", "B", 100, 25, 1, ts=3))
    assert book.orders[1].size == 75
    assert state["cum_trade_shares_A"] == 25
    assert state["cum_fill_shares_B"] == 25
    assert state["cum_removed_shares_B"] == 25
    assert state["cum_execution_removed_shares_B"] == 25
    assert state["cum_cancel_shares_B"] == 0
    assert state["cum_cancel_lifetime_count"] == 0


def test_partial_fill_cancel_with_additional_cancel_splits_removals():
    book = two_sided()
    book.apply(record("F", "B", 100, 10, 1, ts=3, flags=0))
    state = book.apply(record("C", "B", 100, 25, 1, ts=3))
    assert state["cum_execution_removed_shares_B"] == 10
    assert state["cum_cancel_shares_B"] == 15


def test_modify_size_is_absolute_and_move_updates_all_levels():
    book = two_sided()
    book.apply(record("M", "B", 100, 70, 1, ts=3))
    assert book.orders[1].size == 70
    assert book.orders[1].born_ns == 1
    assert book.snapshot()["cum_modify_remove_shares_B"] == 30
    book.apply(record("M", "B", 99, 150, 1, ts=4))
    state = book.snapshot()
    assert state["bid"] == 99 * PRICE_SCALE
    assert state["bid_size"] == 150
    assert 100 * PRICE_SCALE not in book.bid_levels
    assert state["cum_modify_add_shares_B"] == 150
    assert state["cum_modify_remove_shares_B"] == 100
    assert book.orders[1].born_ns == 4


def test_modify_upsert_is_explicit_and_audited():
    strict = two_sided()
    strict.apply(record("M", "B", 99, 50, 9, ts=3))
    assert not strict.valid and 9 not in strict.orders
    assert strict.diagnostics["unknown_modify_invalidations"] == 1
    upsert = Book(trusted_empty_start=True, unknown_modify="add")
    upsert.apply(record("M", "B", 99, 50, 9))
    assert upsert.orders[9].size == 50
    assert upsert.diagnostics["modify_upserts"] == 1
    with pytest.raises(ValueError):
        Book(unknown_modify="anything")


@pytest.mark.parametrize("bad", [
    record("A", "B", 100, 100, 1, ts=3),
    record("C", "B", 100, 1, 99, ts=3),
    record("C", "B", 100, 101, 1, ts=3),
    record("C", "A", 100, 1, 1, ts=3),
    record("C", "B", 99, 1, 1, ts=3),
    record("M", "A", 100, 100, 1, ts=3),
    record("A", "N", 100, 100, 99, ts=3),
    record("A", "B", 100, 0, 99, ts=3),
    record("X", "B", 100, 100, 99, ts=3),
])
def test_semantic_failure_invalid_until_authoritative_clear(bad):
    book = two_sided()
    book.apply(bad)
    assert not book.valid
    book.apply(record("A", "B", 99, 10, 10, ts=4))
    assert not book.valid and 10 not in book.orders
    book.apply(record("R", "N", 0, 0, 0, ts=5))
    book.apply(record("A", "B", 100, 10, 10, ts=6))
    book.apply(record("A", "A", 101, 10, 11, ts=7))
    assert book.valid


@pytest.mark.parametrize("flag", [F_BAD_TS_RECV, F_MAYBE_BAD_BOOK, F_TOB, F_MBP])
def test_bad_real_flags_invalidate_even_informational_records(flag):
    book = two_sided()
    state = book.apply(record("N", "N", 0, 0, 0, ts=3, flags=flag | F_LAST))
    assert not state["valid"]


def test_snapshot_clear_add_bad_ts_allowed_but_not_counted_as_flow():
    book = Book()
    flags = F_SNAPSHOT | F_BAD_TS_RECV
    assert book.apply(record("R", "N", 0, 0, 0, ts=5, flags=flags)) is None
    assert book.apply(record("A", "B", 100, 100, 1, ts=5, flags=flags)) is None
    assert not book.snapshot()["valid"]
    state = book.apply(record("A", "A", 101, 200, 2, ts=5, flags=flags | F_LAST))
    assert state["valid"]
    assert state["cum_add_shares_B"] == state["cum_add_shares_A"] == 0
    assert book.orders[1].born_ns is None
    book.apply(record("C", "B", 100, 25, 1, ts=6))
    assert book.snapshot()["cum_cancel_lifetime_count"] == 0
    assert book.diagnostics["left_censored_cancels"] == 1
    assert book.initialization_source == "snapshot_clear"


def test_snapshot_without_clear_does_not_initialize():
    book = Book()
    book.apply(record(flags=F_SNAPSHOT | F_BAD_TS_RECV | F_LAST))
    assert not book.initialized and not book.orders
    assert book.diagnostics["snapshot_without_clear"] == 1


def test_empty_undefined_timestamp_snapshot_does_not_invent_time():
    book = Book()
    state = book.apply(record("R", "N", 0, 0, 0, ts=UNDEF_TIMESTAMP,
                              flags=F_SNAPSHOT | F_BAD_TS_RECV | F_LAST))
    assert state["initialized"] and not state["valid"]
    assert state["time_ns"] is None


def test_atomic_emission_and_reading_mid_event_is_invalid():
    book = two_sided()
    assert book.apply(record("A", "B", 99, 30, 3, ts=3, flags=0)) is None
    assert not book.snapshot()["valid"]
    state = book.apply(record("N", "N", 0, 0, 0, ts=3))
    assert state["valid"] and state["last_complete_ns"] == 3
    assert book.apply(record("N", "N", 0, 0, 0, ts=4), emit=False) is None
    assert book.snapshot()["valid"]


def test_crossed_invalid_locked_valid_and_audited():
    book = two_sided()
    state = book.apply(record("M", "B", 102, 100, 1, ts=3))
    assert not state["valid"] and book.diagnostics["crossed_events"] == 1
    state = book.apply(record("M", "B", 101, 100, 1, ts=4))
    assert state["valid"] and book.diagnostics["locked_events"] == 1
    assert state["spread_bps"] == 0


def test_uint64_ids_are_never_floats_or_rounded():
    book = cleared()
    big = (1 << 64) - 2
    book.apply(record(order_id=big))
    book.apply(record(order_id=big - 1, price=99, ts=2))
    assert set(book.orders) == {big, big - 1}
    book.apply(record("C", "B", 100, 100, big, ts=3))
    assert set(book.orders) == {big - 1}
    with pytest.raises(TypeError):
        book.apply(record(order_id=float(big), ts=4))
    r = record(order_id=20, ts=4)
    r["price"] = 100.0
    with pytest.raises(TypeError):
        book.apply(r)


def test_price_raw_precision_and_levels_beyond_top_ten():
    book = cleared()
    for n in range(12):
        r = record("A", "B", 100, n + 1, n + 1, ts=n + 1)
        r["price"] = 100 * PRICE_SCALE - n
        book.apply(r)
    book.apply(record("A", "A", 101, 100, 20, ts=13))
    state = book.snapshot()
    assert len(book.bid_levels) == 12
    assert state["bid_depth_10"] == sum(range(1, 11))
    assert state["bid_depth_5"] == 15


def test_receive_time_reversal_invalidates_and_future_reads_do_not_use_future():
    book = two_sided()
    before = deepcopy(book.snapshot())
    assert book.snapshot(time_ns=3)["last_complete_ns"] == 2
    assert book.snapshot() == before
    with pytest.raises(ValueError):
        book.snapshot(time_ns=1)
    book.apply(record("N", "N", 0, 0, 0, ts=1))
    assert not book.valid
    assert book.diagnostics["receive_time_reversals"] == 1


def test_namespace_is_publisher_instrument_not_channel():
    market = MarketBooks(trusted_empty_start=True)
    for publisher, instrument, channel, price in [(2, 10, 0, 100), (5, 10, 0, 99), (2, 11, 0, 98)]:
        market.apply(record(publisher_id=publisher, instrument_id=instrument, channel_id=channel, price=price))
    market.apply(record("M", "B", 100, 50, 1, ts=2, publisher_id=2, instrument_id=10, channel_id=1))
    assert len(market.books) == 3
    assert market.books[2, 10].orders[1].size == 50
    assert market.books[5, 10].orders[1].size == 100
    assert market.books[2, 10].channels_seen == {0, 1}
    with pytest.raises(ValueError):
        market.books[2, 10].apply(record(publisher_id=5, instrument_id=10))


def test_native_record_attribute_input_and_bytes_actions():
    book = cleared()
    r = record()
    r["action"], r["side"] = b"A", b"B"
    book.apply(SimpleNamespace(**r))
    assert book.orders[1].price == 100 * PRICE_SCALE


@pytest.mark.parametrize("ts", [-1, UNDEF_TIMESTAMP, 1])
def test_unsafe_clear_timestamp_cannot_reinitialize(ts):
    book = two_sided()
    book.apply(record("R", "N", 0, 0, 0, ts=ts))
    assert not book.valid
    assert len(book.orders) == 2
    assert book.last_time_ns == 2
    book.apply(record("R", "N", 0, 0, 0, ts=3))
    assert book.initialized and not book.orders


@pytest.mark.parametrize("action", ["T", "F", "C", "M", "N"])
def test_snapshot_contains_only_clear_and_add(action):
    book = Book()
    book.apply(record("R", "N", 0, 0, 0, ts=1, flags=F_SNAPSHOT | F_BAD_TS_RECV))
    book.apply(record(action, "B", 100, 10, 1, ts=1, flags=F_SNAPSHOT | F_BAD_TS_RECV | F_LAST))
    assert not book._healthy
    assert book.diagnostics["invalid_snapshot_actions"] == 1


def test_randomized_mutations_match_independent_order_dictionary():
    import random

    rng = random.Random(20261026)
    book = cleared()
    oracle = {}
    next_id = (1 << 63) + 1
    for ts in range(1, 1501):
        action = rng.choice(["A", "A", "C", "M"]) if oracle else "A"
        if action == "A":
            side = rng.choice(["B", "A"])
            price = rng.randrange(80, 101) if side == "B" else rng.randrange(101, 122)
            size = rng.randrange(1, 1000)
            oid = next_id
            next_id += 1
            oracle[oid] = [side, price * PRICE_SCALE, size]
        else:
            oid = rng.choice(list(oracle))
            side, rawprice, oldsize = oracle[oid]
            price = rawprice // PRICE_SCALE
            if action == "C":
                size = rng.randrange(1, oldsize + 1)
                oracle[oid][2] -= size
                if not oracle[oid][2]:
                    del oracle[oid]
            else:
                price = rng.randrange(80, 101) if side == "B" else rng.randrange(101, 122)
                size = rng.randrange(1, 1000)
                oracle[oid][1:] = [price * PRICE_SCALE, size]
        book.apply(record(action, side, price, size, oid, ts=ts), emit=False)
        expected = {"A": {}, "B": {}}
        for s, p, quantity in oracle.values():
            aggregate = expected[s].setdefault(p, [0, 0])
            aggregate[0] += quantity
            aggregate[1] += 1
        assert set(book.orders) == set(oracle)
        for side_key, levels in [("B", book.bid_levels), ("A", book.ask_levels)]:
            assert {p: [x.size, x.count] for p, x in levels.items()} == expected[side_key]
        state = book.snapshot()
        bid_prices = sorted(expected["B"], reverse=True)
        ask_prices = sorted(expected["A"])
        assert state["bid"] == (bid_prices[0] if bid_prices else None)
        assert state["ask"] == (ask_prices[0] if ask_prices else None)
        for depth in (1, 5, 10):
            assert state[f"bid_depth_{depth}"] == sum(expected["B"][p][0] for p in bid_prices[:depth])
            assert state[f"ask_depth_{depth}"] == sum(expected["A"][p][0] for p in ask_prices[:depth])


def test_real_dbn_message_types_preserve_ids_and_prices():
    import databento_dbn as dbn

    book = Book()
    for ts, action, side, oid, price, size in [
        (0, "R", "N", 0, UNDEF_PRICE, 0),
        (1, "A", "B", (1 << 64) - 2, 100 * PRICE_SCALE + 1, 100),
        (2, "A", "A", (1 << 64) - 3, 101 * PRICE_SCALE + 1, 100),
    ]:
        message = dbn.MBOMsg(publisher_id=2, instrument_id=1, ts_event=ts,
                             ts_recv=ts, action=dbn.Action(action), side=dbn.Side(side), order_id=oid,
                             price=price, size=size, flags=F_LAST, channel_id=0)
        book.apply(message, emit=False)
    state = book.snapshot()
    assert state["valid"]
    assert state["bid"] == 100 * PRICE_SCALE + 1
    assert (1 << 64) - 2 in book.orders


@pytest.mark.parametrize("publisher", [2, 5, 6, 7, 8, 9, 43, 52])
def test_legacy_startup_clear_explicit_full_day_optin(publisher):
    book = Book(allow_legacy_startup_clear=True)
    startup = record("R", "N", 0, 0, 0, ts=100, flags=F_BAD_TS_RECV,
                     publisher_id=publisher, sequence=0)
    startup["price"] = UNDEF_PRICE
    assert book.apply(startup) is None
    assert book.initialized and not book.event_complete
    assert book.last_time_ns is None and book.last_complete_ns is None
    assert book.initialization_source == "legacy_synthetic_session_clear"
    assert not book.flow
    book.apply(record("A", "B", 100, 100, 1, ts=101, publisher_id=publisher))
    book.apply(record("A", "A", 101, 100, 2, ts=102, publisher_id=publisher))
    assert book.valid
    assert book.diagnostics["legacy_startup_clears"] == 1
    # The same shape later in the stream is an unsafe clock, not a bypass.
    startup["ts_recv"] = 103
    book.apply(startup)
    assert not book.valid and len(book.orders) == 2


@pytest.mark.parametrize("changed,value", [("sequence", 1), ("order_id", 1),
                                           ("price", 0), ("size", 1), ("side", "B")])
def test_startup_bad_clear_requires_canonical_signature(changed, value):
    book = Book(allow_legacy_startup_clear=True)
    startup = record("R", "N", 0, 0, 0, flags=F_BAD_TS_RECV, sequence=0)
    startup["price"] = UNDEF_PRICE
    startup[changed] = value
    book.apply(startup)
    assert not book.initialized


def test_native_cboe_unknown_side_execution_uses_absolute_modify_then_zero():
    book = two_sided()
    # Actual BYX pattern: T.N, F.N, M.A(new remaining size), with an
    # execution price different from the resting order's displayed price.
    book.apply(record("T", "N", 100.9, 40, 0, ts=3, flags=0))
    book.apply(record("F", "N", 100.9, 40, 2, ts=3, flags=0))
    state = book.apply(record("M", "A", 101, 160, 2, ts=3))
    assert state["ask_size"] == 160
    assert state["cum_execution_removed_shares_A"] == 40
    assert state["cum_removed_shares_A"] == 40
    assert state["cum_cancel_shares_A"] == 0
    book.apply(record("T", "N", 100.9, 160, 0, ts=4, flags=0))
    book.apply(record("F", "N", 100.9, 160, 2, ts=4, flags=0))
    state = book.apply(record("M", "A", 101, 0, 2, ts=4))
    assert 2 not in book.orders and not book.ask_levels
    assert state["cum_execution_removed_shares_A"] == 200
    assert state["cum_removed_shares_A"] == 200
    assert state["cum_cancel_shares_A"] == 0
    assert state["cum_trade_shares_N"] == 200
    assert state["cum_cancel_lifetime_count"] == 0
    assert book._healthy and book.diagnostics["zero_size_modify_removals"] == 1


def test_total_flow_conserves_net_size_and_does_not_count_price_relocation():
    book = two_sided()
    book.apply(record("M", "B", 100, 120, 1, ts=3))
    state = book.snapshot()
    assert state["cum_add_shares_B"] == 120
    assert state["cum_removed_shares_B"] == 0
    book.apply(record("M", "B", 99, 120, 1, ts=4))
    state = book.snapshot()
    assert state["cum_add_shares_B"] == 120
    assert state["cum_removed_shares_B"] == 0
    assert state["cum_modify_remove_shares_B"] == 120  # Relocation audit, separate from net flows.
    for ts, action, size, remaining in [(5, "C", 30, 90), (6, "M", 60, 60), (7, "M", 0, 0)]:
        book.apply(record(action, "B", 99, size, 1, ts=ts))
        state = book.snapshot()
        assert state["cum_add_shares_B"] - state["cum_removed_shares_B"] == remaining
        assert state["cum_cancel_shares_B"] == 120 - remaining
    assert 1 not in book.orders


@pytest.mark.parametrize("policy", ["invalidate", "add"])
def test_zero_size_modify_cannot_upsert_an_unknown_order(policy):
    book = Book(trusted_empty_start=True, unknown_modify=policy)
    book.apply(record("M", "B", 100, 0, 99))
    assert not book._healthy and not book.orders
    assert book.diagnostics["unmatched_modifies"] == 1
    assert not book.diagnostics["modify_upserts"]


def test_native_mapping_and_attribute_adapters_have_identical_full_state():
    import databento_dbn as dbn
    from types import MappingProxyType

    oid = (1 << 64) - 4
    records = [record("R", "N", 0, 0, 0, ts=0),
               record("A", "B", 100, 100, oid, ts=1),
               record("A", "A", 101, 200, 2, ts=2),
               record("T", "N", 100, 20, 0, ts=3, flags=0),
               record("F", "N", 100, 20, oid, ts=3, flags=0),
               record("M", "B", 100, 80, oid, ts=3),
               record("M", "B", 99, 90, oid, ts=4),
               record("C", "B", 99, 15, oid, ts=5),
               record("F", "B", 99, 75, oid, ts=6, flags=0),
               record("C", "B", 99, 75, oid, ts=6),
               record("A", "B", 98, 30, 20, ts=7)]
    for n, r in enumerate(records):
        r.update(publisher_id=2, instrument_id=1, channel_id=0, sequence=n)
    books = []
    for mode in ("dict", "mappingproxy", "namespace", "native"):
        book = Book()
        for r in records:
            if mode == "native":
                msg = dbn.MBOMsg(**dict(r, ts_event=r["ts_recv"], action=dbn.Action(r["action"]), side=dbn.Side(r["side"])))
            elif mode == "mappingproxy":
                msg = MappingProxyType(r)
            elif mode == "namespace":
                msg = SimpleNamespace(**r)
            else:
                msg = r
            book.apply(msg, emit=False)
        books.append(book)
    for book in books[1:]:
        assert book.orders == books[0].orders
        assert book.level_snapshot() == books[0].level_snapshot()
        assert book.flow == books[0].flow
        assert book.diagnostics == books[0].diagnostics
        assert book.snapshot() == books[0].snapshot()
