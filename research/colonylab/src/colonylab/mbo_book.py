"""Streaming, integer-exact reconstruction of native displayed MBO books.

One :class:`Book` belongs to one publisher/instrument pair.  Order IDs are
Python integers, never floats.  Native feeds must not have F_TOB/F_MBP set.
The conservative default requires an observed cleaR before the book is usable;
merely requesting midnight is not evidence of an empty opening book.

Databento conventions used here (checked 2026-10-26):
https://databento.com/docs/examples/order-book/order-tracking
https://databento.com/docs/examples/order-book/limit-order-book
https://databento.com/docs/conventions/mbo-snapshot
https://databento.com/docs/datasets/xnas-itch
https://databento.com/docs/datasets/xnys-pillar
https://databento.com/docs/datasets/bats-pitch
https://databento.com/blog/improvements-mbo-publishing

T and F are informational: C removes a quantity, while M sets absolute remaining
size, including a terminal size of zero. Native Cboe executions can update the
resting quantity through M rather than C. Snapshots are state, not flow. State may only be read after
F_LAST.  Data-quality or semantic failures invalidate state until a later R.
Unlike the illustrative vendor LOB's M-upsert, unknown M invalidates by default;
``unknown_modify='add'`` is available explicitly and is always audited.
"""

from __future__ import annotations

from bisect import bisect_left, insort
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from math import nan
from operator import index
from typing import Any

PRICE_SCALE = 1_000_000_000
UNDEF_PRICE = (1 << 63) - 1
UNDEF_TIMESTAMP = (1 << 64) - 1
F_LAST = 128
F_TOB = 64
F_SNAPSHOT = 32
F_MBP = 16
F_BAD_TS_RECV = 8
F_MAYBE_BAD_BOOK = 4


def _field(record: Any, name: str, default: Any = None) -> Any:
    return record.get(name, default) if isinstance(record, Mapping) else getattr(record, name, default)


def _integer(value: Any) -> int:
    """Reject rounded floating IDs/prices rather than quietly casting them."""
    return index(value)


@dataclass(slots=True)
class Order:
    side: str
    price: int
    size: int
    born_ns: int | None


@dataclass(slots=True)
class Level:
    size: int = 0
    count: int = 0


class Book:
    """Reconstruct one native displayed order book without retaining history.

    ``apply(record, emit=False)`` avoids allocating snapshots for every event.
    Call ``snapshot()`` at forecast anchors; its ``valid`` flag also requires
    a completed atomic event and a two-sided, uncrossed book.  ``initialized``
    distinguishes trustworthy empty/one-sided books from missing initial state.
    ``trusted_empty_start`` is an explicit, externally justified opt-in, never
    inferred from the first record or wall-clock time.
    ``allow_legacy_startup_clear`` separately permits only a canonical first
    synthetic R in older equity files whose full-session provenance has been
    checked by the caller. Its bad receive clock is excluded from state times.
    """

    def __init__(
        self,
        *,
        publisher_id: int | None = None,
        instrument_id: int | None = None,
        trusted_empty_start: bool = False,
        unknown_modify: str = "invalidate",
        allow_legacy_startup_clear: bool = False,
    ) -> None:
        if unknown_modify not in {"invalidate", "add"}:
            raise ValueError("unknown_modify must be 'invalidate' or 'add'")
        self.publisher_id = publisher_id
        self.instrument_id = instrument_id
        self.unknown_modify = unknown_modify
        self.allow_legacy_startup_clear = bool(allow_legacy_startup_clear)
        self.orders: dict[int, Order] = {}
        self.bid_levels: dict[int, Level] = {}
        self.ask_levels: dict[int, Level] = {}
        self._bid_prices: list[int] = []
        self._ask_prices: list[int] = []
        self.diagnostics: Counter[str] = Counter()
        self.flow: Counter[str] = Counter()
        self.channels_seen: set[int] = set()
        self.initialized = bool(trusted_empty_start)
        self.initialization_source = "trusted_empty_start" if trusted_empty_start else "none"
        self._healthy = bool(trusted_empty_start)
        self.event_complete = bool(trusted_empty_start)
        self.segment = 0
        self.last_time_ns: int | None = None
        self.last_complete_ns: int | None = None
        self.last_state_change_ns: int | None = None
        self._snapshot_active = False
        self._pending_fills: dict[int, int] = {}

    @property
    def valid(self) -> bool:
        if not (self.initialized and self._healthy and self.event_complete):
            return False
        return bool(self._bid_prices and self._ask_prices and self._bid_prices[-1] <= self._ask_prices[0])

    def _invalidate(self, reason: str) -> None:
        self.diagnostics[reason] += 1
        if self._healthy:
            self.segment += 1
        self._healthy = False

    def _clear(self, snapshot: bool) -> None:
        self.orders.clear()
        self.bid_levels.clear()
        self.ask_levels.clear()
        self._bid_prices.clear()
        self._ask_prices.clear()
        self._pending_fills.clear()
        self.initialized = True
        self._healthy = True
        self._snapshot_active = snapshot
        self.initialization_source = "snapshot_clear" if snapshot else "observed_clear"
        self.segment += 1
        self.diagnostics["clears"] += 1

    def _level_change(self, side: str, price: int, size_delta: int, count_delta: int) -> None:
        levels, prices = (self.bid_levels, self._bid_prices) if side == "B" else (self.ask_levels, self._ask_prices)
        level = levels.get(price)
        if level is None:
            level = levels[price] = Level()
            insort(prices, price)
        level.size += size_delta
        level.count += count_delta
        if not level.count:
            if level.size:
                raise AssertionError("nonzero size on empty price level")
            del levels[price]
            prices.pop(bisect_left(prices, price))
        elif level.size <= 0 or level.count < 0:
            raise AssertionError("invalid aggregate level")

    def _add(self, order_id: int, side: str, price: int, size: int, time_ns: int, snapshot: bool) -> bool:
        if order_id in self.orders:
            self._invalidate("duplicate_adds")
            return False
        self.orders[order_id] = Order(side, price, size, None if snapshot else time_ns)
        self._level_change(side, price, size, 1)
        if not snapshot:
            self.flow[f"cum_add_shares_{side}"] += size
            self.flow[f"cum_add_count_{side}"] += 1
        return True

    def _removal_flow(self, order_id: int, order: Order, size: int, time_ns: int) -> None:
        """Account net quantity removed through C or an absolute M decrease."""
        side = order.side
        executed = min(self._pending_fills.get(order_id, 0), size)
        if executed:
            remaining_fill = self._pending_fills[order_id] - executed
            if remaining_fill:
                self._pending_fills[order_id] = remaining_fill
            else:
                self._pending_fills.pop(order_id, None)
        cancelled = size - executed
        self.flow[f"cum_removed_shares_{side}"] += size
        self.flow[f"cum_execution_removed_shares_{side}"] += executed
        self.flow[f"cum_cancel_shares_{side}"] += cancelled
        self.flow[f"cum_cancel_count_{side}"] += bool(cancelled)
        if cancelled and order.born_ns is not None:
            # Quantity-weighted ages cover partial cancels; one event count is
            # also available. Snapshot orders are explicitly left-censored.
            lifetime = time_ns - order.born_ns
            self.flow["cum_cancel_lifetime_ns"] += lifetime
            self.flow["cum_cancel_lifetime_count"] += 1
            self.flow["cum_cancel_age_share_ns"] += lifetime * cancelled
            self.flow["cum_cancel_age_shares"] += cancelled
        elif cancelled:
            self.diagnostics["left_censored_cancels"] += 1

    def _cancel(self, order_id: int, side: str, price: int, size: int, time_ns: int) -> bool:
        order = self.orders.get(order_id)
        if order is None:
            self._invalidate("unmatched_cancels")
            return False
        if order.side != side or order.price != price:
            self._invalidate("cancel_identity_mismatches")
            return False
        if size > order.size:
            self._invalidate("oversize_cancels")
            return False
        self._removal_flow(order_id, order, size, time_ns)
        order.size -= size
        self._level_change(side, price, -size, -int(order.size == 0))
        if not order.size:
            del self.orders[order_id]
        return True

    def _modify(self, order_id: int, side: str, price: int, size: int, time_ns: int, snapshot: bool) -> bool:
        order = self.orders.get(order_id)
        if order is None:
            self.diagnostics["unmatched_modifies"] += 1
            if self.unknown_modify == "add" and size > 0:
                self.diagnostics["modify_upserts"] += 1
                return self._add(order_id, side, price, size, time_ns, snapshot)
            self._invalidate("unknown_modify_invalidations")
            return False
        if order.side != side:
            self._invalidate("modify_side_mismatches")
            return False
        old_price, old_size = order.price, order.size
        if not snapshot:
            net_added = max(0, size - old_size)
            net_removed = max(0, old_size - size)
            if net_added:
                self.flow[f"cum_add_shares_{side}"] += net_added
                self.flow[f"cum_add_count_{side}"] += 1
            if net_removed:
                self._removal_flow(order_id, order, net_removed, time_ns)
        if size == 0:
            self._level_change(side, old_price, -old_size, -1)
            del self.orders[order_id]
            added, removed = 0, old_size
            self.diagnostics["zero_size_modify_removals"] += 1
        elif old_price == price:
            self._level_change(side, price, size - old_size, 0)
            added, removed = max(0, size - old_size), max(0, old_size - size)
        else:
            self._level_change(side, old_price, -old_size, -1)
            self._level_change(side, price, size, 1)
            added, removed = size, old_size
        if not snapshot:
            self.flow[f"cum_modify_add_shares_{side}"] += added
            self.flow[f"cum_modify_remove_shares_{side}"] += removed
            self.flow[f"cum_modify_count_{side}"] += 1
        # Time priority changes with a price change or size increase. This is
        # queue residence since the latest priority-losing change, not a claim
        # about the economic lifetime of the parent order.
        if size and (old_price != price or size > old_size):
            order.born_ns = None if snapshot else time_ns
        if size:
            order.price, order.size = price, size
        return True

    def apply(self, record: Any, *, emit: bool = True) -> dict[str, Any] | None:
        """Apply native DBN record or mapping; emit only a final-event state.

        Integer type errors raise (avoiding float-ID corruption). Semantic or
        data-quality errors are audited and produce invalid state until R.
        Records while invalid are ignored except an authoritative R/snapshot.
        """
        action = _field(record, "action", "N")
        side = _field(record, "side", "N")
        if isinstance(action, bytes):
            action = action.decode("ascii")
        if isinstance(side, bytes):
            side = side.decode("ascii")
        flags = _integer(_field(record, "flags", 0))
        time_ns = _integer(_field(record, "ts_recv", _field(record, "time_ns", 0)))
        order_id = _integer(_field(record, "order_id", 0))
        price = _integer(_field(record, "price", UNDEF_PRICE))
        size = _integer(_field(record, "size", 0))
        publisher = _field(record, "publisher_id")
        instrument = _field(record, "instrument_id")
        channel = _field(record, "channel_id")
        snapshot = bool(flags & F_SNAPSHOT)
        last = bool(flags & F_LAST)
        # Older equity historical data starts with an authoritative synthetic
        # empty-session clear, without F_SNAPSHOT and with unreliable receive
        # time. This narrow opt-in requires independent full-day request
        # provenance. Never infer it from an arbitrary R later in the stream.
        legacy_startup = bool(
            self.allow_legacy_startup_clear and not self.diagnostics["records"]
            and action == "R" and side == "N" and order_id == 0
            and price == UNDEF_PRICE and size == 0
            and flags in {F_BAD_TS_RECV, F_BAD_TS_RECV | F_LAST}
            and _integer(_field(record, "sequence", -1)) == 0
        )
        if legacy_startup:
            last = False  # Do not turn an unreliable synthetic clock into an observation.
        self.diagnostics["records"] += 1
        self.diagnostics[f"action_{action}"] += 1
        if channel is not None:
            self.channels_seen.add(_integer(channel))
        for name, value in (("publisher_id", publisher), ("instrument_id", instrument)):
            if value is not None:
                value = _integer(value)
                previous = getattr(self, name)
                if previous is None:
                    setattr(self, name, value)
                elif previous != value:
                    raise ValueError(f"Book received different {name}: {previous} != {value}")
        self.event_complete = False
        timestamp_safe = True
        if legacy_startup:
            self.diagnostics["legacy_startup_clears"] += 1
        elif time_ns == UNDEF_TIMESTAMP and snapshot and action == "R":
            # Vendor allows this sentinel for an empty snapshot. The state can
            # initialize, but no finite receive-time observation is invented.
            self.diagnostics["undefined_empty_snapshot_timestamp"] += 1
        elif time_ns < 0 or time_ns >= UNDEF_TIMESTAMP:
            self._invalidate("invalid_timestamps")
            timestamp_safe = False
        else:
            if self.last_time_ns is not None and time_ns < self.last_time_ns:
                self._invalidate("receive_time_reversals")
                timestamp_safe = False
            else:
                self.last_time_ns = time_ns
        if flags & (F_TOB | F_MBP):
            self._invalidate("non_native_records")
        if flags & F_MAYBE_BAD_BOOK:
            self._invalidate("maybe_bad_book_flags")
        if flags & F_BAD_TS_RECV and not legacy_startup and not (snapshot and (action == "R" or self._snapshot_active)):
            self._invalidate("bad_receive_timestamp_flags")

        reset_safe = timestamp_safe and not bool(flags & (F_TOB | F_MBP | F_MAYBE_BAD_BOOK))
        if action == "R" and reset_safe and (not flags & F_BAD_TS_RECV or snapshot or legacy_startup):
            self._clear(snapshot)
            if legacy_startup:
                self.initialization_source = "legacy_synthetic_session_clear"
            self.last_state_change_ns = self.last_time_ns
        elif snapshot and not self._snapshot_active:
            self._invalidate("snapshot_without_clear")
        elif snapshot and action != "A":
            self._invalidate("invalid_snapshot_actions")
        elif self._healthy and self.initialized:
            if action in {"A", "C", "M"}:
                if side not in {"A", "B"} or price <= 0 or price >= UNDEF_PRICE or size < 0 or (size == 0 and action != "M") or order_id < 0 or order_id >= (1 << 64):
                    self._invalidate("invalid_order_fields")
                else:
                    if action == "A":
                        changed = self._add(order_id, side, price, size, time_ns, snapshot)
                    elif action == "C":
                        changed = self._cancel(order_id, side, price, size, time_ns)
                    else:
                        changed = self._modify(order_id, side, price, size, time_ns, snapshot)
                    if changed:
                        self.last_state_change_ns = self.last_time_ns
            elif action == "F":
                if not snapshot:
                    if side not in {"A", "B", "N"} or size <= 0 or order_id < 0 or order_id >= (1 << 64):
                        self._invalidate("invalid_fill_fields")
                    else:
                        if order_id not in self.orders:
                            self.diagnostics["unmatched_fills"] += 1
                        self._pending_fills[order_id] = self._pending_fills.get(order_id, 0) + size
                        self.flow[f"cum_fill_shares_{side}"] += size
            elif action == "T":
                if not snapshot:
                    if side not in {"A", "B", "N"} or size <= 0 or price <= 0 or price >= UNDEF_PRICE:
                        self._invalidate("invalid_trade_fields")
                    else:
                        self.flow[f"cum_trade_shares_{side}"] += size
                        self.flow[f"cum_trade_count_{side}"] += 1
            elif action != "N":
                self._invalidate("unknown_actions")
        else:
            self.diagnostics["ignored_uninitialized_or_invalid"] += 1
        if snapshot:
            self.diagnostics["snapshot_records"] += 1
        if last:
            self.event_complete = True
            self.last_complete_ns = self.last_time_ns
            self.diagnostics["completed_events"] += 1
            if self._healthy and self._bid_prices and self._ask_prices:
                if self._bid_prices[-1] > self._ask_prices[0]:
                    self.diagnostics["crossed_events"] += 1
                elif self._bid_prices[-1] == self._ask_prices[0]:
                    self.diagnostics["locked_events"] += 1
            self._pending_fills.clear()
            self._snapshot_active = False
            if emit:
                return self.snapshot()
        return None

    def snapshot(self, time_ns: int | None = None) -> dict[str, Any]:
        """Read state without mutation; no forward extrapolation is asserted.

        ``time_ns`` is an optional observation timestamp and must be no earlier
        than the last applied record. ``last_complete_ns`` remains the actual
        data timestamp for independent freshness checks.
        """
        if time_ns is not None and self.last_time_ns is not None and time_ns < self.last_time_ns:
            raise ValueError("snapshot time precedes applied data")
        bp = self._bid_prices[-10:][::-1]
        ap = self._ask_prices[:10]
        bid, ask = (bp[0] if bp else None), (ap[0] if ap else None)
        result: dict[str, Any] = {
            "time_ns": self.last_time_ns if time_ns is None else time_ns,
            "last_complete_ns": self.last_complete_ns,
            "last_state_change_ns": self.last_state_change_ns,
            "valid": self.valid,
            "initialized": self.initialized,
            "event_complete": self.event_complete,
            "segment": self.segment,
            "publisher_id": self.publisher_id,
            "instrument_id": self.instrument_id,
            "bid": bid,
            "ask": ask,
            "bid_size": self.bid_levels[bid].size if bid is not None else 0,
            "ask_size": self.ask_levels[ask].size if ask is not None else 0,
            "bid_count": self.bid_levels[bid].count if bid is not None else 0,
            "ask_count": self.ask_levels[ask].count if ask is not None else 0,
            "mid": (bid + ask) / (2 * PRICE_SCALE) if bid is not None and ask is not None else nan,
            "spread_bps": 2e4 * (ask - bid) / (ask + bid) if bid is not None and ask is not None else nan,
            "order_count": len(self.orders),
            "bid_level_count": len(self.bid_levels),
            "ask_level_count": len(self.ask_levels),
            "initialization_source": self.initialization_source,
        }
        for depth in (1, 5, 10):
            bsize = sum(self.bid_levels[p].size for p in bp[:depth])
            asize = sum(self.ask_levels[p].size for p in ap[:depth])
            result[f"bid_depth_{depth}"] = bsize
            result[f"ask_depth_{depth}"] = asize
            result[f"depth_imbalance_{depth}"] = (bsize - asize) / (bsize + asize) if bsize + asize else nan
        result["depth_imbalance"] = result["depth_imbalance_1"]
        for side in ("B", "A", "N"):
            for name in ("add_shares", "add_count", "removed_shares", "cancel_shares", "cancel_count", "execution_removed_shares", "modify_add_shares", "modify_remove_shares", "modify_count", "trade_shares", "trade_count", "fill_shares"):
                key = f"cum_{name}_{side}"
                result[key] = self.flow[key]
        for name in ("cancel_lifetime_ns", "cancel_lifetime_count", "cancel_age_share_ns", "cancel_age_shares"):
            result[f"cum_{name}"] = self.flow[f"cum_{name}"]
        return result

    def level_snapshot(self, depth: int = 10) -> tuple[tuple[int, int, int, int, int, int], ...]:
        """Exact MBP-style price/size/count tuples, including empty levels.

        Each tuple is ``bid_px, ask_px, bid_sz, ask_sz, bid_ct, ask_ct``.  This
        diagnostic view preserves raw prices and does not assert book health;
        callers must separately check ``valid`` and ``event_complete``.
        """
        if depth < 1:
            raise ValueError("depth must be positive")
        bids = self._bid_prices[-depth:][::-1]
        asks = self._ask_prices[:depth]
        rows = []
        for position in range(depth):
            bid = bids[position] if position < len(bids) else None
            ask = asks[position] if position < len(asks) else None
            blevel = self.bid_levels.get(bid) if bid is not None else None
            alevel = self.ask_levels.get(ask) if ask is not None else None
            rows.append((bid if bid is not None else UNDEF_PRICE,
                         ask if ask is not None else UNDEF_PRICE,
                         blevel.size if blevel else 0, alevel.size if alevel else 0,
                         blevel.count if blevel else 0, alevel.count if alevel else 0))
        return tuple(rows)


class MarketBooks:
    """Route native records by publisher/instrument, never by float symbol IDs."""

    def __init__(self, **book_options: Any) -> None:
        self.books: dict[tuple[int, int], Book] = {}
        self.book_options = book_options

    def apply(self, record: Any, *, emit: bool = True) -> dict[str, Any] | None:
        key = (_integer(_field(record, "publisher_id")), _integer(_field(record, "instrument_id")))
        book = self.books.get(key)
        if book is None:
            book = self.books[key] = Book(publisher_id=key[0], instrument_id=key[1], **self.book_options)
        return book.apply(record, emit=emit)
