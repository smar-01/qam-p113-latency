"""US equity venues and their Databento per-venue (proprietary feed) datasets.

Dataset codes are checked against the installed databento package's `Dataset` enum
at import time (see `check_against_package`). Whether each dataset serves the `mbp-1`
schema for a given date is checked live by pull.py with `metadata.list_schemas`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Venue:
    dataset: str  # Databento dataset code
    venue: str    # MIC-style venue code, matches databento Publisher.venue
    name: str


# Every lit US equity exchange with a per-venue Databento dataset (databento 0.89.0).
VENUES: tuple[Venue, ...] = (
    Venue("XNAS.ITCH", "XNAS", "Nasdaq"),
    Venue("XBOS.ITCH", "XBOS", "Nasdaq Texas (BX)"),
    Venue("XPSX.ITCH", "XPSX", "Nasdaq PSX"),
    Venue("XNYS.PILLAR", "XNYS", "NYSE"),
    Venue("ARCX.PILLAR", "ARCX", "NYSE Arca"),
    Venue("XASE.PILLAR", "XASE", "NYSE American"),
    Venue("XCHI.PILLAR", "XCHI", "NYSE Texas"),
    Venue("XCIS.PILLAR", "XCIS", "NYSE National"),
    Venue("BATS.PITCH", "BATS", "Cboe BZX"),
    Venue("BATY.PITCH", "BATY", "Cboe BYX"),
    Venue("EDGA.PITCH", "EDGA", "Cboe EDGA"),
    Venue("EDGX.PITCH", "EDGX", "Cboe EDGX"),
    Venue("MEMX.MEMOIR", "MEMX", "MEMX"),
    Venue("EPRL.DOM", "EPRL", "MIAX Pearl"),
    Venue("IEXG.TOPS", "IEXG", "IEX"),
)

# A smaller default set for a first, cheap pull: the big four families plus IEX.
DEFAULT_DATASETS: tuple[str, ...] = (
    "XNAS.ITCH", "XNYS.PILLAR", "ARCX.PILLAR", "BATS.PITCH", "EDGX.PITCH", "IEXG.TOPS",
)

BY_DATASET = {v.dataset: v for v in VENUES}
BY_VENUE = {v.venue: v for v in VENUES}


def check_against_package() -> list[str]:
    """Return dataset codes in VENUES that the installed databento package does not know."""
    import databento as db

    known = {d.value for d in db.Dataset}
    return [v.dataset for v in VENUES if v.dataset not in known]


def venue_for_publisher(publisher_id: int) -> str:
    """Map a DBN publisher_id to its venue code using the package's Publisher table."""
    import databento as db

    return db.Publisher.from_int(int(publisher_id)).venue.value
