"""Loading real DBN bytes, the venue table, and pull.py's no-key / cost-only behaviour."""
import databento as db
import databento_dbn as dbn
import numpy as np
import pytest

import pull
from p113 import venues
from p113.load import bbo_changes, from_store


def test_venue_datasets_exist_in_installed_package():
    assert venues.check_against_package() == []


def test_publisher_ids_map_to_venues():
    for v in venues.VENUES:
        pubs = [p for p in db.Publisher if p.dataset.value == v.dataset]
        assert pubs, v.dataset
        assert venues.venue_for_publisher(pubs[0].to_int()) == v.venue


def _mbp1(pub, ts_event, ts_recv, bid, ask, seq):
    lv = dbn.BidAskPair(bid_px=bid, ask_px=ask, bid_sz=100, ask_sz=200, bid_ct=1, ask_ct=1)
    return dbn.MBP1Msg(publisher_id=pub, instrument_id=1, ts_event=ts_event, price=bid, size=100,
                       action=dbn.Action.ADD, side=dbn.Side.BID, depth=0, flags=130,
                       ts_recv=ts_recv, ts_in_delta=0, sequence=seq, levels=lv)


def test_from_store_reads_dbn_mbp1_bytes():
    t0 = 1_760_000_000_000_000_000
    meta = dbn.Metadata(dataset="XNAS.ITCH", start=t0, end=t0 + 10**9, stype_in=dbn.SType.RAW_SYMBOL,
                        stype_out=dbn.SType.INSTRUMENT_ID, schema=dbn.Schema.MBP_1, symbols=["AAPL"])
    buf = bytes(meta.encode())
    buf += bytes(_mbp1(2, t0 + 100, t0 + 900, 189_500_000_000, 189_510_000_000, 1))
    buf += bytes(_mbp1(2, t0 + 200, t0 + 1_000, 189_500_000_000, 189_510_000_000, 2))  # size-only, dropped
    buf += bytes(_mbp1(2, t0 + 300, t0 + 1_100, 189_510_000_000, dbn.UNDEF_PRICE, 3))
    q = bbo_changes(from_store(db.DBNStore.from_bytes(buf)))
    assert list(q["venue"]) == ["XNAS", "XNAS"]
    assert list(q["ts_event"]) == [t0 + 100, t0 + 300]
    assert list(q["ts_recv"]) == [t0 + 900, t0 + 1_100]
    assert q["bid"].tolist() == [189.5, 189.51]
    assert q["ask"].iloc[0] == 189.51 and np.isnan(q["ask"].iloc[1])


def test_pull_without_key_stops_before_any_api_call(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    monkeypatch.setattr(pull, "DATA", tmp_path / "raw")
    monkeypatch.setattr(db, "Historical", lambda *a, **k: pytest.fail("client created without a key"))
    assert pull.main(["--cost-only"]) == 2
    assert not (tmp_path / "raw").exists()


class FakeClient:
    calls = []

    def __init__(self, key):
        self.metadata, self.timeseries = self, self

    def list_schemas(self, dataset):
        FakeClient.calls.append(("list_schemas", dataset))
        return ["mbo", "mbp-1"]

    def get_cost(self, **kw):
        FakeClient.calls.append(("get_cost", kw["dataset"]))
        return 1.25

    def get_billable_size(self, **kw):
        return 2_000_000_000

    def get_range(self, **kw):
        FakeClient.calls.append(("get_range", kw["dataset"]))
        kw["path"].write_bytes(b"x")


def test_cost_only_prices_and_downloads_nothing(monkeypatch, tmp_path):
    FakeClient.calls = []
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-not-a-key")
    monkeypatch.setattr(pull, "DATA", tmp_path / "raw")
    monkeypatch.setattr(pull, "ROOT", tmp_path)
    monkeypatch.setattr(db, "Historical", FakeClient)
    assert pull.main(["--cost-only", "--datasets", "XNAS.ITCH", "XNYS.PILLAR"]) == 0
    kinds = [c[0] for c in FakeClient.calls]
    assert kinds.count("get_cost") == 2 and "get_range" not in kinds
    assert not (tmp_path / "raw").exists()


def test_download_refused_above_max_cost(monkeypatch, tmp_path):
    FakeClient.calls = []
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-not-a-key")
    monkeypatch.setattr(pull, "DATA", tmp_path / "raw")
    monkeypatch.setattr(pull, "ROOT", tmp_path)
    monkeypatch.setattr(db, "Historical", FakeClient)
    assert pull.main(["--datasets", "XNAS.ITCH", "--max-cost", "1.0", "--yes"]) == 3
    assert "get_range" not in [c[0] for c in FakeClient.calls]


def test_download_after_cost_check(monkeypatch, tmp_path):
    FakeClient.calls = []
    monkeypatch.setenv("DATABENTO_API_KEY", "dummy-not-a-key")
    monkeypatch.setattr(pull, "DATA", tmp_path / "raw")
    monkeypatch.setattr(pull, "ROOT", tmp_path)
    monkeypatch.setattr(db, "Historical", FakeClient)
    assert pull.main(["--datasets", "XNAS.ITCH", "--max-cost", "5", "--yes"]) == 0
    kinds = [c[0] for c in FakeClient.calls]
    assert kinds.index("get_cost") < kinds.index("get_range")
    assert list((tmp_path / "raw" / "XNAS.ITCH").glob("mbp-1_*.dbn.zst"))
