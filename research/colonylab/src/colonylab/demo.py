"""Reproducible, explicitly synthetic data for exercising the research pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_demo() -> pd.DataFrame:
    """Return three synthetic sessions; never interpret their results as evidence.

    The generator changes arrival, size and direction distributions every minute.
    A weak, artificial relationship between order flow and subsequent midpoint
    changes makes the end-to-end workflow visible. No real data are represented.
    """
    rng = np.random.default_rng(4317)
    sessions: list[pd.DataFrame] = []
    mid = 100.0
    for day in ("2025-01-06", "2025-01-07", "2025-01-08"):
        count = 6000
        times = pd.date_range(f"{day} 14:30:00", periods=count, freq="100ms", tz="UTC")
        regime = (np.arange(count) // 600) % 3
        trade = rng.random(count) < np.choose(regime, [0.30, 0.78, 0.48])
        # Directional blocks alternate so that training does not just learn a trend.
        block_sign = np.where((np.arange(count) // 300) % 2 == 0, 1, -1)
        buy_probability = np.where(regime == 1, 0.5 + 0.36 * block_sign, 0.50)
        sign = np.where(rng.random(count) < buy_probability, 1, -1)
        volume_scale = np.choose(regime, [30.0, 180.0, 75.0])
        sizes = np.maximum(1, rng.lognormal(np.log(volume_scale), 0.5)).astype(int)
        mids = np.empty(count)
        pressure = 0.0
        for i in range(count):
            pressure = 0.93 * pressure + (float(sign[i]) if trade[i] else 0.0)
            mid += rng.normal(0, 0.003) + 0.00012 * pressure
            mids[i] = mid
        spread = np.choose(regime, [0.02, 0.03, 0.06])
        bid = np.round(mids - spread / 2, 4)
        ask = np.round(mids + spread / 2, 4)
        side = np.where(sign > 0, "B", "A")
        side[rng.random(count) < 0.04] = "N"
        frame = pd.DataFrame(
            {
                "ts_event": times,
                "ts_recv": times + pd.Timedelta("300us"),
                "symbol": "SYNTH",
                "instrument_id": 1,
                "publisher_id": "SYNTHETIC",
                "action": np.where(trade, "T", "M"),
                "side": np.where(trade, side, "N"),
                "price": np.where(sign > 0, ask, bid),
                "size": np.where(trade, sizes, 0),
                "bid_px_00": bid,
                "ask_px_00": ask,
                "bid_sz_00": rng.integers(100, 1000, count),
                "ask_sz_00": rng.integers(100, 1000, count),
                "synthetic": True,
            }
        )
        sessions.append(frame)
    result = pd.concat(sessions, ignore_index=True)
    result.attrs["synthetic"] = True
    return result
