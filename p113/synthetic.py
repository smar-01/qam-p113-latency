"""Synthetic multi-venue quote streams with a known answer.

One leader venue's quote follows a tick random walk. Each follower copies every leader
quote change after `lag + noise`. Inside the volatile window, moves are more frequent
and larger, and follower lags are multiplied by `volatile_lag_mult`.

Clocks:
    ts_event = the venue's own matching-engine time (true time; clocks assumed in sync)
    ts_recv  = ts_event + that venue's feed delay to the capture point
So on ts_event a follower's lag is `lag`; on ts_recv it is lag + feed[f] - feed[leader].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TICK = 0.01


def make_quotes(leader: str = "XNAS",
                lags_ns: dict[str, int] | None = None,
                feed_ns: dict[str, int] | None = None,
                jitter_ns: int = 5_000,
                seconds: int = 3_600,
                calm_rate: float = 2.0,
                volatile_rate: float = 20.0,
                volatile_window: tuple[float, float] = (0.4, 0.6),
                volatile_lag_mult: float = 1.0,
                symbol: str = "SYN",
                t0_ns: int = 1_760_533_800_000_000_000,  # 2025-10-15 13:10 UTC, arbitrary
                seed: int = 7) -> pd.DataFrame:
    lags_ns = lags_ns if lags_ns is not None else {"XNYS": 50_000, "BATS": 200_000}
    feed_ns = feed_ns if feed_ns is not None else {}
    rng = np.random.default_rng(seed)

    # leader move times: Poisson, rate depends on regime
    v0, v1 = volatile_window[0] * seconds, volatile_window[1] * seconds
    times, regime = [], []
    s = 0.0
    while s < seconds:
        vol = v0 <= s < v1
        s += rng.exponential(1.0 / (volatile_rate if vol else calm_rate))
        times.append(s)
        regime.append(vol)
    times, regime = np.array(times[:-1]), np.array(regime[:-1])
    steps = rng.choice([-1, 1], size=len(times)) * np.where(regime, rng.integers(1, 4, len(times)), 1)
    bid_ticks = 10_000 + np.cumsum(steps)  # $100.00 start
    t_true = t0_ns + np.rint(times * 1e9).astype(np.int64)

    def venue_frame(venue, t_event):
        bid = bid_ticks * TICK
        df = pd.DataFrame({
            "ts_event": t_event,
            "venue": venue,
            "symbol": symbol,
            "bid": np.round(bid, 2),
            "ask": np.round(bid + TICK, 2),
            "bid_sz": 100,
            "ask_sz": 100,
        })
        # opening quote one second before the first move so every venue has a state
        first = df.iloc[[0]].copy()
        first["ts_event"] = t0_ns - 1_000_000_000
        first["bid"], first["ask"] = np.round(10_000 * TICK, 2), np.round(10_001 * TICK, 2)
        df = pd.concat([first, df], ignore_index=True)
        df["ts_recv"] = df["ts_event"] + feed_ns.get(venue, 0)
        return df

    frames = [venue_frame(leader, t_true)]
    for v, lag in lags_ns.items():
        lag_each = np.where(regime, lag * volatile_lag_mult, lag)
        noise = rng.normal(0, jitter_ns, len(t_true)).clip(-3 * jitter_ns, 3 * jitter_ns)
        frames.append(venue_frame(v, t_true + np.rint(lag_each + noise).astype(np.int64)))
    q = pd.concat(frames, ignore_index=True)
    cols = ["ts_recv", "ts_event", "venue", "symbol", "bid", "ask", "bid_sz", "ask_sz"]
    return q[cols].sort_values(["ts_recv", "venue"], kind="stable").reset_index(drop=True)
