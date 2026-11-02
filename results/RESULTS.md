# p113 first results (run 2026-11-02)

Data: Databento MBP-1 (top of book per venue) for AAPL, SPY and MSFT on 6 venues: Nasdaq (XNAS), NYSE (XNYS),
NYSE Arca (ARCX), Cboe BZX (BATS), Cboe EDGX (EDGX), IEX (IEXG). Three days, picked by SPY's high-low range
among 2026-08-03 to 2026-10-30.

| day | SPY range | why |
|---|---|---|
| 2026-08-04 | 1.98% | widest day in the window |
| 2026-10-15 | 1.14% | middle, the first day pulled |
| 2026-10-13 | 0.37% | narrowest day in the window |

Reaction latency: after a leader venue's best bid or ask moves to a new price, how long until the follower's
quote reaches that price. Cases where the follower was already there are dropped. No match within 10 ms is censored.
Tables and charts per day are in the dated folders here.

## What we see

Median of the 30 pair medians, in microseconds:

| day | Databento clock (ts_recv) | exchange clocks (ts_event) |
|---|---|---|
| 2026-08-04 | 354.6 | 222.0 |
| 2026-10-15 | 472.5 | 337.0 |
| 2026-10-13 | 413.1 | 275.8 |

1. **Most venue pairs react in a few hundred microseconds.** Excluding IEX, the median pair is about 200-350 us.
2. **Cboe BZX and EDGX move almost together:** 9-29 us depending on day and clock.
3. **IEX stands apart.** Other venues reach IEX's price in about 400-585 us on ts_recv, consistent with its 350 us
   speed bump. Moves that start on IEX get matched slowly (660-2,240 us) and it has the lowest leadership share.
4. **Nasdaq's lead depends on the clock.** On ts_event it has the highest leadership share each day (0.68-0.71). On
   ts_recv it is about level with Cboe BZX (0.61-0.66).
5. **Latency is higher in volatile stretches inside a day.** Within regular hours, volatile seconds are about 12-22%
   slower than calm ones, in 27-30 of 30 pairs on every day and clock, also with the first and last 5 minutes dropped
   (`rth_regimes.csv`).
6. **A wider day is not a slower day.** The widest day has the lowest latency of the three, so the within-day effect
   does not carry over to day-level volatility in this sample.

## What this can't say yet

- Three days and three symbols. These are descriptive numbers with no confidence intervals.
- Only 24-29% of behind cases match within 10 ms. The rest are censored, so the medians describe the cases where the
  follower reached the price, and a longer horizon could change them. This is the first thing to test.
- "Leader" means moved first, not caused. A follower may be reacting to something else, such as its own order flow.
- ts_recv includes each feed's path to Databento's capture point. ts_event compares different exchanges' clocks.
  Neither is pure propagation time, and the two give different pair rankings.
- Lead-lag peak correlations on 10/15 are very small (mostly under 0.05), so the peak lags are weak evidence.
- Volatile seconds use a trailing 5-minute realized vol. Volume and quote traffic may rise at the same times, and this
  split does not separate them.
- Odd lots and auctions get no special treatment. MBO is not analyzed yet.

## Checks

- 19 unit tests recover known lags on synthetic data.
- `checks/independent_check.py` recomputes 6 symbol-pair cases from the raw files with separate code. Match counts and
  medians were identical to the pipeline on 2026-10-15.
- Locked or crossed quotes were 0.000% of regular-hours rows for AAPL and SPY on the 4 venues that check covers.

## Next

Test the 10 ms horizon, add more days and symbols with bootstrap intervals, add the remaining venues, then move to MBO
as Databento suggested.
