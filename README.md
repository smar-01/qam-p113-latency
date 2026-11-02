# p113: Exchange latency and market reaction study

Fordham QAM capstone (Fall 2026), client Databento. This project measures how
quickly one venue's bid or ask reaches another venue's changed price, alongside
midpoint responses, lead-lag diagnostics and descriptive volatility regimes.

## Numerical and input corrections

The standalone default is **numerical-only**. `analysis_mode.json` records the
rules used for each run. No external research package is required.

- Midpoint event detection and response ratios use native integer bid+ask
  prices. This removes float-only zero-price-change events and their nearly
  zero response-ratio denominators. Synthetic dollar quotes are quantized on
  the nanodollar lattice before averaging.
- Lead-lag increments are summed in fixed 10-microsecond bins anchored at the
  UTC epoch before energy normalization. The result is a bounded, uncentered
  normalized cross-product, not a Pearson correlation. Zero energy is undefined.
- DBN loading retains original integer prices, flags and raw size/state updates.
  Each invocation must use one aligned request interval across its input files.
- Volatility is calculated separately on each clock from raw states, retaining
  size-only freshness updates. It requires a full 300-return warmup and explicit
  supported observations. Unavailable regimes remain `unknown`.
- Download caches include the full symbol set and exact intraday start/end in
  their identity. Reuse requires a successful completion receipt, matching byte
  count and SHA-256. An existing unverified or changed cache is rejected.

**The bid/ask reaction-latency algorithm and its original event selection remain
unchanged.** Its matched medians are conditional on matching within the chosen
horizon. The volatility freshness rules do not filter reaction-latency events;
this patch does not supply a full flags/ties/freshness-aware latency engine.

## Setup and offline verification

Use Python 3.12 and the existing pinned requirements:

```text
python -m venv .venv
# Activate .venv using your platform's normal activation command.
python -m pip install -r requirements.txt
python -m pytest
python analyze.py --synthetic --clock both
python analyze.py --synthetic --clock both --event-support complete --out outputs/synthetic_complete
```

Both synthetic commands run without an API key or network requests. The
`--numerical-only` flag explicitly acknowledges the default mode; it does not
change its rules. Tests include integer-midpoint counterexamples, an independent
dense-grid normalization check, cache identity/integrity, known lags, unsupported
endpoints and raw-state volatility support.

## Midpoint-response support policy

`--event-support legacy` is the default. It retains the starter's per-offset
cohorts and last-state carrying so arithmetic fixes can be assessed separately
from observation-policy changes.

`--event-support complete` uses raw rows, a strict prior baseline and one
continuously observed source/follower cohort across all requested offsets. It
excludes unsupported request endpoints. It does not resolve conflicting tied
states, add a quote-age rule to midpoint responses or qualify bid/ask latency.
The complete cohort is fixed separately for each follower, so compare the
reported `n` counts as well as ratios. Signed response ratios can legitimately
be negative or exceed one and are not clipped.

## Real-data acquisition and analysis

The API key is read only from `DATABENTO_API_KEY`; it is never written to disk.
Without a key, `pull.py` stops before constructing an API client or writing data.
First price a request, then choose an explicit cost ceiling:

```text
python pull.py --cost-only
python pull.py --max-cost 0 --yes
python analyze.py --data data/raw --clock both --out outputs/real
```

Defaults are AAPL, SPY and MSFT; MBP-1; October 15, 2026 (UTC day); Nasdaq,
NYSE, Arca, Cboe BZX, Cboe EDGX and IEX. Use `--symbols`, `--start`, `--end` and
`--datasets` to select exact requests. `--datasets all` selects the venue table.
The cost gate remains in place. `--schema mbo` can retrieve MBO, but this
analysis reads MBP-1 only.

Analyze each aligned date/window separately. Mixing requests with different
start/end boundaries is rejected. Preserve the original DBN metadata and cache
receipts; changing a filename is not evidence of matching request identity.

## Outputs and interpretation

Each output directory contains `analysis_mode.json`. Each selected clock gets
its own `latency_summary_*`, `latency_matrix_*`, `leadership_share_*`,
`latency_summary_by_regime_*`, `realized_vol_*`, `leadlag_*`, `event_study_*` and
`latency_events_*.parquet` files. Figures label the corrected cross-product and
signed response ratio explicitly. In particular, volatility now uses
`realized_vol_ts_recv.csv` and `realized_vol_ts_event.csv` rather than one shared
receive-clock regime file.

`ts_recv` includes feed transport to the capture point; `ts_event` compares
exchanges' own clocks. Neither is pure propagation time. Counting leadership is
not Hasbrouck information share, and matching does not identify causality.
Volatility labels are retrospective; these outputs do not establish forecasting
skill or statistical significance. Quote-quality gaps, stale observations,
auctions and odd-lot conventions still require a separately specified quality
study. Numerical corrections alone do not solve these measurement issues.

Existing `results/` documents describe earlier runs; they are not automatically
regenerated or validated by this source patch. Re-run the appropriate saved
input windows with explicit policies before replacing published figures.
