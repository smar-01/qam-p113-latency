# p113: Exchange latency and market reaction study

Fordham QAM capstone (Fall 2026), client Databento. Question: when one US equity venue
changes its quote (say Nasdaq), how fast do the other venues (NYSE, Arca, Cboe, IEX, ...)
follow? Does that lead-lag change in volatile periods?

## Setup

```
py -3.12 -m venv C:\fordham-s4-classes\venvs\p113
C:\fordham-s4-classes\venvs\p113\Scripts\python -m pip install -r requirements.txt
```

All commands below run from this folder with that venv's `python`.

## Run it now, without a key

```
python -m pytest              # synthetic venues with known lags; checks the code recovers them
python analyze.py --synthetic # full pipeline on synthetic data -> outputs/synthetic/
```

## First real pull

The key is read only from the env var `DATABENTO_API_KEY`. It is never written to disk.

```
$env:DATABENTO_API_KEY = "<your key>"     # PowerShell, this window only
python pull.py --cost-only                # prices the pull; downloads nothing
python pull.py --max-cost 0 --yes         # downloads only if the quote is <= $0
python analyze.py                         # -> outputs/real/
```

Defaults: AAPL, SPY, MSFT; MBP-1; 2026-10-15 (UTC day); six venues (XNAS.ITCH,
XNYS.PILLAR, ARCX.PILLAR, BATS.PITCH, EDGX.PITCH, IEXG.TOPS). Change with `--symbols`,
`--start`, `--end`, `--datasets` (`--datasets all` = all 15 venues in `p113/venues.py`).
Pick a day with a big move in a liquid name (Eric's advice on the kickoff call).
`pull.py` first checks each dataset offers the schema (`metadata.list_schemas`), then prices
every dataset (`metadata.get_cost`, `get_billable_size`), and only then downloads. It refuses
to download above `--max-cost` (default $0). Files land in `data/raw/<dataset>/` as
`.dbn.zst` (gitignored). Existing files are not downloaded again.

## What it measures

- **Reaction latency.** For each change in a leader venue's best bid (or ask) to price P,
  the time until the follower's best bid (ask) reaches P. If the follower is already at P
  when the leader moves, it was not behind and the event is left out ("already"). No match
  within `--horizon-ms` (default 10 ms) is "censored".
- **Lead-lag.** Cross-correlation of the two venues' mid changes at lags from -2 ms to
  +2 ms in 10 µs steps, computed on raw event times (no fixed grid). Positive lag = the second
  venue moves after the first. LLR > 1 means the first venue leads.
- **Leadership share** (information-share-style). For venue i: matched events where i moved
  first, divided by all matched events involving i. 1 = always first, 0.5 = even, 0 = always
  follows. It is a counting measure, not Hasbrouck's information share.
- **Event study.** After each leader mid change, the share of that move each venue has
  absorbed at offsets from -100 µs to 10 ms. 1.0 = caught up.
- **Calm vs volatile.** Cross-venue median mid every second; 5-minute trailing realized vol;
  at or above the day's 75th percentile (`--vol-q`) = volatile. Every latency event is tagged.

## Clocks

Everything runs twice, once per clock:
- `ts_recv`: Databento's capture time. One clock for all venues, but it includes each feed's
  path to Databento's capture point, so a venue farther away looks slower.
- `ts_event`: each exchange's own matching-engine time. Closer to when it happened, but it
  compares different exchanges' clocks, so small differences (single µs) are not trustworthy.

## Outputs (`outputs/<synthetic|real>/`, one set per clock)

| file | what it is |
|---|---|
| `heatmap_<clock>.png`, `latency_matrix_<clock>.csv` | median latency, leader rows x follower columns (µs); blank = fewer than 20 matches |
| `distributions_<leader>_<clock>.png` | latency histograms (log µs) after each leader change, one line per follower |
| `latency_summary_<clock>.csv` | per pair: events, events where the follower was behind, matches, match rate, p10/median/p90/mean µs |
| `latency_summary_by_regime_<clock>.csv`, `regimes_<leader>_<clock>.png` | the same split calm vs volatile |
| `leadlag_<leader>_<clock>.png`, `leadlag_curves_<clock>.csv`, `leadlag_summary_<clock>.csv` | cross-correlation curves; peak lag, peak correlation, LLR per pair |
| `leadership_share_<clock>.csv` | one number per venue, see above |
| `event_study_<leader>_<symbol>_<clock>.png/.csv` | share of the leader's move absorbed over time |
| `latency_events_<clock>.parquet` | every leader event with its follower result and regime |
| `realized_vol.csv` | the 1-second realized vol series and regime labels |

## Assumptions and limits

- MBP-1 (top of book per venue). The client prefers MBO for the exact instant an order
  lands; `--schema mbo` will pull it, but the analysis reads MBP-1 only so far.
- Only price changes count; size-only updates are dropped.
- Odd lots, locked/crossed quotes and auction periods are not treated specially yet.
- The synthetic tests prove the measurement recovers a known lag; they say nothing about
  real venue behavior.

## Layout

```
pull.py        cost check + download (needs DATABENTO_API_KEY)
analyze.py     everything else
p113/venues.py     venue <-> Databento dataset table, checked against the installed package
p113/load.py       DBN -> one tidy quote table
p113/latency.py    reaction latency, summaries, matrix, leadership share
p113/leadlag.py    cross-correlation, LLR, event study
p113/regimes.py    realized vol and calm/volatile labels
p113/figures.py    plots
p113/synthetic.py  known-lag test data
tests/         pytest
```
