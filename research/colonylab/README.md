# ColonyLab: first Databento experiment

This research project is integrated into the public
[`smar-01/qam-p113-latency` repository](https://github.com/smar-01/qam-p113-latency)
under `research/colonylab/`. It includes the original execution-context work,
network and candle studies, full-session MBO/SIRS experiment, and the later
latency, Apple-event and stationarity extensions. The source and protocols are
preserved; the combined repository's [main README](../../README.md) explains
installation and the test command that covers both projects.

From the combined repository root, activate your Python environment, run
`python -m pip install -r requirements-research.txt`, then `python -m pytest`.
`python -m colonylab.cli demo` runs the synthetic execution-context demonstration;
`python research/colonylab/scripts/run_mbo_sir.py --help` describes the separate
MBO/SIRS driver without acquiring data. Project defaults write inside this
directory's ignored `data/` and `out/` folders.

The historical command examples below run from `research/colonylab/` and show a
project-local `.venv`. If using the combined repository's activated environment,
use `python` in place of `.\.venv\Scripts\python.exe` in those examples.
The [earlier-study results bundle](results/earlier/README.md) covers clustering,
network prediction, candle comparisons and [full-session MBO/SIRS findings](#full-session-mbo-venue-network-and-recurrent-sirs-test).

For the current team review, start with the tracked
[results and validation bundle](results/README.md). It summarizes the corrected
team-code experiment, native venue latency, Apple events and ADF/KPSS diagnostics
without requiring market-data downloads. See [comparison source provenance](comparisons/README.md)
for the pinned original and corrected implementations.

This project tests whether anonymous execution contexts can be grouped into recurring patterns with different later price responses. The first live API test succeeded using AAPL from EQUS.MINI, MBP-1, October 23, 2026, 10:00–10:10 AM New York.

The downloaded sample has **21,687 records and 1,149 executed trades**. Databento estimated **$0.001938968897**, and the download was allowed only below a fresh **$0.01 estimate threshold**. This is an estimate, not a final billing statement or a provider-enforced spending cap. Subsequent analysis reads the saved file without further charges.

The default one-second feature run fits three exploratory mixture components using 266 training events and describes 129 later holdout events. Of those holdout events, 89 have an estimated direction and 40 remain unknown; none have feed-confirmed direction. These counts reflect freshness, timestamp ambiguity and the five-earlier-trade requirement. A separate ten-second feature sensitivity run retains 330 holdout events, with two feed-confirmed and 257 estimated directions.

## Run locally

For Linux and Codex cloud installation, see [CLOUD_SETUP.md](CLOUD_SETUP.md). The source is versioned in Git; the October 28 `data/` and `out/` research snapshot is packaged as private release archives with a SHA-256 manifest. The newer November studies are summarized in the tracked results bundle; their additional raw data and full local outputs are not part of that older release.

Use PowerShell in this folder with the combined environment activated, or create a project-local environment. The project uses the official Databento Python SDK; no OpenAI API key is needed.

```powershell
# Open a masked local prompt if you need to set or replace the Databento key.
python scripts/setup_key.py

# Free account authentication, dataset/schema and date-range checks.
.\.venv\Scripts\python.exe -m colonylab.cli doctor

# Free estimate for the exact request in pilot.toml. Does not download data.
.\.venv\Scripts\python.exe -m colonylab.cli plan

# Analyze the real file already downloaded. No network access or charges.
.\.venv\Scripts\python.exe -m colonylab.cli analyze data/pilot/sample.dbn.zst

# Offline software demonstration; all its data and results are synthetic.
.\.venv\Scripts\python.exe -m colonylab.cli demo

# Automated checks; no credentials or network access used.
.\.venv\Scripts\python.exe -m pytest -q
```

For a fresh installation: `python -m venv --system-site-packages .venv`, then `.\.venv\Scripts\python.exe -m pip install -e . --no-build-isolation`. Python 3.11 or later is required. Development tests also need pytest.

## Download another small sample

Edit `pilot.toml` with explicit symbols and timezone-aware timestamps, then run `plan` and inspect `out/pilot-plan.json`. The pilot restricts requests to at most 10 symbols, 60 minutes and 1,000,000 records. `ALL_SYMBOLS` is disabled. To execute a reviewed request:

```powershell
.\.venv\Scripts\python.exe -m colonylab.cli fetch --max-cost-usd 0.01 --out data/pilot_next
.\.venv\Scripts\python.exe -m colonylab.cli analyze data/pilot_next/sample.dbn.zst --out out/pilot_next
```

The fetch command requotes the exact request, including its record limit, before downloading. It refuses an estimate above the supplied budget and refuses to overwrite completed downloads. If conversion is interrupted after saving the DBN, it can re-export the saved file without downloading it again. A recovered file may still require inspection for truncation after a network interruption; recovery is not a completeness guarantee. Prices are converted using the current SDK's `price_type="float"` API.

The key is stored in `.env`, which is ignored by Git. Data and generated results are also excluded from Git history and preserved separately in the private research release described in `CLOUD_SETUP.md`. Do not paste the key into chat or put it in `pilot.toml`. Error output redacts the configured key and its Basic-auth encoding.

## What the analysis does

Each trade anchors an execution context. Features describe **strictly preceding** activity within the configured window: trade count and size, directional volume imbalance and persistence, interarrival gaps, recent returns and volatility, spread and depth. The current initiating direction is used only when measuring the later signed response. At least five earlier trades, a full observed feature window and a fresh baseline price are required.

A scaler and a three-component Gaussian mixture are fitted only to the chronological training block. Membership probabilities for later events come from that frozen model. The training block ends at least the longest outcome horizon before the holdout block begins. Colony numbers are arbitrary and need not match across runs.

The default response horizons are 1 ms, 10 ms, 100 ms, 1 second, 10 seconds and 1 minute. The baseline is strictly before the event; the horizon observation is at or before the target time. No future quote filling or extrapolation past the sample end is allowed. Tied conflicting quote states and stale observations are handled conservatively. Receive timestamps are preferred when present, with nanosecond precision preserved.

MBP-1 uses midpoint responses. The `trades` schema supports transaction-price responses, which include bid/ask bounce. MBO is deliberately rejected by this first pipeline: it needs order-book reconstruction and different features.

Supplied B/A aggressor sides are distinguished from estimates. For unavailable sides, a strictly prior fresh midpoint comparison supplies an explicitly labelled estimate when possible; midpoint ties remain unknown. Tick-rule signs are diagnostic only and do not drive signed-response results. In the first sample, **1,131 of 1,149 trade sides are unavailable**, so dependence on estimated signs is a major limitation.

## Outputs

Real-data outputs are under `out/pilot/`; synthetic outputs are under `out/demo/`.

| File | Contents |
| --- | --- |
| `response_curve.png` | Holdout signed midpoint response by colony and horizon |
| `response_curve.csv` | Probability-weighted means, negative-response fractions and sample support |
| `coverage.csv` | Missing, stale, unknown-direction and usable-event counts |
| `event_memberships.csv` | Chronological split, membership probabilities and sign sources |
| `event_features.csv` | Past-only model inputs |
| `event_responses.csv` | Auditable event-level horizon outcomes |
| `colony_profiles.csv` | Training feature summaries for interpreting colonies |
| `run_summary.json` | Run settings, sample counts and methodological limitations |

The raw DBN, CSV export, request and estimated-cost receipt are under `data/pilot/`. `out/doctor.json` contains the free connection checks. A ten-second feature-window sensitivity run can be reproduced with `analyze data/pilot/sample.dbn.zst --window-ms 10000 --out out/pilot_10s`; it reuses the same data and does not create independent evidence.

## Cboe venue, maker-opportunity and retail context

Public Cboe reports are now downloaded separately under `data/cboe/`, without API keys or Databento charges. The venue history covers 204 observed dates in 2026 through October 23. For October 23, it contains 21 reporting centers; FINRA facilities account for about 50.21% of reported shares. That fraction describes off-exchange reporting, not retail participation.

The maker report is a **BZX weekly average-daily opportunity report**, observed for October 19–23. AAPL has 2,876,785 shares of volume opportunity against BZX ADV of 1,137,661. Its raw ratio of 2.52868 means **252.87% of venue ADV**, not 2.53% and not executed market-maker volume. AAPL ranks around the 65.86th percentile on that ratio among positive-ADV symbols in this snapshot. Zero-ADV symbols have no defined opportunity ratio.

The retail report provides symbol/side submitted and executed shares, missed and exhausted liquidity, and price-improvement measures. Raw side codes B/S/T/E are retained separately; T is not a total row. The current integration does not assume report-specific meanings for T/E or assign these rows to our individual trades. Cboe describes RPI price improvement per share in dollars relative to the far NBBO; the raw CSV field is preserved. Report-specific definitions for effective-spread weighting, median order size and `shares_exec_div_bats_vol` remain unverified. Zero quality values on zero-execution rows are masked in derived execution-quality columns. Submitted minus executed shares need not equal missed plus exhausted opportunity.

```powershell
# Public Cboe reports; reuses files already downloaded.
.\.venv\Scripts\python.exe scripts/fetch_cboe_context.py --date 2026-10-23

# Offline parsing, descriptive context and event-table enrichment.
.\.venv\Scripts\python.exe -m colonylab.cli context --date 2026-10-23 --symbol AAPL
```

Outputs under `out/context/` include daily venue shares, concentration measures, observed-date lags, an AAPL maker/retail snapshot, plots and `events_with_context.csv`. The latter attaches context to the 1,149 event rows while retaining colony membership and split fields. These external fields are **not used in model fitting or prediction**. Publication times and historical revisions are unverified: even a previous-date row is not automatically proven available at a specific intraday time. `lag1` means the previous date present in the file, with calendar gaps reported.

`symbol_opportunity_context.csv` and `opportunity_associations.csv` also compare positive-ADV maker-report symbols against RPI execution quality, separately for each raw side code. The reported rank correlations exclude rows without executions and do not claim statistical significance. This exploratory cross-section compares different venue and time scopes (weekly BZX opportunity and daily RPI outcomes); dollar-per-share improvement can also reflect stock price, spread, tick size and liquidity. It is not a test of colony identity or causal opportunity effects.

The first comparison matches 10,128 symbols, with 3,051 symbol-side rows having executions. For raw side B (1,427 symbols), opportunity/ADV versus price improvement has Spearman rho about -0.116; raw opportunity shares give about +0.013. For side S (1,107 symbols), the corresponding values are about -0.194 and -0.100. The change with scale is a reason to model liquidity and security characteristics before interpreting opportunity as an explanatory signal. See `opportunity_comparison.png` for the separate raw-side plots and `opportunity_comparison.json` for exclusions and limitations.

The rolling maker CSV has no period column. `data/cboe/report_periods.json` records the period observed on Cboe's report page for the saved snapshot. If the file is replaced, update and verify that period before joining it to other events. Same-day daily totals and a weekly report including that day are retrospective context only. A single symbol-day has constant context, so it cannot identify whether context explains differences between its intraday colonies.

Sources: [Cboe historical volumes](https://www.cboe.com/markets/us/equities/market-statistics/historical-market-volume), [Maker Opportunity definitions](https://www.cboe.com/us/equities/market_statistics/maker_report/), [RPI daily reports](https://www.cboe.com/markets/us/equities/market-statistics/retail-price-improvement/daily), [RPI methodology](https://www.cboe.com/insights/posts/retail-price-improvement-provides-unique-opportunities-for-retail-investors-and-liquidity-providers). Market statistics are attributed to Cboe Exchange, Inc.

## Historical data access checks

The historical-access audit separates catalogue/schema support from successful downloads. It lists all catalogue datasets, then tests the US-equity feeds relevant to this project using explicit symbols, one-second intraday requests and small daily-bar requests. Intraday probes confirm retrieval and event fields; they do not initialize complete books or estimate cross-venue influence.

```powershell
# Small, resumable historical-access audit; each download is cost estimated.
.\.venv\Scripts\python.exe scripts/check_databento_access.py --budget 0.10 --workers 4

# Public-source checks only; large IEX captures and Nasdaq report bodies are not downloaded.
.\.venv\Scripts\python.exe scripts/check_public_sources.py

# Build an access inventory and observed-native-venue volume panel from saved results.
.\.venv\Scripts\python.exe scripts/build_access_report.py
```

Evidence is saved under `out/access/`; probe DBNs and exact request manifests are under `data/access_audit/`. Completed, provenance-matched files are reused. Cost estimates are reserved before retrieval, including failed requests that might have delivered partial data; the estimate threshold is not a provider-enforced billing cap. Parallel workers share a synchronized estimate budget.

The authenticated WRDS check on October 26, 2026 found that **NYSE Daily TAQ is not included** in the current institutional subscription. The query form explicitly disables submission as not subscribed; sample TAQ products are included. No WRDS credentials were collected or saved by the agent.

The completed Databento audit retrieved actual daily bars from all 20 selected equity products and actual MBO from 13 individual exchange feeds. The native-feed panel has 45 stock/venue rows across AAPL, MSFT and SPY on 15 exchange venues for October 23. IEX supplies TOPS/MBP-1 rather than native MBO in this catalogue. Blue Ocean's daytime sample was empty, while a separate midnight-UTC control returned 36 MBO records and 8 MBP-1 rows. No permission denials or retrieval errors occurred. Combined estimated reservations for the initial, resumed and overnight checks were approximately $0.00433, below the $0.10 threshold; the final billed cost was not checked.

FINRA's public endpoint returned real AAPL weekly ATS and non-ATS volume records, with publication dates. Those bounded samples are not a complete volume panel. IEX's public historical manifest is accessible; its large raw captures were left undownloaded. Nasdaq monthly symbol-report endpoints respond, but academic-use authorization has not been verified. NYSE's public consolidated-volume sample is accessible; access to full licensed history has not been established. Raw request/status evidence is in `public_sources.json`.

`stock_venue_daily_volume.csv` uses successful daily-bar downloads from individual exchange feeds, with canonical publisher/venue mappings. Its weights are **shares among the observed native feeds only**, excluding overlapping composites, combined products, TRFs and overnight ATSs. Missing bars are not treated as zero. The CSV records which symbols were actually tested for MBO separately from daily-bar coverage. UTC-day bar conventions and trade/session eligibility must be reconciled before claiming full-market venue shares. A later network study needs overlapping venue events and initialized book state, along with lagged stock-specific weights.

An additional empirical cross-check found that Cboe RPI executed shares divided by `shares_exec_div_bats_vol` matches the downloaded BYX daily volume for all ten executing symbol/side observations across AAPL, MSFT and SPY. The BYX denominators were 84,042, 33,631 and 252,759 shares respectively. This validates the linkage for those observations; it does not establish the field's universal dictionary or publication timing. Evidence is in `rpi_volume_crosscheck.csv` and its JSON summary.

## Aligned exchange-network pilot

The separate network pipeline uses Nasdaq TotalView (`XNAS.ITCH`), NYSE Integrated (`XNYS.PILLAR`) and Cboe BZX (`BATS.PITCH`) for AAPL, MSFT and SPY. It requests the same 14:00–14:01 UTC window on October 21, 22 and 23, 2026. October 21–22 calibrate the shock thresholds and prediction models; October 23 is held out. This is a short technical and exploratory test, with one minute per date.

```powershell
# Reproduce both analyses from the completed local download, without network calls.
.\.venv\Scripts\python.exe scripts/run_network_pilot.py --analyze-only

# Cost-estimate every explicit request before any fresh download.
.\.venv\Scripts\python.exe scripts/run_network_pilot.py --plan-only

# Fetch any missing requests and run both analyses; completed matching files are reused.
.\.venv\Scripts\python.exe scripts/run_network_pilot.py --budget 0.50 --workers 3
```

The download plan, exact request manifests and DBNs preserve provenance. The estimate ceiling applies to the combined requests before downloads; it is not a provider-enforced billing cap. Databento documents that record counts and billable sizes can overreport ranges that are not multiples of ten minutes. Completeness therefore requires a successful SDK stream, a parseable saved DBN and an unreached record cap, rather than equality with the metadata count. Failed streams are preserved and excluded from analysis.

MBP-1 supplies already-computed top-of-book states; the pilot does not reconstruct a full queue from a mid-session MBO sample. Only completed atomic events (`F_LAST`) are used as quote states. Snapshot, inaccurate receive-clock and possibly corrupted book flags create barriers, as do conflicting states at tied timestamps. A new valid observation cannot fill across a barrier. The analysis uses the receive clock consistently across feeds, retaining nanosecond precision. Arrival order still includes transport delays.

Each venue/stock shock threshold is the larger of half a cent and the 75th percentile of nonzero adjacent midpoint changes on the training dates. The event study requires quotes no older than 250 ms. Shocks are separated by at least 100 ms. The directed event study asks whether a **new same-direction target shock** appears more than 2 ms after a source shock and within 10, 100 or 1,000 ms. It separately reports midpoint markouts and repeats the longer horizons with a 10 ms exclusion band. These are price-shock associations; liquidity changes and MBO activity are not yet transmission predictors.

Timing placebos move source triggers forward by 5, 15 and 30 seconds within the same window, without wrapping. Observed and shifted responses use exactly the same eligible source-event denominator. The longest shift removes much of the one-minute window; support is reported for each edge and horizon. No independent-event significance or causal contagion rate is inferred.

The prediction comparison uses 10 ms bins. Its baseline includes strictly preceding own-venue shock counts over 10, 100 and 1,000 ms, elapsed time and observed SPY shock activity. SPY targets instead use AAPL/MSFT activity as a limited market proxy. The network model adds past same-stock shocks on the other venues. Both models use the same eligible rows, training-only scaling and fixed L2 logistic settings. Log loss and Brier scores compare the frozen models on October 23; a positive baseline-minus-network loss favors the network. This benchmark predicts shock arrival, not its direction, profitability or causal impact.

Outputs are in `out/network_pilot/`: `network_summary.json`, `edge_response.csv`, `timing_placebos.csv`, `clock_sensitivity.csv`, `quote_quality.csv`, `shock_events.csv`, `shock_thresholds.csv`, `mbo_activity.csv`, `venue_network.png`, `response_curve.png`, `prediction_summary.json`, `prediction_scores.csv`, `prediction_coverage.csv`, `prediction_test_bins.csv` and `prediction_coefficients.csv`. MBO outputs are action counts for observed deltas; they do not establish order lifetimes, full depth, maker identity or quote ownership. Run `scripts/plot_network_prediction.py` to plot the saved scores as `prediction_comparison.png`, without refitting.

The completed real-data run retrieved **600,862 records in all 18 requests**, with no capped files. Of those, 497,834 are MBO and 103,028 are MBP-1. Its combined conservative metadata cost estimate was **$0.365525**, below the $0.50 estimate ceiling; the final bill was not checked. There are 27 stock/venue/date quote groups and 4,058 de-duplicated shocks. The saved MBP-1 sample has no flagged bad clocks, snapshots, possibly bad books, conflicting tied states or invalid quote states after atomic-event selection; sparse updates and freshness requirements still reduce support.

The held-out prediction comparison scored **32,817 of 51,204 candidate stock/venue bins (64.1%)**. The prediction benchmark requires a fully observed preceding second, quotes at most one second old and no historical invalid-state barriers on any required feed. Coverage exclusions select more active periods. These are dependent observations, not 32,817 independent experiments.

| Held-out loss | Training-rate constant | Own history + market proxy | Plus other venues |
| --- | ---: | ---: | ---: |
| Log loss | 0.108869 | 0.098868 | 0.097496 |
| Brier score | 0.024244 | 0.023135 | 0.022957 |

The network reduced pooled log loss by **1.39% relative to the baseline**, and improved log loss and Brier score in 7 of 9 stock/venue tasks. AAPL on BZX and Nasdaq worsened. The weakest task, MSFT on BZX, has only 19 positive held-out bins. This is a small, uneven descriptive gain in one held-out minute, without significance or evidence of trading value. Pooled ranking metrics partly reflect differences in task base rates and should not be read as pure intraday predictive skill.

The 100 ms event study has 1,089 matched edge-events: 1,053 SPY, 36 AAPL and zero MSFT. Observed same-direction response probability is **23.32%**, versus **30.76%** for shifted controls, an excess of **−7.44 percentage points**. This does not demonstrate negative influence or inhibition: near-synchronous target jumps inside the excluded 2 ms band can already consume the target's 100 ms shock separation interval. The result is sensitive to event timing and de-duplication. It does not support a positive delayed-contagion claim under this specification. A later study needs longer windows, multiple holdout dates and predeclared alternative event definitions; the present thresholds and models were not tuned to improve the held-out result.

Public venue weights, maker opportunity and retail price improvement remain retrospective context. Their published aggregation and publication times must be aligned over multiple dates before testing whether they explain edge variation. The current outputs are measurements to inform a recurrent network model; they do not fit SIR, SIS or Hawkes parameters.

References: [record-count estimation](https://databento.com/docs/api-historical/metadata/metadata-get-record-count?lang=python), [billable-size estimation](https://databento.com/docs/api-historical/metadata/metadata-get-billable-size?lang=python), [timestamp and flag conventions](https://databento.com/docs/conventions/common-fields-enums-types).

## Wider historical replication

`wide_network.toml` records the expanded design before retrieving or inspecting its outcomes. It keeps AAPL, MSFT and SPY and adds NYSE Arca and Cboe EDGX to Nasdaq, NYSE and Cboe BZX. Each of ten dates has the same 14:00–14:10 UTC window. October 5–9 train the thresholds and models; October 12–16 are five separate held-out dates. These dates are disjoint from the first exploratory sample. The expansion increases aligned time from three minutes to 100 minutes per venue and tests 20 directed venue pairs rather than six.

```powershell
# Read the saved design and check the size of every request before retrieval.
.\.venv\Scripts\python.exe scripts/run_wide_network.py --plan-only

# Retrieve missing matching files, then evaluate the fixed five-day holdout.
.\.venv\Scripts\python.exe scripts/run_wide_network.py

# Repeat analyses entirely offline from a complete saved manifest.
.\.venv\Scripts\python.exe scripts/run_wide_network.py --analyze-only

# Plot the saved prediction comparison.
.\.venv\Scripts\python.exe scripts/plot_network_prediction.py --out-dir out/wide_network
```

The 50 requests use MBP-1 for the initial broader venue comparison. They preserve the full ten-minute windows and stop before retrieval if size exceeds the design limits; no automatic one-minute fallback is applied. Request size checks finish before any download. The current configuration sets `billing_assumed_included = true` at the user's instruction: `get_cost` calls and monetary ceilings are skipped. Skipped cost fields are null with explicit provenance; this records the user's included-access assumption rather than a verified invoice. Record/byte limits, complete-stream checks and exact cache validation remain active. Separate `data/wide_network/` and `out/wide_network/` paths preserve the original pilot. Exact configuration, training/test dates and request provenance are saved with the manifest. Earlier cost estimates below are historical metadata receipts from before this policy change.

Thresholds, clocks, shock separation, response horizons, timing controls and logistic settings retain the pilot specification. Metrics are reported for each held-out date and task, alongside pooled results. The broader prediction export omits the large per-bin CSV while retaining scores, coefficients and coverage evidence. Adding venues makes the common-coverage requirement stricter: a missing or stale required predictor feed excludes that bin from both models. Coverage is part of the result and is not relaxed after seeing scores.

This replication tests whether the initial associations repeat across more dates and venues. The timing/de-duplication limitation still applies. Expanding this sample does not by itself turn temporal association into causal influence or fit an SIR model. Full-session MBO reconstruction, additional intraday periods, and publication-aligned maker/RPI predictors are separate extensions.

The wider download completed all **50 requests and 4,988,927 MBP-1 records**, matching the summed ten-minute metadata counts, with no capped files or retrieval errors. The combined estimate was **$0.446045**; the final bill was not checked. The analysis has 150 stock/venue/date groups and 201,551 detected shocks. It excludes 188,062 non-final intermediate records and observes no snapshot flags, bad receive-clock flags, possibly bad-book flags, conflicting timestamp ties or invalid book states. In `quote_quality.csv`, feed-wide counts repeat on each symbol row; deduplicate dataset/date before summing those columns.

The five held-out dates provide **283,888 matched 100 ms edge-events**, versus 1,089 in the first pilot. Support now includes 14,278 AAPL and 6,769 MSFT edge-events, although SPY still accounts for 92.6%. The observed response rate is **20.85%**, versus **24.91%** for timing placebos, an excess of **−4.06 percentage points**. All five dates have negative excess under the unchanged definition. This repeats the original event-study result with more support; the clock-exclusion and shock-separation issue still prevents interpreting negative excess as inhibition.

The full set of predeclared response horizons is mixed. Each horizon has its own matched sample, so differences across rows also reflect coverage selection.

| Response horizon | Matched edge-events | Observed response | Shifted controls | Excess |
| --- | ---: | ---: | ---: | ---: |
| 10 ms | 298,593 | 3.81% | 1.99% | +1.82 pp |
| 100 ms | 283,888 | 20.85% | 24.91% | −4.06 pp |
| 1 second | 267,207 | 93.52% | 92.87% | +0.65 pp |

The frozen prediction models score **2,971,955 held-out task bins**, retaining 66.24% of 4,486,700 candidate bins. Retention is 78.46% for AAPL, 66.96% for MSFT and 53.29% for SPY. The intersections select active periods; this result applies to that observable sample. `prediction_feed_coverage.csv` identifies which required feeds exclude bins; those exclusions overlap and must not be added together.

| Held-out loss | Training-rate constant | Own history + market proxy | Plus other venues |
| --- | ---: | ---: | ---: |
| Log loss | 0.106850 | 0.098389 | 0.097048 |
| Brier score | 0.022855 | 0.022162 | 0.022016 |

Adding strictly past same-stock venue shocks reduces pooled log loss by **1.363%**. All fifteen stock/venue tasks improve log loss when pooled over the five test dates; Brier improves in thirteen, with small deteriorations for MSFT on EDGX and Nasdaq. Each test date improves both pooled loss measures. This is a replication of modest incremental predictive information. The wider run changes dates, window length and venue set together; it does not isolate the marginal benefit of the two added venues through a same-date three-versus-five-venue comparison.

| Held-out date | Scored task bins | Relative log-loss reduction |
| --- | ---: | ---: |
| October 12 | 503,631 | 1.73% |
| October 13 | 586,774 | 1.32% |
| October 14 | 669,924 | 1.21% |
| October 15 | 595,953 | 1.42% |
| October 16 | 615,673 | 1.28% |

`prediction_scores_by_day.csv` and `prediction_task_scores_by_day.csv` retain the daily metrics. `prediction_comparison.png` shows the fifteen tasks; `replication_by_day.png` compares the five dates. Generate the latter with `scripts/plot_wide_replication.py`. The daily weighted losses reproduce the pooled losses, the timing placebos use matched denominators, and **146 focused acquisition, engine and prediction checks pass**. These dependent windows provide no independent-bin significance, causal effect, maker attribution or trading-profit estimate.

## Volume candles versus time candles

```powershell
# Offline comparison using all 50 completed wider-sample files.
.\.venv\Scripts\python.exe scripts/run_volume_comparison.py
```

The candle comparison uses the same October 5–9 training dates, October 12–16 test dates, five venues and three stocks. It changes the candle representation while preserving the next-10-ms shock labels and forecast anchors, which makes the loss comparison meaningful. These test dates have already been inspected in the earlier study, so this is an exploratory paired comparison; independent confirmation needs additional dates.

Only genuine MBP-1 action-`T` prints contribute shares, including trade records without `F_LAST`. Fill records are not added, and identical prints are not blindly deduplicated. The raw sample has 235,979 valid trade prints totaling 10,413,210 shares, including 188,062 prints without `F_LAST`. There are no invalid or unresolved same-time trade groups in this sample. Unknown-side volume is 2,783,888 shares (26.73%); unknown aggressor sides are retained and measured as a fraction, rather than invented.

Each venue/stock volume quantum is fitted from **training shares divided by the full requested training seconds**, rounded to at least one share. The quantum targets one second of average training activity and stays frozen on test dates. A volume candle closes when a whole receive-timestamp group reaches that quantum. A large print can overshoot; the observed excess is reported and no synthetic split executions are created. Incomplete trailing candles are excluded. Candles reset each date. Volume duration includes waiting time since the previous close; one-second time candles retain their full scheduled duration. Empty time intervals have no candle or fabricated carry-forward prices.

Time candles contain trades in `[start,end)` and become available at `end <= prediction_time`. Volume candles contain their closing print and require `close < prediction_time`. Thus all underlying predictor trades strictly precede the forecast anchor. Unsafe prints create barriers, and the last completed candle must be at most ten seconds old. All three models use the intersection of valid time and volume candle features on every required same-stock venue; the reference model is refitted on exactly that same sample.

The reference contains the existing own-venue, market-proxy and cross-venue shock histories. Two alternatives add nine candle features per venue: return, range, volume relative to the frozen quantum, trade count, duration, signed-volume fraction, unknown-side fraction, VWAP/close difference and age. Both candle families have the same number of feature columns. Scaling and fixed L2 logistic models are fitted only on training rows. A positive time-minus-volume loss means volume candles help on this specified target. This does not yet change the forecasting clock itself to a next-volume-bar outcome.

Results and auditable candles are saved under `out/volume_comparison/`: `volume_comparison_summary.json`, `volume_thresholds.csv`, `volume_candles.csv`, `time_candles.csv`, `candle_statistics.csv`, `trade_quality.csv`, `candle_model_scores.csv`, `candle_daily_scores.csv`, `candle_coverage.csv`, `candle_feed_coverage.csv` and `candle_coefficients.csv`.

The completed run builds **55,599 volume candles and 45,690 one-second candles**. Completed volume plus discarded incomplete tails accounts for every input share, with no invalid-trade losses. Median volume-candle duration is 567 ms and the 95th percentile is 6.67 seconds across the full sample. Median threshold overshoot is 32 shares. All fifteen share thresholds use exactly 3,000 requested training seconds and range from 13 to 266 shares.

The three models score **2,513,892 identical held-out bins**, retaining 84.59% of the 2,971,955 quote-eligible bins after requiring both candle families. The reference is refitted on this common cohort; its loss therefore describes this cohort rather than the earlier, larger quote-only sample.

| Model on common rows | Log loss | Brier score |
| --- | ---: | ---: |
| Existing network reference | 0.100723 | 0.023127 |
| Reference + one-second candles | 0.100929 | 0.023144 |
| Reference + volume candles | 0.100680 | 0.023105 |

Volume candles reduce pooled log loss by **0.247% versus one-second candles**, winning on all five dates and thirteen of fifteen venue/stock tasks. Their pooled improvement over the existing network reference is **0.042%**, positive on two of five dates and ten of fifteen tasks. The time-candle addition worsens the reference by about 0.205%. These figures support a modest preference for volume over one-second candle features in this setup; the incremental value beyond the network model is tiny and varies by date.

| Test date | Volume gain versus time candles | Volume gain versus network reference |
| --- | ---: | ---: |
| October 12 | +0.067% | −0.100% |
| October 13 | +0.331% | +0.196% |
| October 14 | +0.323% | +0.196% |
| October 15 | +0.263% | −0.011% |
| October 16 | +0.160% | −0.139% |

Positive entries are relative log-loss reductions. Generate the plots with `scripts/plot_volume_comparison.py`: `volume_vs_time.png` compares all fifteen tasks, and `volume_clock_by_day.png` shows daily performance and realized candle durations. Raw-volume conservation, all date boundaries, threshold provenance and daily-to-pooled arithmetic were independently checked; **203 focused tests pass**. No parameter grid was searched after inspecting these scores, and the repeated test-date use remains exploratory.

## Fresh-date volume-candle confirmation

The fresh test was frozen in `fresh_confirmation.toml` and `fresh_confirmation_protocol.json` before retrieval or scoring of these dates. It expands to eight native venues: Nasdaq, NYSE, BZX, Arca, EDGX, MEMX, BYX and EDGA. The same AAPL, MSFT and SPY targets use twenty training sessions from August 3–28, 2026 and twenty fresh held-out sessions from August 31–September 28, excluding the September 7 exchange holiday. Each session covers 14:00–14:30 UTC (10:00–10:30 New York time). All 320 venue/day MBP-1 requests must complete without reaching their record caps.

```powershell
# Included historical access: no price lookup or monetary ceiling.
.\.venv\Scripts\python.exe scripts/run_wide_network.py --config fresh_confirmation.toml --fetch-only --download-workers 8
# Offline streaming calibration, candle panels, fixed paired models and inference.
.\.venv\Scripts\python.exe scripts/run_fresh_confirmation.py
```

The primary comparison refits the existing network reference and adds completed volume-candle features, using identical rows. It omits the exploratory time-candle model and its coverage mask. The volume quantum still targets one second of full requested training exposure, candle state has a ten-second maximum age, and every closing trade must strictly precede its forecast. No venue, stock or test date is removed after inspecting outcomes. Quote and candle retention by required feed and date are part of the result.

Forecast anchors are spaced 100 ms apart for this larger experiment. **The outcome remains a shock in the next 10 ms**, and all prior-history features retain their original 10/100/1000-ms windows. This is deterministic thinning of the original forecasting grid. These results therefore measure volume's incremental value on the new paired sample; their absolute losses are not directly comparable with the October 10-ms-grid sample.

Training determines shock thresholds, volume thresholds, scaling and fixed L2 `C=1` logistic coefficients. Normalized inputs and per-day panels are checkpointed on disk, and models are fitted one stock/venue task at a time. Primary loss totals are pooled with their observation denominators, rather than averaging daily percentages. Paired uncertainty resamples entire dates, preserving all stocks and venues together; the predeclared secondary dependence check resamples consecutive five-session blocks. Both use 10,000 replicates with fixed seeds. Twenty held-out dates supply only about four non-overlapping week-length blocks, so the intervals are approximate. A positive pooled effect and a positive lower five-session-block interval support incremental prediction in this particular window; they do not establish causal venue effects or profitable execution.

`--download-workers` changes transport concurrency only; the manifest records the actual setting while preserving the frozen research configuration. After a failed transfer run has stopped, `--recover-interrupted` preserves recognized failed-stream prefixes inside their request caches and retries the exact requests under included access. Completed streams and successful files are reused. An unmarked partial stream never counts as complete. The offline `scripts/audit_fresh_confirmation.py` rereads the actual source files, checks every share and task/date denominator, and independently reconstructs the reported intervals.

The completed study retrieved **64,753,218 MBP-1 records across all 320 requests**. Every intended held-out stock/venue/date has paired observations, and all 48 model fits across 24 stock/venue tasks converged. The fixed volume-candle addition **worsens the existing network's held-out prediction loss**:

| Metric on identical held-out rows | Network reference | Reference + volume candles | Relative volume improvement |
| --- | ---: | ---: | ---: |
| Log loss | 0.074764599 | 0.075641197 | −1.1725% |
| Brier score | 0.015568659 | 0.015662855 | −0.6050% |

Negative improvement means higher loss. The paired-date 95% interval for relative log-loss gain is **[−1.506%, −0.877%]**; the five-session block interval is **[−1.505%, −0.878%]**. The block interval for relative Brier gain is **[−0.925%, −0.348%]**. Volume worsens pooled log loss on **all twenty test dates and 23 of 24 stock/venue tasks**. Its Brier score improves on one date and three tasks. Each of the three stocks and all eight venues has worse pooled log loss. The predeclared confirmation rule is not met; the small October exploratory gain does not replicate under this frozen specification.

The comparison scores **594,354 paired held-out task bins**, retaining **6.905% of 8,607,736 candidates** and **40.518% of 1,466,872 quote-eligible bins**. AAPL supplies 441,590 rows, MSFT 103,284 and SPY 49,480. Candidate retention is 15.374%, 3.600% and 1.726% respectively; daily overall retention ranges from 1.567% to 12.607%. Thirty of 480 held-out task/dates contain no positive shock bins, although none has zero observations. Sparse BYX/EDGA updates, quote freshness, the market-proxy requirements and the all-venue completed-candle intersection restrict the result to jointly observable active periods. The files are complete; these exclusions arise from the frozen eligibility rules. They prevent generalizing the loss estimates to the entire requested window or full trading day.

Detailed evidence is under `out/fresh_confirmation/`: `confirmation_summary.json`, `task_daily_scores.csv`, `daily_scores.csv`, `coverage.csv`, `feed_coverage.csv`, `volume_thresholds.csv`, `shock_thresholds.csv`, `coefficients.csv` and the exact source/code/protocol fingerprints in `analysis_provenance.json`. **289 focused tests pass**, including independent paired-loss arithmetic, training-only calibration/scaling, strict candle availability and unchanged 10-ms labels on the thinned grid. The independent raw-data audit also passes (`audit.json`): all 320 source requests, 960 candle task/dates, 24 training quanta and 480 held-out task/dates agree, with exact share conservation and independently reproduced bootstrap intervals. The result concerns these fixed volume features, target, venues and coverage cohort; it does not establish that every volume-candle design fails or identify a cause of the deterioration.

## Full-session MBO venue network and recurrent SIRS test

`mbo_sir.toml` and `mbo_sir_protocol.json` define a separate experiment using AAPL on Nasdaq, NYSE, BZX, Arca, EDGX, MEMX, BYX and EDGA. Ten training sessions run June 1–12, 2026; twenty held-out sessions run June 15–July 14, excluding June 19 and July 3. Each feed is replayed from 00:00 through 20:00 UTC, retaining every displayed order and price level. Forecasts cover the full 13:30–20:00 UTC trading session. SPY's completed one-second BBO observations provide a market control. The capped startup probes are excluded from the experiment.

```powershell
# Included historical access; no monetary metadata or price ceiling.
.\.venv\Scripts\python.exe scripts/run_mbo_sir.py --fetch-only --download-workers 24
# Optional concurrent local replay, while the preceding download is active.
.\.venv\Scripts\python.exe scripts/run_mbo_sir.py --replay-available
# Retry only failed completed requests, preserving interrupted bytes.
.\.venv\Scripts\python.exe scripts/run_mbo_sir.py --repair-failed-only
# Record the provider's date-level availability and quality conditions.
.\.venv\Scripts\python.exe scripts/run_mbo_sir.py --conditions-only
# Offline reconstruction, training and held-out scoring after retrieval finishes.
.\.venv\Scripts\python.exe scripts/run_mbo_sir.py --analyze-only
# Independently rebuild every prediction/loss from saved JSON parameters.
.\.venv\Scripts\python.exe scripts/audit_mbo_sir.py --config mbo_sir.toml
# Scientific figures from the completed result tables.
.\.venv\Scripts\python.exe -m colonylab.mbo_sir_plot --out-dir out/mbo_sir
```

The integer order book handles adds, absolute modifies, quantity cancels and authoritative clears. Informational trades/fills do not remove an order twice. Complete events are exposed only after `F_LAST`, and predictor records strictly precede forecast anchors. The initial legacy synthetic clear is accepted only under a verified midnight-start request and its audited zero/sentinel signature. Unsafe flags, semantic failures and incomplete events create observation barriers. Quiet initialized books remain observable; book age is a predictor rather than a blanket quote-freshness exclusion. Every venue/day must match the provider's top-ten prices, shares and order counts in an independent five-minute MBP-10 window before fitting.

In this observed-state SIRS approximation, a qualifying midpoint shock starts 100 ms of infection, then 100 ms of refractoriness, then susceptibility. Shocks during infection/refractoriness do not restart the process. Per-venue shock thresholds are the training-only 75th percentile of nonzero absolute midpoint changes, floored at half a cent. Susceptible targets are evaluated every 100 ms for a shock in the next 10 ms. Sources are observed strictly before a fixed two-millisecond lead. A positive directed coefficient adds to the target's conditional hazard; the probability uses `1-exp(-0.01*hazard)`.

Four models share exact forecast timestamps and labels: an independent MBO hazard, the SIRS network, the previous logistic network family refitted with the new labels and MBO controls, and SIRS with source activity shifted thirty seconds into the past without wrapping. Controls include depth, imbalance, adds/removals/trades, order count, book age, own recent shocks, time of day and common market returns. Recovery and immunity durations are fixed definitions, not estimated epidemic parameters. The primary test compares SIRS with the independent hazard using paired whole-date losses and a five-session block bootstrap. Secondary logistic and shifted-source comparisons remain visible. The prior MBP/candle experiment uses different stocks, dates, labels and coverage; its absolute loss cannot measure this experiment's improvement.

Checkpoints and evidence are retained under `out/mbo_sir/`: exact raw-request manifests, compressed-file hashes and a fresh raw-data audit; reconstruction caches and `book_audit.csv`; training-only `shock_thresholds.csv`; feature caches and `coverage.csv`; fitted parameters and directed `edges.csv`; identical-cohort daily losses; and `summary.json` with inference, convergence and numerical guards. The network concerns unsigned shock risk in displayed liquidity. It does not identify makers or hidden orders, establish causal venue impact, estimate epidemiological R0 or demonstrate trading profit.

The provider's date-level condition catalogue, retrieved October 27, 2026 UTC and saved in `dataset_conditions.json`, flags Arca on June 24 and July 14 as degraded. Native AAPL bad-book flags corroborate this: only 23,193 of 234,000 scheduled Arca anchors remain valid on June 24, and 38,725 on July 14. The last valid anchors occur at 14:08:39.2 and 14:34:32.4 UTC respectively; no later clear restores a valid two-sided anchor. The common all-venue quality mask therefore restricts every model's forecasts on those dates. The frozen study retains both dates. Complete transfers and agreement with the same provider's MBP-10 data do not certify upstream feed completeness, and the provider catalogue does not explain the underlying incident.

Retrieval completed all **510 requests and 532,658,530 total records**, including **527,567,665 native MBO records**. All 240 venue/day books pass their validation windows: **4,228,249 matched MBP-10 keys, zero mismatches and zero missing keys**. The independent full-file audit reparses and hashes every source file, with no capped, truncated or failed requests. No full-session order-semantics failures are observed. These checks establish reconstruction consistency; the validation window is five minutes per venue/day, and native quality flags remain active throughout each full session.

The held-out cohort contains **27,688,883 identical task forecasts and 324,223 positive shock labels**, retaining **73.955% of 37,440,000 scheduled anchors**. Quality and market support retain 34,143,184 anchors (91.194%) before target susceptibility removes infectious/refractory states. Every intended venue/date has eligible observations. Training-only shock thresholds range from $0.005 on Nasdaq and Arca to $1.31 on BYX. Consequently the unsigned local-midpoint shock definition represents different dollar moves on different venues; it does not measure a uniform move in the consolidated stock price.

All **32 model fits converge**, with **zero numerical-guard rows**. The SIRS network adds a small predictive gain over the independent MBO hazard, but **does not improve on the refitted logistic network**:

| Model on identical held-out rows | Log loss | Brier score |
| --- | ---: | ---: |
| Independent MBO hazard | 0.057137049 | 0.011373377 |
| SIRS with source activity shifted 30 seconds into the past | 0.057125379 | 0.011372861 |
| SIRS network | 0.057102707 | 0.011372198 |
| Refitted logistic network | 0.056881849 | 0.011361266 |

| Reference for SIRS comparison | Relative SIRS log-loss reduction | Paired five-session-block 95% interval |
| --- | ---: | ---: |
| Independent MBO hazard | +0.0601% | [+0.0393%, +0.0739%] |
| Refitted logistic network | −0.3883% | [−0.4302%, −0.3394%] |
| Past-shifted SIRS | +0.0397% | [+0.0206%, +0.0528%] |

Positive reductions favor SIRS. SIRS improves both losses versus the independent hazard on all twenty test dates and all eight pooled venue tasks. Its Brier gain is only 0.0104%. The logistic network beats SIRS on both losses on all twenty dates and all eight pooled venue tasks. SIRS beats the past-shifted control on eighteen dates for log loss and nineteen for Brier. The primary confirmation flag is true because it tests the independent-hazard comparison; it does **not** mean that SIRS beats the existing network family. Twenty dates supply only about four non-overlapping five-session blocks, so the reported intervals remain approximate.

The fitted graph has 48 rates above 10⁻⁸ per second among 56 possible directed edges. These are conditional rates when a source is infectious, without volume or event-frequency weighting, and have no edge-level confidence intervals. A large coefficient on a rare, wide quote move does not establish that the venue dominates price discovery. `network_beta.png` displays the associations; `sirs_model_comparison.png` shows all three held-out comparisons. The result supports weak cross-venue predictive information under this SIRS specification while retaining the logistic model as the stronger predictor on this sample.

**387 focused software checks pass.** The final full independent result audit also passes (`result_audit.json`): all 510 raw requests, 240 replay/panel tasks, 640 held-out task/model/date score rows, training-only scaling, 32 parameter sets, all three paired bootstrap comparisons and all 56 directed rates agree. Predictions are independently reconstructed from JSON parameters without loading fitted-model pickles; maximum task loss-sum discrepancies are 3.64×10⁻¹² for log loss and 9.10×10⁻¹³ for Brier. All four scientific figures have been visually checked. Presentation-only plot corrections and their source/test hashes are explicitly recorded in `provenance.json`, with the original analysis snapshot preserved separately.

## Native-event venue latency on the saved MBP-1 studies

The offline latency extension adapts the measurement ideas from the team's
[qam-p113-latency project](https://github.com/smar-01/qam-p113-latency/tree/62d6dcd8af9ecd7136096dbcce3eb40371987422)
to ColonyLab's source integrity, quote-quality barriers and date-level validation.
It independently implements the measurement; it does not import the other project's code.

```powershell
# Reads existing local files only. No API client, key, or new download is needed.
.\.venv\Scripts\python.exe scripts/run_latency.py --workers 2
# Independently verify every source hash, output denominator and date-block interval.
.\.venv\Scripts\python.exe scripts/audit_latency.py --out-dir out/latency
# Independently decode selected raw files, rebuild states and recompute full pairs.
.\.venv\Scripts\python.exe scripts/audit_latency_raw.py --out out/latency
# Publish the audited findings, matrices and descriptive leadership counts locally.
.\.venv\Scripts\python.exe scripts/build_latency_readout.py --out-dir out/latency
# Rebuild presentation/inference from verified completed day checkpoints.
.\.venv\Scripts\python.exe scripts/run_latency.py --summarize-only
```

The inputs are the complete `fresh_confirmation.toml` and `wide_network.toml`
acquisition grids: **370 native MBP-1 requests, 69,742,145 records and 50 dates**.
The first study has three symbols, eight venues and forty thirty-minute windows;
the second has three symbols, five venues and ten ten-minute windows. Studies,
symbols and the original training/held-out date groups remain separate. This is
a **new exploratory reanalysis of already studied data**, not a new preregistered
latency confirmation. The saved MBP-1 studies do not contain IEX.

`latency_protocol.json` fixes both clocks, 1/10/100-ms matching horizons, the earlier
MBP network's 250-ms observed-quote-age limit, a full trailing five-minute volatility window and
five-session date-block uncertainty. After each bid/ask price change, the matcher
finds the first strictly later follower price at or beyond the new price in that
move's direction. Equal-time followers already at the price are classified as
already there. A later leader reversal does not cancel the original target.

Prices remain integer nanodollars and timestamps remain integer nanoseconds.
Only completed `F_LAST` states enter the book path. Unsafe flags, snapshots,
missing/nonpositive/locked/crossed quotes, conflicting same-clock states,
within-symbol clock reversals and stale-state gaps are observation barriers.
Size-only updates refresh quote age before price-state compression. Sources are
checked against exact receipts, stream-completion markers, DBN metadata and the
original saved SHA-256 identities. Exchange-clock support ends conservatively at
the last observed exchange timestamp because the requests were selected by the
capture clock.

The primary matched quantiles and match rates require the **whole chosen horizon
observable in both feeds**, including for early matches. Already-at-price,
unknown-prior-state, quality-truncated and request-end-truncated cases are counted
separately. Conditional matched medians accompany match rates, nonmatch counts and
restricted mean waiting times that include nonmatches at the horizon cap. The
main tables separate stocks; a second table separates bid and ask events.

Volatility uses a supported cross-venue median midpoint on each selected clock,
with at least half the intended venues observed and a full 300-return warmup.
The within-window top quartile is a retrospective descriptive regime, not a
prospective forecasting signal. Inference estimates an **equal-date mean of daily
conditional medians**, with calm/volatile comparisons paired on the same dates.
Whole-date blocks keep all stocks and venue pairs together. Only primary tasks
with all twenty original held-out dates and at least twenty matches per date
receive exploratory marginal intervals. These intervals do not control familywise
error, and twenty dates contain only about four non-overlapping weekly blocks.

Results are written to `out/latency/`. Start with the audited `findings.html`,
then `index.html` for all eight figures and `summary.json` for machine-readable scope.
`daily_latency.csv` contains latency, support and censoring for every study/date,
clock, stock, pair, horizon and regime; `daily_latency_by_side.csv` provides the
bid/ask sensitivity. `inference.csv` contains date-level estimates and intervals;
`quote_quality.csv` records exclusions. `leadlag_summary.csv`, `leadlag_curves.csv`,
`event_study.csv`, `realized_volatility.csv` and `latency_histograms.csv` retain the
other descriptive measurements. `leadership_share.csv` is the outgoing matched
count divided by all matched counts involving a venue; it is not Hasbrouck's
information share or a causal leadership estimate. The six primary
`latency_matrix_<clock>_<symbol>.csv` files require full twenty-date support.
Cross-products use sparse, aggregated 10-us
increments with bounded energy normalization, rather than the starter's event
energy normalization. Event-study offsets share one continuously observed cohort.
Day checkpoints bind source hashes, protocol and computation-code fingerprints
to all output files; interrupted runs reuse only matching complete checkpoints.

Every pair/side/date has an independent leader-event count and sampled plain-loop
first-passage checks in `day_audits.csv`. The final table audit independently
reconstructs arithmetic and bootstrap results. A separate raw-data audit decodes
the first/last training and held-out dates of each study, independently rebuilds
quality states, and recomputes full Nasdaq/BZX results. Audit coverage is reported
explicitly; a sampled audit is not described as a second full reconstruction of
all 69.7 million records. `freshness_sensitivity.csv/.json` also preserve a limited
250-versus-1000-ms check on five common training dates. Its changed event cohorts
and the driver-only fingerprint difference are recorded; it has no confidence
intervals and is not a second full-study confirmation.

The completed 250-ms run has **345,974 sampled independent first-passage checks
with zero mismatches**. Both scientific audits pass: all 370 original source
hashes, 131,760 daily result rows and 5,472 inference rows reconcile. The separate
raw audit independently checks 9,434,570 records, 312 quote series and 864 full
pooled/bid/ask summaries on eight selected dates. **117 focused latency checks
pass.** The full regression run has 920 passes and one existing mocked-download
test that failed on Windows temporary-directory creation and passed on targeted
retry; the full-run and retry reports are retained separately.

At the primary 10-ms horizon, 151 of 168 receive-clock
stock/pair tasks and 149 of 168 exchange-clock tasks meet full twenty-date support.
The median across those supported pair estimates is **360.6 microseconds on the
receive clock and 286.4 microseconds on exchange clocks**. Each pair estimate is
an equal-date mean of daily conditional matched-event medians, not a pooled-event
median. Across all fully observed primary pair-events, **25.56% and 24.92%** match
within 10 ms respectively. Most fully observed events do not match within that
horizon, so these conditional delays must accompany the censoring statistics.

The primary calm/volatile comparison is **unavailable under the stated quality
rules**. Neither AAPL nor MSFT has a fully supported five-minute volatility window;
SPY has only four to eight paired dates depending on the venue pair, short of the
required twenty. No primary regime intervals are produced. This neither replicates
nor rejects the team's volatile-period result; it identifies a coverage limit of
these saved windows under the conservative quote-age and volatility rules.

Receive-clock delays include feed transport and exchange clocks can disagree.
Price matching does not identify causal propagation or pure infrastructure
latency. Quality flags do not guarantee unreported upstream completeness; no
saved provider date-condition catalogue covers these August–October MBP-1 dates.
The MBO model's 100-ms grids and midpoint-only jump caches are not substituted for
event-level bid/ask quotes. The measurements concern these saved intraday windows,
not complete sessions or the whole US equity market.

## Controlled experiment fixing the team's analysis code

`comparisons/qam_p113_corrected/` is a corrected local copy of the team's starter.
The original checkout is unchanged. Exact integer midpoint arithmetic removes
float-only price moves; cross-products now aggregate fixed-bin increments before
normalization; complete event-study support excludes extrapolated endpoints.
Request-cache identities include the exact symbol set and intraday times, and
the corrected default analysis uses the existing ColonyLab quality matcher.
Volatility is calculated separately for each clock from raw size/state updates.

```powershell
# Identical-input original/numerical/support comparisons on the ten-date saved study.
.\.venv\Scripts\python.exe scripts/reproduce_team_code_corrections.py --workers 2
.\.venv\Scripts\python.exe scripts/audit_team_code_corrections.py --out out/team_code_corrections
.\.venv\Scripts\python.exe scripts/report_team_code_corrections.py --out out/team_code_corrections
# Corrected standalone entry point, with local ColonyLab quality controls by default.
.\.venv\Scripts\python.exe comparisons/qam_p113_corrected/analyze.py --synthetic --out out/team_corrected_demo
```

The controlled experiment reads **50 saved requests and 4,988,927 records** for
AAPL, MSFT and SPY on five venues over ten October dates, 14:00–14:10 UTC. It
does not download data or recreate the teammates' three original full-day runs;
those raw files and IEX are absent. Original source snapshots, raw/code hashes,
shared-input quote fingerprints and complete day checkpoints are retained under
`out/team_code_corrections/`. Start with its `findings.html` and `summary.json`.

Numerical-only fixes preserve the original quote-event cohort. They remove
**122 float-only midpoint triggers across 300 venue/stock/clock series**, out of
2,793,998 floating midpoint-event detections. The largest absolute mean AAPL
event-study ratio falls from **129,185,438 to 2.169**, and MSFT from **8,850,376 to
2.346**. SPY's maximum remains about 1.217. Signed ratios may legitimately be
negative or exceed one; no clipping to a plausible-looking range is applied.
The large changes arise from removing near-zero false denominators.

All **1,200 day/stock/clock/directed-pair bid/ask reaction comparisons remain
exactly identical** under the numerical-only fix. Fixed-bin cross-products change
195 of 600 peak-lag point estimates; the median absolute change among those
changed tasks is 20 microseconds. Both bin alignment and normalization change,
so these shifts are not attributed solely to the denominator correction.

Quality/support-policy effects are kept separate, using the previously audited
wide-study outputs on these exact same raw requests. Their observation cohorts
and denominators differ, so changes in match rates are not presented as better
forecasting or faster venues. The stricter full-window volatility calculation
has no classified calm/volatile seconds in these short windows. This is a
coverage limitation, not a rejection of a volatility effect.

The corrected default driver was also run on all five actual October 15 source
files, **825,502 raw records**. All 120 stock/pair/clock summary rows agree with
the earlier audited quality implementation for twelve count fields and six
latency/wait statistics; this check is saved in
`out/team_code_corrections/corrected_driver_2026-10-15/driver_smoke.json`.
All 44 focused correction tests and all 19 copied teammate tests pass. The
independent correction audit passes, including all 50 newly hashed sources,
ten source/code-bound day checkpoints, unchanged input fingerprints and full
comparison-grid arithmetic. `corrections.patch` and `correction_files.json`
retain the source diff and corrected-file fingerprints for review.

## Saved-data Apple corporate-event trial

`aapl_event_calendar.json` records official Apple/SEC event evidence and the
saved-data coverage gaps. `aapl_event_trial.json` fixes the exploratory trial
before calculating its contrasts. The covered case is the June 8, 2026 WWDC
keynote, scheduled for 17:00 UTC. Saved windows do not cover the actual fall
launch, earnings release/call or filing acceptance times; date-only press
releases receive no invented intraday anchor.

```powershell
.\.venv\Scripts\python.exe scripts/run_aapl_event_trial.py --workers 2
.\.venv\Scripts\python.exe scripts/audit_aapl_event_trial.py
.\.venv\Scripts\python.exe scripts/report_aapl_event_trial.py
```

The trial uses only existing AAPL MBO reconstruction caches and saved raw files:
one event date plus 25 reference dates, including six same-clock reference
Mondays. Every cache/raw identity is freshly hashed and checked against the
previously audited reconstruction. Native integer midpoint paths are validated
against strict-prior integer BBO states. Invalid books, resets and unfinished
atomic updates remain observation barriers. The 100-ms grid supplies descriptive
spread/depth and reconstruction checks, never microsecond event timestamps.

The primary comparison uses 30 minutes before/after the scheduled start, five
fixed venues and all 20 directed pairs, with complete 10-ms observation and at
least 20 matches per pair. All eight venues, 1/100-ms horizons, five-minute
windows and shifted macro-time anchors are sensitivities. Shock thresholds are
fixed at 0.5/1/2 basis points; the primary 1-bp rate thins bursts causally with a
100-ms separation. It does not reuse thresholds fitted on a period containing
the event. Missing venue coverage suppresses the affected fixed-universe
comparison instead of changing its composition.

Start with `out/aapl_event_trial/findings.html`, then `result_audit.json`,
`contrasts.csv` and `event_coverage.csv`. Latency is receive-clock midpoint
first passage for eligible price changes. It is distinct from native bid/ask
latency, exchange-clock timing and a causal information-flow estimate. Shock
activity is reported separately. One covered keynote cannot establish a
difference among product events, press releases, earnings calls and filings;
reference ranks and IQRs remain descriptive, with no event-class p-values or
confidence intervals.

The audited WWDC common-five comparison has conditional latency rising from
360.93 to 430.90 microseconds (+19.39%), while clustered 1-bp shocks rise from
7.60 to 22.35 per venue-minute (+194.04%). Matching within 10 ms changes from
30.48% to 30.03%. The six Monday reference changes have a median latency change
of -32.43 microseconds; the event's +69.97-microsecond change exceeds all six.
Among 23 supported broader reference dates, one has a larger latency increase;
none has a larger shock-rate increase. Two broader dates lack Arca coverage.
The eight-venue event sample has full book coverage but insufficient matches
on several pairs for a fixed-universe latency estimate; its 1-bp shock increase
is smaller (+37.18%). These are descriptive associations, sensitive to horizon,
threshold and venue universe. The final audit passes 416 raw-file and 208 cache
hash checks, 8,652,000 eligible grid-price comparisons and 13,125 direct-scan
matching cases with no mismatches. The focused event/latency test run passes
83 tests.

## Multi-year Apple event and random-reference study

`aapl_multi_event_calendar.json` and `aapl_multi_event_protocol.json` extend
the case study to 2022–2026: five WWDC keynotes, ten earnings releases, ten
earnings calls and three September product launches. Release and call anchors
share ten quarterly bundles; there are eighteen corporate-event bundles, not
twenty-eight independent announcements. Official Business Wire publisher
headers verify release times at 16:30 EDT, thirty minutes before the calls.
These minute-level publication anchors do not identify first receipt by every
market participant. SEC acceptance proxies remain labelled context.

Each anchor has five seeded, randomly drawn same-year/weekday reference
sessions within eight weeks, at the same New York wall-clock time. The
saved candidate pools, permutation, selections and blackout reasons precede
inspection of new market outcomes. Known Apple dates, holidays and conservative
short-session periods are excluded; reference days are not certified news-free.
The exclusion settings are expressed as business weekdays,
not assumed to be identical to exchange trading-session distances.

The comparison uses a fixed Nasdaq/Arca/BZX/EDGX panel. NYSE closing quotes
are not carried into after-hours earnings windows. Native integer MBP-1
observations support both receive and exchange clocks. The existing 250-ms
freshness rule remains primary; 1,000/5,000-ms rules are explicit sensitivities.
Each conditional latency estimate requires all twelve directed pairs to have
at least twenty matches, with full observation at the selected horizon.
Matching probabilities have their own positive-eligible-event support check.
Every venue must satisfy the frozen 90% book-observation requirement; an
unsupported primary estimate is not replaced by a successful sensitivity.

```powershell
.\.venv\Scripts\python.exe scripts/run_multi_event_study.py design
.\.venv\Scripts\python.exe scripts/run_multi_event_study.py plan --workers 4
# Inspect the complete quote in out/aapl_multi_event/download_plan.json.
# The fetch action requires an explicit --max-cost-usd ceiling.
.\.venv\Scripts\python.exe scripts/run_multi_event_study.py analyze --workers 3
.\.venv\Scripts\python.exe scripts/audit_multi_event_study.py
.\.venv\Scripts\python.exe scripts/report_multi_event_study.py
```

An optional `design --reuse-requests <saved-request-grid.json>` preserves exact
existing intervals and adds only uncovered gaps. Each logical event window can
therefore use multiple contiguous, non-overlapping raw fragments. Every
fragment retains its own real DBN metadata, stream receipt and hash; joint
normalization never invents DBN metadata for the combined window. Empty
completed fragments differ from missing/incomplete sources. The current
study reuses 416 completed streams and adds 224 pre-release fragments.

The event-level result is its pre/post change minus the median change on all
five references. Missing controls suppress that excess estimate. Class tables
use equal bundle weights and report planned versus supported counts, year
dependence and leave-one-year-out sensitivities. The already-inspected 2026
WWDC bundle is separate from the 2022–2025 historical replication. Cross-session
class differences remain confounded with trading hours, and no causal effect,
p-value or event-class confidence interval is claimed.

Start with `out/aapl_multi_event/findings.html`, `result_audit.json`,
`event_excess.csv`, `class_summary.csv` and `control_draws.json`. The pilot's
`tie_diagnostics.json` independently decomposes after-hours exclusion into
quote-age gaps and conservative conflicting-timestamp barriers. Native
trade/cancel pairs sharing timestamps and sequence numbers can cause lengthy
unknown intervals under this policy; that is distinct from corrupt prices or
an incomplete download. The report retains those limitations.

The completed acquisition contains **33,338,763 records in 640 streams**
(about 718 MB compressed). The summed stream cost quotes are approximately
**$3.61**, not an invoice. All 115 focused tests pass. The independent audit
verifies all 640 source hashes/receipts, 104 checkpoints, 90,720 aggregate
metric rows and all event/bundle/year tables; 82,243 sampled native matching
cases have no mismatches.

At the primary 250-ms freshness setting, no event has a complete 30-minute,
10-ms-horizon comparison against all five references. At the predeclared
5,000-ms freshness sensitivity, all four historical WWDCs have positive excess
conditional latency: the class median is +10.96 microseconds on receive time
and +6.89 on exchange time. These are maximum quote-age settings, not measured
reaction times. Product-launch medians change sign across clocks (-29.23 versus
+11.15 microseconds), and no earnings release/call has a supported five-control
comparison. After-hours exclusions describe observability and reference support,
not a finding of zero earnings impact. These sensitivity results do not establish
a confirmed primary effect or a stable ranking of event types.

## Separate stationarity diagnostics on saved event windows

`aapl_stationarity_protocol.json` defines an offline ADF/KPSS diagnostic run on
the existing multi-event acquisition. It preserves native quality flags,
ambiguous-timestamp exclusions, freshness barriers and both clocks. Each
pre/post 30-minute window uses a one-second strict-prior grid. Only its earliest
longest unbroken supported run is retained, including breaks between otherwise
valid grid endpoints. No gaps or dates are joined. At least 300 observations
are required for each test; runs with 600 or more price points are flagged
descriptively. These thresholds do not guarantee useful unit-root test power.

The paired tests cover log midpoint levels with a constant, a level/trend
sensitivity, and one-second log returns with a constant. ADF uses BIC lag
selection with a capped Schwert maximum lag; KPSS uses automatic bandwidth.
The output records lags, sample lengths, critical values, warnings and bounded
KPSS p-values. Nominal paired classifications are exploratory. A separate
global Holm correction covers all unique ADF tests; duplicated price paths
reuse their test result rather than enlarging that family.

```powershell
.\.venv\Scripts\python.exe scripts/run_stationarity.py --workers 3
.\.venv\Scripts\python.exe scripts/audit_stationarity.py
.\.venv\Scripts\python.exe scripts/report_stationarity.py
```

Outputs are separate in `out/aapl_stationarity/`, including the exact integer
sampled paths, diagnostic window registry, test statistics, source/code hashes
and HTML readout. This run makes no market-data requests and does not transform
or overwrite the audited latency study. First-passage matching does not require
stationary price levels. ADF/KPSS diagnose possible regular-time model inputs;
they are not applied directly to irregular, censored latency observations.
Short windows, microstructure and structural breaks limit interpretation.
Per-venue tests also do not establish the joint sample or cointegration needed
for a common-price VAR/VECM study. See the version-matched
[ADF documentation](https://www.statsmodels.org/v0.14.4/generated/statsmodels.tsa.stattools.adfuller.html)
and [KPSS documentation](https://www.statsmodels.org/v0.14.4/generated/statsmodels.tsa.stattools.kpss.html).

The saved-data diagnostic evaluates 8,064 planned venue/clock/freshness/window
requests, retaining 3,631 distinct sampled paths. Only 318 paths have enough
unbroken observations for all three specifications. At nominal 5%, constant-only
log levels show the unit-root-compatible pattern in 277/318 paths, the
stationarity-compatible pattern in 4, conflicting rejections in 29 and an
inconclusive result in 8. Log returns show the stationarity-compatible pattern
in 304/318 paths and conflicting rejections in 14. All 318 return ADF tests
still reject a unit root after global Holm correction across the 954 computed
ADF tests. These are dependent selected segments, not independent event counts
or proof of stationarity. Neither clock supplies a 300-point unbroken run at
the primary 250-ms freshness setting; computed tests therefore concern the
1,000/5,000-ms sensitivities. The existing latency conclusions remain unchanged.
All 42 focused diagnostic tests pass. The independent audit recomputes every
paired result and Holm value, verifies all 640 raw source hashes, and rebuilds
192 selected windows from three representative raw source groups without a
mismatch.

## What the first result establishes

The API connection, download, event extraction, probabilistic clustering and held-out horizon calculations work end to end on real data. This establishes technical feasibility for an exploratory aggregate-response study. A ten-minute, one-symbol sample does not establish stable colonies, predictive usefulness, causal impact, participant identity or a trading strategy. Response events overlap; reported effective event counts are measures of probability concentration, not counts of independent observations. No statistical significance or confidence intervals are claimed.

EQUS.MINI has an anonymized aggregate BBO rather than quotes from each underlying venue. Its publisher ID cannot identify the originating exchange, and its order-count fields are always zero. A comparison of Nasdaq, NYSE and Cboe responses requires separate venue feeds and a synchronization design. A value investor's identity or investment horizon cannot be recovered from these prints.

The next research stage should audit feed quality flags and conditions, replicate fixed settings over multiple dates and symbols, use dependence-aware inference, and control for liquidity, volatility and time of day. Only then should external ETF ownership or fund-flow information be added as an explanatory layer. Venue-specific MBO can be assessed separately for order-lifecycle features; an exchange order ID still does not identify an investor.

## Primary references

- [EQUS.MINI feed specification](https://databento.com/docs/datasets/equs-mini)
- [Historical API cost estimator](https://databento.com/docs/api-historical/metadata/metadata-get-cost?lang=python)
- [Timestamping](https://databento.com/docs/architecture/timestamping-guide)
- [Official Databento Python SDK](https://github.com/databento/databento-python)
