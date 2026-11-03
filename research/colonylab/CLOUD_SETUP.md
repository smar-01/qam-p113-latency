# Local and cloud Python setup

The combined public repository is
[`smar-01/qam-p113-latency`](https://github.com/smar-01/qam-p113-latency).
ColonyLab lives under `research/colonylab/`; the original standalone project
remains at the repository root. Use Python 3.11 for the validated combined
environment; the legacy standalone requirements describe a separate Python
3.12 environment.

## Combined environment

From the combined repository root, create and activate an environment using
your platform's normal activation command, then install and check both projects:

```text
python -m venv .venv
# Activate .venv using your platform's normal activation command.
python -m pip install -r requirements-research.txt
python -m pytest
python analyze.py --synthetic --clock both
python -m colonylab.cli demo
```

The research requirements install the nested editable package and combined
dependency constraints. The root pytest configuration collects the standalone
tests and ColonyLab's main test suite. Historical comparison fixtures are
exercised by explicit regression tests rather than recursively collecting
their duplicate upstream test names. Installation needs package-registry
access; tests and synthetic demonstrations need neither credentials nor
market-data downloads.

In a Linux cloud environment, the equivalent explicit commands are:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-research.txt
MPLBACKEND=Agg .venv/bin/python -m pytest -q --basetemp .tmp/pytest-cloud
MPLBACKEND=Agg .venv/bin/python -m colonylab.cli demo
```

The imported `scripts/setup_cloud.sh` remains a project-local installer for
ColonyLab alone. The combined setup above also covers the standalone project
and its root test configuration. No setup command acquires market data.

## Results and saved research data

A public clone includes all research source, protocols, tests and documentation,
the [earlier-study aggregate results](results/earlier/README.md), and the
[latency/event/stationarity publication bundle](results/README.md). These bundles
provide findings and validation evidence without raw observations or local
output directories. Reading them requires no API key or private archive.

The original October 28 research snapshot was stored separately in a restricted
archive: 6,695 `data/` and `out/` files, 22,511,155,154 uncompressed bytes across
15 ZIP parts. `research-archive-manifest.json` records original relative paths,
sizes and SHA-256 identities. Those raw-data archives are not distributed by
this public repository, and a public clone does not grant access to them.
The additional November raw data and full local outputs are also excluded.

For a separately authorized holder of those archive files, the imported
`scripts/research_archive.py` can verify and restore them using the manifest;
run it from the nested ColonyLab directory and inspect `--help` for explicit
archive and destination paths. It verifies hashes and refuses to overwrite
differing files. Restoration preserves historical bytes. Absolute paths in
old output provenance do not automatically relocate, so restoring a snapshot
does not make every original audit executable under another workspace. Fresh
runs record their actual local inputs; audited JSON is not silently rewritten.

## Credentials and live requests

`.env`, credentials, environments, raw data and full output directories remain
outside Git. Synthetic tests and demonstrations do not need a key. Live
Databento operations require a separately configured credential and the
explicit request, availability and cost controls of the selected pipeline.
The frozen `wide_network.toml`, `fresh_confirmation.toml` and `mbo_sir.toml`
retain `billing_assumed_included = true` as the original account's historical
policy. Their included-access settings do not verify entitlement or free
acquisition for another account. Review the configuration and run the selected
pipeline's explicit planning and acquisition controls for your own access
before downloading.

An included-access setting is not a provider-enforced spending cap or a
guarantee of data entitlement. Setup does not run those operations.
