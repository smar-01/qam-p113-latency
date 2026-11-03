#!/usr/bin/env bash
# Install the project for Linux cloud development; never fetch market data.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

python_bin="${PYTHON_BIN:-python3}"
"$python_bin" -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11 or later is required"'
"$python_bin" -m venv .venv
.venv/bin/python -m pip install 'setuptools>=65' wheel
.venv/bin/python -m pip install --editable . pytest

printf '%s\n' \
  'Cloud dependencies installed.' \
  'Run offline tests: MPLBACKEND=Agg .venv/bin/python -m pytest -q' \
  'Run the synthetic demonstration: MPLBACKEND=Agg .venv/bin/python -m colonylab.cli demo'
