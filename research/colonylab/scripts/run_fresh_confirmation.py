"""Score the frozen eight-venue fresh-date volume-candle confirmation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from colonylab.config import safe_error
from colonylab.network_wide_data import read_wide_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "fresh_confirmation.toml")
    args = parser.parse_args()
    try:
        from colonylab.fresh_confirmation import run_fresh_confirmation

        config_path = args.config.resolve()
        config = read_wide_config(config_path)
        out_dir = (ROOT / config["out_dir"]).resolve()
        result = run_fresh_confirmation(
            out_dir / "data_manifest.json", config_path, out_dir, anchor_step_ms=100,
        )
        print(json.dumps(result, indent=2, default=str, allow_nan=False), flush=True)
    except Exception as exc:
        print(safe_error(exc), file=sys.stderr, flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
