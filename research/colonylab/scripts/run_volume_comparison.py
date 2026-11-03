"""Compare volume and time candles using the cached wider research sample."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from colonylab.config import safe_error
from colonylab.network_wide_data import load_wide_frames
from colonylab.volume_prediction import run_volume_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "out/wide_network/data_manifest.json")
    parser.add_argument("--shocks", type=Path, default=ROOT / "out/wide_network/shock_events.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out/volume_comparison")
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text())
        frames = load_wide_frames(args.manifest)
        result = run_volume_comparison(
            frames, args.shocks, args.out_dir, training_days=manifest["training_days"],
            held_out_days=manifest["held_out_days"], progress=lambda s: print(s, flush=True))
        print(json.dumps({"pooled": result["pooled"], "summary": str(args.out_dir / "volume_comparison_summary.json")}, indent=2))
    except Exception as exc:
        print(safe_error(exc), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
