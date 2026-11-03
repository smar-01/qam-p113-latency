"""Offline native-event latency analysis of already saved, verified MBP-1 studies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def prepare_output_directories(protocol_path: Path, out_dir: Path) -> None:
    """Create shared worker parents serially to avoid Windows/OneDrive races."""
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    directories = [out_dir, out_dir / "days"]
    for config_name in protocol["study_configs"]:
        config_path = ROOT / config_name
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        base = out_dir / "days" / config_path.stem
        directories.append(base)
        directories.extend(base / day for day in config["training_days"] + config["held_out_days"])
    for directory in directories:
        for attempt in range(6):
            try:
                directory.mkdir(parents=True, exist_ok=True)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(.1 * (attempt + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "latency_protocol.json")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "latency")
    parser.add_argument("--workers", type=int, default=2, choices=range(1, 5))
    parser.add_argument("--summarize-only", action="store_true", help="Verify and summarize completed day checkpoints")
    args = parser.parse_args()
    from colonylab.latency_pipeline import run_latency
    if not args.summarize_only:
        prepare_output_directories(args.protocol.resolve(), args.out.resolve())
    result = run_latency(ROOT, args.protocol.resolve(), args.out.resolve(),
                         workers=args.workers, summarize_only=args.summarize_only)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
