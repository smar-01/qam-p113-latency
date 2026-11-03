"""Diagnose saved native price levels and log returns without new downloads."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "aapl_stationarity_protocol.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out/aapl_stationarity")
    parser.add_argument("--workers", type=int, choices=[1, 2, 3, 4], default=3)
    args = parser.parse_args()
    from colonylab.stationarity_pipeline import analyze
    result = analyze(ROOT, args.config, args.out_dir, workers=args.workers)
    print(json.dumps({"status": result["status"], **result["counts"]}), flush=True)


if __name__ == "__main__":
    main()
