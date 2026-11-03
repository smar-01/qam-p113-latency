"""Run the source-verified AAPL corporate-event case study on saved data only."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "aapl_event_trial.json")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "aapl_event_trial")
    parser.add_argument("--workers", type=int, choices=[1, 2, 3, 4], default=2)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    from colonylab.aapl_event_trial import run
    result = run(ROOT, args.protocol, args.out, workers=args.workers, summarize_only=args.summarize_only)
    print(json.dumps({key: result[key] for key in ("status", "event_count", "reference_days", "source_cache_tasks", "independent_matching_cases", "independent_matching_mismatches")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
