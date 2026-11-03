"""Portable entry point for the saved original-versus-corrected experiment.

The historical runner stays byte-identical to its audited version. This wrapper
replaces only its machine-specific baseline location with a verified source
subset pinned to the original team commit. It never downloads data.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "comparisons/qam_p113_baseline"


def verified_baseline(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((BASELINE / "source_manifest.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        source = directory / row["path"]
        if not source.is_file() or hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != row["sha256"]:
            raise ValueError(f"Baseline differs from pinned team source {manifest['commit']}: {row['path']}")
    return directory


def verify_existing_snapshot(out):
    """Never silently resume against another original implementation."""
    snapshot = Path(out).resolve() / "original_source"
    manifest = json.loads((BASELINE / "source_manifest.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        source = snapshot / row["path"]
        if source.exists() and (not source.is_file() or hashlib.sha256(
                source.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != row["sha256"]):
            raise ValueError(f"Existing output snapshot differs from pinned baseline: {row['path']}; use another --out")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--original-dir", type=Path, default=BASELINE,
                        help="Original source directory; used files must match the pinned baseline after LF normalization")
    parser.add_argument("--out", type=Path, default=ROOT / "out/team_code_corrections")
    args, remainder = parser.parse_known_args(argv)
    directory = verified_baseline(args.original_dir)
    verify_existing_snapshot(args.out)
    # Reuse the historical implementation without changing its audited bytes.
    path = ROOT / "scripts/run_team_code_corrections.py"
    spec = importlib.util.spec_from_file_location("portable_team_correction_runner", path)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    runner.ORIGINAL = directory
    return runner.main(["--out", str(args.out), *remainder])


if __name__ == "__main__":
    main()
