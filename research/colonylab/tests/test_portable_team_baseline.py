"""Pinned comparison fixtures work without the original author's checkout."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("portable_team_baseline_test", ROOT / "scripts/reproduce_team_code_corrections.py")
portable = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(portable)


def test_shipped_baseline_is_complete_and_pinned():
    verified = portable.verified_baseline(portable.BASELINE)
    manifest = json.loads((verified / "source_manifest.json").read_text())
    assert manifest["commit"] == "62d6dcd8af9ecd7136096dbcce3eb40371987422"
    assert len(manifest["files"]) == 8


def test_relocated_baseline_is_accepted_without_any_external_checkout(tmp_path):
    target = tmp_path / "baseline"
    shutil.copytree(portable.BASELINE, target)
    assert portable.verified_baseline(target) == target.resolve()


def test_mutated_baseline_is_rejected_before_comparison(tmp_path):
    target = tmp_path / "baseline"
    shutil.copytree(portable.BASELINE, target)
    (target / "p113/leadlag.py").write_text("changed\n")
    with pytest.raises(ValueError, match="Baseline differs.*p113/leadlag.py"):
        portable.verified_baseline(target)


def test_external_crlf_checkout_matches_canonical_git_source(tmp_path):
    target = tmp_path / "baseline"
    shutil.copytree(portable.BASELINE, target)
    source = target / "p113/leadlag.py"
    source.write_bytes(source.read_bytes().replace(b"\n", b"\r\n"))
    assert portable.verified_baseline(target) == target.resolve()


def test_changed_existing_output_snapshot_cannot_be_reused(tmp_path):
    snapshot = tmp_path / "original_source/p113"
    snapshot.mkdir(parents=True)
    (snapshot / "load.py").write_text("another implementation\n")
    with pytest.raises(ValueError, match="Existing output snapshot differs.*p113/load.py"):
        portable.verify_existing_snapshot(tmp_path)
