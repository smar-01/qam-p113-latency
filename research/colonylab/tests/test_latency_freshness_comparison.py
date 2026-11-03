"""The sensitivity comparator must not hide changed matching code."""
import importlib.util
from pathlib import Path

import pytest


script = Path(__file__).resolve().parents[1] / "scripts" / "compare_latency_freshness.py"
spec = importlib.util.spec_from_file_location("compare_latency_freshness", script)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def identity():
    return [{"path": f"src/colonylab/{name}.py", "sha256": "0" * 64}
            for name in ("latency", "latency_data", "latency_pipeline")]


def test_driver_difference_requires_explicit_opt_in_and_is_reported():
    first, second = identity(), identity()
    first[-1]["sha256"] = "1" * 64
    with pytest.raises(ValueError, match="explicitly allowed"):
        comparison._compare_code(first, second, allow_driver_change=False)
    changed = comparison._compare_code(first, second, allow_driver_change=True)
    assert changed == [{"path": "src/colonylab/latency_pipeline.py", "primary_sha256": "1" * 64,
                        "reference_sha256": "0" * 64}]


@pytest.mark.parametrize("position", [0, 1])
def test_driver_exception_cannot_allow_matching_or_loader_code_change(position):
    first, second = identity(), identity()
    first[position]["sha256"] = "1" * 64
    with pytest.raises(ValueError, match="matching/quote-loader"):
        comparison._compare_code(first, second, allow_driver_change=True)
