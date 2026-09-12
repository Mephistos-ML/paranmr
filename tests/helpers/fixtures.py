"""Materialize immutable integration fixtures into a writable test case."""

from __future__ import annotations

import shutil
from pathlib import Path


_DATA_ROOT = Path("tests/data")
_CONFIG_ROOT = Path("tests/integration/experimental/configs")


def materialize_example_fixture(*, tmp_path: Path, system: str) -> Path:
    """Create a test-local case with canonical data and test-only configs."""
    root = tmp_path / system
    shutil.copytree(_DATA_ROOT / system / "DATA", root / "DATA")
    shutil.copytree(_CONFIG_ROOT / system / "SIMULATIONS", root / "SIMULATIONS")
    return root
