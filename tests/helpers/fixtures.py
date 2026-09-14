"""Materialize immutable integration fixtures into a writable test case."""

from __future__ import annotations

import shutil
from pathlib import Path

_DATA_ROOT = Path("tests/data")
_CONFIG_ROOT = Path("tests/integration/experimental/configs")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def repository_path(*parts: str) -> Path:
    """Return an absolute path inside the checked-out repository."""
    return _REPOSITORY_ROOT.joinpath(*parts)


def materialize_canonical_fixture(*, tmp_path: Path, system: str) -> Path:
    """Create a test-local case with canonical data and test-only configs."""
    root = tmp_path / system
    shutil.copytree(_REPOSITORY_ROOT / _DATA_ROOT / system / "DATA", root / "DATA")
    shutil.copytree(
        _REPOSITORY_ROOT / _CONFIG_ROOT / system / "SIMULATIONS",
        root / "SIMULATIONS",
    )
    return root
