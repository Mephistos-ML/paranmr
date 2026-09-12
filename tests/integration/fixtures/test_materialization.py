# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

from pathlib import Path

import pytest

from tests.helpers.fixtures import materialize_example_fixture


@pytest.mark.integration
@pytest.mark.parametrize("system", ["P3FeCl", "DyL1", "FeH", "YbL8"])
def test_materialize_example_fixture_creates_writable_case(
    tmp_path: Path, system: str
) -> None:
    """Canonical input data and test configs form one isolated runnable case."""
    root = materialize_example_fixture(tmp_path=tmp_path, system=system)

    assert (root / "DATA").is_dir()
    assert (root / "SIMULATIONS").is_dir()

    marker = root / "generated-by-test.txt"
    marker.write_text("ok", encoding="utf-8")
    assert marker.read_text(encoding="utf-8") == "ok"
