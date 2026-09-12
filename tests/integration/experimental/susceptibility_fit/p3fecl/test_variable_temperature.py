# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Variable-temperature susceptibility fitting against the P3FeCl fixture."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.helpers.fixtures import materialize_example_fixture
from tests.helpers.cli import run_paranmr


@pytest.mark.integration
def test_variable_temperature_split_fit(tmp_path: Path) -> None:
    """Fit the P3FeCl variable-temperature reference workflow end to end."""
    root = materialize_example_fixture(tmp_path=tmp_path, system="P3FeCl")
    cwd = root / "SIMULATIONS" / "Fitting"
    environment = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
        "XDG_CACHE_HOME": str(tmp_path / "xdg-cache"),
    }
    result = run_paranmr(
        ["--hide", "fit_susc", "P3FeCl_VT_Fitting.yml"], cwd=cwd, env=environment
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (cwd / "P3FeCl_VT_Fitting" / "susceptibility_tensor.csv").exists()
