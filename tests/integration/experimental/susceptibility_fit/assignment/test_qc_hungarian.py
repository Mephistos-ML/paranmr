# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Integration coverage for QC hyperfine Hungarian assignment fitting."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.helpers.cli import run_paranmr
from tests.helpers.fixtures import materialize_canonical_fixture


def _cli_env(tmp_path: Path) -> dict[str, str]:
    return {
        **os.environ,
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
        "XDG_CACHE_HOME": str(tmp_path / "xdg-cache"),
    }


@pytest.mark.integration
def test_qc_hungarian_assignment_fit(tmp_path: Path) -> None:
    """Run the canonical P3FeCl QC Hungarian VT fitting workflow."""
    root = materialize_canonical_fixture(tmp_path=tmp_path, system="P3FeCl")
    cwd = root / "SIMULATIONS" / "Fitting"
    result = run_paranmr(
        ["--hide", "fit_susc", "P3FeCl_VT_Fitting.yml"],
        cwd=cwd,
        env=_cli_env(tmp_path),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    output = cwd / "P3FeCl_VT_Fitting"
    tensor = pd.read_csv(
        output / "susceptibility_tensor.csv",
        comment="#",
        encoding="utf-8-sig",
    )
    assert len(tensor) == 6
    assert np.isfinite(tensor.select_dtypes("number").to_numpy()).all()
