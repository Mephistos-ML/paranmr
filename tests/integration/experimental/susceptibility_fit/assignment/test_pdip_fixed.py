# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Integration coverage for PDIP fixed-assignment susceptibility fitting."""

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
def test_pdip_fixed_assignment_fit(tmp_path: Path) -> None:
    """Run the canonical DyL1 PDIP fixed-assignment workflow."""
    root = materialize_canonical_fixture(tmp_path=tmp_path, system="DyL1")
    cwd = root / "SIMULATIONS" / "Fitting" / "Standart_Fit"
    result = run_paranmr(
        ["--hide", "fit_susc", "DyL1_1H_Fitting.yml"],
        cwd=cwd,
        env=_cli_env(tmp_path),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    output = cwd / "DyL1_1H_Fitting"
    tensor = pd.read_csv(
        output / "susceptibility_tensor.csv",
        comment="#",
        encoding="utf-8-sig",
    )
    assert tensor.shape[0] == 1
    assert np.isfinite(tensor.select_dtypes("number").to_numpy()).all()

    peak_data = pd.read_csv(
        output / "peak_data_302.15_K.csv",
        comment="#",
        encoding="utf-8-sig",
    )
    assert peak_data["linewidth_exp (ppm)"].notna().all()
    assert (output / "objective_map_ax_rho_over_ax_302.15_K.pdf").is_file()
