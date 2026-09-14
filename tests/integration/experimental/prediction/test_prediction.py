# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Integration tests for canonical predict example workflows.

These tests exercise public, user-facing prediction examples that represent
stable happy-path configurations for the main supported input combinations.
"""

import os
from pathlib import Path

import pytest

from tests.helpers.cli import run_paranmr
from tests.helpers.fixtures import materialize_canonical_fixture


def _cli_env(tmp_path: Path) -> dict[str, str]:
    return {**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(tmp_path / "mpl")}


@pytest.mark.integration
def test_predict_with_qc_hfc_and_qc_susceptibility(tmp_path: Path):
    """Run the canonical P3FeCl prediction example end-to-end.

    This example covers the public happy-path combination of QC-derived
    hyperfine input with QC-derived susceptibility input, including the
    relaxation-enabled prediction workflow.
    """
    root = materialize_canonical_fixture(tmp_path=tmp_path, system="P3FeCl")
    cwd = root / "SIMULATIONS" / "Prediction"
    cmd = ["paranmr", "--hide", "predict", "P3FeCl_Prediction.yml"]
    result = run_paranmr(cmd[1:], cwd=cwd, env=_cli_env(tmp_path))

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\nstdout:\n"
        f"{result.stdout}\nstderr:\n{result.stderr}"
    )

    expected_output = cwd / "P3FeCl_13C_Prediction" / "susceptibility_tensor.csv"
    assert expected_output.exists(), f"Expected output file missing: {expected_output}"


@pytest.mark.integration
def test_predict_with_pdip_hfc_and_csv_susceptibility(tmp_path: Path):
    """Run the canonical DyL1 prediction example end-to-end.

    This example covers the public happy-path combination of point-dipole
    hyperfine input from XYZ coordinates with CSV-based susceptibility input,
    including the relaxation-enabled prediction workflow.
    """
    root = materialize_canonical_fixture(tmp_path=tmp_path, system="DyL1")
    cwd = root / "SIMULATIONS" / "Prediction"
    cmd = ["paranmr", "--hide", "predict", "DyL1_1H_Prediction.yml"]
    result = run_paranmr(cmd[1:], cwd=cwd, env=_cli_env(tmp_path))

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\nstdout:\n"
        f"{result.stdout}\nstderr:\n{result.stderr}"
    )

    expected_output = cwd / "DyL1_1H_Prediction" / "susceptibility_tensor.csv"
    assert expected_output.exists(), f"Expected output file missing: {expected_output}"
