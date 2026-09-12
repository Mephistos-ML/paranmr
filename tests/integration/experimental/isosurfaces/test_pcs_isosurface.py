# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Integration test for the PCS isosurface CLI workflow."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration
def test_calc_pcs_iso_with_csv_susceptibility_creates_cube_file():
    """Run the PCS isosurface workflow with CSV-based susceptibility input.

    This test exercises the CLI-driven ``calc_pcs_iso`` path on the internal
    DyL1 fixture using susceptibility data loaded from a CSV file and verifies
    that at least one cube file is produced.
    """
    cwd = Path("tests/data/pipelines/pcs_isosurface/csv_susceptibility_dyl1")
    cmd = [
        "paranmr",
        "calc_pcs_iso",
        "Chi_DyL1_from_B20.csv",
        "302.15",
        "structure.xyz",
        "Dy1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)

    if result.returncode != 0 and "missing" in result.stderr.lower():
        pytest.xfail(f"Test failed due to missing dependency:\n{result.stderr}")

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\nstdout:\n"
        f"{result.stdout}\nstderr:\n{result.stderr}"
    )

    cube_file = sorted(cwd.glob("*.cube"))
    assert cube_file, "Expected one .cube file to be created"


@pytest.mark.integration
def test_calc_pcs_iso_with_nevpt2_susceptibility_creates_cube_file():
    """Run the PCS isosurface workflow with NEVPT2 susceptibility input.

    This test exercises the CLI-driven ``calc_pcs_iso`` path on the internal
    P3FeCl fixture using susceptibility data loaded from a NEVPT2 output file
    and verifies that at least one cube file is produced.
    """
    cwd = Path("tests/data/pipelines/pcs_isosurface/nevpt2_susceptibility_p3fecl")
    cmd = [
        "paranmr",
        "calc_pcs_iso",
        "P3FeCl_Susceptibility_NEVPT2.out",
        "298.00",
        "structure.xyz",
        "Fe1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)

    if result.returncode != 0 and "missing" in result.stderr.lower():
        pytest.xfail(f"Test failed due to missing dependency:\n{result.stderr}")

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\nstdout:\n"
        f"{result.stdout}\nstderr:\n{result.stderr}"
    )

    cube_file = sorted(cwd.glob("*.cube"))
    assert cube_file, "Expected one .cube file to be created"
