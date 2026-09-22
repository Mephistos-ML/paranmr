# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Integration test for the spin-Hamiltonian CLI workflow."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.helpers.cli import run_simpnmr_x
from tests.helpers.fixtures import repository_path


@pytest.mark.integration
def test_get_sh_with_isoaxrho_fit_csv_runs_successfully(tmp_path: Path):
    """Run the ``get_sh`` CLI workflow on an ``isoaxrho`` fit result.

    This test exercises the CLI-driven spin-Hamiltonian extraction path using
    an internal susceptibility-fit CSV fixture and verifies successful command
    execution together with creation of the expected output CSV.
    """
    cwd = tmp_path / "spinham"
    shutil.copytree(repository_path("tests", "data", "pipelines", "spinham"), cwd)
    cmd = ["simpnmr-x", "get_sh", "--spin", "2.0", "isoaxrho_fit.csv"]
    result = run_simpnmr_x(cmd[1:], cwd=cwd)

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\nstdout:\n"
        f"{result.stdout}\nstderr:\n{result.stderr}"
    )
    expected_output = cwd / "spin_hamiltonian_from_chiT.csv"
    assert expected_output.exists(), f"Expected output file missing: {expected_output}"
