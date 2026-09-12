# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Integration test for the canonical correlation-time-fitting workflow.

This module exercises the public, user-facing ``fit_corr_time`` example as a
stable happy-path integration case for the CLI pipeline.
"""

import os
from pathlib import Path

import pytest

from tests.helpers.fixtures import materialize_canonical_fixture
from tests.helpers.cli import run_paranmr


def _cli_env(tmp_path: Path) -> dict[str, str]:
    return {**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(tmp_path / "mpl")}


@pytest.mark.integration
def test_fit_corr_time(tmp_path: Path):
    """Run the canonical ``fit_corr_time`` CLI workflow.

    This integration test executes the canonical test configuration and asserts
    that the pipeline completes successfully and
    produces the expected diagnostics CSV artifact.
    """
    root = materialize_canonical_fixture(tmp_path=tmp_path, system="FeH")
    cwd = root / "SIMULATIONS" / "Fit_Correlation_Time"
    cmd = ["paranmr", "--hide", "fit_corr_time", "FeH_fit_corr_time.yml"]
    result = run_paranmr(cmd[1:], cwd=cwd, env=_cli_env(tmp_path))

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\nstdout:\n"
        f"{result.stdout}\nstderr:\n{result.stderr}"
    )

    expected_output = cwd / "FeH_Correlation_Time_Fit" / "corr_time_fit_diagnostics.csv"
    assert expected_output.exists(), f"Expected output file missing: {expected_output}"
