# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Integration tests for canonical susceptibility-fitting workflows.

These tests exercise public, user-facing ``fit_susc`` examples that represent
stable happy-path configurations for the main supported fitting setups.
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _cli_env(tmp_path: Path) -> dict[str, str]:
    mpl_cache = tmp_path / "matplotlib"
    xdg_cache = tmp_path / "xdg-cache"
    mpl_cache.mkdir()
    xdg_cache.mkdir()

    return {
        **os.environ,
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(mpl_cache),
        "XDG_CACHE_HOME": str(xdg_cache),
    }


@pytest.mark.integration
def test_fit_susc_with_pdip_hfc_isoaxrho_and_permutation_assignment(tmp_path: Path):
    """Run the canonical ``fit_susc`` workflow with PDIP hyperfine input.

    This public happy-path case uses point-dipole hyperfine data from XYZ
    coordinates together with the ``isoaxrho`` susceptibility fit model and
    permutation-based assignment.
    """
    cwd = Path("examples/DyL1/SIMULATIONS/Fitting/Standart_Fit")
    cmd = ["paranmr", "--hide", "fit_susc", "DyL1_1H_Fitting.yml"]
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=cwd, env=_cli_env(tmp_path)
    )

    if result.returncode != 0 and "missing" in result.stderr.lower():
        pytest.xfail(f"Test failed due to missing dependency:\n{result.stderr}")

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\nstdout:\n"
        f"{result.stdout}\nstderr:\n{result.stderr}"
    )

    expected_output = cwd / "DyL1_1H_Fitting" / "susceptibility_tensor.csv"
    assert expected_output.exists(), f"Expected output file missing: {expected_output}"

    peak_output = cwd / "DyL1_1H_Fitting" / "peak_data_302.15_K.csv"
    assert peak_output.exists(), f"Expected output file missing: {peak_output}"

    peak_data = pd.read_csv(peak_output, comment="#", encoding="utf-8-sig")
    assert "linewidth_exp (ppm)" in peak_data.columns
    linewidths = peak_data["linewidth_exp (ppm)"].to_numpy(dtype=float)
    assert np.isfinite(linewidths).all()


@pytest.mark.integration
def test_fit_susc_with_qc_hfc_split_model_and_hungarian_assignment(tmp_path: Path):
    """Run the canonical variable-temperature ``fit_susc`` workflow.

    This public happy-path case uses DFT-derived hyperfine input together with
    the ``split`` susceptibility fit model and Hungarian-based assignment. It
    exercises the variable-temperature fitting path, where the workflow fits a
    shared susceptibility model across multiple experiments recorded at
    different temperatures.
    """
    cwd = Path("examples/P3FeCl/SIMULATIONS/Fitting")
    cmd = ["paranmr", "--hide", "fit_susc", "P3FeCl_VT_Fitting.yml"]
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=cwd, env=_cli_env(tmp_path)
    )

    if result.returncode != 0 and "missing" in result.stderr.lower():
        pytest.xfail(f"Test failed due to missing dependency:\n{result.stderr}")

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\nstdout:\n"
        f"{result.stdout}\nstderr:\n{result.stderr}"
    )

    expected_output = cwd / "P3FeCl_VT_Fitting" / "susceptibility_tensor.csv"
    assert expected_output.exists(), f"Expected output file missing: {expected_output}"


@pytest.mark.integration
def test_fit_susc_moments_weighted_ls_smoke(tmp_path: Path):
    """Run the canonical ``fit_susc`` moments workflow using weighted LS.

    This public happy-path case exercises the assignment-free moments branch
    with the ``isoaxrho_euler`` susceptibility model and the ``Weighted_LS_Obj``
    example dataset. It asserts that the workflow completes, writes diagnostics,
    and produces finite moment outputs with a finite weighted score.
    """
    cwd = Path("examples/DyL1/SIMULATIONS/Fitting/Moments/Weighted_LS_Obj")
    cmd = ["paranmr", "--hide", "fit_susc", "DyL1_1H_Fitting_moments_iso_ax_rho.yml"]
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=cwd, env=_cli_env(tmp_path)
    )

    if result.returncode != 0 and "missing" in result.stderr.lower():
        pytest.xfail(f"Test failed due to missing dependency:\n{result.stderr}")

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\nstdout:\n"
        f"{result.stdout}\nstderr:\n{result.stderr}"
    )

    expected_output = (
        cwd
        / "DyL1_1H_Fitting_Moments_iso_ax_rho"
        / "moment_fit_diagnostics_302.15_K.csv"
    )
    assert expected_output.exists(), f"Expected output file missing: {expected_output}"

    diagnostics = pd.read_csv(expected_output, comment="#", encoding="utf-8-sig")
    assert set(diagnostics["quantity"]) == {"observed", "calculated"}

    numeric_cols = [column for column in diagnostics.columns if column != "quantity"]
    assert np.isfinite(diagnostics[numeric_cols].to_numpy(dtype=float)).all()

    file_text = expected_output.read_text(encoding="utf-8-sig")
    weighted_score_line = next(
        line for line in file_text.splitlines() if line.startswith("# weighted_score = ")
    )
    weighted_score = float(weighted_score_line.split("=", maxsplit=1)[1].strip())
    assert np.isfinite(weighted_score)

    linewidth_model_output = (
        cwd
        / "DyL1_1H_Fitting_Moments_iso_ax_rho"
        / "linewidth_model_302.15_K.csv"
    )
    assert linewidth_model_output.exists(), (
        f"Expected output file missing: {linewidth_model_output}"
    )

    linewidth_model = pd.read_csv(
        linewidth_model_output, comment="#", encoding="utf-8-sig"
    )
    assert list(linewidth_model.columns) == ["linewidth_method", "p1", "p2"]
    assert linewidth_model.shape == (1, 3)
    assert linewidth_model["linewidth_method"].iloc[0] == "r6"
    assert np.isfinite(float(linewidth_model["p1"].iloc[0]))
    assert np.isfinite(float(linewidth_model["p2"].iloc[0]))

    peak_output = (
        cwd / "DyL1_1H_Fitting_Moments_iso_ax_rho" / "peak_data_302.15_K.csv"
    )
    assert peak_output.exists(), f"Expected output file missing: {peak_output}"

    peak_data = pd.read_csv(peak_output, comment="#", encoding="utf-8-sig")
    assert "linewidth_r6_fit (ppm)" in peak_data.columns
    linewidths = peak_data["linewidth_r6_fit (ppm)"].to_numpy(dtype=float)
    assert np.isfinite(linewidths).all()
    assert np.all(linewidths > 0.0)

    susc_output = (
        cwd / "DyL1_1H_Fitting_Moments_iso_ax_rho" / "susceptibility_tensor.csv"
    )
    assert susc_output.exists(), f"Expected output file missing: {susc_output}"

    susc = pd.read_csv(susc_output, comment="#", encoding="utf-8-sig")
    chi_ax = float(susc["chi_ax (Å^3)"].iloc[0])
    assert np.isfinite(chi_ax)
