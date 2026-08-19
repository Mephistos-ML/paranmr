# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Integration tests for canonical susceptibility-fitting workflows.

These tests exercise public, user-facing ``fit_susc`` examples that represent
stable happy-path configurations for the main supported fitting setups.
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paranmr.app.policies.averaging import detect_methyl_group_records
from paranmr.core.domain.mol import Molecule
from paranmr.tools.coords import xyz_fmt as xyzf


def _cli_env(tmp_path: Path) -> dict[str, str]:
    mpl_cache = tmp_path / "matplotlib"
    xdg_cache = tmp_path / "xdg-cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)

    return {
        **os.environ,
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(mpl_cache),
        "XDG_CACHE_HOME": str(xdg_cache),
    }


def _read_generated_hyperfines_table(path: Path) -> pd.DataFrame:
    """Read a generated hyperfines CSV with a free-form metadata preamble."""

    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header_index = next(
        index for index, line in enumerate(lines) if line.startswith("atom_label")
    )
    return pd.read_csv(path, skiprows=header_index, encoding="utf-8-sig")


def _reference_signal_partitions(
    *,
    labels_csv: Path,
) -> set[frozenset[str]]:
    """Return the fixed-assignment proton partition from the labels reference."""

    labels = pd.read_csv(labels_csv, encoding="utf-8-sig")
    return {
        frozenset(group["atom_label"].astype(str).tolist())
        for _, group in labels.groupby("signal_label", sort=False)
    }


def _reference_signal_groups(
    *,
    labels_csv: Path,
) -> list[frozenset[str]]:
    """Return ordered fixed-assignment proton groups from the labels reference."""

    labels = pd.read_csv(labels_csv, encoding="utf-8-sig")
    return [
        frozenset(group["atom_label"].astype(str).tolist())
        for _, group in labels.groupby("signal_label", sort=False)
    ]


def _reference_signal_partitions_with_methyls(
    *,
    labels_csv: Path,
    xyz_file: Path,
) -> set[frozenset[str]]:
    """Return the reference proton partition refined by the methyl policy."""

    reference_partition = _reference_signal_partitions(labels_csv=labels_csv)
    labels, coords = xyzf.load_xyz(str(xyz_file), check=False)
    molecule = Molecule.from_labels_coords(labels, coords, elements="H")
    methyl_groups = [
        frozenset(group.proton_labels)
        for group in detect_methyl_group_records(molecule)
    ]

    refined_partition: set[frozenset[str]] = set()
    for reference_group in reference_partition:
        matching_methyls = [
            methyl_group
            for methyl_group in methyl_groups
            if methyl_group.issubset(reference_group)
        ]
        if matching_methyls and frozenset().union(*matching_methyls) == reference_group:
            refined_partition.update(matching_methyls)
        else:
            refined_partition.add(reference_group)
    return refined_partition


def _reference_signal_groups_with_methyls(
    *,
    labels_csv: Path,
    xyz_file: Path,
) -> list[frozenset[str]]:
    """Return ordered reference proton groups refined by the methyl policy."""

    reference_groups = _reference_signal_groups(labels_csv=labels_csv)
    labels, coords = xyzf.load_xyz(str(xyz_file), check=False)
    molecule = Molecule.from_labels_coords(labels, coords, elements="H")
    methyl_groups = [
        frozenset(group.proton_labels)
        for group in detect_methyl_group_records(molecule)
    ]

    refined_groups: list[frozenset[str]] = []
    for reference_group in reference_groups:
        matching_methyls = [
            methyl_group
            for methyl_group in methyl_groups
            if methyl_group.issubset(reference_group)
        ]
        if matching_methyls and frozenset().union(*matching_methyls) == reference_group:
            refined_groups.extend(matching_methyls)
        else:
            refined_groups.append(reference_group)
    return refined_groups


def _partition_protons_by_sorted_shift(
    *,
    hyperfines_csv: Path,
    group_sizes: list[int],
) -> set[frozenset[str]]:
    """Partition proton labels by sorted fitted shift using reference group sizes."""

    table = _read_generated_hyperfines_table(hyperfines_csv)
    protons = table[table["atom_label ()"].astype(str).str.startswith("H")].copy()
    protons = protons.sort_values("δ_total (ppm)", kind="mergesort")

    atom_labels = protons["atom_label ()"].astype(str).tolist()
    assert sum(group_sizes) == len(atom_labels), (
        "Reference group sizes do not match the number of fitted proton rows"
    )

    groups: list[frozenset[str]] = []
    start = 0
    for size in group_sizes:
        stop = start + size
        groups.append(frozenset(atom_labels[start:stop]))
        start = stop
    return set(groups)


def _generated_signal_partitions(
    *,
    hyperfines_csv: Path,
) -> set[frozenset[str]]:
    """Return the proton partition encoded in generated fitted hyperfines output."""

    table = _read_generated_hyperfines_table(hyperfines_csv)
    protons = table[table["atom_label ()"].astype(str).str.startswith("H")].copy()
    return {
        frozenset(group["atom_label ()"].astype(str).tolist())
        for _, group in protons.groupby("signal_label ()", sort=False)
    }


def _group_centers_from_hyperfines(
    *,
    hyperfines_csv: Path,
    proton_groups: set[frozenset[str]] | list[frozenset[str]],
) -> dict[frozenset[str], float]:
    """Return fitted group centers as mean proton shifts for each proton group."""

    table = _read_generated_hyperfines_table(hyperfines_csv)
    protons = table[table["atom_label ()"].astype(str).str.startswith("H")].copy()

    centers: dict[frozenset[str], float] = {}
    for proton_group in proton_groups:
        mask = protons["atom_label ()"].isin(proton_group)
        matched = protons.loc[mask, "δ_total (ppm)"].to_numpy(dtype=float)
        assert len(matched) == len(proton_group), (
            "Fitted proton rows do not match the requested proton group"
        )
        centers[proton_group] = float(np.mean(matched))
    return centers


def _range_based_ppm_tolerance(
    *,
    centers_by_group: dict[frozenset[str], float],
    fraction: float = 0.03,
) -> float:
    """Return a ppm tolerance as a fraction of the fitted spectral range."""

    center_values = np.asarray(list(centers_by_group.values()), dtype=float)
    spectral_range = float(np.max(center_values) - np.min(center_values))
    return fraction * spectral_range


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

    objective_map_output = (
        cwd / "DyL1_1H_Fitting" / "objective_map_ax_rho_over_ax_302.15_K.pdf"
    )
    assert objective_map_output.exists(), (
        f"Expected output file missing: {objective_map_output}"
    )

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
    score_line = next(
        line for line in file_text.splitlines() if line.startswith("# score = ")
    )
    score = float(score_line.split("=", maxsplit=1)[1].strip())
    assert np.isfinite(score)

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

    moment_jacobian_output = (
        cwd
        / "DyL1_1H_Fitting_Moments_iso_ax_rho"
        / "moment_jacobian_302.15_K.csv"
    )
    assert moment_jacobian_output.exists(), (
        f"Expected output file missing: {moment_jacobian_output}"
    )

    moment_jacobian = pd.read_csv(
        moment_jacobian_output, comment="#", encoding="utf-8-sig"
    )
    assert list(moment_jacobian.columns) == ["quantity", "ax", "rho_over_ax"]
    assert moment_jacobian.shape == (6, 3)
    jacobian_values = moment_jacobian.drop(columns=["quantity"]).to_numpy(
        dtype=float
    )
    assert np.isfinite(jacobian_values).all()

    moment_jacobian_heatmap_output = (
        cwd
        / "DyL1_1H_Fitting_Moments_iso_ax_rho"
        / "moment_jacobian_heatmap_302.15_K.pdf"
    )
    assert moment_jacobian_heatmap_output.exists(), (
        f"Expected output file missing: {moment_jacobian_heatmap_output}"
    )

    moment_covariance_heatmap_output = (
        cwd
        / "DyL1_1H_Fitting_Moments_iso_ax_rho"
        / "moment_covariance_heatmap_302.15_K.pdf"
    )
    assert not moment_covariance_heatmap_output.exists()

    objective_map_output = (
        cwd
        / "DyL1_1H_Fitting_Moments_iso_ax_rho"
        / "objective_map_ax_rho_over_ax_302.15_K.pdf"
    )
    assert objective_map_output.exists(), (
        f"Expected output file missing: {objective_map_output}"
    )

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


@pytest.mark.integration
def test_fit_susc_moments_gmm_recovers_dyl1_reference_proton_partition(
    tmp_path: Path,
):
    """Check that DyL1 GMM moment fitting recovers fixed proton groups and positions."""

    dyl1_root = tmp_path / "DyL1"
    shutil.copytree(Path("examples/DyL1"), dyl1_root)

    fixed_cwd = dyl1_root / "SIMULATIONS" / "Fitting" / "Standart_Fit"
    fixed_cmd = ["paranmr", "--hide", "fit_susc", "DyL1_1H_Fitting.yml"]
    fixed_result = subprocess.run(
        fixed_cmd,
        capture_output=True,
        text=True,
        cwd=fixed_cwd,
        env=_cli_env(tmp_path / "fixed"),
    )

    if fixed_result.returncode != 0 and "missing" in fixed_result.stderr.lower():
        pytest.xfail(f"Test failed due to missing dependency:\n{fixed_result.stderr}")

    assert fixed_result.returncode == 0, (
        f"Command failed with return code {fixed_result.returncode}\nstdout:\n"
        f"{fixed_result.stdout}\nstderr:\n{fixed_result.stderr}"
    )

    gmm_cwd = dyl1_root / "SIMULATIONS" / "Fitting" / "Moments" / "GMM"
    gmm_cmd = ["paranmr", "--hide", "fit_susc", "DyL1_1H_GMM_Fitting_moments_iso_ax_rho.yml"]
    gmm_result = subprocess.run(
        gmm_cmd,
        capture_output=True,
        text=True,
        cwd=gmm_cwd,
        env=_cli_env(tmp_path / "gmm"),
    )

    if gmm_result.returncode != 0 and "missing" in gmm_result.stderr.lower():
        pytest.xfail(f"Test failed due to missing dependency:\n{gmm_result.stderr}")

    assert gmm_result.returncode == 0, (
        f"Command failed with return code {gmm_result.returncode}\nstdout:\n"
        f"{gmm_result.stdout}\nstderr:\n{gmm_result.stderr}"
    )

    labels_csv = dyl1_root / "DATA" / "LABELS" / "DyL1_1H_Labels.csv"
    fixed_hyperfines_csv = (
        fixed_cwd / "DyL1_1H_Fitting" / "hyperfines_and_fitted_shifts_302.15_K.csv"
    )
    fixed_susc_csv = fixed_cwd / "DyL1_1H_Fitting" / "susceptibility_tensor.csv"
    gmm_hyperfines_csv = (
        gmm_cwd
        / "DyL1_1H_Fitting_Moments_iso_ax_rho"
        / "hyperfines_and_fitted_shifts_302.15_K.csv"
    )
    gmm_susc_csv = (
        gmm_cwd
        / "DyL1_1H_Fitting_Moments_iso_ax_rho"
        / "susceptibility_tensor.csv"
    )

    assert fixed_hyperfines_csv.exists(), (
        f"Expected output file missing: {fixed_hyperfines_csv}"
    )
    assert fixed_susc_csv.exists(), f"Expected output file missing: {fixed_susc_csv}"
    assert gmm_hyperfines_csv.exists(), (
        f"Expected output file missing: {gmm_hyperfines_csv}"
    )
    assert gmm_susc_csv.exists(), f"Expected output file missing: {gmm_susc_csv}"

    expected_partition = _reference_signal_partitions(labels_csv=labels_csv)
    reference_group_sizes = [
        len(group) for group in _reference_signal_groups(labels_csv=labels_csv)
    ]
    recovered_partition = _partition_protons_by_sorted_shift(
        hyperfines_csv=gmm_hyperfines_csv,
        group_sizes=reference_group_sizes,
    )

    assert recovered_partition == expected_partition

    fixed_centers = _group_centers_from_hyperfines(
        hyperfines_csv=fixed_hyperfines_csv,
        proton_groups=expected_partition,
    )
    gmm_centers = _group_centers_from_hyperfines(
        hyperfines_csv=gmm_hyperfines_csv,
        proton_groups=recovered_partition,
    )
    ppm_tolerance = _range_based_ppm_tolerance(centers_by_group=fixed_centers)

    for proton_group in expected_partition:
        assert abs(gmm_centers[proton_group] - fixed_centers[proton_group]) < ppm_tolerance

    fixed_susc = pd.read_csv(fixed_susc_csv, comment="#", encoding="utf-8-sig")
    gmm_susc = pd.read_csv(gmm_susc_csv, comment="#", encoding="utf-8-sig")
    fixed_chi_ax = float(fixed_susc["chi_ax (Å^3)"].iloc[0])
    gmm_chi_ax = float(gmm_susc["chi_ax (Å^3)"].iloc[0])

    assert np.isfinite(fixed_chi_ax)
    assert np.isfinite(gmm_chi_ax)
    assert abs(gmm_chi_ax - fixed_chi_ax) < 0.01


@pytest.mark.integration
def test_fit_susc_moments_gmm_recovers_ybl8_reference_proton_partition(
    tmp_path: Path,
):
    """Check that YbL8 GMM moment fitting recovers methyl-aware proton groups."""

    ybl8_root = tmp_path / "YbL8"
    shutil.copytree(Path("examples/YbL8"), ybl8_root)

    fixed_cwd = ybl8_root / "SIMULATIONS" / "Fitting" / "Standart_Fit"
    fixed_cmd = ["paranmr", "--hide", "fit_susc", "YbL8_PD_fit_standart.yml"]
    fixed_result = subprocess.run(
        fixed_cmd,
        capture_output=True,
        text=True,
        cwd=fixed_cwd,
        env=_cli_env(tmp_path / "fixed-ybl8"),
    )

    if fixed_result.returncode != 0 and "missing" in fixed_result.stderr.lower():
        pytest.xfail(f"Test failed due to missing dependency:\n{fixed_result.stderr}")

    assert fixed_result.returncode == 0, (
        f"Command failed with return code {fixed_result.returncode}\nstdout:\n"
        f"{fixed_result.stdout}\nstderr:\n{fixed_result.stderr}"
    )

    gmm_cwd = ybl8_root / "SIMULATIONS" / "Fitting" / "Moments" / "GMM"
    gmm_cmd = ["paranmr", "--hide", "fit_susc", "YbL8_PD_GMM_fit_momens.yml"]
    gmm_result = subprocess.run(
        gmm_cmd,
        capture_output=True,
        text=True,
        cwd=gmm_cwd,
        env=_cli_env(tmp_path / "gmm-ybl8"),
    )

    if gmm_result.returncode != 0 and "missing" in gmm_result.stderr.lower():
        pytest.xfail(f"Test failed due to missing dependency:\n{gmm_result.stderr}")

    assert gmm_result.returncode == 0, (
        f"Command failed with return code {gmm_result.returncode}\nstdout:\n"
        f"{gmm_result.stdout}\nstderr:\n{gmm_result.stderr}"
    )

    labels_csv = ybl8_root / "DATA" / "LABELS" / "YbL8_labels.csv"
    xyz_file = ybl8_root / "DATA" / "HFC" / "YbL8.xyz"
    fixed_hyperfines_csv = (
        fixed_cwd / "YbL8_1H_PD_fit_Standart" / "hyperfines_and_fitted_shifts_302.15_K.csv"
    )
    fixed_susc_csv = (
        fixed_cwd / "YbL8_1H_PD_fit_Standart" / "susceptibility_tensor.csv"
    )
    gmm_hyperfines_csv = (
        gmm_cwd
        / "YbL8_1H_PD_fit_Moments_norm_cov_fixed_L"
        / "hyperfines_and_fitted_shifts_302.15_K.csv"
    )
    gmm_susc_csv = (
        gmm_cwd
        / "YbL8_1H_PD_fit_Moments_norm_cov_fixed_L"
        / "susceptibility_tensor.csv"
    )

    assert fixed_hyperfines_csv.exists(), (
        f"Expected output file missing: {fixed_hyperfines_csv}"
    )
    assert fixed_susc_csv.exists(), f"Expected output file missing: {fixed_susc_csv}"
    assert gmm_hyperfines_csv.exists(), (
        f"Expected output file missing: {gmm_hyperfines_csv}"
    )
    assert gmm_susc_csv.exists(), f"Expected output file missing: {gmm_susc_csv}"

    expected_partition = _reference_signal_partitions_with_methyls(
        labels_csv=labels_csv,
        xyz_file=xyz_file,
    )
    expected_groups = _reference_signal_groups_with_methyls(
        labels_csv=labels_csv,
        xyz_file=xyz_file,
    )
    recovered_partition = _generated_signal_partitions(
        hyperfines_csv=gmm_hyperfines_csv,
    )

    assert recovered_partition == expected_partition

    fixed_centers = _group_centers_from_hyperfines(
        hyperfines_csv=fixed_hyperfines_csv,
        proton_groups=expected_groups,
    )
    gmm_centers = _group_centers_from_hyperfines(
        hyperfines_csv=gmm_hyperfines_csv,
        proton_groups=recovered_partition,
    )
    ppm_tolerance = _range_based_ppm_tolerance(centers_by_group=fixed_centers)

    for proton_group in expected_groups:
        assert abs(gmm_centers[proton_group] - fixed_centers[proton_group]) < ppm_tolerance

    fixed_susc = pd.read_csv(fixed_susc_csv, comment="#", encoding="utf-8-sig")
    gmm_susc = pd.read_csv(gmm_susc_csv, comment="#", encoding="utf-8-sig")
    fixed_chi_ax = float(fixed_susc["chi_ax (Å^3)"].iloc[0])
    gmm_chi_ax = float(gmm_susc["chi_ax (Å^3)"].iloc[0])

    assert np.isfinite(fixed_chi_ax)
    assert np.isfinite(gmm_chi_ax)
    assert abs(gmm_chi_ax - fixed_chi_ax) < 0.01
