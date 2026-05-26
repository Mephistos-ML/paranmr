# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Integration tests for hyperfine benchmark workflows."""

import os
import subprocess
from pathlib import Path

import pytest

from paranmr.io.csv.csv_util import read_csv_safe

DATA_DIR = Path("tests/data/sources/hfc/qc/orca/version_6")
HFC_FILE = DATA_DIR / "P3FeCl_HFC.out"
CHEM_LABELS_FILE = DATA_DIR / "P3FeCl_Chemical_Labels_13C.csv"


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


def _write_benchmark_input(tmp_path: Path, *, project_dir: Path) -> Path:
    input_file = tmp_path / "benchmark.yml"
    input_file.write_text(
        "\n".join(
            [
                "project:",
                f"  name: {project_dir}",
                "",
                "hyperfine:",
                "  - functional: B3LYP",
                f"    file: {HFC_FILE.resolve()}",
                "",
                "chem_labels:",
                f"  file: {CHEM_LABELS_FILE.resolve()}",
                "",
                "nuclei:",
                "  include: [C]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return input_file


@pytest.mark.integration
def test_benchmark_a_fc_with_orca6_hfc(tmp_path: Path):
    """Run the A_fc benchmark workflow with ORCA 6 hyperfine data."""
    project_dir = tmp_path / "A_FC_Benchmark"
    input_file = _write_benchmark_input(tmp_path, project_dir=project_dir)

    result = subprocess.run(
        ["paranmr", "--hide", "benchmark", "a_fc", str(input_file)],
        capture_output=True,
        text=True,
        env=_cli_env(tmp_path),
    )

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\nstdout:\n"
        f"{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert (project_dir / "B3LYP_C_A_FC_benchmark_spread.pdf").exists()
    assert (project_dir / "C_A_FC_benchmark_max_curve.pdf").exists()

    report = read_csv_safe(project_dir / "A_FC_benchmark_max.csv")
    assert list(report.columns) == [
        "chem_label",
        "nucleus",
        "functional",
        "max (ppm A-3)",
        "min (ppm A-3)",
        "range",
    ]
    assert len(report) == 1
    row = report.iloc[0]
    assert row["nucleus"] == "C"
    assert row["functional"] == "B3LYP"
    assert row["range"] == pytest.approx(
        (row["max (ppm A-3)"] - row["min (ppm A-3)"]) / row["max (ppm A-3)"]
    )


@pytest.mark.integration
def test_benchmark_a_sd_with_orca6_hfc(tmp_path: Path):
    """Run the A_sd benchmark workflow with ORCA 6 hyperfine data."""
    project_dir = tmp_path / "A_SD_Benchmark"
    input_file = _write_benchmark_input(tmp_path, project_dir=project_dir)

    result = subprocess.run(
        ["paranmr", "--hide", "benchmark", "a_sd", str(input_file)],
        capture_output=True,
        text=True,
        env=_cli_env(tmp_path),
    )

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\nstdout:\n"
        f"{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert (project_dir / "B3LYP_C_A_SD_benchmark_spread.pdf").exists()
    assert (project_dir / "C_A_SD_benchmark_max_curve.pdf").exists()
