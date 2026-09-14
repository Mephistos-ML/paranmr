# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""End-to-end GMM validation against a committed seeded YbL8 fixture."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.helpers.cli import run_paranmr

_YBL8_ROOT = Path(__file__).resolve().parents[5] / "tests" / "data" / "YbL8"
_YBL8_DATA = _YBL8_ROOT / "DATA"
_GMM_FIXTURE = _YBL8_ROOT / "SYNTHETIC" / "GMM"


def _cli_env(tmp_path: Path) -> dict[str, str]:
    return {
        **os.environ,
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
        "XDG_CACHE_HOME": str(tmp_path / "xdg-cache"),
    }


def _materialize_gmm_config(tmp_path: Path) -> Path:
    """Create a runnable copy of the committed GMM fixture."""
    config = yaml.safe_load(
        (_GMM_FIXTURE / "gmm_config.yml").read_text(encoding="utf-8")
    )
    config["project"]["name"] = str(tmp_path / "paranmr_gmm_fitted_output")
    config["hyperfine"]["file"] = str(_YBL8_DATA / "HFC" / "YbL8.xyz")
    config["diamagnetic"]["file"] = str(_YBL8_DATA / "DIA" / "LuL8_DIA_NMR.out")
    config["diamagnetic_ref"]["file"] = str(_YBL8_DATA / "DIA" / "tms_ref.out")
    config["experiment"]["files"] = str(_GMM_FIXTURE / "generated_shifts.csv")
    config_path = tmp_path / "gmm_config.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


@pytest.mark.integration
def test_gmm_recovers_seeded_synthetic_ybl8_shifts(tmp_path: Path) -> None:
    """Fit all χ/R6 variables to the committed seeded YbL8 fixture."""
    gmm_config_path = _materialize_gmm_config(tmp_path)
    gmm_config = yaml.safe_load(gmm_config_path.read_text(encoding="utf-8"))
    assert gmm_config["assignment"]["method"] == "moments"
    assert gmm_config["assignment"]["moment_objective"]["type"] == "gmm"
    assert all(
        value[0] == "fit" for value in gmm_config["susc_fit"]["variables"].values()
    )
    assert all(
        value[0] == "fit" for value in gmm_config["linewidth"]["variables"].values()
    )
    generated_peaks = pd.read_csv(
        _GMM_FIXTURE / "generated_shifts.csv",
        comment="#",
        encoding="utf-8-sig",
    )
    expected_centers = np.sort(generated_peaks["shift (ppm)"].to_numpy(dtype=float))

    result = run_paranmr(
        ["--hide", "fit_susc", gmm_config_path.name],
        cwd=gmm_config_path.parent,
        env=_cli_env(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    output = tmp_path / "paranmr_gmm_fitted_output"
    peak_data = pd.read_csv(
        output / "peak_data_302.15_K.csv", comment="#", encoding="utf-8-sig"
    )
    recovered_centers = np.sort(peak_data["δ_total_avg (ppm)"].to_numpy(dtype=float))
    assert recovered_centers == pytest.approx(expected_centers, abs=2e-3)
