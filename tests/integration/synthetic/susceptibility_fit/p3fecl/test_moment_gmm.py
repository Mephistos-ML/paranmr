# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""End-to-end GMM recovery for a label-averaged DFT-HFC P3FeCl fixture."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

from tests.helpers.cli import run_simpnmr_x

_P3FECL_ROOT = Path(__file__).resolve().parents[5] / "tests" / "data" / "P3FeCl"
_P3FECL_DATA = _P3FECL_ROOT / "DATA"
_GMM_FIXTURE = _P3FECL_ROOT / "SYNTHETIC" / "GMM"


def _cli_env(tmp_path: Path) -> dict[str, str]:
    return {
        **os.environ,
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
        "XDG_CACHE_HOME": str(tmp_path / "xdg-cache"),
    }


def _materialize_gmm_config(tmp_path: Path) -> Path:
    """Create a runnable copy of the committed P3FeCl fixture."""
    config = yaml.safe_load(
        (_GMM_FIXTURE / "gmm_config.yml").read_text(encoding="utf-8")
    )
    config["project"]["name"] = str(tmp_path / "p3fecl_gmm_fitted_output")
    config["hyperfine"]["file"] = str(_P3FECL_DATA / "HFC" / "P3FeCl_HFC.out")
    config["signal_labels"]["file"] = str(
        _P3FECL_DATA / "LABELS" / "P3FeCl_Signal_Labels_13C.csv"
    )
    config["experiment"]["files"] = str(_GMM_FIXTURE / "generated_shifts.csv")
    config_path = tmp_path / "gmm_config.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


@pytest.mark.integration
def test_gmm_recovers_label_averaged_synthetic_p3fecl_tensor(tmp_path: Path) -> None:
    """Recover P3FeCl anisotropy with known isotropic and linewidth channels."""
    config_path = _materialize_gmm_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["assignment"]["max_moment_order"] == 10
    assert config["susc_fit"]["average_shifts"] == "all"
    assert config["susc_fit"]["variables"]["iso"] == ["fix", 0.243408]
    assert all(
        value == ["fit", 0.0]
        for name, value in config["susc_fit"]["variables"].items()
        if name != "iso"
    )

    result = run_simpnmr_x(
        ["--hide", "fit_susc", config_path.name],
        cwd=config_path.parent,
        env=_cli_env(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    truth = json.loads((_GMM_FIXTURE / "truth.json").read_text(encoding="utf-8"))
    recovered = pd.read_csv(
        tmp_path / "p3fecl_gmm_fitted_output" / "susceptibility_tensor.csv",
        comment="#",
        encoding="utf-8-sig",
    ).iloc[0]
    columns = {
        "dxx": "dchi_xx (Å^3)",
        "dyy": "dchi_yy (Å^3)",
        "dxy": "dchi_xy (Å^3)",
        "dxz": "dchi_xz (Å^3)",
        "dyz": "dchi_yz (Å^3)",
    }
    for parameter, column in columns.items():
        assert recovered[column] == pytest.approx(truth[parameter], abs=1e-8)
