# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""End-to-end GMM validation against a committed seeded YbL8 fixture."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from tests.helpers.cli import run_paranmr
from tests.helpers.gmm import (
    assert_gmm_fit_config,
    assert_gmm_recovers_synthetic_truth,
)

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
    config["signal_labels"]["file"] = str(_YBL8_DATA / "LABELS" / "YbL8_labels.csv")
    config["experiment"]["files"] = str(_GMM_FIXTURE / "generated_shifts.csv")
    config_path = tmp_path / "gmm_config.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


@pytest.mark.integration
def test_gmm_recovers_seeded_synthetic_ybl8_shifts(tmp_path: Path) -> None:
    """Recover seeded YbL8 shifts and R6 linewidth parameters."""
    gmm_config_path = _materialize_gmm_config(tmp_path)
    gmm_config = yaml.safe_load(gmm_config_path.read_text(encoding="utf-8"))
    assert_gmm_fit_config(gmm_config)
    assert gmm_config["linewidth"]["variables"] == {
        "p1": ["fit", 1.0, [0.0, 1000000.0]],
        "p2": ["fit", 0.01, [0.001, 10.0]],
    }

    result = run_paranmr(
        ["--hide", "fit_susc", gmm_config_path.name],
        cwd=gmm_config_path.parent,
        env=_cli_env(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    output = tmp_path / "paranmr_gmm_fitted_output"
    assert_gmm_recovers_synthetic_truth(
        output_dir=output,
        generated_shifts_file=_GMM_FIXTURE / "generated_shifts.csv",
        truth_file=_GMM_FIXTURE / "truth.json",
    )
