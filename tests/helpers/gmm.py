"""Shared assertions for synthetic GMM recovery tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

_TENSOR_COLUMNS = {
    "dxx": "dchi_xx (Å^3)",
    "dyy": "dchi_yy (Å^3)",
    "dxy": "dchi_xy (Å^3)",
    "dxz": "dchi_xz (Å^3)",
    "dyz": "dchi_yz (Å^3)",
    "dzz": "dchi_zz (Å^3)",
}
_REQUIRED_TRUTH_FIELDS = {"iso", *_TENSOR_COLUMNS, "p1", "p2"}


def assert_gmm_fit_config(config: dict[str, Any]) -> None:
    """Require both fixtures to exercise the same fitted GMM parameters."""
    assert config["assignment"]["method"] == "moments"
    assert config["assignment"]["max_moment_order"] == 10
    assert config["susc_fit"]["average_shifts"] == "all"
    variables = config["susc_fit"]["variables"]
    assert set(variables) == {"iso", "dxx", "dyy", "dxy", "dxz", "dyz"}
    assert variables["iso"] == ["fix", 0.0]
    assert all(
        value == ["fit", 0.0] for name, value in variables.items() if name != "iso"
    )
    assert config["linewidth"]["method"] == "r6"
    assert set(config["linewidth"]["variables"]) == {"p1", "p2"}
    assert all(value[0] == "fit" for value in config["linewidth"]["variables"].values())


def _read_gmm_truth(truth_file: Path) -> dict[str, Any]:
    """Read the common JSON truth schema used by every synthetic GMM fixture."""
    truth = json.loads(truth_file.read_text(encoding="utf-8"))
    assert _REQUIRED_TRUTH_FIELDS.issubset(truth)
    assert np.isfinite([float(truth[name]) for name in _REQUIRED_TRUTH_FIELDS]).all()
    return truth


def assert_gmm_recovers_synthetic_truth(
    *,
    output_dir: Path,
    generated_shifts_file: Path,
    truth_file: Path,
    temperature_k: float = 302.15,
) -> None:
    """Compare recovered peak centers, the full Δχ tensor, and R6 truth."""
    truth = _read_gmm_truth(truth_file)
    assert float(truth["iso"]) == pytest.approx(0.0, abs=1e-12)
    temperature = f"{temperature_k:.2f}_K"

    generated = pd.read_csv(generated_shifts_file, comment="#", encoding="utf-8-sig")
    expected_shifts = np.sort(generated["shift (ppm)"].to_numpy(dtype=float))
    peak_data = pd.read_csv(
        output_dir / f"peak_data_{temperature}.csv",
        comment="#",
        encoding="utf-8-sig",
    )
    recovered_shifts = np.sort(peak_data["δ_total_avg (ppm)"].to_numpy(dtype=float))
    assert recovered_shifts.size == expected_shifts.size
    assert np.isfinite(recovered_shifts).all()
    assert recovered_shifts == pytest.approx(expected_shifts, abs=1e-6)

    tensor = pd.read_csv(
        output_dir / "susceptibility_tensor.csv",
        comment="#",
        encoding="utf-8-sig",
    )
    assert tensor.shape[0] == 1
    recovered_tensor = tensor.iloc[0]
    for parameter, column in _TENSOR_COLUMNS.items():
        assert recovered_tensor[column] == pytest.approx(truth[parameter], abs=2e-6)

    linewidth = pd.read_csv(
        output_dir / f"linewidth_model_{temperature}.csv",
        comment="#",
        encoding="utf-8-sig",
    ).iloc[0]
    assert linewidth["p1"] == pytest.approx(truth["p1"], abs=0.05)
    assert linewidth["p2"] == pytest.approx(truth["p2"], abs=5e-5)
