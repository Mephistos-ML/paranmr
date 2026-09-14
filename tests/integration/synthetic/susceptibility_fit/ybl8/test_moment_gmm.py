# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""End-to-end GMM validation against seeded ParaNMR-Synth YbL8 cases.

Install the optional generator dependency with ``pip install '.[synthetic]'``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.helpers.cli import run_paranmr


_YBL8_DATA = Path(__file__).resolve().parents[5] / "tests" / "data" / "YbL8" / "DATA"


def _cli_env(tmp_path: Path) -> dict[str, str]:
    return {
        **os.environ,
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
        "XDG_CACHE_HOME": str(tmp_path / "xdg-cache"),
    }


def _generation_config(seed: int, config_type: type[Any]) -> Any:
    """Build the seeded YbL8 generator contract from canonical raw inputs."""
    return config_type.from_mapping(
        {
            "project": {"name": "synthetic_ybl8_gmm", "n_cases": 1, "seed": seed},
            "hyperfine": {
                "method": "pdip",
                "file": str(_YBL8_DATA / "HFC" / "YbL8.xyz"),
                "paramagnetic_centre": [0.0, 0.0, 0.0],
                "spin": 0.5,
                "orbit": 3.0,
                "total_momentum_J": 3.5,
            },
            "nuclei": {"include": "H"},
            "diamagnetic": {
                "method": "dft",
                "file": str(_YBL8_DATA / "DIA" / "LuL8_DIA_NMR.out"),
            },
            "diamagnetic_ref": {
                "method": "dft",
                "file": str(_YBL8_DATA / "DIA" / "tms_ref.out"),
            },
            "experiment": {"temperature_k": 302.15, "magnetic_field_t": 4.7},
            "moments": {"number_of_moments": 10},
            "linewidth": {"method": "r6"},
            "susceptibility": {"model": "isoaxrho_euler"},
        }
    )


def _find_generated_gmm_config(case_dir: Path) -> Path:
    """Locate the runnable GMM YAML emitted by ParaNMR-Synth.

    The generator owns the output directory layout and may change directory or
    file naming without changing the generated YAML contract.  Select the
    configuration by its contents rather than coupling the test to a path.
    """
    candidates = sorted(
        path
        for pattern in ("*.yml", "*.yaml")
        for path in case_dir.rglob(pattern)
    )
    matches: list[Path] = []
    for path in candidates:
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(document, dict):
            continue
        assignment = document.get("assignment")
        objective = (
            assignment.get("moment_objective")
            if isinstance(assignment, dict)
            else None
        )
        if (
            isinstance(assignment, dict)
            and assignment.get("method") == "moments"
            and isinstance(objective, dict)
            and objective.get("type") == "gmm"
        ):
            matches.append(path)

    if len(matches) != 1:
        raise AssertionError(
            "ParaNMR-Synth must emit exactly one GMM fitting YAML; found: "
            + ", ".join(str(path.relative_to(case_dir)) for path in matches)
        )
    return matches[0]


@pytest.mark.integration
@pytest.mark.parametrize("seed", [20260912])
def test_gmm_recovers_seeded_synthetic_ybl8_shifts(
    tmp_path: Path, seed: int
) -> None:
    """Fit all χ/R6 variables to a seeded, unlabeled YbL8 synthetic spectrum."""
    try:
        from paranmr_synth.app.pipelines.dataset_export import generate_dataset
        from paranmr_synth.cfg.dataset import DatasetGenerationConfig
    except ModuleNotFoundError:
        pytest.fail(
            "Synthetic GMM anchor requires ParaNMR-Synth; install with "
            "`pip install '.[synthetic]'`."
        )

    root = generate_dataset(
        config=_generation_config(seed, DatasetGenerationConfig),
        output_dir=tmp_path / "synthetic_data",
    )
    case_dir = next((root / "cases").iterdir())
    gmm_config_path = _find_generated_gmm_config(case_dir)
    gmm_config = yaml.safe_load(gmm_config_path.read_text(encoding="utf-8"))
    assert gmm_config["assignment"]["method"] == "moments"
    assert gmm_config["assignment"]["moment_objective"]["type"] == "gmm"
    assert all(
        value[0] == "fit" for value in gmm_config["susc_fit"]["variables"].values()
    )
    assert all(
        value[0] == "fit"
        for value in gmm_config["linewidth"]["variables"].values()
    )
    generated_peaks = pd.read_csv(
        case_dir / "DATA" / "PARA" / "generated_shifts.csv",
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

    output = gmm_config_path.parent / "paranmr_gmm_fitted_output"
    peak_data = pd.read_csv(
        output / "peak_data_302.15_K.csv", comment="#", encoding="utf-8-sig"
    )
    recovered_centers = np.sort(peak_data["δ_total_avg (ppm)"].to_numpy(dtype=float))
    assert recovered_centers == pytest.approx(expected_centers, abs=2e-3)
