# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

import numpy as np
import pytest

from simpnmr_x.core.domain.tensor import Susceptibility
from simpnmr_x.core.fitting.susceptibility.models.isoaxrho_euler import (
    IsoAxRhoEulerFitter,
)
from simpnmr_x.core.fitting.susceptibility.models.split import SplitFitter


def _split_parameters_from_tensor(tensor: np.ndarray) -> dict[str, float]:
    iso = float(np.trace(tensor) / 3.0)
    deviatoric = tensor - iso * np.eye(3)
    return {
        "iso": iso,
        "dxx": float(deviatoric[0, 0]),
        "dyy": float(deviatoric[1, 1]),
        "dxy": float(deviatoric[0, 1]),
        "dxz": float(deviatoric[0, 2]),
        "dyz": float(deviatoric[1, 2]),
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "iso": 0.11,
            "ax": 0.32,
            "rho_over_ax": 0.10,
            "alpha": 0.0,
            "beta": 0.0,
            "gamma": 0.0,
        },
        {
            "iso": -0.08,
            "ax": -0.24,
            "rho_over_ax": 0.22,
            "alpha": 31.0,
            "beta": 74.0,
            "gamma": 192.0,
        },
    ],
)
def test_split_parameters_reconstruct_euler_tensor(parameters):
    tensor = IsoAxRhoEulerFitter.totensor(parameters)

    assert SplitFitter.totensor(_split_parameters_from_tensor(tensor)) == pytest.approx(
        tensor
    )


@pytest.mark.unit
def test_split_tensor_is_symmetric_and_has_requested_trace():
    tensor = SplitFitter.totensor(
        {"iso": 0.17, "dxx": 0.25, "dyy": -0.17, "dxy": 0.04, "dxz": -0.13, "dyz": 0.07}
    )

    assert tensor == pytest.approx(tensor.T)
    assert np.trace(tensor) == pytest.approx(3.0 * 0.17)


@pytest.mark.unit
@pytest.mark.unit
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "iso": 0.11,
            "ax": 0.32,
            "rho_over_ax": 0.10,
            "alpha": 31.0,
            "beta": 74.0,
            "gamma": 192.0,
        },
        {
            "iso": -0.08,
            "ax": -0.24,
            "rho_over_ax": 0.22,
            "alpha": 51.0,
            "beta": 48.0,
            "gamma": 17.0,
        },
    ],
)
def test_susceptibility_euler_decomposition_preserves_split_tensor(parameters):
    tensor = IsoAxRhoEulerFitter.totensor(parameters)
    split_tensor = SplitFitter.totensor(_split_parameters_from_tensor(tensor))
    decomposition = Susceptibility(split_tensor, 300.0).decomposition
    reported = {
        "iso": decomposition.iso,
        "ax": decomposition.axiality,
        "rho_over_ax": decomposition.rhombicity / decomposition.axiality,
        "alpha": decomposition.alpha,
        "beta": decomposition.beta,
        "gamma": decomposition.gamma,
    }

    assert reported["rho_over_ax"] >= 0.0
    assert reported["rho_over_ax"] <= 1.0 / 3.0 + 1.0e-12
    assert IsoAxRhoEulerFitter.totensor(reported) == pytest.approx(split_tensor)


@pytest.mark.unit
def test_susceptibility_reports_zero_gamma_for_axial_tensor():
    tensor = IsoAxRhoEulerFitter.totensor(
        {
            "iso": 0.11,
            "ax": 0.32,
            "rho_over_ax": 0.0,
            "alpha": 31.0,
            "beta": 74.0,
            "gamma": 192.0,
        }
    )
    decomposition = Susceptibility(tensor, 300.0).decomposition

    assert decomposition.rhombicity == pytest.approx(0.0)
    assert decomposition.gamma == pytest.approx(0.0)


@pytest.mark.unit
def test_susceptibility_reports_zero_angles_for_isotropic_tensor():
    decomposition = Susceptibility(np.eye(3) * 0.11, 300.0).decomposition

    assert decomposition.__dict__ == pytest.approx(
        {
            "iso": 0.11,
            "axiality": 0.0,
            "rhombicity": 0.0,
            "alpha": 0.0,
            "beta": 0.0,
            "gamma": 0.0,
        }
    )
