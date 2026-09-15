# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

import numpy as np
import pytest

from paranmr.core.fitting.susceptibility.models.isoaxrho_euler import (
    IsoAxRhoEulerFitter,
)
from paranmr.core.fitting.susceptibility.parameterization.traceless import (
    cartesian_parameters_from_euler,
    euler_parameters_from_cartesian,
    tensor_from_cartesian_parameters,
)


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
def test_cartesian_parameters_reconstruct_current_euler_tensor(parameters):
    isotropic, coordinates = cartesian_parameters_from_euler(parameters)

    tensor = tensor_from_cartesian_parameters(
        isotropic=isotropic,
        coordinates=coordinates,
    )

    assert tensor == pytest.approx(IsoAxRhoEulerFitter.totensor(parameters))


@pytest.mark.unit
def test_cartesian_tensor_is_symmetric_and_has_requested_trace():
    tensor = tensor_from_cartesian_parameters(
        isotropic=0.17,
        coordinates=np.asarray([0.21, -0.08, 0.04, -0.13, 0.07]),
    )

    assert tensor == pytest.approx(tensor.T)
    assert np.trace(tensor) == pytest.approx(3.0 * 0.17)


@pytest.mark.unit
def test_cartesian_tensor_rejects_nonfive_coordinate_vector():
    with pytest.raises(ValueError, match=r"shape \(5,\)"):
        tensor_from_cartesian_parameters(
            isotropic=0.0,
            coordinates=np.zeros(6),
        )


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
def test_euler_parameters_from_cartesian_preserve_tensor(parameters):
    isotropic, coordinates = cartesian_parameters_from_euler(parameters)

    reported = euler_parameters_from_cartesian(
        isotropic=isotropic,
        coordinates=coordinates,
    )

    assert reported["rho_over_ax"] >= 0.0
    assert reported["rho_over_ax"] <= 1.0 / 3.0 + 1.0e-12
    assert IsoAxRhoEulerFitter.totensor(reported) == pytest.approx(
        IsoAxRhoEulerFitter.totensor(parameters)
    )


@pytest.mark.unit
def test_euler_parameters_from_cartesian_reports_zero_gamma_for_axial_tensor():
    isotropic, coordinates = cartesian_parameters_from_euler(
        {
            "iso": 0.11,
            "ax": 0.32,
            "rho_over_ax": 0.0,
            "alpha": 31.0,
            "beta": 74.0,
            "gamma": 192.0,
        }
    )

    reported = euler_parameters_from_cartesian(
        isotropic=isotropic,
        coordinates=coordinates,
    )

    assert reported["rho_over_ax"] == pytest.approx(0.0)
    assert reported["gamma"] == pytest.approx(0.0)


@pytest.mark.unit
def test_euler_parameters_from_cartesian_reports_zero_angles_for_isotropic_tensor():
    reported = euler_parameters_from_cartesian(
        isotropic=0.11,
        coordinates=np.zeros(5),
    )

    assert reported == pytest.approx(
        {
            "iso": 0.11,
            "ax": 0.0,
            "rho_over_ax": 0.0,
            "alpha": 0.0,
            "beta": 0.0,
            "gamma": 0.0,
        }
    )
