# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Tests for susceptibility tensor Euler angle conventions."""

import numpy as np

from paranmr.core.domain.tensor import Susceptibility
from paranmr.core.fitting.susceptibility.models.isoaxrho_euler import (
    IsoAxRhoEulerFitter,
)


def test_susceptibility_euler_angles_round_trip_isoaxrho_euler_tensor():
    """CSV-style Euler angles should be reusable by the Euler fitter."""

    parameters = {
        "iso": 0.2,
        "ax": 1.3,
        "rho_over_ax": 0.12,
        "alpha": 90.0,
        "beta": 30.0,
        "gamma": 40.0,
    }

    tensor = IsoAxRhoEulerFitter.totensor(parameters)
    susceptibility = Susceptibility(tensor)
    susceptibility.iso = parameters["iso"]

    recovered = {
        "iso": float(susceptibility.iso),
        "ax": float(susceptibility.axiality),
        "rho_over_ax": float(
            susceptibility.rhombicity / susceptibility.axiality
        ),
        "alpha": float(susceptibility.alpha),
        "beta": float(susceptibility.beta),
        "gamma": float(susceptibility.gamma),
    }

    recovered_tensor = IsoAxRhoEulerFitter.totensor(recovered)

    np.testing.assert_allclose(recovered_tensor, tensor, atol=1.0e-12)


def test_susceptibility_euler_angles_round_trip_for_sign_reflected_axiality():
    """Euler extraction should also round-trip the opposite axiality branch."""

    parameters = {
        "iso": 0.0,
        "ax": -0.8,
        "rho_over_ax": 0.1,
        "alpha": 180.0,
        "beta": 90.0,
        "gamma": 0.0,
    }

    tensor = IsoAxRhoEulerFitter.totensor(parameters)
    susceptibility = Susceptibility(tensor)
    susceptibility.iso = parameters["iso"]

    recovered = {
        "iso": float(susceptibility.iso),
        "ax": float(susceptibility.axiality),
        "rho_over_ax": float(
            susceptibility.rhombicity / susceptibility.axiality
        ),
        "alpha": float(susceptibility.alpha),
        "beta": float(susceptibility.beta),
        "gamma": float(susceptibility.gamma),
    }

    recovered_tensor = IsoAxRhoEulerFitter.totensor(recovered)

    np.testing.assert_allclose(recovered_tensor, tensor, atol=1.0e-12)
