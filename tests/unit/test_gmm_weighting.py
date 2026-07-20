# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

import numpy as np
import pytest

from paranmr.core.fitting.susceptibility.jacobian.types import (
    MOMENT_JACOBIAN_PARAMETER_NAMES,
    MomentJacobianResult,
)
from paranmr.core.fitting.susceptibility.moments.descriptors import MOMENT_NAMES
from paranmr.core.fitting.susceptibility.objectives.moments.gmm.weighting import (
    build_gmm_weighting_matrix,
    estimate_gmm_covariance_from_jacobian,
    normalize_moment_jacobian,
)


@pytest.mark.unit
def test_estimate_gmm_covariance_from_jacobian_returns_regularized_symmetric_matrix():
    jacobian = MomentJacobianResult(
        temperature=302.15,
        moment_names=MOMENT_NAMES,
        parameter_names=MOMENT_JACOBIAN_PARAMETER_NAMES,
        values=np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
    )

    covariance = estimate_gmm_covariance_from_jacobian(jacobian)

    assert covariance.shape == (6, 6)
    assert np.allclose(covariance, covariance.T)
    assert np.all(np.diag(covariance) > 0.0)


@pytest.mark.unit
def test_normalize_moment_jacobian_scales_rows_by_observed_moments():
    jacobian = MomentJacobianResult(
        temperature=302.15,
        moment_names=MOMENT_NAMES,
        parameter_names=MOMENT_JACOBIAN_PARAMETER_NAMES,
        values=np.asarray(
            [
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 18.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 32.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 72.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
    )

    normalized = normalize_moment_jacobian(
        jacobian=jacobian,
        observed_moments={
            "m1": 2.0,
            "m2": 4.0,
            "m3": 6.0,
            "m4": 8.0,
            "m5": 10.0,
            "m6": 12.0,
        },
    )

    expected = np.zeros((6, 8), dtype=float)
    expected[0, 0] = 1.0
    expected[1, 1] = 2.0
    expected[2, 2] = 3.0
    expected[3, 3] = 4.0
    expected[4, 4] = 5.0
    expected[5, 5] = 6.0
    assert normalized.values == pytest.approx(expected)


@pytest.mark.unit
def test_build_gmm_weighting_matrix_inverts_covariance():
    covariance = np.diag([1.0, 4.0, 9.0, 16.0, 25.0, 36.0]).astype(float)

    weighting = build_gmm_weighting_matrix(covariance)

    assert weighting == pytest.approx(
        np.diag([1.0, 0.25, 1.0 / 9.0, 1.0 / 16.0, 1.0 / 25.0, 1.0 / 36.0])
    )
