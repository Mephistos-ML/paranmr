# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

import numpy as np
import pytest

from paranmr.core.fitting.susceptibility.objectives.moments.gmm.covariance import (
    MonteCarloMomentCovarianceConfig,
    estimate_moment_covariance_from_monte_carlo,
)
from paranmr.core.fitting.susceptibility.objectives.moments.gmm.weighting import (
    build_gmm_weighting_matrix,
)


@pytest.mark.unit
def test_estimate_moment_covariance_from_monte_carlo_returns_symmetric_matrix():
    estimate = estimate_moment_covariance_from_monte_carlo(
        observed_peaks={
            "center": np.asarray([-10.0, 5.0, 20.0], dtype=float),
            "sigma": np.asarray([1.0, 2.0, 1.5], dtype=float),
            "area_norm": np.asarray([0.2, 0.3, 0.5], dtype=float),
        },
        moment_names=("m1", "m2", "m3", "m4", "m5", "m6"),
        config=MonteCarloMomentCovarianceConfig(
            n_samples=200,
            shift_sigma_abs=0.02,
            width_sigma_rel=0.05,
            random_seed=12345,
        ),
    )

    assert estimate.covariance.shape == (6, 6)
    assert np.allclose(estimate.covariance, estimate.covariance.T)
    assert np.all(np.diag(estimate.covariance) >= 0.0)


@pytest.mark.unit
def test_build_gmm_weighting_matrix_inverts_covariance():
    covariance = np.diag([1.0, 4.0, 9.0, 16.0, 25.0, 36.0]).astype(float)

    weighting = build_gmm_weighting_matrix(covariance)

    assert weighting == pytest.approx(
        np.diag([1.0, 0.25, 1.0 / 9.0, 1.0 / 16.0, 1.0 / 25.0, 1.0 / 36.0])
    )
