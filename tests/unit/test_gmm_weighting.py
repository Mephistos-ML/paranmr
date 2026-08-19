# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

import numpy as np
import pytest
from paranmr.core.fitting.susceptibility.moments.descriptors import (
    compute_gaussian_mixture_moments,
)
from paranmr.core.fitting.susceptibility.objectives.moments.gmm.covariance import (
    MonteCarloMomentCovarianceConfig,
    estimate_moment_covariance_from_monte_carlo,
)
from paranmr.core.fitting.susceptibility.objectives.moments.gmm.weighting import (
    build_gmm_weighting_matrix,
)
from paranmr.io.csv.fit import save_moment_weighting_matrix


@pytest.mark.unit
def test_estimate_moment_covariance_from_monte_carlo_returns_symmetric_matrix():
    observed_peaks = {
        "center": np.asarray([-10.0, 5.0, 20.0], dtype=float),
        "sigma": np.asarray([1.0, 2.0, 1.5], dtype=float),
        "area_norm": np.asarray([0.2, 0.3, 0.5], dtype=float),
    }
    moment_names = ("m1", "m2", "m3", "m4", "m5", "m6")
    raw_experimental_moments = compute_gaussian_mixture_moments(
        centers=observed_peaks["center"],
        sigmas=observed_peaks["sigma"],
        area_norm=observed_peaks["area_norm"],
        moment_labels=moment_names,
    )
    estimate = estimate_moment_covariance_from_monte_carlo(
        observed_peaks=observed_peaks,
        raw_experimental_moments=raw_experimental_moments,
        moment_names=moment_names,
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
    assert np.any(np.diag(estimate.covariance) > 0.0)


@pytest.mark.unit
def test_build_gmm_weighting_matrix_inverts_covariance():
    covariance = np.diag([1.0, 4.0, 9.0, 16.0, 25.0, 36.0]).astype(float)

    weighting = build_gmm_weighting_matrix(covariance)

    assert weighting == pytest.approx(
        np.diag([1.0, 0.25, 1.0 / 9.0, 1.0 / 16.0, 1.0 / 25.0, 1.0 / 36.0])
    )


@pytest.mark.unit
def test_save_moment_weighting_matrix_writes_csv(tmp_path):
    output = tmp_path / "moment_weighting_matrix_302.15_K.csv"
    save_moment_weighting_matrix(
        weighting_matrix=np.asarray([[2.0, 0.5], [0.5, 1.0]], dtype=float),
        moment_names=("m1", "m2"),
        file_name=str(output),
        temperature=302.15,
        verbose=False,
    )

    content = output.read_text(encoding="utf-8-sig")
    assert "matrix = gmm_weighting" in content
    assert "quantity,m1,m2" in content
    assert "m1,2.000000e+00,5.000000e-01" in content
