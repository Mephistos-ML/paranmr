# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Build weighting matrices for covariance-driven GMM fits."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def build_gmm_weighting_matrix(
    covariance: NDArray[np.float64],
    *,
    ridge_factor: float = 1.0e-8,
) -> NDArray[np.float64]:
    """Return the inverse weighting matrix ``W = S^{-1}`` for GMM."""

    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("GMM covariance matrix must be square")
    if not np.allclose(covariance, covariance.T):
        raise ValueError("GMM covariance matrix must be symmetric")

    n_moments = covariance.shape[0]
    average_scale = float(np.trace(covariance) / n_moments) if n_moments else 1.0
    if not np.isfinite(average_scale) or average_scale <= 0.0:
        average_scale = 1.0
    ridge = ridge_factor * average_scale
    covariance = covariance + ridge * np.eye(n_moments, dtype=float)

    weighting = np.linalg.inv(covariance)
    return 0.5 * (weighting + weighting.T)
