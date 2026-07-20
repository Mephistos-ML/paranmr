# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Build covariance and weighting matrices for two-step GMM fits."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from paranmr.core.fitting.susceptibility.jacobian.types import (
    MOMENT_JACOBIAN_PARAMETER_NAMES,
    MomentJacobianResult,
)
from paranmr.core.fitting.susceptibility.moments.descriptors import MOMENT_NAMES


def normalize_moment_jacobian(
    *,
    jacobian: MomentJacobianResult,
    observed_moments: dict[str, float],
) -> MomentJacobianResult:
    """Return the Jacobian of normalized calculated moments.

    With ``m_n^norm = m_n^calc / m_n^exp``, each Jacobian row is scaled by the
    corresponding observed raw moment.
    """

    missing = [name for name in MOMENT_NAMES if name not in observed_moments]
    if missing:
        raise ValueError(
            "Cannot normalize moment Jacobian without observed moments for: "
            + ", ".join(missing)
        )

    scales = np.asarray(
        [float(observed_moments[name]) for name in MOMENT_NAMES],
        dtype=float,
    )
    zero_like = [
        name
        for name, scale in zip(MOMENT_NAMES, scales)
        if np.isclose(scale, 0.0, atol=1e-12, rtol=0.0)
    ]
    if zero_like:
        raise ValueError(
            "Cannot normalize moment Jacobian by observed moment values "
            "that are zero or too close to zero: "
            + ", ".join(zero_like)
        )

    normalized_values = np.asarray(jacobian.values, dtype=float) / scales[:, None]
    return MomentJacobianResult(
        temperature=float(jacobian.temperature),
        moment_names=MOMENT_NAMES,
        parameter_names=MOMENT_JACOBIAN_PARAMETER_NAMES,
        values=normalized_values,
    )


def estimate_gmm_covariance_from_jacobian(
    jacobian: MomentJacobianResult,
    *,
    ridge_factor: float = 1.0e-8,
) -> NDArray[np.float64]:
    """Estimate a regularized moment covariance matrix from the Jacobian.

    The first working approximation uses ``J J^T`` evaluated at the stage-1
    solution and adds a small isotropic ridge term to guarantee invertibility.
    """

    values = np.asarray(jacobian.values, dtype=float)
    covariance = values @ values.T
    covariance = 0.5 * (covariance + covariance.T)

    n_moments = covariance.shape[0]
    average_scale = float(np.trace(covariance) / n_moments) if n_moments else 1.0
    if not np.isfinite(average_scale) or average_scale <= 0.0:
        average_scale = 1.0
    ridge = ridge_factor * average_scale
    covariance = covariance + ridge * np.eye(n_moments, dtype=float)
    return covariance


def build_gmm_weighting_matrix(
    covariance: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the inverse weighting matrix ``W = S^{-1}`` for GMM."""

    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("GMM covariance matrix must be square")
    if not np.allclose(covariance, covariance.T):
        raise ValueError("GMM covariance matrix must be symmetric")

    weighting = np.linalg.inv(covariance)
    return 0.5 * (weighting + weighting.T)
