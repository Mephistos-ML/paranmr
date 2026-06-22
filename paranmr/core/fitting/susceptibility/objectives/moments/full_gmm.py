# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Full-covariance GMM moment objective transform."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from paranmr.core.fitting.susceptibility.objectives.moments.state import (
    config_float,
)


def full_gmm_transform(
    *,
    moment_names: tuple[str, ...],
    covariance: NDArray[np.float64],
    variances: NDArray[np.float64],
    variance_floor: float,
    objective_config: dict,
) -> tuple[NDArray[np.float64], dict]:
    """Build a full-covariance bootstrap GMM residual transform."""
    shrinkage = config_float(
        objective_config,
        "covariance_regularization",
        config_float(objective_config, "shrinkage", 0.1),
    )
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("covariance_regularization must be between 0 and 1")

    covariance = np.asarray(covariance, dtype=float)
    safe_variances = np.maximum(np.asarray(variances, dtype=float), variance_floor)
    diagonal_covariance = np.diag(safe_variances)
    floored_covariance = covariance.copy()
    np.fill_diagonal(floored_covariance, safe_variances)
    regularized_covariance = (
        (1.0 - shrinkage) * floored_covariance + shrinkage * diagonal_covariance
    )
    regularized_covariance = 0.5 * (
        regularized_covariance + regularized_covariance.T
    )

    eigenvalues, eigenvectors = np.linalg.eigh(regularized_covariance)
    safe_eigenvalues = np.maximum(eigenvalues, variance_floor)
    transform = np.diag(1.0 / np.sqrt(safe_eigenvalues)) @ eigenvectors.T
    precision = eigenvectors @ np.diag(1.0 / safe_eigenvalues) @ eigenvectors.T
    effective_weights = np.sqrt(np.maximum(np.diag(precision), 0.0))

    diagnostics = {
        "weights": {
            name: float(value)
            for name, value in zip(moment_names, effective_weights)
        },
        "variances": {
            name: float(value) for name, value in zip(moment_names, variances)
        },
        "covariance": covariance,
        "regularized_covariance": regularized_covariance,
        "precision": precision,
        "covariance_regularization": shrinkage,
        "variance_floor": variance_floor,
        "covariance_condition_number": float(
            np.max(safe_eigenvalues) / np.min(safe_eigenvalues)
        ),
    }
    return transform, diagnostics
