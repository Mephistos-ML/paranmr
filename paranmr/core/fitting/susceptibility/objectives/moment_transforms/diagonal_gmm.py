# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Diagonal GMM moment objective transform."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from paranmr.core.fitting.susceptibility.objectives.moment_transforms.state import (
    weights_from_transform,
)


def diagonal_gmm_transform(
    *,
    moment_names: tuple[str, ...],
    covariance: NDArray[np.float64],
    variances: NDArray[np.float64],
    variance_floor: float,
) -> tuple[NDArray[np.float64], dict]:
    """Build a diagonal inverse-standard-deviation GMM transform."""
    safe_variances = np.maximum(variances, variance_floor)
    transform = np.diag(1.0 / np.sqrt(safe_variances))
    diagnostics = {
        "weights": weights_from_transform(moment_names, transform),
        "variances": {
            name: float(value) for name, value in zip(moment_names, variances)
        },
        "covariance": covariance,
        "variance_floor": variance_floor,
    }
    return transform, diagnostics
