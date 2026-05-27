# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Compute moment-matching quantities for susceptibility fitting.

This module contains numerical helpers for comparing distributions of chemical
shifts. The current proof of concept uses peak positions only; later extensions
can add area- or width-aware moment sources without changing the core moment
calculation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


DEFAULT_MOMENT_ORDERS = (1, 2, 3, 4, 5, 6)


def compute_moments(
    values: NDArray[np.float64],
    orders: tuple[int, ...] = DEFAULT_MOMENT_ORDERS,
    type: str = "central",
) -> dict[int | str, float]:
    """Compute moments or central moment descriptors for a value array.

    Args:
        values: One-dimensional numeric values.
        orders: Positive integer moment orders to compute.
        type: Moment definition. Supported values are ``"central"`` and ``"raw"``.
            For ``"central"``, the first moment is the mean and higher moments are
            standardized around that mean. Returned values are ``mean``, ``std``,
            ``skewness``, ``kurtosis``, ``standardized_5``, and ``standardized_6``.

    Returns:
        Mapping from moment order or descriptor name to moment value.

    Raises:
        ValueError: If values are empty, an order is invalid, central moments are
            undefined, or type is unknown.
    """

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot compute moments for an empty value array")

    if any(order < 1 for order in orders):
        raise ValueError("Moment orders must be positive integers")

    type_normalized = type.strip().lower()
    if type_normalized == "central":
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std == 0.0:
            raise ValueError(
                "Cannot compute central moments for zero-variance values"
            )

        standardized = (arr - mean) / std
        return {
            "mean": mean,
            "std": std,
            "skewness": float(np.mean(standardized**3)),
            "kurtosis": float(np.mean(standardized**4)),
            "standardized_5": float(np.mean(standardized**5)),
            "standardized_6": float(np.mean(standardized**6)),
        }
    elif type_normalized == "raw":
        base = arr
    else:
        raise ValueError("Moment type must be 'central' or 'raw'")

    return {order: float(np.mean(base**order)) for order in orders}


def compute_moment_residuals(
    calculated: dict[int | str, float],
    experimental: dict[int | str, float],
    normalize: bool = True,
) -> dict[int | str, float]:
    """Compute moment residuals as calculated minus experimental values.

    Args:
        calculated: Moment vector computed from calculated shifts.
        experimental: Moment vector computed from experimental shifts.
        normalize: Whether to scale dimensional residuals. When ``True``, ``mean``
            and ``std`` residuals are divided by experimental ``std``; all
            standardized shape-moment residuals are left unchanged.

    Returns:
        Mapping from moment name/order to residual value.

    Raises:
        ValueError: If moment keys differ or normalization is requested without a
            nonzero experimental ``std``.
    """

    if calculated.keys() != experimental.keys():
        raise ValueError("Calculated and experimental moment keys must match")

    residuals = {
        moment_name: calculated[moment_name] - experimental[moment_name]
        for moment_name in experimental.keys()
    }
    if not normalize:
        return residuals

    if "std" not in experimental:
        raise ValueError("Cannot normalize moment residuals without experimental std")

    std = float(experimental["std"])
    if std == 0.0:
        raise ValueError("Cannot normalize moment residuals with zero experimental std")

    normalized = residuals.copy()
    if "mean" in normalized:
        normalized["mean"] = normalized["mean"] / std
    normalized["std"] = normalized["std"] / std

    return normalized
