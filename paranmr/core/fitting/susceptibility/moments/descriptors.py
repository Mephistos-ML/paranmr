# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Gaussian mixture moment descriptors and residuals."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

MOMENT_NAMES = (
    "mean",
    "std",
    "skewness",
    "kurtosis",
    "standardized_5",
    "standardized_6",
)


def gaussian_mixture_moments(
    centers: ArrayLike,
    sigmas: ArrayLike,
    area_norm: ArrayLike,
) -> dict[str, float]:
    """Compute standardized moments for a normalized Gaussian mixture.

    Args:
        centers: Gaussian component centers.
        sigmas: Gaussian component standard deviations.
        area_norm: Normalized component areas. Values must sum to one.

    Returns:
        Mapping with ``mean``, ``std``, ``skewness``, ``kurtosis``,
        ``standardized_5``, and ``standardized_6``.

    Raises:
        ValueError: If arrays do not have matching shapes, sigmas are not positive,
            weights are negative, weights do not sum to one, or the mixture has
            zero variance.
    """

    centers_arr = np.asarray(centers, dtype=float)
    sigmas_arr = np.asarray(sigmas, dtype=float)
    weights_arr = np.asarray(area_norm, dtype=float)

    if centers_arr.shape != sigmas_arr.shape or centers_arr.shape != weights_arr.shape:
        raise ValueError("Gaussian mixture arrays must have matching shapes")
    if np.any(sigmas_arr <= 0.0):
        raise ValueError("Gaussian mixture sigmas must be positive")
    if np.any(weights_arr < 0.0):
        raise ValueError("Gaussian mixture normalized areas must be non-negative")
    if not np.isclose(np.sum(weights_arr), 1.0):
        raise ValueError("Gaussian mixture normalized areas must sum to one")

    mean = float(np.sum(weights_arr * centers_arr))
    delta = centers_arr - mean
    variance = float(np.sum(weights_arr * (delta**2 + sigmas_arr**2)))
    if variance <= 0.0:
        raise ValueError("Gaussian mixture moments are undefined for zero variance")

    std = float(np.sqrt(variance))
    central_3 = float(
        np.sum(weights_arr * (delta**3 + 3.0 * delta * sigmas_arr**2))
    )
    central_4 = float(
        np.sum(
            weights_arr
            * (
                delta**4
                + 6.0 * delta**2 * sigmas_arr**2
                + 3.0 * sigmas_arr**4
            )
        )
    )
    central_5 = float(
        np.sum(
            weights_arr
            * (
                delta**5
                + 10.0 * delta**3 * sigmas_arr**2
                + 15.0 * delta * sigmas_arr**4
            )
        )
    )
    central_6 = float(
        np.sum(
            weights_arr
            * (
                delta**6
                + 15.0 * delta**4 * sigmas_arr**2
                + 45.0 * delta**2 * sigmas_arr**4
                + 15.0 * sigmas_arr**6
            )
        )
    )

    return {
        "mean": mean,
        "std": std,
        "skewness": central_3 / std**3,
        "kurtosis": central_4 / std**4,
        "standardized_5": central_5 / std**5,
        "standardized_6": central_6 / std**6,
    }


def gaussian_mixture_moment_residuals(
    calculated: dict[str, float],
    observed: dict[str, float],
    normalize: bool = True,
) -> dict[str, float]:
    """Compute Gaussian mixture moment residuals as calculated minus observed.

    Args:
        calculated: Moment vector computed from theoretical/calculated peaks.
        observed: Moment vector computed from observed peaks.
        normalize: Whether to scale dimensional residuals. When ``True``, mean
            and standard-deviation residuals are divided by observed ``std``;
            all standardized shape-moment residuals are left unchanged.

    Returns:
        Mapping from moment name to residual value.

    Raises:
        ValueError: If moment keys differ or normalization is requested without a
            nonzero observed ``std``.
    """

    if calculated.keys() != observed.keys():
        raise ValueError("Calculated and observed moment keys must match")

    residuals = {
        moment_name: calculated[moment_name] - observed[moment_name]
        for moment_name in observed.keys()
    }
    if not normalize:
        return residuals

    if "std" not in observed:
        raise ValueError("Cannot normalize moment residuals without observed std")

    observed_std = float(observed["std"])
    if observed_std == 0.0:
        raise ValueError("Cannot normalize moment residuals with zero observed std")

    normalized = residuals.copy()
    if "mean" in normalized:
        normalized["mean"] = normalized["mean"] / observed_std
    normalized["std"] = normalized["std"] / observed_std

    return normalized


def moment_residual_norm(residuals: dict[str, float]) -> float:
    """Compute the Euclidean norm of a moment residual vector.

    Args:
        residuals: Mapping from moment name to residual value.

    Returns:
        Euclidean norm of residual values.
    """

    values = np.asarray(list(residuals.values()), dtype=float)
    return float(np.linalg.norm(values))
