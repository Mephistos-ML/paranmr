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


def normalize_gaussian_mixture_moment_vectors(
    *,
    observed: dict[str, float],
    calculated: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Normalize observed and calculated moments for objective evaluation.

    The observed standard deviation defines the comparison scale for dimensional
    moments. Shape moments are already standardized by their definitions and are
    therefore copied unchanged.

    Args:
        observed: Raw Gaussian-mixture moments from experimental peaks.
        calculated: Raw Gaussian-mixture moments from calculated peaks.

    Returns:
        ``(normalized_observed, normalized_calculated)`` moment mappings.

    Raises:
        ValueError: If moment keys differ or observed ``std`` is zero/missing.
    """

    if calculated.keys() != observed.keys():
        raise ValueError("Calculated and observed moment keys must match")
    if "std" not in observed:
        raise ValueError("Cannot normalize moment vectors without observed std")

    observed_std = float(observed["std"])
    if observed_std == 0.0:
        raise ValueError("Cannot normalize moment vectors with zero observed std")

    normalized_observed = {name: float(value) for name, value in observed.items()}
    if "mean" in normalized_observed:
        normalized_observed["mean"] = normalized_observed["mean"] / observed_std
    normalized_observed["std"] = normalized_observed["std"] / observed_std

    normalized_calculated = {
        name: float(value) for name, value in calculated.items()
    }
    if "mean" in normalized_calculated:
        normalized_calculated["mean"] = normalized_calculated["mean"] / observed_std
    normalized_calculated["std"] = normalized_calculated["std"] / observed_std

    return normalized_observed, normalized_calculated

