# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Compute Gaussian representations for moment-matching diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from paranmr.core.spectrum.kernels import (
    gaussian_fwhm_to_sigma,
    gaussian_height_from_area,
)


def gaussian_peak_representation(
    centers: ArrayLike,
    fwhm: ArrayLike,
    areas: ArrayLike,
) -> dict[str, NDArray[np.float64]]:
    """Project peak descriptors into a pure-Gaussian representation.

    Args:
        centers: Peak centers in the axis units.
        fwhm: Peak full widths at half maximum in the axis units.
        areas: Integrated peak areas.

    Returns:
        Mapping with ``center``, ``fwhm``, ``sigma``, ``height``, ``area``,
        ``area_norm``, and ``l_to_g`` arrays. The returned ``l_to_g`` is always
        zero because the representation is pure Gaussian.

    Raises:
        ValueError: If input arrays do not have the same shape or Gaussian
            parameters are invalid, or the total area is zero.
    """

    center_arr = np.asarray(centers, dtype=float)
    fwhm_arr = np.asarray(fwhm, dtype=float)
    area_arr = np.asarray(areas, dtype=float)

    if center_arr.shape != fwhm_arr.shape or center_arr.shape != area_arr.shape:
        raise ValueError("Gaussian peak arrays must have matching shapes")

    sigma_arr = gaussian_fwhm_to_sigma(fwhm_arr)
    height_arr = gaussian_height_from_area(area_arr, sigma_arr)
    total_area = float(np.sum(area_arr))
    if total_area <= 0.0:
        raise ValueError("Gaussian peak total area must be positive")

    return {
        "center": center_arr,
        "fwhm": fwhm_arr,
        "sigma": sigma_arr,
        "height": height_arr,
        "area": area_arr,
        "area_norm": area_arr / total_area,
        "l_to_g": np.zeros_like(center_arr, dtype=float),
    }


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


def apply_moment_weights(
    residuals: dict[str, float],
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Apply per-moment weights to a residual vector.

    Args:
        residuals: Mapping from moment name to residual value.
        weights: Optional mapping from moment name to multiplicative weight.
            Missing weights default to one.

    Returns:
        Weighted residual vector with the same keys as ``residuals``.

    Raises:
        ValueError: If an unknown moment name or a negative weight is provided.
    """

    if weights is None:
        weights = {}

    unknown = set(weights) - set(residuals)
    if unknown:
        raise ValueError(
            "Moment weights contain unknown moment(s): "
            + ", ".join(sorted(unknown))
        )

    weighted = {}
    for moment_name, residual in residuals.items():
        weight = float(weights.get(moment_name, 1.0))
        if weight < 0.0:
            raise ValueError("Moment weights must be non-negative")
        weighted[moment_name] = residual * weight

    return weighted


def moment_residual_norm(residuals: dict[str, float]) -> float:
    """Compute the Euclidean norm of a moment residual vector.

    Args:
        residuals: Mapping from moment name to residual value.

    Returns:
        Euclidean norm of residual values.
    """

    values = np.asarray(list(residuals.values()), dtype=float)
    return float(np.linalg.norm(values))
