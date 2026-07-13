# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Gaussian mixture moment descriptors and residuals."""

from __future__ import annotations

from math import comb

import numpy as np
from numpy.typing import ArrayLike

MAX_MOMENT_ORDER = 6


def moment_n(order: int) -> str:
    """Return the canonical public name for a moment descriptor order."""
    if order < 1:
        raise ValueError("Moment order must be positive")
    return f"m{order}"


MOMENT_ORDERS = tuple(range(1, MAX_MOMENT_ORDER + 1))
MOMENT_NAMES = tuple(moment_n(order) for order in MOMENT_ORDERS)


def compute_first_moment(
    centers: ArrayLike,
    area_norm: ArrayLike,
) -> float:
    """Return the first raw moment (spectral mean) of a Gaussian mixture."""

    centers_arr = np.asarray(centers, dtype=float)
    weights_arr = np.asarray(area_norm, dtype=float)
    if centers_arr.shape != weights_arr.shape:
        raise ValueError("Gaussian mixture arrays must have matching shapes")
    return float(np.sum(weights_arr * centers_arr))


def _compute_single_central_moment(
    *,
    weights: np.ndarray,
    delta: np.ndarray,
    sigmas: np.ndarray,
    order: int,
) -> float:
    """Return a central moment of the Gaussian mixture about the global mean."""
    total = np.zeros_like(weights, dtype=float)
    for inner_order in range(order + 1):
        if inner_order == 0:
            component_moment = np.ones_like(sigmas, dtype=float)
        elif inner_order % 2 == 1:
            component_moment = np.zeros_like(sigmas, dtype=float)
        else:
            double_factorial = 1
            for value in range(inner_order - 1, 0, -2):
                double_factorial *= value
            component_moment = double_factorial * sigmas**inner_order
        if np.all(component_moment == 0.0):
            continue
        total += (
            comb(order, inner_order)
            * delta ** (order - inner_order)
            * component_moment
        )
    return float(np.sum(weights * total))


def compute_central_moments(
    *,
    centers: ArrayLike,
    sigmas: ArrayLike,
    area_norm: ArrayLike,
    mean: float,
    min_order: int = 2,
    max_order: int = MAX_MOMENT_ORDER,
) -> dict[str, float]:
    """Return Gaussian-mixture central moments for the requested order range."""

    centers_arr = np.asarray(centers, dtype=float)
    sigmas_arr = np.asarray(sigmas, dtype=float)
    weights_arr = np.asarray(area_norm, dtype=float)

    if centers_arr.shape != sigmas_arr.shape or centers_arr.shape != weights_arr.shape:
        raise ValueError("Gaussian mixture arrays must have matching shapes")
    if min_order < 2:
        raise ValueError("Central moments must start from order 2")
    if max_order < min_order:
        raise ValueError("Maximum moment order must be >= minimum moment order")

    delta = centers_arr - float(mean)
    return {
        moment_n(order): _compute_single_central_moment(
            weights=weights_arr,
            delta=delta,
            sigmas=sigmas_arr,
            order=order,
        )
        for order in range(min_order, max_order + 1)
    }


def compute_gaussian_mixture_moments(
    centers: ArrayLike,
    sigmas: ArrayLike,
    area_norm: ArrayLike,
) -> dict[str, float]:
    """Compute raw moment descriptors for a normalized Gaussian mixture.

    Args:
        centers: Gaussian component centers.
        sigmas: Gaussian component standard deviations.
        area_norm: Normalized component areas. Values must sum to one.

    Returns:
        Mapping with ``m1`` (mean) and ``m2``-``mN`` (second to
        ``MAX_MOMENT_ORDER`` central moments).

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

    mean = compute_first_moment(centers_arr, weights_arr)
    central_moments = compute_central_moments(
        centers=centers_arr,
        sigmas=sigmas_arr,
        area_norm=weights_arr,
        mean=mean,
        min_order=2,
        max_order=MAX_MOMENT_ORDER,
    )

    return {"m1": mean, **central_moments}


def normalize_gaussian_mixture_moment_vectors(
    *,
    observed: dict[str, float],
    calculated: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Normalize observed and calculated moments for objective evaluation.

    Each observed and calculated descriptor pair is converted to a ratio
    against the observed descriptor value.

    Args:
        observed: Raw Gaussian-mixture moments from experimental peaks.
        calculated: Raw Gaussian-mixture moments from calculated peaks.

    Returns:
        ``(normalized_observed, normalized_calculated)`` moment mappings.

    Raises:
        ValueError: If moment keys differ or any observed descriptor is
            missing, zero, or numerically too close to zero.
    """

    if calculated.keys() != observed.keys():
        raise ValueError("Calculated and observed moment keys must match")
    missing = set(MOMENT_NAMES) - set(observed)
    if missing:
        raise ValueError(
            "Cannot normalize moment vectors without keys: "
            + ", ".join(sorted(missing))
        )

    zero_like = [
        name
        for name in MOMENT_NAMES
        if np.isclose(float(observed[name]), 0.0, atol=1e-12, rtol=0.0)
    ]
    if zero_like:
        raise ValueError(
            "Cannot normalize moment vectors by observed descriptor values "
            "that are zero or too close to zero: "
            + ", ".join(zero_like)
        )

    normalized_observed = {
        name: 1.0
        for name in observed.keys()
    }
    normalized_calculated = {
        name: float(calculated[name]) / float(observed[name])
        for name in calculated.keys()
    }

    return normalized_observed, normalized_calculated
