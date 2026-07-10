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


def moment_power(order: int) -> int:
    """Return the observed-std power used to normalize a moment descriptor."""
    if order < 1:
        raise ValueError("Moment order must be positive")
    return 1 if order in (1, 2) else order


MOMENT_ORDERS = tuple(range(1, MAX_MOMENT_ORDER + 1))
MOMENT_NAMES = tuple(moment_n(order) for order in MOMENT_ORDERS)


def _gaussian_component_central_moment(
    order: int,
    sigma: np.ndarray,
) -> np.ndarray:
    """Return the central moment of one Gaussian component."""
    if order < 0:
        raise ValueError("Moment order must be non-negative")
    if order == 0:
        return np.ones_like(sigma, dtype=float)
    if order % 2 == 1:
        return np.zeros_like(sigma, dtype=float)

    double_factorial = 1
    for value in range(order - 1, 0, -2):
        double_factorial *= value
    return double_factorial * sigma**order


def _gaussian_mixture_central_moment(
    *,
    weights: np.ndarray,
    delta: np.ndarray,
    sigmas: np.ndarray,
    order: int,
) -> float:
    """Return a central moment of the Gaussian mixture about the global mean."""
    total = np.zeros_like(weights, dtype=float)
    for inner_order in range(order + 1):
        component_moment = _gaussian_component_central_moment(
            inner_order,
            sigmas,
        )
        if np.all(component_moment == 0.0):
            continue
        total += (
            comb(order, inner_order)
            * delta ** (order - inner_order)
            * component_moment
        )
    return float(np.sum(weights * total))


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
        Mapping with ``m1`` (mean), ``m2`` (spectral standard deviation),
        and ``m3``-``mN`` (third to ``MAX_MOMENT_ORDER`` central moments).

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
    variance = _gaussian_mixture_central_moment(
        weights=weights_arr,
        delta=delta,
        sigmas=sigmas_arr,
        order=2,
    )
    if variance <= 0.0:
        raise ValueError("Gaussian mixture moments are undefined for zero variance")

    std = float(np.sqrt(variance))
    raw_moments_by_order = {1: mean, 2: std}
    for order in range(3, MAX_MOMENT_ORDER + 1):
        raw_moments_by_order[order] = _gaussian_mixture_central_moment(
            weights=weights_arr,
            delta=delta,
            sigmas=sigmas_arr,
            order=order,
        )

    return {
        moment_n(order): float(raw_moments_by_order[order])
        for order in MOMENT_ORDERS
    }


def normalize_gaussian_mixture_moment_vectors(
    *,
    observed: dict[str, float],
    calculated: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Normalize observed and calculated moments for objective evaluation.

    The observed standard deviation defines the comparison scale for every
    moment component. Mean and standard deviation are scaled by the first power
    of the observed standard deviation, while the third to sixth central
    moments are scaled by the corresponding powers.

    Args:
        observed: Raw Gaussian-mixture moments from experimental peaks.
        calculated: Raw Gaussian-mixture moments from calculated peaks.

    Returns:
        ``(normalized_observed, normalized_calculated)`` moment mappings.

    Raises:
        ValueError: If moment keys differ or observed ``m2`` is zero/missing.
    """

    if calculated.keys() != observed.keys():
        raise ValueError("Calculated and observed moment keys must match")
    if "m2" not in observed:
        raise ValueError("Cannot normalize moment vectors without observed m2")

    observed_std = float(observed["m2"])
    if observed_std == 0.0:
        raise ValueError("Cannot normalize moment vectors with zero observed m2")

    powers = {
        moment_n(order): moment_power(order)
        for order in MOMENT_ORDERS
    }
    missing = set(MOMENT_NAMES) - set(observed)
    if missing:
        raise ValueError(
            "Cannot normalize moment vectors without keys: "
            + ", ".join(sorted(missing))
        )

    normalized_observed = {
        name: float(observed[name]) / observed_std ** powers[name]
        for name in observed.keys()
    }
    normalized_calculated = {
        name: float(calculated[name]) / observed_std ** powers[name]
        for name in calculated.keys()
    }

    return normalized_observed, normalized_calculated
