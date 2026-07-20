# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Analytical derivatives of Gaussian-mixture moments."""

from __future__ import annotations

from math import comb

import numpy as np
from numpy.typing import ArrayLike, NDArray

from paranmr.core.fitting.susceptibility.moments.descriptors import (
    MAX_MOMENT_ORDER,
    MOMENT_ORDERS,
    compute_first_moment,
)


def differentiate_moments_by_centers(
    *,
    centers: ArrayLike,
    sigmas: ArrayLike,
    area_norm: ArrayLike,
) -> NDArray[np.float64]:
    """Return analytical moment derivatives with respect to centers.

    The returned matrix has one row per public moment descriptor ``m1``-``mN``
    and one column per Gaussian component center.
    """

    centers_arr, sigmas_arr, weights_arr = _validate_gaussian_mixture_inputs(
        centers=centers,
        sigmas=sigmas,
        area_norm=area_norm,
    )
    mean = compute_first_moment(centers_arr, weights_arr)
    delta = centers_arr - mean
    shifted_moments = _compute_shifted_component_moments(
        delta=delta,
        sigmas=sigmas_arr,
        max_order=MAX_MOMENT_ORDER - 1,
    )

    jacobian = np.zeros((MAX_MOMENT_ORDER, len(centers_arr)), dtype=float)
    jacobian[0, :] = weights_arr
    for order in MOMENT_ORDERS[1:]:
        component_previous = shifted_moments[order - 1]
        component_average = float(np.sum(weights_arr * component_previous))
        jacobian[order - 1, :] = (
            order * weights_arr * (component_previous - component_average)
        )
    return jacobian


def differentiate_moments_by_sigmas(
    *,
    centers: ArrayLike,
    sigmas: ArrayLike,
    area_norm: ArrayLike,
) -> NDArray[np.float64]:
    """Return analytical moment derivatives with respect to sigmas.

    The returned matrix has one row per public moment descriptor ``m1``-``mN``
    and one column per Gaussian component standard deviation.
    """

    centers_arr, sigmas_arr, weights_arr = _validate_gaussian_mixture_inputs(
        centers=centers,
        sigmas=sigmas,
        area_norm=area_norm,
    )
    mean = compute_first_moment(centers_arr, weights_arr)
    delta = centers_arr - mean
    shifted_moments = _compute_shifted_component_moments(
        delta=delta,
        sigmas=sigmas_arr,
        max_order=MAX_MOMENT_ORDER - 2,
    )

    jacobian = np.zeros((MAX_MOMENT_ORDER, len(centers_arr)), dtype=float)
    for order in MOMENT_ORDERS[1:]:
        component_two_lower = shifted_moments[order - 2]
        jacobian[order - 1, :] = (
            order
            * (order - 1)
            * weights_arr
            * sigmas_arr
            * component_two_lower
        )
    return jacobian


def _compute_shifted_component_moments(
    *,
    delta: np.ndarray,
    sigmas: np.ndarray,
    max_order: int,
) -> dict[int, np.ndarray]:
    """Return component moments about the global mixture mean."""

    moments: dict[int, np.ndarray] = {0: np.ones_like(delta, dtype=float)}
    if max_order >= 1:
        moments[1] = delta.astype(float, copy=True)
    for order in range(2, max_order + 1):
        moments[order] = _differentiate_single_shifted_moment(
            delta=delta,
            sigmas=sigmas,
            order=order,
        )
    return moments


def _differentiate_single_shifted_moment(
    *,
    delta: np.ndarray,
    sigmas: np.ndarray,
    order: int,
) -> np.ndarray:
    """Return one component-wise moment about the global mixture mean."""

    total = np.zeros_like(delta, dtype=float)
    for inner_order in range(order + 1):
        total += (
            comb(order, inner_order)
            * delta ** (order - inner_order)
            * _normal_central_moment(sigmas=sigmas, order=inner_order)
        )
    return total


def _normal_central_moment(
    *,
    sigmas: np.ndarray,
    order: int,
) -> np.ndarray:
    """Return the central moment of a Gaussian component of given order."""

    if order == 0:
        return np.ones_like(sigmas, dtype=float)
    if order % 2 == 1:
        return np.zeros_like(sigmas, dtype=float)

    double_factorial = 1
    for value in range(order - 1, 0, -2):
        double_factorial *= value
    return double_factorial * sigmas**order


def _validate_gaussian_mixture_inputs(
    *,
    centers: ArrayLike,
    sigmas: ArrayLike,
    area_norm: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return centers_arr, sigmas_arr, weights_arr
