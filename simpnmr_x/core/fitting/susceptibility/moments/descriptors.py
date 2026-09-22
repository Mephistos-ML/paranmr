# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Gaussian-mixture raw moment descriptors."""

from __future__ import annotations

from math import comb

import numpy as np
from numpy.typing import ArrayLike


def moment_n(order: int) -> str:
    """Return the canonical public name for a moment descriptor order."""
    if order < 0:
        raise ValueError("Moment order must be non-negative")
    return f"m{order}"


def moment_order(label: str) -> int:
    """Return the integer order encoded in a canonical moment label."""
    if not isinstance(label, str) or not label.startswith("m"):
        raise ValueError(f"Invalid moment label {label!r}")
    try:
        order = int(label[1:])
    except ValueError as exc:
        raise ValueError(f"Invalid moment label {label!r}") from exc
    if order < 0:
        raise ValueError(f"Invalid moment label {label!r}")
    return order


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


def _normal_raw_moment(
    *,
    sigmas: np.ndarray,
    order: int,
) -> np.ndarray:
    """Return the raw moment of a centered Gaussian component of given order."""

    if order == 0:
        return np.ones_like(sigmas, dtype=float)
    if order % 2 == 1:
        return np.zeros_like(sigmas, dtype=float)

    double_factorial = 1
    for value in range(order - 1, 0, -2):
        double_factorial *= value
    return double_factorial * sigmas**order


def compute_single_gaussian_mixture_raw_moment(
    *,
    centers: ArrayLike,
    sigmas: ArrayLike,
    area_norm: ArrayLike,
    order: int,
) -> float:
    """Return the raw moment of the requested order for a Gaussian mixture."""

    centers_arr = np.asarray(centers, dtype=float)
    sigmas_arr = np.asarray(sigmas, dtype=float)
    weights_arr = np.asarray(area_norm, dtype=float)

    if centers_arr.shape != sigmas_arr.shape or centers_arr.shape != weights_arr.shape:
        raise ValueError("Gaussian mixture arrays must have matching shapes")

    total = np.zeros_like(weights_arr, dtype=float)
    for inner_order in range(order + 1):
        component_moment = _normal_raw_moment(
            sigmas=sigmas_arr,
            order=inner_order,
        )
        if np.all(component_moment == 0.0):
            continue
        total += (
            comb(order, inner_order)
            * centers_arr ** (order - inner_order)
            * component_moment
        )
    return float(np.sum(weights_arr * total))


def _compute_gaussian_mixture_raw_moments_up_to(
    *,
    centers: np.ndarray,
    sigmas: np.ndarray,
    area_norm: np.ndarray,
    max_order: int,
) -> np.ndarray:
    """Return Gaussian-mixture raw moments from order zero to ``max_order``."""

    moments = np.empty(max_order + 1, dtype=float)
    component_two_lower = np.ones_like(centers, dtype=float)
    moments[0] = float(np.sum(area_norm * component_two_lower))
    if max_order == 0:
        return moments

    component_one_lower = centers.astype(float, copy=True)
    moments[1] = float(np.sum(area_norm * component_one_lower))
    sigma_squared = sigmas**2
    for order in range(2, max_order + 1):
        component_current = (
            centers * component_one_lower
            + (order - 1) * sigma_squared * component_two_lower
        )
        moments[order] = float(np.sum(area_norm * component_current))
        component_two_lower = component_one_lower
        component_one_lower = component_current
    return moments


def compute_gaussian_mixture_moments(
    *,
    centers: ArrayLike,
    sigmas: ArrayLike,
    area_norm: ArrayLike,
    moment_labels: tuple[str, ...],
) -> dict[str, float]:
    """Compute raw moments for a normalized Gaussian mixture.

    Args:
        centers: Gaussian component centers.
        sigmas: Gaussian component standard deviations.
        area_norm: Normalized component areas. Values must sum to one.
        moment_labels: Ordered moment labels requested by the caller.

    Returns:
        Mapping in the requested label order. Each label ``mN`` receives the
        raw moment of order ``N``. ``m0`` is the normalized total area.

    Raises:
        ValueError: If arrays do not have matching shapes, sigmas are not positive,
            weights are negative or weights do not sum to one.
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
    if not moment_labels:
        raise ValueError("At least one moment label is required")

    orders = tuple(moment_order(label) for label in moment_labels)
    moments = _compute_gaussian_mixture_raw_moments_up_to(
        centers=centers_arr,
        sigmas=sigmas_arr,
        area_norm=weights_arr,
        max_order=max(orders),
    )
    return {label: float(moments[order]) for label, order in zip(moment_labels, orders)}
