# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Gaussian peak representations for moment matching."""

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
