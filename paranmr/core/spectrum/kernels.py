# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Define spectral line-shape kernels.

Provides Gaussian and Lorentzian peak functions for spectrum construction.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def gaussian_fwhm_to_sigma(fwhm: ArrayLike) -> NDArray[np.float64]:
    """Convert Gaussian full width at half maximum to standard deviation.

    Args:
        fwhm: Gaussian full width at half maximum values.

    Returns:
        Standard deviation values in the same units as ``fwhm``.

    Raises:
        ValueError: If any FWHM value is not positive.
    """

    fwhm_arr = np.asarray(fwhm, dtype=float)
    if np.any(fwhm_arr <= 0.0):
        raise ValueError("Gaussian FWHM values must be positive")

    return fwhm_arr / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def gaussian_height_from_area(
    area: ArrayLike,
    sigma: ArrayLike,
) -> NDArray[np.float64]:
    """Compute Gaussian peak height from integrated area and sigma.

    Args:
        area: Integrated Gaussian peak areas.
        sigma: Gaussian standard deviations in the axis units.

    Returns:
        Gaussian peak heights in inverse axis units.

    Raises:
        ValueError: If any area is negative or any sigma is not positive.
    """

    area_arr = np.asarray(area, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)

    if np.any(area_arr < 0.0):
        raise ValueError("Gaussian peak areas must be non-negative")
    if np.any(sigma_arr <= 0.0):
        raise ValueError("Gaussian sigma values must be positive")

    return area_arr / (sigma_arr * np.sqrt(2.0 * np.pi))


def gaussian(x: ArrayLike, fwhm: float, b: float, area: float) -> NDArray:
    """Evaluates a Gaussian peak with a given position, width, and area.

    The functional form is:

        ``g(x) = area/(c*sqrt(2*pi)) * exp(-(x-b)^2/(2*c^2))``

    where ``c = fwhm/(2*sqrt(2*ln(2)))``.

    Args:
        x: Coordinate grid.
        fwhm: Full width at half maximum.
        b: Peak position.
        area: Peak area.

    Returns:
        Array of ``g(x)`` values.
    """

    c = gaussian_fwhm_to_sigma(fwhm)

    a = gaussian_height_from_area(1.0, c)

    gaus = a * np.exp(-((x - b) ** 2) / (2 * c**2))

    gaus *= area

    return gaus


def lorentzian(x: ArrayLike, fwhm, x0, area) -> NDArray:
    """Evaluates a Lorentzian peak with a given position, width, and area.

    The functional form is:

        ``L(x) = (0.5*area*fwhm/pi) / ((x-x0)^2 + (0.5*fwhm)^2)``

    Args:
        x: Coordinate grid.
        fwhm: Full width at half maximum.
        x0: Peak position.
        area: Peak area.

    Returns:
        Array of ``L(x)`` values.
    """

    lor = 0.5 * fwhm / np.pi
    lor *= 1.0 / ((x - x0) ** 2 + (0.5 * fwhm) ** 2)

    lor *= area

    return lor
