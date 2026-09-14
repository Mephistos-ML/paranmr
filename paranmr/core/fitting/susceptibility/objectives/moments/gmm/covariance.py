# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Jacobian-based covariance propagation for moment-based GMM fitting."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import NDArray

from paranmr.core.fitting.susceptibility.jacobian.moments import (
    differentiate_moments_by_centers,
    differentiate_moments_by_sigmas,
)


@dataclass(frozen=True)
class JacobianMomentCovarianceConfig:
    """Measurement uncertainty used for Jacobian covariance propagation."""

    shift_sigma_abs: float
    width_sigma_rel: float


@dataclass(frozen=True)
class MomentCovarianceEstimate:
    """Structured Jacobian-propagated estimate of moment covariance."""

    method: str
    moment_names: tuple[str, ...]
    covariance: NDArray[np.float64]
    shift_sigma_abs: float
    width_sigma_rel: float
    input_names: tuple[str, ...]
    input_covariance: NDArray[np.float64]


def estimate_moment_covariance_from_jacobian(
    *,
    observed_peaks: dict[str, NDArray[np.float64]],
    raw_experimental_moments: dict[str, float],
    moment_names: tuple[str, ...],
    config: JacobianMomentCovarianceConfig,
) -> MomentCovarianceEstimate:
    """Propagate peak measurement uncertainty into relative moment space.

    The input covariance assumes independent center and width measurements.
    Width uncertainty is relative to each measured Gaussian FWHM width.
    """

    centers = np.asarray(observed_peaks["center"], dtype=float)
    fwhm = np.asarray(observed_peaks["fwhm"], dtype=float)
    sigmas = np.asarray(observed_peaks["sigma"], dtype=float)
    area_norm = np.asarray(observed_peaks["area_norm"], dtype=float)

    center_jacobian = differentiate_moments_by_centers(
        centers=centers,
        sigmas=sigmas,
        area_norm=area_norm,
        moment_labels=moment_names,
    )
    sigma_jacobian = differentiate_moments_by_sigmas(
        centers=centers,
        sigmas=sigmas,
        area_norm=area_norm,
        moment_labels=moment_names,
    )
    scales = np.asarray(
        [float(raw_experimental_moments[name]) for name in moment_names],
        dtype=float,
    )
    if np.any(np.isclose(scales, 0.0)):
        raise ValueError("Cannot propagate covariance through zero-valued moments")
    fwhm_to_sigma = 2.0 * sqrt(2.0 * np.log(2.0))
    width_jacobian = sigma_jacobian / fwhm_to_sigma
    condition_jacobian = np.hstack((center_jacobian, width_jacobian))
    condition_jacobian = condition_jacobian / scales[:, np.newaxis]
    standard_deviations = np.concatenate(
        (
            np.full(centers.shape, config.shift_sigma_abs, dtype=float),
            config.width_sigma_rel * fwhm,
        )
    )
    input_covariance = np.diag(standard_deviations**2)
    covariance = condition_jacobian @ input_covariance @ condition_jacobian.T
    covariance = 0.5 * (covariance + covariance.T)
    return MomentCovarianceEstimate(
        method="jacobian",
        moment_names=moment_names,
        covariance=covariance,
        shift_sigma_abs=config.shift_sigma_abs,
        width_sigma_rel=config.width_sigma_rel,
        input_names=tuple(
            [f"center[{index}]" for index in range(centers.size)]
            + [f"width[{index}]" for index in range(fwhm.size)]
        ),
        input_covariance=input_covariance,
    )
