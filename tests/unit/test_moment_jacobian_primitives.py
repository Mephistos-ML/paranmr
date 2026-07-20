# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

import numpy as np
import pytest

from paranmr.core.fitting.susceptibility.jacobian.moments import (
    differentiate_moments_by_centers,
    differentiate_moments_by_sigmas,
)
from paranmr.core.fitting.susceptibility.moments.descriptors import (
    MOMENT_NAMES,
    compute_gaussian_mixture_moments,
)


def _moment_vector(
    *,
    centers: np.ndarray,
    sigmas: np.ndarray,
    area_norm: np.ndarray,
) -> np.ndarray:
    moments = compute_gaussian_mixture_moments(
        centers=centers,
        sigmas=sigmas,
        area_norm=area_norm,
    )
    return np.asarray([moments[name] for name in MOMENT_NAMES], dtype=float)


def _finite_difference_by_centers(
    *,
    centers: np.ndarray,
    sigmas: np.ndarray,
    area_norm: np.ndarray,
    step: float = 1e-7,
) -> np.ndarray:
    jacobian = np.zeros((len(MOMENT_NAMES), len(centers)), dtype=float)
    for index in range(len(centers)):
        centers_forward = centers.copy()
        centers_backward = centers.copy()
        centers_forward[index] += step
        centers_backward[index] -= step
        jacobian[:, index] = (
            _moment_vector(
                centers=centers_forward,
                sigmas=sigmas,
                area_norm=area_norm,
            )
            - _moment_vector(
                centers=centers_backward,
                sigmas=sigmas,
                area_norm=area_norm,
            )
        ) / (2.0 * step)
    return jacobian


def _finite_difference_by_sigmas(
    *,
    centers: np.ndarray,
    sigmas: np.ndarray,
    area_norm: np.ndarray,
    step: float = 1e-7,
) -> np.ndarray:
    jacobian = np.zeros((len(MOMENT_NAMES), len(sigmas)), dtype=float)
    for index in range(len(sigmas)):
        sigmas_forward = sigmas.copy()
        sigmas_backward = sigmas.copy()
        sigmas_forward[index] += step
        sigmas_backward[index] -= step
        jacobian[:, index] = (
            _moment_vector(
                centers=centers,
                sigmas=sigmas_forward,
                area_norm=area_norm,
            )
            - _moment_vector(
                centers=centers,
                sigmas=sigmas_backward,
                area_norm=area_norm,
            )
        ) / (2.0 * step)
    return jacobian


@pytest.mark.unit
def test_differentiate_moments_by_centers_matches_finite_difference():
    centers = np.asarray([-1.5, 0.3, 2.2], dtype=float)
    sigmas = np.asarray([0.4, 0.7, 0.5], dtype=float)
    area_norm = np.asarray([0.2, 0.5, 0.3], dtype=float)

    analytical = differentiate_moments_by_centers(
        centers=centers,
        sigmas=sigmas,
        area_norm=area_norm,
    )
    finite_difference = _finite_difference_by_centers(
        centers=centers,
        sigmas=sigmas,
        area_norm=area_norm,
    )

    assert analytical == pytest.approx(finite_difference, rel=1e-6, abs=1e-8)


@pytest.mark.unit
def test_differentiate_moments_by_sigmas_matches_finite_difference():
    centers = np.asarray([-1.5, 0.3, 2.2], dtype=float)
    sigmas = np.asarray([0.4, 0.7, 0.5], dtype=float)
    area_norm = np.asarray([0.2, 0.5, 0.3], dtype=float)

    analytical = differentiate_moments_by_sigmas(
        centers=centers,
        sigmas=sigmas,
        area_norm=area_norm,
    )
    finite_difference = _finite_difference_by_sigmas(
        centers=centers,
        sigmas=sigmas,
        area_norm=area_norm,
    )

    assert analytical == pytest.approx(finite_difference, rel=1e-6, abs=1e-8)
