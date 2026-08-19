# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Monte Carlo covariance estimation for moment-based GMM fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from paranmr.core.fitting.susceptibility.moments.descriptors import (
    build_normalized_moment_vectors,
    compute_gaussian_mixture_moments,
)


@dataclass(frozen=True)
class MonteCarloMomentCovarianceConfig:
    """Configuration for Monte Carlo moment-covariance estimation."""

    n_samples: int
    shift_sigma_abs: float
    width_sigma_rel: float
    random_seed: int | None = None


@dataclass(frozen=True)
class MomentCovarianceEstimate:
    """Structured Monte Carlo estimate of the moment covariance matrix."""

    method: str
    moment_names: tuple[str, ...]
    covariance: NDArray[np.float64]
    n_samples: int
    random_seed: int | None
    shift_sigma_abs: float
    width_sigma_rel: float


def estimate_moment_covariance_from_monte_carlo(
    *,
    observed_peaks: dict[str, NDArray[np.float64]],
    raw_experimental_moments: dict[str, float],
    moment_names: tuple[str, ...],
    config: MonteCarloMomentCovarianceConfig,
) -> MomentCovarianceEstimate:
    """Estimate covariance in normalized moment space via Monte Carlo sampling."""

    rng = np.random.default_rng(config.random_seed)
    centers = np.asarray(observed_peaks["center"], dtype=float)
    sigmas = np.asarray(observed_peaks["sigma"], dtype=float)
    area_norm = np.asarray(observed_peaks["area_norm"], dtype=float)

    samples = np.empty((config.n_samples, len(moment_names)), dtype=float)
    for i in range(config.n_samples):
        perturbed_centers = centers + rng.normal(
            loc=0.0,
            scale=config.shift_sigma_abs,
            size=centers.shape,
        )
        sigma_factors = 1.0 + rng.normal(
            loc=0.0,
            scale=config.width_sigma_rel,
            size=sigmas.shape,
        )
        perturbed_sigmas = sigmas * np.maximum(sigma_factors, 1.0e-12)
        moments = compute_gaussian_mixture_moments(
            centers=perturbed_centers,
            sigmas=perturbed_sigmas,
            area_norm=area_norm,
            moment_labels=moment_names,
        )
        normalized_moments = build_normalized_moment_vectors(
            observed=raw_experimental_moments,
            calculated=moments,
            moment_names=moment_names,
        )
        samples[i, :] = np.asarray(
            [normalized_moments.calculated[name] for name in moment_names],
            dtype=float,
        )

    covariance = np.cov(samples, rowvar=False, ddof=1)
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    return MomentCovarianceEstimate(
        method="monte_carlo",
        moment_names=moment_names,
        covariance=covariance,
        n_samples=config.n_samples,
        random_seed=config.random_seed,
        shift_sigma_abs=config.shift_sigma_abs,
        width_sigma_rel=config.width_sigma_rel,
    )
