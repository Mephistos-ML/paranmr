# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Bootstrap uncertainty estimates for moment-matching objectives."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from paranmr.core.fitting.susceptibility.moments.descriptors import (
    gaussian_mixture_moment_residuals,
    gaussian_mixture_moments,
)
from paranmr.core.fitting.susceptibility.moments.gaussian import (
    gaussian_peak_representation,
)


def bootstrap_moment_residual_covariance(
    *,
    observed_centers: ArrayLike,
    widths_ppm: ArrayLike,
    areas: ArrayLike,
    observed_moments: dict[str, float],
    bootstrap_config: dict | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate moment-residual covariance by perturbing the peak table."""

    if bootstrap_config is None:
        bootstrap_config = {}

    samples = _bootstrap_int(bootstrap_config, "samples", 1000)
    if samples < 2:
        raise ValueError("Bootstrap moment objectives require at least two samples")

    center_sigma = _bootstrap_float(
        bootstrap_config,
        "center_sigma_ppm",
        _bootstrap_float(bootstrap_config, "centre_sigma_ppm", 0.0),
    )
    linewidth_rel_sigma = _bootstrap_float(
        bootstrap_config, "linewidth_relative_sigma", 0.0
    )
    area_rel_sigma = _bootstrap_float(bootstrap_config, "area_relative_sigma", 0.0)

    if center_sigma < 0.0:
        raise ValueError("center_sigma_ppm must be non-negative")
    if linewidth_rel_sigma < 0.0:
        raise ValueError("linewidth_relative_sigma must be non-negative")
    if area_rel_sigma < 0.0:
        raise ValueError("area_relative_sigma must be non-negative")
    if center_sigma == 0.0 and linewidth_rel_sigma == 0.0 and area_rel_sigma == 0.0:
        raise ValueError(
            "Bootstrap moment objectives require at least one positive peak "
            "uncertainty"
        )

    seed = bootstrap_config.get("seed")
    rng = np.random.default_rng(None if seed in (None, "") else int(seed))

    centers_arr = np.asarray(observed_centers, dtype=float)
    widths_arr = np.asarray(widths_ppm, dtype=float)
    areas_arr = np.asarray(areas, dtype=float)
    eps = np.finfo(float).tiny

    residual_samples = []
    for _ in range(samples):
        sample_centers = centers_arr + rng.normal(
            loc=0.0, scale=center_sigma, size=centers_arr.shape
        )
        sample_widths = widths_arr * (
            1.0
            + rng.normal(
                loc=0.0, scale=linewidth_rel_sigma, size=widths_arr.shape
            )
        )
        sample_areas = areas_arr * (
            1.0
            + rng.normal(loc=0.0, scale=area_rel_sigma, size=areas_arr.shape)
        )
        sample_widths = np.maximum(sample_widths, eps)
        sample_areas = np.maximum(sample_areas, eps)

        sample_peaks = gaussian_peak_representation(
            centers=sample_centers,
            fwhm=sample_widths,
            areas=sample_areas,
        )
        sample_moments = gaussian_mixture_moments(
            centers=sample_peaks["center"],
            sigmas=sample_peaks["sigma"],
            area_norm=sample_peaks["area_norm"],
        )
        sample_residuals = gaussian_mixture_moment_residuals(
            calculated=sample_moments,
            observed=observed_moments,
            normalize=True,
        )
        residual_samples.append(
            [sample_residuals[name] for name in observed_moments.keys()]
        )

    residual_arr = np.asarray(residual_samples, dtype=float)
    covariance = np.cov(residual_arr, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    variances = np.diag(covariance)
    return covariance, variances


def _bootstrap_int(config: dict, key: str, default: int) -> int:
    value = config.get(key, default)
    if value in (None, ""):
        value = default
    return int(value)


def _bootstrap_float(config: dict, key: str, default: float) -> float:
    value = config.get(key, default)
    if value in (None, ""):
        value = default
    return float(value)
