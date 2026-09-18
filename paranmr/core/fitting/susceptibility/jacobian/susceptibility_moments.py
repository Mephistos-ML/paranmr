# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Analytical moment derivatives in Cartesian split susceptibility coordinates."""

from __future__ import annotations

from numpy.typing import NDArray

from paranmr.core.domain.mol import Nucleus
from paranmr.core.fitting.susceptibility.jacobian.moments import (
    differentiate_moments_by_centers,
)
from paranmr.core.fitting.susceptibility.jacobian.susceptibility_centers import (
    differentiate_centers_by_split_parameter,
)
from paranmr.core.fitting.susceptibility.models.split import SplitFitter
from paranmr.core.fitting.susceptibility.moments.forward import (
    calculated_signal_packages_from_parameters,
    package_areas,
    package_centers,
    package_linewidths,
    sort_packages_by_center,
)
from paranmr.core.fitting.susceptibility.moments.gaussian import (
    gaussian_peak_representation,
)


def differentiate_moments_by_split_parameter(
    *,
    parameter_name: str,
    parameters: dict[str, float],
    nuclei: list[Nucleus],
    linewidths_by_label: dict[str, float],
    moment_labels: tuple[str, ...],
    average_labels: tuple[tuple[str, ...], ...] = (),
) -> NDArray:
    """Return analytical moment derivatives for one split tensor coordinate."""

    packages = sort_packages_by_center(
        calculated_signal_packages_from_parameters(
            model=SplitFitter,
            parameters=parameters,
            nuclei=nuclei,
            include_diamagnetic=True,
            average_labels=average_labels,
        )
    )
    peaks = gaussian_peak_representation(
        centers=package_centers(packages),
        fwhm=package_linewidths(packages, linewidths_by_label),
        areas=package_areas(packages),
    )
    d_moments_by_centers = differentiate_moments_by_centers(
        centers=peaks["center"],
        sigmas=peaks["sigma"],
        area_norm=peaks["area_norm"],
        moment_labels=moment_labels,
    )
    d_centers = differentiate_centers_by_split_parameter(
        parameter_name=parameter_name,
        parameters=parameters,
        nuclei=nuclei,
        average_labels=average_labels,
    )
    return d_moments_by_centers @ d_centers
