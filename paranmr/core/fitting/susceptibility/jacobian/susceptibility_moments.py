# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Analytical moment derivatives for susceptibility parameters."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from paranmr.core.domain.mol import Nucleus
from paranmr.core.fitting.susceptibility.jacobian.moments import (
    differentiate_moments_by_centers,
)
from paranmr.core.fitting.susceptibility.jacobian.susceptibility_centers import (
    ShiftOnlyIsoAxRhoModel,
    differentiate_centers_by_alpha,
    differentiate_centers_by_beta,
    differentiate_centers_by_gamma,
    differentiate_centers_by_susc_ax,
    differentiate_centers_by_susc_iso,
    differentiate_centers_by_susc_rho_over_ax,
)
from paranmr.core.fitting.susceptibility.moments.forward import (
    calculated_signal_packages_from_parameters,
    package_centers,
    sort_packages_by_center,
)
from paranmr.core.fitting.susceptibility.moments.gaussian import (
    gaussian_peak_representation,
)


def differentiate_moments_by_susc_ax(
    *,
    parameters: dict[str, float],
    nuclei: list[Nucleus],
    linewidths_by_label: dict[str, float],
    average_labels: tuple[tuple[str, ...], ...] = (),
) -> NDArray[np.float64]:
    """Return analytical moment derivatives with respect to susceptibility axiality."""

    return differentiate_moments_by_center_derivative(
        parameters=parameters,
        nuclei=nuclei,
        linewidths_by_label=linewidths_by_label,
        average_labels=average_labels,
        d_centers=differentiate_centers_by_susc_ax(
            parameters=parameters,
            nuclei=nuclei,
            average_labels=average_labels,
        ),
    )


def differentiate_moments_by_susc_iso(
    *,
    parameters: dict[str, float],
    nuclei: list[Nucleus],
    linewidths_by_label: dict[str, float],
    average_labels: tuple[tuple[str, ...], ...] = (),
) -> NDArray[np.float64]:
    """Return analytical moment derivatives with respect to susceptibility isotropy."""

    return differentiate_moments_by_center_derivative(
        parameters=parameters,
        nuclei=nuclei,
        linewidths_by_label=linewidths_by_label,
        average_labels=average_labels,
        d_centers=differentiate_centers_by_susc_iso(
            parameters=parameters,
            nuclei=nuclei,
            average_labels=average_labels,
        ),
    )


def differentiate_moments_by_susc_rho_over_ax(
    *,
    parameters: dict[str, float],
    nuclei: list[Nucleus],
    linewidths_by_label: dict[str, float],
    average_labels: tuple[tuple[str, ...], ...] = (),
) -> NDArray[np.float64]:
    """Return analytical moment derivatives with respect to rhombicity ratio."""

    return differentiate_moments_by_center_derivative(
        parameters=parameters,
        nuclei=nuclei,
        linewidths_by_label=linewidths_by_label,
        average_labels=average_labels,
        d_centers=differentiate_centers_by_susc_rho_over_ax(
            parameters=parameters,
            nuclei=nuclei,
            average_labels=average_labels,
        ),
    )


def differentiate_moments_by_alpha(
    *,
    parameters: dict[str, float],
    nuclei: list[Nucleus],
    linewidths_by_label: dict[str, float],
    average_labels: tuple[tuple[str, ...], ...] = (),
) -> NDArray[np.float64]:
    """Return analytical moment derivatives with respect to ``alpha``."""

    return differentiate_moments_by_center_derivative(
        parameters=parameters,
        nuclei=nuclei,
        linewidths_by_label=linewidths_by_label,
        average_labels=average_labels,
        d_centers=differentiate_centers_by_alpha(
            parameters=parameters,
            nuclei=nuclei,
            average_labels=average_labels,
        ),
    )


def differentiate_moments_by_beta(
    *,
    parameters: dict[str, float],
    nuclei: list[Nucleus],
    linewidths_by_label: dict[str, float],
    average_labels: tuple[tuple[str, ...], ...] = (),
) -> NDArray[np.float64]:
    """Return analytical moment derivatives with respect to ``beta``."""

    return differentiate_moments_by_center_derivative(
        parameters=parameters,
        nuclei=nuclei,
        linewidths_by_label=linewidths_by_label,
        average_labels=average_labels,
        d_centers=differentiate_centers_by_beta(
            parameters=parameters,
            nuclei=nuclei,
            average_labels=average_labels,
        ),
    )


def differentiate_moments_by_gamma(
    *,
    parameters: dict[str, float],
    nuclei: list[Nucleus],
    linewidths_by_label: dict[str, float],
    average_labels: tuple[tuple[str, ...], ...] = (),
) -> NDArray[np.float64]:
    """Return analytical moment derivatives with respect to ``gamma``."""

    return differentiate_moments_by_center_derivative(
        parameters=parameters,
        nuclei=nuclei,
        linewidths_by_label=linewidths_by_label,
        average_labels=average_labels,
        d_centers=differentiate_centers_by_gamma(
            parameters=parameters,
            nuclei=nuclei,
            average_labels=average_labels,
        ),
    )


def differentiate_moments_by_center_derivative(
    *,
    parameters: dict[str, float],
    nuclei: list[Nucleus],
    linewidths_by_label: dict[str, float],
    average_labels: tuple[tuple[str, ...], ...],
    d_centers: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return moment derivatives for a supplied center-derivative vector."""

    packages = sort_packages_by_center(
        calculated_signal_packages_from_parameters(
            model=ShiftOnlyIsoAxRhoModel(),
            parameters=parameters,
            nuclei=nuclei,
            include_diamagnetic=True,
            average_labels=average_labels,
        )
    )
    centers = package_centers(packages)
    fwhm = np.asarray(
        [
            _package_linewidth_for_jacobian(
                package=package,
                linewidths_by_label=linewidths_by_label,
            )
            for package in packages
        ],
        dtype=float,
    )
    peaks = gaussian_peak_representation(
        centers=centers,
        fwhm=fwhm,
        areas=np.ones(len(packages), dtype=float),
    )
    d_moments_by_centers = differentiate_moments_by_centers(
        centers=peaks["center"],
        sigmas=peaks["sigma"],
        area_norm=peaks["area_norm"],
    )
    return d_moments_by_centers @ d_centers


def package_linewidth_for_jacobian(
    *,
    package,
    linewidths_by_label: dict[str, float],
) -> float:
    """Return package linewidth under the current package-label precedence contract."""

    if package.label in linewidths_by_label:
        return float(linewidths_by_label[package.label])
    return float(
        np.mean([linewidths_by_label[atom_label] for atom_label in package.atom_labels])
    )


_package_linewidth_for_jacobian = package_linewidth_for_jacobian
