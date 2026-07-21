# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Analytical linewidth-side derivatives for moment Jacobians."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from paranmr.core.fitting.susceptibility.linewidths import (
    SusceptibilityLinewidthInputs,
    predict_r6_widths_by_atom_label,
)
from paranmr.core.fitting.susceptibility.jacobian.moments import (
    differentiate_moments_by_sigmas,
)
from paranmr.core.fitting.susceptibility.moments.forward import (
    CalculatedSignalPackage,
    package_centers,
    package_linewidths,
)
from paranmr.core.fitting.susceptibility.moments.gaussian import (
    gaussian_peak_representation,
)

_GAUSSIAN_FWHM_TO_SIGMA_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))


def differentiate_sigmas_by_linewidth_parameters(
    *,
    packages: list[CalculatedSignalPackage],
    linewidth_inputs: SusceptibilityLinewidthInputs,
) -> NDArray[np.float64]:
    """Return analytical sigma derivatives with respect to ``p1`` and ``p2``.

    The returned matrix has one row per calculated signal package and two
    columns ordered as ``(p1, p2)``.
    """

    mean_inv_r6_by_atom_label = linewidth_inputs.mean_inv_r6_by_atom_label
    if mean_inv_r6_by_atom_label is None:
        raise ValueError(
            "Susceptibility Jacobian evaluation requires atom-level mean 1/r^6 "
            "values for the R6 linewidth model."
        )

    jacobian = np.zeros((len(packages), 2), dtype=float)
    for row_index, package in enumerate(packages):
        package_mean_inv_r6 = _package_mean_inv_r6(
            package=package,
            mean_inv_r6_by_atom_label=mean_inv_r6_by_atom_label,
        )
        jacobian[row_index, 0] = (
            package_mean_inv_r6 / _GAUSSIAN_FWHM_TO_SIGMA_FACTOR
        )
        jacobian[row_index, 1] = 1.0 / _GAUSSIAN_FWHM_TO_SIGMA_FACTOR
    return jacobian


def differentiate_moments_by_linewidth_parameters(
    *,
    packages: list[CalculatedSignalPackage],
    linewidth_inputs: SusceptibilityLinewidthInputs,
    linewidth_vars_by_name: dict[str, float],
    moment_labels: tuple[str, ...],
) -> NDArray[np.float64]:
    """Return analytical moment derivatives with respect to ``p1`` and ``p2``.

    The returned matrix has one row per public moment descriptor ``m1``-``mN``
    and two columns ordered as ``(p1, p2)``.
    """

    if "p1" not in linewidth_vars_by_name or "p2" not in linewidth_vars_by_name:
        raise ValueError(
            "Linewidth Jacobian evaluation requires linewidth variables 'p1' "
            "and 'p2'."
        )

    centers = package_centers(packages)
    linewidths_by_label = predict_r6_widths_by_atom_label(
        linewidth_inputs=linewidth_inputs,
        linewidth_vars_by_name=linewidth_vars_by_name,
    )
    fwhm = package_linewidths(packages, linewidths_by_label)
    peaks = gaussian_peak_representation(
        centers=centers,
        fwhm=fwhm,
        areas=np.ones(len(packages), dtype=float),
    )

    d_moments_by_sigmas = differentiate_moments_by_sigmas(
        centers=peaks["center"],
        sigmas=peaks["sigma"],
        area_norm=peaks["area_norm"],
        moment_labels=moment_labels,
    )
    d_sigmas_by_linewidth = differentiate_sigmas_by_linewidth_parameters(
        packages=packages,
        linewidth_inputs=linewidth_inputs,
    )
    return d_moments_by_sigmas @ d_sigmas_by_linewidth


def _package_mean_inv_r6(
    *,
    package: CalculatedSignalPackage,
    mean_inv_r6_by_atom_label: dict[str, float],
) -> float:
    if package.label in mean_inv_r6_by_atom_label:
        return float(mean_inv_r6_by_atom_label[package.label])

    missing = [
        atom_label
        for atom_label in package.atom_labels
        if atom_label not in mean_inv_r6_by_atom_label
    ]
    if missing:
        raise ValueError(
            "Missing atom-level mean 1/r^6 values for linewidth Jacobian "
            "evaluation: "
            + ", ".join(missing)
        )
    return float(
        np.mean(
            [mean_inv_r6_by_atom_label[atom_label] for atom_label in package.atom_labels]
        )
    )
