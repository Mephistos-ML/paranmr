# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Forward representations for susceptibility moment fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from paranmr.core.domain.mol import Nucleus
from paranmr.core.fitting.susceptibility.moments.descriptors import (
    compute_gaussian_mixture_moments,
)
from paranmr.core.fitting.susceptibility.moments.gaussian import (
    gaussian_peak_representation,
)


@dataclass(frozen=True)
class CalculatedSignalPackage:
    """Calculated signal package used for assignment-free fitting."""

    label: str
    atom_labels: tuple[str, ...]
    center: float


def calculated_signal_packages_from_parameters(
    model,
    parameters: dict[str, float],
    nuclei: list[Nucleus],
    include_diamagnetic: bool = True,
) -> list[CalculatedSignalPackage]:
    """Compute calculated signal packages from model parameters."""

    trial_shifts = model.model(parameters, nuclei)
    label_to_total_shift = {
        nuc.label: trial_shifts[nuc.label]
        + (nuc.shift.dia if include_diamagnetic else 0.0)
        for nuc in nuclei
    }
    nucleus_by_label = {nuc.label: nuc for nuc in nuclei}

    return [
        CalculatedSignalPackage(
            label=nucleus_by_label[label].label,
            atom_labels=(label,),
            center=float(total_shift),
        )
        for label, total_shift in label_to_total_shift.items()
    ]


def sort_packages_by_center(
    packages: list[CalculatedSignalPackage],
) -> list[CalculatedSignalPackage]:
    """Return calculated signal packages sorted by center."""

    return sorted(packages, key=lambda package: package.center)


def package_centers(packages: list[CalculatedSignalPackage]) -> NDArray:
    """Return package centers in the existing package order."""

    return np.asarray([package.center for package in packages], dtype=float)


def package_centers_sorted_by_center(
    packages: list[CalculatedSignalPackage],
) -> NDArray:
    """Return calculated package centers sorted in ppm space."""

    return package_centers(sort_packages_by_center(packages))


def package_linewidths(
    packages: list[CalculatedSignalPackage],
    linewidths_by_label: dict[str, float],
) -> NDArray:
    """Return package linewidths in the existing package order."""

    missing = [
        package.label
        for package in packages
        if package.label not in linewidths_by_label
        and not all(label in linewidths_by_label for label in package.atom_labels)
    ]
    if missing:
        raise ValueError(
            "Calculated linewidths are missing for package label(s): "
            + ", ".join(missing)
        )
    return np.asarray(
        [
            _package_linewidth(package, linewidths_by_label)
            for package in packages
        ],
        dtype=float,
    )


def calculated_moments_from_parameters(
    *,
    model,
    parameters: dict[str, float],
    nuclei: list[Nucleus],
    linewidths_by_label: dict[str, float],
    include_diamagnetic: bool,
) -> dict[str, float]:
    """Compute Gaussian-mixture moments for a calculated parameter set."""

    packages = calculated_signal_packages_from_parameters(
        model=model,
        parameters=parameters,
        nuclei=nuclei,
        include_diamagnetic=include_diamagnetic,
    )
    sorted_packages = sort_packages_by_center(packages)
    centers = package_centers(sorted_packages)
    calculated_widths_ppm = package_linewidths(sorted_packages, linewidths_by_label)

    calculated_peaks = gaussian_peak_representation(
        centers=centers,
        fwhm=calculated_widths_ppm,
        areas=np.ones(len(sorted_packages), dtype=float),
    )
    return compute_gaussian_mixture_moments(
        centers=calculated_peaks["center"],
        sigmas=calculated_peaks["sigma"],
        area_norm=calculated_peaks["area_norm"],
    )


def _package_linewidth(
    package: CalculatedSignalPackage,
    linewidths_by_label: dict[str, float],
) -> float:
    if package.label in linewidths_by_label:
        return linewidths_by_label[package.label]
    return float(
        np.mean([linewidths_by_label[label] for label in package.atom_labels])
    )
