# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

import numpy as np
import pytest

from paranmr.core.fitting.susceptibility.jacobian.linewidth import (
    differentiate_moments_by_linewidth_parameters,
    differentiate_sigmas_by_linewidth_parameters,
)
from paranmr.core.fitting.susceptibility.linewidths import (
    SusceptibilityLinewidthInputs,
    predict_r6_widths_by_atom_label,
)
from paranmr.core.fitting.susceptibility.moments.descriptors import (
    compute_gaussian_mixture_moments,
)
from paranmr.core.fitting.susceptibility.moments.forward import (
    CalculatedSignalPackage,
    package_centers,
    package_linewidths,
    sort_packages_by_center,
)
from paranmr.core.fitting.susceptibility.moments.gaussian import (
    gaussian_peak_representation,
)
from paranmr.core.spectrum.kernels import gaussian_fwhm_to_sigma

MOMENT_LABELS = tuple(f"m{order}" for order in range(1, 7))


def _package_sigmas(
    *,
    packages: list[CalculatedSignalPackage],
    linewidth_inputs: SusceptibilityLinewidthInputs,
    p1: float,
    p2: float,
) -> np.ndarray:
    linewidths_by_atom_label = predict_r6_widths_by_atom_label(
        linewidth_inputs=linewidth_inputs,
        linewidth_vars_by_name={"p1": p1, "p2": p2},
    )
    package_fwhm = package_linewidths(packages, linewidths_by_atom_label)
    return gaussian_fwhm_to_sigma(package_fwhm)


@pytest.mark.unit
def test_differentiate_sigmas_by_linewidth_parameters_matches_finite_difference():
    packages = sort_packages_by_center(
        [
            CalculatedSignalPackage(
                label="H3",
                atom_labels=("H3", "H4", "H5"),
                center=-2.0,
            ),
            CalculatedSignalPackage(
                label="H1",
                atom_labels=("H1",),
                center=1.0,
            ),
            CalculatedSignalPackage(
                label="H2",
                atom_labels=("H2",),
                center=3.0,
            ),
        ]
    )
    linewidth_inputs = SusceptibilityLinewidthInputs(
        mean_inv_r6_by_atom_label={
            "H1": 2.0,
            "H2": 5.0,
            "H3": 3.0,
            "H4": 4.0,
            "H5": 6.0,
        }
    )
    p1 = 1000.0
    p2 = 0.5
    step = 1e-6

    analytical = differentiate_sigmas_by_linewidth_parameters(
        packages=packages,
        linewidth_inputs=linewidth_inputs,
    )

    finite_difference = np.zeros_like(analytical)
    finite_difference[:, 0] = (
        _package_sigmas(
            packages=packages,
            linewidth_inputs=linewidth_inputs,
            p1=p1 + step,
            p2=p2,
        )
        - _package_sigmas(
            packages=packages,
            linewidth_inputs=linewidth_inputs,
            p1=p1 - step,
            p2=p2,
        )
    ) / (2.0 * step)
    finite_difference[:, 1] = (
        _package_sigmas(
            packages=packages,
            linewidth_inputs=linewidth_inputs,
            p1=p1,
            p2=p2 + step,
        )
        - _package_sigmas(
            packages=packages,
            linewidth_inputs=linewidth_inputs,
            p1=p1,
            p2=p2 - step,
        )
    ) / (2.0 * step)

    assert analytical == pytest.approx(finite_difference, rel=1e-6, abs=1e-7)


@pytest.mark.unit
def test_differentiate_sigmas_by_linewidth_parameters_uses_package_label_precedence():
    packages = [
        CalculatedSignalPackage(
            label="H3",
            atom_labels=("H3", "H4", "H5"),
            center=-2.0,
        )
    ]
    linewidth_inputs = SusceptibilityLinewidthInputs(
        mean_inv_r6_by_atom_label={
            "H3": 3.0,
            "H4": 4.0,
            "H5": 6.0,
        }
    )

    jacobian = differentiate_sigmas_by_linewidth_parameters(
        packages=packages,
        linewidth_inputs=linewidth_inputs,
    )

    factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
    assert jacobian[0, 0] == pytest.approx(3.0 / factor)
    assert jacobian[0, 1] == pytest.approx(1.0 / factor)


@pytest.mark.unit
def test_differentiate_moments_by_linewidth_parameters_matches_finite_difference():
    packages = sort_packages_by_center(
        [
            CalculatedSignalPackage(
                label="H3",
                atom_labels=("H3", "H4", "H5"),
                center=-2.0,
            ),
            CalculatedSignalPackage(
                label="H1",
                atom_labels=("H1",),
                center=1.0,
            ),
            CalculatedSignalPackage(
                label="H2",
                atom_labels=("H2",),
                center=3.0,
            ),
        ]
    )
    linewidth_inputs = SusceptibilityLinewidthInputs(
        mean_inv_r6_by_atom_label={
            "H1": 2.0,
            "H2": 5.0,
            "H3": 3.0,
            "H4": 4.0,
            "H5": 6.0,
        }
    )
    p1 = 1000.0
    p2 = 0.5
    step = 1e-6

    analytical = differentiate_moments_by_linewidth_parameters(
        packages=packages,
        linewidth_inputs=linewidth_inputs,
        linewidth_vars_by_name={"p1": p1, "p2": p2},
        moment_labels=MOMENT_LABELS,
    )

    centers = package_centers(packages)

    def _moment_vector_for_linewidths(*, p1: float, p2: float) -> np.ndarray:
        linewidths_by_label = predict_r6_widths_by_atom_label(
            linewidth_inputs=linewidth_inputs,
            linewidth_vars_by_name={"p1": p1, "p2": p2},
        )
        fwhm = package_linewidths(packages, linewidths_by_label)
        peaks = gaussian_peak_representation(
            centers=centers,
            fwhm=fwhm,
            areas=np.ones(len(packages), dtype=float),
        )
        moments = compute_gaussian_mixture_moments(
            centers=peaks["center"],
            sigmas=peaks["sigma"],
            area_norm=peaks["area_norm"],
            moment_labels=MOMENT_LABELS,
        )
        return np.asarray([moments[name] for name in MOMENT_LABELS], dtype=float)

    finite_difference = np.zeros_like(analytical)
    finite_difference[:, 0] = (
        _moment_vector_for_linewidths(p1=p1 + step, p2=p2)
        - _moment_vector_for_linewidths(p1=p1 - step, p2=p2)
    ) / (2.0 * step)
    finite_difference[:, 1] = (
        _moment_vector_for_linewidths(p1=p1, p2=p2 + step)
        - _moment_vector_for_linewidths(p1=p1, p2=p2 - step)
    ) / (2.0 * step)

    assert analytical == pytest.approx(finite_difference, rel=1e-6, abs=1e-7)
