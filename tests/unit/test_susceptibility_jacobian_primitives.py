# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

import numpy as np
import pytest

from paranmr.core.domain.mol import Nucleus
from paranmr.core.domain.tensor import Hyperfine
from paranmr.core.fitting.susceptibility.jacobian.susceptibility_centers import (
    differentiate_centers_by_alpha,
    differentiate_centers_by_beta,
    differentiate_centers_by_gamma,
    differentiate_centers_by_susc_ax,
    differentiate_centers_by_susc_iso,
    differentiate_centers_by_susc_rho_over_ax,
)
from paranmr.core.fitting.susceptibility.jacobian.susceptibility_moments import (
    differentiate_moments_by_alpha,
    differentiate_moments_by_beta,
    differentiate_moments_by_gamma,
    differentiate_moments_by_susc_ax,
    differentiate_moments_by_susc_iso,
    differentiate_moments_by_susc_rho_over_ax,
)
from paranmr.core.fitting.susceptibility.moments.descriptors import (
    compute_gaussian_mixture_moments,
)
from paranmr.core.fitting.susceptibility.moments.forward import (
    calculated_signal_packages_from_parameters,
    package_centers,
    package_linewidths,
    sort_packages_by_center,
)
from paranmr.core.fitting.susceptibility.moments.gaussian import (
    gaussian_peak_representation,
)
from paranmr.core.fitting.susceptibility.models.isoaxrho_euler import (
    IsoAxRhoEulerFitter,
)

MOMENT_LABELS = tuple(f"m{order}" for order in range(1, 7))


def _test_nuclei() -> list[Nucleus]:
    return [
        Nucleus(
            label="H1",
            coord=[0.0, 0.0, 0.0],
            A=Hyperfine(
                tensor_full=np.array(
                    [
                        [1.2, 0.1, 0.0],
                        [0.1, -0.5, 0.2],
                        [0.0, 0.2, 0.7],
                    ],
                    dtype=float,
                )
            ),
        ),
        Nucleus(
            label="H2",
            coord=[1.0, 0.0, 0.0],
            A=Hyperfine(
                tensor_full=np.array(
                    [
                        [-0.3, 0.0, 0.1],
                        [0.0, 0.8, -0.2],
                        [0.1, -0.2, 0.4],
                    ],
                    dtype=float,
                )
            ),
        ),
        Nucleus(
            label="H3",
            coord=[2.0, 0.0, 0.0],
            A=Hyperfine(
                tensor_full=np.array(
                    [
                        [0.5, -0.1, 0.0],
                        [-0.1, 0.2, 0.0],
                        [0.0, 0.0, -0.9],
                    ],
                    dtype=float,
                )
            ),
        ),
    ]


@pytest.mark.unit
def test_differentiate_centers_by_susc_ax_matches_finite_difference_isoaxrho():
    nuclei = _test_nuclei()
    parameters = {
        "iso": 0.0,
        "ax": 0.12,
        "rho_over_ax": 0.08,
    }
    step = 1e-7

    analytical = differentiate_centers_by_susc_ax(
        parameters=parameters,
        nuclei=nuclei,
    )

    def _sorted_centers(ax_value: float) -> np.ndarray:
        packages = sort_packages_by_center(
            calculated_signal_packages_from_parameters(
                model=_IsoModel(),
                parameters={
                    "iso": parameters["iso"],
                    "ax": ax_value,
                    "rho_over_ax": parameters["rho_over_ax"],
                },
                nuclei=nuclei,
                include_diamagnetic=True,
                average_labels=(),
            )
        )
        return np.asarray([package.center for package in packages], dtype=float)

    finite_difference = (
        _sorted_centers(parameters["ax"] + step)
        - _sorted_centers(parameters["ax"] - step)
    ) / (2.0 * step)

    assert analytical == pytest.approx(finite_difference, rel=1e-6, abs=1e-8)


@pytest.mark.unit
def test_differentiate_centers_by_susc_ax_matches_finite_difference_euler():
    nuclei = _test_nuclei()
    parameters = {
        "iso": 0.0,
        "ax": 0.12,
        "rho_over_ax": 0.08,
        "alpha": 25.0,
        "beta": 40.0,
        "gamma": 75.0,
    }
    step = 1e-7

    analytical = differentiate_centers_by_susc_ax(
        parameters=parameters,
        nuclei=nuclei,
    )

    def _sorted_centers(ax_value: float) -> np.ndarray:
        packages = sort_packages_by_center(
            calculated_signal_packages_from_parameters(
                model=_EulerModel(),
                parameters={
                    **parameters,
                    "ax": ax_value,
                },
                nuclei=nuclei,
                include_diamagnetic=True,
                average_labels=(),
            )
        )
        return np.asarray([package.center for package in packages], dtype=float)

    finite_difference = (
        _sorted_centers(parameters["ax"] + step)
        - _sorted_centers(parameters["ax"] - step)
    ) / (2.0 * step)

    assert analytical == pytest.approx(finite_difference, rel=1e-6, abs=1e-8)


@pytest.mark.unit
def test_differentiate_centers_by_susc_ax_averages_grouped_package_centers():
    nuclei = _test_nuclei()
    parameters = {
        "iso": 0.0,
        "ax": 0.12,
        "rho_over_ax": 0.08,
    }
    average_labels = (("H1", "H2"),)

    derivatives = differentiate_centers_by_susc_ax(
        parameters=parameters,
        nuclei=nuclei,
        average_labels=average_labels,
    )

    packages = sort_packages_by_center(
        calculated_signal_packages_from_parameters(
            model=_IsoModel(),
            parameters=parameters,
            nuclei=nuclei,
            include_diamagnetic=True,
            average_labels=average_labels,
        )
    )
    grouped_index = next(
        index
        for index, package in enumerate(packages)
        if package.atom_labels == ("H1", "H2")
    )

    d_tensor_by_ax = np.array(
        [
            [-1.0 / 3.0 + parameters["rho_over_ax"], 0.0, 0.0],
            [0.0, -1.0 / 3.0 - parameters["rho_over_ax"], 0.0],
            [0.0, 0.0, 2.0 / 3.0],
        ],
        dtype=float,
    )
    expected_average = float(
        np.mean(
            [
                np.trace(d_tensor_by_ax @ nucleus.A.tensor_full) / 3.0
                for nucleus in nuclei[:2]
            ]
        )
    )
    assert derivatives[grouped_index] == pytest.approx(expected_average)


@pytest.mark.unit
def test_differentiate_moments_by_susc_ax_matches_finite_difference_isoaxrho():
    nuclei = _test_nuclei()
    parameters = {
        "iso": 0.0,
        "ax": 0.12,
        "rho_over_ax": 0.08,
    }
    linewidths_by_label = {
        "H1": 1.1,
        "H2": 0.9,
        "H3": 1.3,
    }
    step = 1e-7

    analytical = differentiate_moments_by_susc_ax(
        parameters=parameters,
        nuclei=nuclei,
        linewidths_by_label=linewidths_by_label,
        moment_labels=MOMENT_LABELS,
    )

    def _moment_vector(ax_value: float) -> np.ndarray:
        packages = sort_packages_by_center(
            calculated_signal_packages_from_parameters(
                model=_IsoModel(),
                parameters={
                    "iso": parameters["iso"],
                    "ax": ax_value,
                    "rho_over_ax": parameters["rho_over_ax"],
                },
                nuclei=nuclei,
                include_diamagnetic=True,
                average_labels=(),
            )
        )
        peaks = gaussian_peak_representation(
            centers=package_centers(packages),
            fwhm=package_linewidths(packages, linewidths_by_label),
            areas=np.ones(len(packages), dtype=float),
        )
        moments = compute_gaussian_mixture_moments(
            centers=peaks["center"],
            sigmas=peaks["sigma"],
            area_norm=peaks["area_norm"],
            moment_labels=MOMENT_LABELS,
        )
        return np.asarray([moments[name] for name in MOMENT_LABELS], dtype=float)

    finite_difference = (
        _moment_vector(parameters["ax"] + step)
        - _moment_vector(parameters["ax"] - step)
    ) / (2.0 * step)

    assert analytical == pytest.approx(finite_difference, rel=1e-6, abs=1e-8)


@pytest.mark.unit
def test_differentiate_moments_by_susc_ax_matches_finite_difference_euler():
    nuclei = _test_nuclei()
    parameters = {
        "iso": 0.0,
        "ax": 0.12,
        "rho_over_ax": 0.08,
        "alpha": 25.0,
        "beta": 40.0,
        "gamma": 75.0,
    }
    linewidths_by_label = {
        "H1": 1.1,
        "H2": 0.9,
        "H3": 1.3,
    }
    step = 1e-7

    analytical = differentiate_moments_by_susc_ax(
        parameters=parameters,
        nuclei=nuclei,
        linewidths_by_label=linewidths_by_label,
        moment_labels=MOMENT_LABELS,
    )

    def _moment_vector(ax_value: float) -> np.ndarray:
        packages = sort_packages_by_center(
            calculated_signal_packages_from_parameters(
                model=_EulerModel(),
                parameters={
                    **parameters,
                    "ax": ax_value,
                },
                nuclei=nuclei,
                include_diamagnetic=True,
                average_labels=(),
            )
        )
        peaks = gaussian_peak_representation(
            centers=package_centers(packages),
            fwhm=package_linewidths(packages, linewidths_by_label),
            areas=np.ones(len(packages), dtype=float),
        )
        moments = compute_gaussian_mixture_moments(
            centers=peaks["center"],
            sigmas=peaks["sigma"],
            area_norm=peaks["area_norm"],
            moment_labels=MOMENT_LABELS,
        )
        return np.asarray([moments[name] for name in MOMENT_LABELS], dtype=float)

    finite_difference = (
        _moment_vector(parameters["ax"] + step)
        - _moment_vector(parameters["ax"] - step)
    ) / (2.0 * step)

    assert analytical == pytest.approx(finite_difference, rel=1e-6, abs=1e-8)


@pytest.mark.unit
def test_differentiate_centers_by_susc_rho_over_ax_matches_finite_difference_isoaxrho():
    nuclei = _test_nuclei()
    parameters = {
        "iso": 0.0,
        "ax": 0.12,
        "rho_over_ax": 0.08,
    }
    step = 1e-7

    analytical = differentiate_centers_by_susc_rho_over_ax(
        parameters=parameters,
        nuclei=nuclei,
    )

    def _sorted_centers(rho_value: float) -> np.ndarray:
        packages = sort_packages_by_center(
            calculated_signal_packages_from_parameters(
                model=_IsoModel(),
                parameters={
                    "iso": parameters["iso"],
                    "ax": parameters["ax"],
                    "rho_over_ax": rho_value,
                },
                nuclei=nuclei,
                include_diamagnetic=True,
                average_labels=(),
            )
        )
        return np.asarray([package.center for package in packages], dtype=float)

    finite_difference = (
        _sorted_centers(parameters["rho_over_ax"] + step)
        - _sorted_centers(parameters["rho_over_ax"] - step)
    ) / (2.0 * step)

    assert analytical == pytest.approx(finite_difference, rel=1e-6, abs=1e-8)


@pytest.mark.unit
def test_differentiate_centers_by_susc_rho_over_ax_matches_finite_difference_euler():
    nuclei = _test_nuclei()
    parameters = {
        "iso": 0.0,
        "ax": 0.12,
        "rho_over_ax": 0.08,
        "alpha": 25.0,
        "beta": 40.0,
        "gamma": 75.0,
    }
    step = 1e-7

    analytical = differentiate_centers_by_susc_rho_over_ax(
        parameters=parameters,
        nuclei=nuclei,
    )

    def _sorted_centers(rho_value: float) -> np.ndarray:
        packages = sort_packages_by_center(
            calculated_signal_packages_from_parameters(
                model=_EulerModel(),
                parameters={
                    **parameters,
                    "rho_over_ax": rho_value,
                },
                nuclei=nuclei,
                include_diamagnetic=True,
                average_labels=(),
            )
        )
        return np.asarray([package.center for package in packages], dtype=float)

    finite_difference = (
        _sorted_centers(parameters["rho_over_ax"] + step)
        - _sorted_centers(parameters["rho_over_ax"] - step)
    ) / (2.0 * step)

    assert analytical == pytest.approx(finite_difference, rel=1e-6, abs=1e-8)


@pytest.mark.unit
def test_differentiate_centers_by_susc_iso_matches_finite_difference_isoaxrho():
    nuclei = _test_nuclei()
    parameters = {
        "iso": 0.0,
        "ax": 0.12,
        "rho_over_ax": 0.08,
    }
    step = 1e-7

    analytical = differentiate_centers_by_susc_iso(
        parameters=parameters,
        nuclei=nuclei,
    )

    def _sorted_centers(iso_value: float) -> np.ndarray:
        packages = sort_packages_by_center(
            calculated_signal_packages_from_parameters(
                model=_IsoModel(),
                parameters={
                    "iso": iso_value,
                    "ax": parameters["ax"],
                    "rho_over_ax": parameters["rho_over_ax"],
                },
                nuclei=nuclei,
                include_diamagnetic=True,
                average_labels=(),
            )
        )
        return np.asarray([package.center for package in packages], dtype=float)

    finite_difference = (
        _sorted_centers(parameters["iso"] + step)
        - _sorted_centers(parameters["iso"] - step)
    ) / (2.0 * step)

    assert analytical == pytest.approx(finite_difference, rel=1e-6, abs=1e-8)


@pytest.mark.unit
def test_differentiate_moments_by_susc_rho_over_ax_matches_finite_difference_isoaxrho():
    nuclei = _test_nuclei()
    parameters = {
        "iso": 0.0,
        "ax": 0.12,
        "rho_over_ax": 0.08,
    }
    linewidths_by_label = {
        "H1": 1.1,
        "H2": 0.9,
        "H3": 1.3,
    }
    step = 1e-7

    analytical = differentiate_moments_by_susc_rho_over_ax(
        parameters=parameters,
        nuclei=nuclei,
        linewidths_by_label=linewidths_by_label,
        moment_labels=MOMENT_LABELS,
    )

    def _moment_vector(rho_value: float) -> np.ndarray:
        packages = sort_packages_by_center(
            calculated_signal_packages_from_parameters(
                model=_IsoModel(),
                parameters={
                    "iso": parameters["iso"],
                    "ax": parameters["ax"],
                    "rho_over_ax": rho_value,
                },
                nuclei=nuclei,
                include_diamagnetic=True,
                average_labels=(),
            )
        )
        peaks = gaussian_peak_representation(
            centers=package_centers(packages),
            fwhm=package_linewidths(packages, linewidths_by_label),
            areas=np.ones(len(packages), dtype=float),
        )
        moments = compute_gaussian_mixture_moments(
            centers=peaks["center"],
            sigmas=peaks["sigma"],
            area_norm=peaks["area_norm"],
            moment_labels=MOMENT_LABELS,
        )
        return np.asarray([moments[name] for name in MOMENT_LABELS], dtype=float)

    finite_difference = (
        _moment_vector(parameters["rho_over_ax"] + step)
        - _moment_vector(parameters["rho_over_ax"] - step)
    ) / (2.0 * step)

    assert analytical == pytest.approx(finite_difference, rel=1e-6, abs=1e-8)


@pytest.mark.unit
def test_differentiate_moments_by_susc_rho_over_ax_matches_finite_difference_euler():
    nuclei = _test_nuclei()
    parameters = {
        "iso": 0.0,
        "ax": 0.12,
        "rho_over_ax": 0.08,
        "alpha": 25.0,
        "beta": 40.0,
        "gamma": 75.0,
    }
    linewidths_by_label = {
        "H1": 1.1,
        "H2": 0.9,
        "H3": 1.3,
    }
    step = 1e-7

    analytical = differentiate_moments_by_susc_rho_over_ax(
        parameters=parameters,
        nuclei=nuclei,
        linewidths_by_label=linewidths_by_label,
        moment_labels=MOMENT_LABELS,
    )

    def _moment_vector(rho_value: float) -> np.ndarray:
        packages = sort_packages_by_center(
            calculated_signal_packages_from_parameters(
                model=_EulerModel(),
                parameters={
                    **parameters,
                    "rho_over_ax": rho_value,
                },
                nuclei=nuclei,
                include_diamagnetic=True,
                average_labels=(),
            )
        )
        peaks = gaussian_peak_representation(
            centers=package_centers(packages),
            fwhm=package_linewidths(packages, linewidths_by_label),
            areas=np.ones(len(packages), dtype=float),
        )
        moments = compute_gaussian_mixture_moments(
            centers=peaks["center"],
            sigmas=peaks["sigma"],
            area_norm=peaks["area_norm"],
            moment_labels=MOMENT_LABELS,
        )
        return np.asarray([moments[name] for name in MOMENT_LABELS], dtype=float)

    finite_difference = (
        _moment_vector(parameters["rho_over_ax"] + step)
        - _moment_vector(parameters["rho_over_ax"] - step)
    ) / (2.0 * step)

    assert analytical == pytest.approx(finite_difference, rel=1e-6, abs=1e-8)


@pytest.mark.unit
def test_differentiate_moments_by_susc_iso_matches_finite_difference_isoaxrho():
    nuclei = _test_nuclei()
    parameters = {
        "iso": 0.0,
        "ax": 0.12,
        "rho_over_ax": 0.08,
    }
    linewidths_by_label = {
        "H1": 1.1,
        "H2": 0.9,
        "H3": 1.3,
    }
    step = 1e-7

    analytical = differentiate_moments_by_susc_iso(
        parameters=parameters,
        nuclei=nuclei,
        linewidths_by_label=linewidths_by_label,
        moment_labels=MOMENT_LABELS,
    )

    def _moment_vector(iso_value: float) -> np.ndarray:
        packages = sort_packages_by_center(
            calculated_signal_packages_from_parameters(
                model=_IsoModel(),
                parameters={
                    "iso": iso_value,
                    "ax": parameters["ax"],
                    "rho_over_ax": parameters["rho_over_ax"],
                },
                nuclei=nuclei,
                include_diamagnetic=True,
                average_labels=(),
            )
        )
        peaks = gaussian_peak_representation(
            centers=package_centers(packages),
            fwhm=package_linewidths(packages, linewidths_by_label),
            areas=np.ones(len(packages), dtype=float),
        )
        moments = compute_gaussian_mixture_moments(
            centers=peaks["center"],
            sigmas=peaks["sigma"],
            area_norm=peaks["area_norm"],
            moment_labels=MOMENT_LABELS,
        )
        return np.asarray([moments[name] for name in MOMENT_LABELS], dtype=float)

    finite_difference = (
        _moment_vector(parameters["iso"] + step)
        - _moment_vector(parameters["iso"] - step)
    ) / (2.0 * step)

    assert analytical == pytest.approx(finite_difference, rel=1e-6, abs=1e-8)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("parameter_name", "helper"),
    [
        ("alpha", differentiate_centers_by_alpha),
        ("beta", differentiate_centers_by_beta),
        ("gamma", differentiate_centers_by_gamma),
    ],
)
def test_differentiate_centers_by_euler_angle_matches_finite_difference(
    parameter_name, helper
):
    nuclei = _test_nuclei()
    parameters = {
        "iso": 0.0,
        "ax": 0.12,
        "rho_over_ax": 0.08,
        "alpha": 25.0,
        "beta": 40.0,
        "gamma": 75.0,
    }
    step = 1e-7

    analytical = helper(
        parameters=parameters,
        nuclei=nuclei,
    )

    def _sorted_centers(angle_value: float) -> np.ndarray:
        shifted = {**parameters, parameter_name: angle_value}
        packages = sort_packages_by_center(
            calculated_signal_packages_from_parameters(
                model=_EulerModel(),
                parameters=shifted,
                nuclei=nuclei,
                include_diamagnetic=True,
                average_labels=(),
            )
        )
        return np.asarray([package.center for package in packages], dtype=float)

    finite_difference = (
        _sorted_centers(parameters[parameter_name] + step)
        - _sorted_centers(parameters[parameter_name] - step)
    ) / (2.0 * step)

    assert analytical == pytest.approx(finite_difference, rel=1e-6, abs=1e-8)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("parameter_name", "helper"),
    [
        ("alpha", differentiate_moments_by_alpha),
        ("beta", differentiate_moments_by_beta),
        ("gamma", differentiate_moments_by_gamma),
    ],
)
def test_differentiate_moments_by_euler_angle_matches_finite_difference(
    parameter_name, helper
):
    nuclei = _test_nuclei()
    parameters = {
        "iso": 0.0,
        "ax": 0.12,
        "rho_over_ax": 0.08,
        "alpha": 25.0,
        "beta": 40.0,
        "gamma": 75.0,
    }
    linewidths_by_label = {
        "H1": 1.1,
        "H2": 0.9,
        "H3": 1.3,
    }
    step = 1e-7

    analytical = helper(
        parameters=parameters,
        nuclei=nuclei,
        linewidths_by_label=linewidths_by_label,
        moment_labels=MOMENT_LABELS,
    )

    def _moment_vector(angle_value: float) -> np.ndarray:
        shifted = {**parameters, parameter_name: angle_value}
        packages = sort_packages_by_center(
            calculated_signal_packages_from_parameters(
                model=_EulerModel(),
                parameters=shifted,
                nuclei=nuclei,
                include_diamagnetic=True,
                average_labels=(),
            )
        )
        peaks = gaussian_peak_representation(
            centers=package_centers(packages),
            fwhm=package_linewidths(packages, linewidths_by_label),
            areas=np.ones(len(packages), dtype=float),
        )
        moments = compute_gaussian_mixture_moments(
            centers=peaks["center"],
            sigmas=peaks["sigma"],
            area_norm=peaks["area_norm"],
            moment_labels=MOMENT_LABELS,
        )
        return np.asarray([moments[name] for name in MOMENT_LABELS], dtype=float)

    finite_difference = (
        _moment_vector(parameters[parameter_name] + step)
        - _moment_vector(parameters[parameter_name] - step)
    ) / (2.0 * step)

    assert analytical == pytest.approx(finite_difference, rel=1e-6, abs=1e-8)


class _IsoModel:
    @staticmethod
    def model(parameters, nuclei):
        tensor = np.array(
            [
                [
                    -parameters["ax"] / 3.0
                    + parameters["rho_over_ax"] * parameters["ax"],
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    -parameters["ax"] / 3.0
                    - parameters["rho_over_ax"] * parameters["ax"],
                    0.0,
                ],
                [0.0, 0.0, 2.0 / 3.0 * parameters["ax"]],
            ],
            dtype=float,
        )
        tensor += np.eye(3) * parameters["iso"]
        return {
            nucleus.label: float(np.trace(tensor @ nucleus.A.tensor_full) / 3.0)
            for nucleus in nuclei
        }


class _EulerModel:
    @staticmethod
    def model(parameters, nuclei):
        return IsoAxRhoEulerFitter.model(parameters, nuclei)
