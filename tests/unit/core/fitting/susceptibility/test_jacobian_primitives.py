# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Finite-difference checks for the Cartesian GMM moment Jacobian."""

import numpy as np
import pytest

from simpnmr_x.core.domain.mol import Nucleus
from simpnmr_x.core.domain.tensor import Hyperfine
from simpnmr_x.core.fitting.susceptibility.jacobian.susceptibility_moments import (
    differentiate_moments_by_split_parameter,
)
from simpnmr_x.core.fitting.susceptibility.models.split import SplitFitter
from simpnmr_x.core.fitting.susceptibility.moments.descriptors import (
    compute_gaussian_mixture_moments,
)
from simpnmr_x.core.fitting.susceptibility.moments.forward import (
    calculated_signal_packages_from_parameters,
    package_areas,
    package_centers,
    package_linewidths,
    sort_packages_by_center,
)
from simpnmr_x.core.fitting.susceptibility.moments.gaussian import (
    gaussian_peak_representation,
)

MOMENT_LABELS = tuple(f"m{order}" for order in range(1, 7))
_PARAMETERS = {
    "iso": 0.04,
    "dxx": 0.12,
    "dyy": -0.08,
    "dxy": 0.03,
    "dxz": -0.02,
    "dyz": 0.05,
}
_LINEWIDTHS = {"H1": 1.1, "H2": 0.9, "H3": 1.3}


def _test_nuclei() -> list[Nucleus]:
    return [
        Nucleus(
            label="H1",
            coord=[0.0, 0.0, 0.0],
            A=Hyperfine(
                tensor_full=np.array(
                    [[1.2, 0.1, 0.0], [0.1, -0.5, 0.2], [0.0, 0.2, 0.7]]
                )
            ),
        ),
        Nucleus(
            label="H2",
            coord=[1.0, 0.0, 0.0],
            A=Hyperfine(
                tensor_full=np.array(
                    [[-0.3, 0.0, 0.1], [0.0, 0.8, -0.2], [0.1, -0.2, 0.4]]
                )
            ),
        ),
        Nucleus(
            label="H3",
            coord=[2.0, 0.0, 0.0],
            A=Hyperfine(
                tensor_full=np.array(
                    [[0.5, -0.1, 0.0], [-0.1, 0.2, 0.0], [0.0, 0.0, -0.9]]
                )
            ),
        ),
    ]


def _split_moments(
    *,
    parameters: dict[str, float],
    nuclei: list[Nucleus],
    average_labels: tuple[tuple[str, ...], ...] = (),
) -> np.ndarray:
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
        fwhm=package_linewidths(packages, _LINEWIDTHS),
        areas=package_areas(packages),
    )
    moments = compute_gaussian_mixture_moments(
        centers=peaks["center"],
        sigmas=peaks["sigma"],
        area_norm=peaks["area_norm"],
        moment_labels=MOMENT_LABELS,
    )
    return np.asarray([moments[name] for name in MOMENT_LABELS])


@pytest.mark.unit
@pytest.mark.parametrize("parameter_name", SplitFitter.VARNAMES)
def test_split_moment_derivative_matches_finite_difference(parameter_name: str):
    nuclei = _test_nuclei()
    step = 1.0e-7
    analytical = differentiate_moments_by_split_parameter(
        parameter_name=parameter_name,
        parameters=_PARAMETERS,
        nuclei=nuclei,
        linewidths_by_label=_LINEWIDTHS,
        moment_labels=MOMENT_LABELS,
    )
    finite_difference = (
        _split_moments(
            parameters={
                **_PARAMETERS,
                parameter_name: _PARAMETERS[parameter_name] + step,
            },
            nuclei=nuclei,
        )
        - _split_moments(
            parameters={
                **_PARAMETERS,
                parameter_name: _PARAMETERS[parameter_name] - step,
            },
            nuclei=nuclei,
        )
    ) / (2.0 * step)

    assert analytical == pytest.approx(finite_difference, rel=1e-6, abs=1e-8)


@pytest.mark.unit
def test_split_moment_derivative_uses_grouped_signal_area():
    nuclei = _test_nuclei()
    average_labels = (("H1", "H2"),)
    step = 1.0e-7
    analytical = differentiate_moments_by_split_parameter(
        parameter_name="dxx",
        parameters=_PARAMETERS,
        nuclei=nuclei,
        linewidths_by_label=_LINEWIDTHS,
        moment_labels=MOMENT_LABELS,
        average_labels=average_labels,
    )
    finite_difference = (
        _split_moments(
            parameters={**_PARAMETERS, "dxx": _PARAMETERS["dxx"] + step},
            nuclei=nuclei,
            average_labels=average_labels,
        )
        - _split_moments(
            parameters={**_PARAMETERS, "dxx": _PARAMETERS["dxx"] - step},
            nuclei=nuclei,
            average_labels=average_labels,
        )
    ) / (2.0 * step)

    assert analytical == pytest.approx(finite_difference, rel=1e-6, abs=1e-8)
