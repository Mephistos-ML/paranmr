# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

import numpy as np
import pytest

from simpnmr_x.core.fitting.susceptibility.moments.forward import (
    CalculatedSignalPackage,
    integral_scale_from_calculated_packages,
    observed_moment_data_from_peaks,
)


@pytest.mark.unit
def test_observed_moment_data_sorts_peaks_and_retains_physical_integral():
    observed = observed_moment_data_from_peaks(
        centers=np.asarray([2.0, -1.0]),
        fwhm=np.asarray([0.6, 0.4]),
        areas=np.asarray([3.0, 1.0]),
        moment_labels=("m1", "m2"),
    )

    assert observed.peaks["center"] == pytest.approx([-1.0, 2.0])
    assert observed.peaks["fwhm"] == pytest.approx([0.4, 0.6])
    assert observed.peaks["area"] == pytest.approx([1.0, 3.0])
    assert observed.moments["m0"] == pytest.approx(4.0)
    assert observed.moments["m1"] == pytest.approx(1.25)


@pytest.mark.unit
def test_integral_scale_uses_calculated_package_areas():
    scale = integral_scale_from_calculated_packages(
        observed_integral=10.0,
        packages=[
            CalculatedSignalPackage("H1", ("H1",), center=1.0, area=2.0),
            CalculatedSignalPackage("H2", ("H2", "H3"), center=2.0, area=3.0),
        ],
    )

    assert scale == pytest.approx(2.0)
