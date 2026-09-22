# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Analytical center derivatives in Cartesian split susceptibility coordinates."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from simpnmr_x.core.domain.mol import Nucleus
from simpnmr_x.core.fitting.susceptibility.jacobian.susceptibility_tensor import (
    differentiate_tensor_by_split_parameter,
)
from simpnmr_x.core.fitting.susceptibility.models.split import SplitFitter
from simpnmr_x.core.fitting.susceptibility.moments.forward import (
    calculated_signal_packages_from_parameters,
    sort_packages_by_center,
)


def differentiate_centers_by_split_parameter(
    *,
    parameter_name: str,
    parameters: dict[str, float],
    nuclei: list[Nucleus],
    average_labels: tuple[tuple[str, ...], ...] = (),
) -> NDArray[np.float64]:
    """Return center derivatives with respect to one split tensor component."""

    d_tensor = differentiate_tensor_by_split_parameter(parameter_name)
    packages = sort_packages_by_center(
        calculated_signal_packages_from_parameters(
            model=SplitFitter,
            parameters=parameters,
            nuclei=nuclei,
            include_diamagnetic=True,
            average_labels=average_labels,
        )
    )
    derivative_by_atom_label = {
        nucleus.label: float(np.trace(d_tensor @ nucleus.A.tensor_full) / 3.0)
        for nucleus in nuclei
    }
    return np.asarray(
        [
            float(
                np.mean(
                    [
                        derivative_by_atom_label[atom_label]
                        for atom_label in package.atom_labels
                    ]
                )
            )
            for package in packages
        ],
        dtype=float,
    )
