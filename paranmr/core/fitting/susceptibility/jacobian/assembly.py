# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Assembly helpers for raw moment Jacobian matrices."""

from __future__ import annotations

import numpy as np

from paranmr.core.domain.mol import Nucleus
from paranmr.core.fitting.susceptibility.jacobian.linewidth import (
    differentiate_moments_by_linewidth_parameters,
)
from paranmr.core.fitting.susceptibility.jacobian.susceptibility_moments import (
    differentiate_moments_by_split_parameter,
)
from paranmr.core.fitting.susceptibility.jacobian.types import MomentJacobianResult
from paranmr.core.fitting.susceptibility.linewidths import (
    SusceptibilityLinewidthInputs,
    predict_r6_widths_by_atom_label,
)
from paranmr.core.fitting.susceptibility.models.split import SplitFitter
from paranmr.core.fitting.susceptibility.moments.forward import (
    calculated_signal_packages_from_parameters,
    sort_packages_by_center,
)


def build_moment_jacobian(
    *,
    temperature: float,
    parameters: dict[str, float],
    nuclei: list[Nucleus],
    linewidth_inputs: SusceptibilityLinewidthInputs,
    linewidth_vars_by_name: dict[str, float],
    moment_names: tuple[str, ...],
    parameter_names: tuple[str, ...],
    average_labels: tuple[tuple[str, ...], ...] = (),
) -> MomentJacobianResult:
    """Build the raw moment Jacobian for the active fit parameters."""

    linewidths_by_label = predict_r6_widths_by_atom_label(
        linewidth_inputs=linewidth_inputs,
        linewidth_vars_by_name=linewidth_vars_by_name,
    )
    packages = sort_packages_by_center(
        calculated_signal_packages_from_parameters(
            model=SplitFitter,
            parameters=parameters,
            nuclei=nuclei,
            include_diamagnetic=True,
            average_labels=average_labels,
        )
    )
    linewidth_derivatives = differentiate_moments_by_linewidth_parameters(
        packages=packages,
        linewidth_inputs=linewidth_inputs,
        linewidth_vars_by_name=linewidth_vars_by_name,
        moment_labels=moment_names,
    )

    derivatives_by_parameter = {
        "p1": linewidth_derivatives[:, 0],
        "p2": linewidth_derivatives[:, 1],
    }
    for name in ("iso", "dxx", "dyy", "dxy", "dxz", "dyz"):
        derivatives_by_parameter[name] = differentiate_moments_by_split_parameter(
            parameter_name=name,
            parameters=parameters,
            nuclei=nuclei,
            linewidths_by_label=linewidths_by_label,
            moment_labels=moment_names,
            average_labels=average_labels,
        )
    missing_parameters = [
        name for name in parameter_names if name not in derivatives_by_parameter
    ]
    if missing_parameters:
        raise ValueError(
            "Moment Jacobian is missing derivative builders for parameter(s): "
            + ", ".join(missing_parameters)
        )

    values = np.column_stack(
        [derivatives_by_parameter[name] for name in parameter_names]
    )

    return MomentJacobianResult(
        temperature=float(temperature),
        moment_names=moment_names,
        parameter_names=parameter_names,
        values=values,
    )
