# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Assembly helpers for full moment Jacobian matrices."""

from __future__ import annotations

import numpy as np

from paranmr.core.domain.mol import Nucleus
from paranmr.core.fitting.susceptibility.jacobian.linewidth import (
    differentiate_moments_by_linewidth_parameters,
)
from paranmr.core.fitting.susceptibility.jacobian.susceptibility_moments import (
    differentiate_moments_by_alpha,
    differentiate_moments_by_beta,
    differentiate_moments_by_gamma,
    differentiate_moments_by_susc_ax,
    differentiate_moments_by_susc_iso,
    differentiate_moments_by_susc_rho_over_ax,
)
from paranmr.core.fitting.susceptibility.jacobian.types import (
    MOMENT_JACOBIAN_PARAMETER_NAMES,
    MomentJacobianResult,
)
from paranmr.core.fitting.susceptibility.linewidths import (
    SusceptibilityLinewidthInputs,
    predict_r6_widths_by_atom_label,
)
from paranmr.core.fitting.susceptibility.moments.descriptors import MOMENT_NAMES


def build_moment_jacobian(
    *,
    temperature: float,
    parameters: dict[str, float],
    nuclei: list[Nucleus],
    linewidth_inputs: SusceptibilityLinewidthInputs,
    linewidth_vars_by_name: dict[str, float],
    average_labels: tuple[tuple[str, ...], ...] = (),
) -> MomentJacobianResult:
    """Build the full canonical ``6 x 8`` moment Jacobian matrix."""

    linewidths_by_label = predict_r6_widths_by_atom_label(
        linewidth_inputs=linewidth_inputs,
        linewidth_vars_by_name=linewidth_vars_by_name,
    )
    packages = _sorted_packages(
        parameters=parameters,
        nuclei=nuclei,
        average_labels=average_labels,
    )
    linewidth_derivatives = differentiate_moments_by_linewidth_parameters(
        packages=packages,
        linewidth_inputs=linewidth_inputs,
        linewidth_vars_by_name=linewidth_vars_by_name,
    )

    values = np.column_stack(
        [
            linewidth_derivatives[:, 0],
            linewidth_derivatives[:, 1],
            differentiate_moments_by_susc_iso(
                parameters=parameters,
                nuclei=nuclei,
                linewidths_by_label=linewidths_by_label,
                average_labels=average_labels,
            ),
            differentiate_moments_by_susc_ax(
                parameters=parameters,
                nuclei=nuclei,
                linewidths_by_label=linewidths_by_label,
                average_labels=average_labels,
            ),
            differentiate_moments_by_susc_rho_over_ax(
                parameters=parameters,
                nuclei=nuclei,
                linewidths_by_label=linewidths_by_label,
                average_labels=average_labels,
            ),
            differentiate_moments_by_alpha(
                parameters=parameters,
                nuclei=nuclei,
                linewidths_by_label=linewidths_by_label,
                average_labels=average_labels,
            ),
            differentiate_moments_by_beta(
                parameters=parameters,
                nuclei=nuclei,
                linewidths_by_label=linewidths_by_label,
                average_labels=average_labels,
            ),
            differentiate_moments_by_gamma(
                parameters=parameters,
                nuclei=nuclei,
                linewidths_by_label=linewidths_by_label,
                average_labels=average_labels,
            ),
        ]
    )

    return MomentJacobianResult(
        temperature=float(temperature),
        moment_names=MOMENT_NAMES,
        parameter_names=MOMENT_JACOBIAN_PARAMETER_NAMES,
        values=values,
    )


def _sorted_packages(
    *,
    parameters: dict[str, float],
    nuclei: list[Nucleus],
    average_labels: tuple[tuple[str, ...], ...],
):
    from paranmr.core.fitting.susceptibility.jacobian.susceptibility_centers import (
        ShiftOnlyIsoAxRhoModel,
    )
    from paranmr.core.fitting.susceptibility.moments.forward import (
        calculated_signal_packages_from_parameters,
        sort_packages_by_center,
    )

    return sort_packages_by_center(
        calculated_signal_packages_from_parameters(
            model=ShiftOnlyIsoAxRhoModel(),
            parameters=parameters,
            nuclei=nuclei,
            include_diamagnetic=True,
            average_labels=average_labels,
        )
    )
