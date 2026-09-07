# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Assembly helpers for normalized moment Jacobian matrices."""

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
from paranmr.core.fitting.susceptibility.jacobian.types import MomentJacobianResult
from paranmr.core.fitting.susceptibility.linewidths import (
    SusceptibilityLinewidthInputs,
    predict_r6_widths_by_atom_label,
)


def build_moment_jacobian(
    *,
    temperature: float,
    parameters: dict[str, float],
    nuclei: list[Nucleus],
    linewidth_inputs: SusceptibilityLinewidthInputs,
    linewidth_vars_by_name: dict[str, float],
    observed_moments: dict[str, float],
    parameter_names: tuple[str, ...],
    average_labels: tuple[tuple[str, ...], ...] = (),
) -> MomentJacobianResult:
    """Build the normalized moment Jacobian for the active fit parameters."""

    raw_jacobian = _build_raw_moment_jacobian(
        temperature=temperature,
        parameters=parameters,
        nuclei=nuclei,
        linewidth_inputs=linewidth_inputs,
        linewidth_vars_by_name=linewidth_vars_by_name,
        moment_names=tuple(observed_moments.keys()),
        parameter_names=parameter_names,
        average_labels=average_labels,
    )
    return _normalize_raw_moment_jacobian(
        jacobian=raw_jacobian,
        observed_moments=observed_moments,
    )


def _build_raw_moment_jacobian(
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
    """Build the raw moment Jacobian for the requested active parameter set."""

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
        moment_labels=moment_names,
    )

    derivatives_by_parameter = {
        "p1": linewidth_derivatives[:, 0],
        "p2": linewidth_derivatives[:, 1],
        "iso": differentiate_moments_by_susc_iso(
            parameters=parameters,
            nuclei=nuclei,
            linewidths_by_label=linewidths_by_label,
            moment_labels=moment_names,
            average_labels=average_labels,
        ),
        "ax": differentiate_moments_by_susc_ax(
            parameters=parameters,
            nuclei=nuclei,
            linewidths_by_label=linewidths_by_label,
            moment_labels=moment_names,
            average_labels=average_labels,
        ),
        "rho_over_ax": differentiate_moments_by_susc_rho_over_ax(
            parameters=parameters,
            nuclei=nuclei,
            linewidths_by_label=linewidths_by_label,
            moment_labels=moment_names,
            average_labels=average_labels,
        ),
        "alpha": differentiate_moments_by_alpha(
            parameters=parameters,
            nuclei=nuclei,
            linewidths_by_label=linewidths_by_label,
            moment_labels=moment_names,
            average_labels=average_labels,
        ),
        "beta": differentiate_moments_by_beta(
            parameters=parameters,
            nuclei=nuclei,
            linewidths_by_label=linewidths_by_label,
            moment_labels=moment_names,
            average_labels=average_labels,
        ),
        "gamma": differentiate_moments_by_gamma(
            parameters=parameters,
            nuclei=nuclei,
            linewidths_by_label=linewidths_by_label,
            moment_labels=moment_names,
            average_labels=average_labels,
        ),
    }
    missing_parameters = [
        name for name in parameter_names if name not in derivatives_by_parameter
    ]
    if missing_parameters:
        raise ValueError(
            "Moment Jacobian is missing derivative builders for parameter(s): "
            + ", ".join(missing_parameters)
        )

    values = np.column_stack([derivatives_by_parameter[name] for name in parameter_names])

    return MomentJacobianResult(
        temperature=float(temperature),
        moment_names=moment_names,
        parameter_names=parameter_names,
        values=values,
    )


def _normalize_raw_moment_jacobian(
    *,
    jacobian: MomentJacobianResult,
    observed_moments: dict[str, float],
) -> MomentJacobianResult:
    """Return the Jacobian of normalized calculated moments."""

    missing = [name for name in jacobian.moment_names if name not in observed_moments]
    if missing:
        raise ValueError(
            "Cannot normalize moment Jacobian without observed moments for: "
            + ", ".join(missing)
        )

    scales = np.asarray(
        [float(observed_moments[name]) for name in jacobian.moment_names],
        dtype=float,
    )
    zero_like = [
        name
        for name, scale in zip(jacobian.moment_names, scales)
        if np.isclose(scale, 0.0, atol=1e-12, rtol=0.0)
    ]
    if zero_like:
        raise ValueError(
            "Cannot normalize moment Jacobian by observed moment values "
            "that are zero or too close to zero: "
            + ", ".join(zero_like)
        )
    normalized_values = np.asarray(jacobian.values, dtype=float) / scales[:, None]
    return MomentJacobianResult(
        temperature=float(jacobian.temperature),
        moment_names=tuple(jacobian.moment_names),
        parameter_names=tuple(jacobian.parameter_names),
        values=normalized_values,
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
