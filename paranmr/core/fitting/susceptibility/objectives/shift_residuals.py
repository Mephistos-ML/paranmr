# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Shift residual definitions for susceptibility fitting objectives."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from paranmr.core.domain.mol import Nucleus

if TYPE_CHECKING:
    from paranmr.core.fitting.susceptibility.models.base import SusceptibilityModel


def shift_residuals(
    model: "SusceptibilityModel",
    parameters: dict[str, float],
    nuclei: list[Nucleus],
    al_to_para_shift: dict[str, float],
    average_labels: list[list[str]] | None = None,
) -> list[float]:
    """Compute residuals between experimental and predicted shifts.

    Args:
        model: Susceptibility model used to predict paramagnetic shifts.
        parameters: Trial parameters used to compute model shifts.
        nuclei: Nuclei for which shifts are computed.
        al_to_para_shift: Mapping from atom label to experimental paramagnetic shift.
        average_labels: Optional groups of atom labels whose predicted shifts are
            averaged prior to residual computation.

    Returns:
        Residuals as experimental minus predicted paramagnetic shifts.
    """

    if average_labels is None:
        average_labels = []

    trial_shifts = model.model(parameters, nuclei)

    weights = {lab: 1.0 for lab in trial_shifts.keys()}
    if average_labels:
        for group in average_labels:
            group_average = np.mean([trial_shifts[lab] for lab in group])
            group_size = len(group)
            for lab in group:
                trial_shifts[lab] = group_average
                weights[lab] = np.sqrt(group_size)

    return [
        (exp_shift - trial_shifts[atom_label]) / weights.get(atom_label, 1.0)
        for atom_label, exp_shift in al_to_para_shift.items()
    ]


def shift_residual_from_float_list(
    new_vals: list[float],
    model: "SusceptibilityModel",
    fit_vars: dict[str, float],
    fix_vars: dict[str, float],
    nuclei: list[Nucleus],
    al_to_para_shift: dict[str, float],
    average_labels: list[list[str]] | None = None,
) -> list[float]:
    """Adapt shift residuals for optimizers that pass a flat float list.

    Args:
        new_vals: New values provided by the optimizer in fit variable order.
        model: Susceptibility model used to predict paramagnetic shifts.
        fit_vars: Fit-variable template mapping names to initial guesses.
        fix_vars: Fixed parameters that remain constant during fitting.
        nuclei: Nuclei for which shifts are computed.
        al_to_para_shift: Mapping from atom label to experimental paramagnetic shift.
        average_labels: Optional groups of atom labels whose predicted shifts are
            averaged prior to residual computation.

    Returns:
        Residual vector for the supplied optimizer values.
    """

    new_fit_vars = {name: guess for guess, name in zip(new_vals, fit_vars.keys())}
    all_vars = {**fix_vars, **new_fit_vars}

    return shift_residuals(
        model=model,
        parameters=all_vars,
        nuclei=nuclei,
        al_to_para_shift=al_to_para_shift,
        average_labels=average_labels,
    )
