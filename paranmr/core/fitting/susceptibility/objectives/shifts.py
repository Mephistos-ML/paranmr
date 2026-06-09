# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Shift-based fitting objective for susceptibility models."""

import copy
import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares, lsq_linear

from paranmr.core.domain.exp import Experiment
from paranmr.core.domain.mol import Molecule, Nucleus
from paranmr.core.fitting.susceptibility.stats import svd_stdev

if TYPE_CHECKING:
    from paranmr.core.fitting.susceptibility.models.base import (
        LinearSusceptibilityModel,
        SusceptibilityModel,
    )

logger = logging.getLogger(__name__)


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


def fit_model_to_shifts(
    model: "SusceptibilityModel",
    molecule: Molecule,
    experiment: Experiment,
    verbose: bool = True,
    average_labels: list[list[str]] | None = None,
) -> None:
    """Fit a susceptibility model to assigned experimental shifts.

    Args:
        model: Susceptibility model to mutate with fitted parameters and metrics.
        molecule: Molecule providing nuclei and geometric information.
        experiment: Experimental data object.
        verbose: If ``False``, suppresses terminal warnings.
        average_labels: Optional groups of atom labels whose predicted shifts are
            averaged prior to residual computation.

    Returns:
        None.
    """

    if average_labels is None:
        average_labels = []

    guess = [val for val in model.fit_vars.values()]
    bounds = np.array([model.BOUNDS[name] for name in model.fit_vars.keys()]).T
    al_to_para_shift = {
        nuc.label: experiment[nuc.chem_label].shift - nuc.shift.dia
        for nuc in molecule.nuclei
    }

    curr_fit = least_squares(
        fun=shift_residual_from_float_list,
        args=(
            model,
            model.fit_vars,
            model.fix_vars,
            molecule.nuclei,
            al_to_para_shift,
            average_labels,
        ),
        x0=guess,
        bounds=bounds,
        jac="3-point",
    )

    model.temperature = experiment.temperature
    curr_fit_dict = {
        name: value for name, value in zip(model.fit_vars.keys(), curr_fit.x)
    }

    if curr_fit.status == 0:
        if verbose:
            logger.warning(
                "Fit at %s K failed - Too many iterations", model.temperature
            )
        model.final_var_values = copy.deepcopy(curr_fit_dict)
        model.fit_stdev = {label: np.nan for label in model.fit_vars.keys()}
        model.fit_status = False
        model.mae = np.nan
        model.rmse = np.nan
        model.r2 = np.nan
        model.adj_r2 = np.nan
        return

    stdev, _ = svd_stdev(curr_fit)
    model.fit_stdev = {
        label: val for label, val in zip(model.fit_vars.keys(), stdev)
    }
    model.fit_status = True
    model.final_var_values = copy.deepcopy(curr_fit_dict)
    for key, val in model.fix_vars.items():
        model.final_var_values[key] = val

    model._post_fit()

    model.mae = np.sum(np.abs(curr_fit.fun)) / len(curr_fit.fun)
    ss_res = np.sum(curr_fit.fun**2)
    model.rmse = np.sqrt(ss_res / len(curr_fit.fun))
    ecs = [al_to_para_shift[nuc.label] for nuc in molecule.nuclei]
    ss_tot = np.sum((ecs - np.mean(ecs)) ** 2)
    model.r2 = 1 - (ss_res / ss_tot)
    model.adj_r2 = 1 - (1 - model.r2) * (len(ecs) - 1) / (
        len(ecs) - len(model.fit_vars) - 1
    )


def fit_linear_model_to_shifts(
    model: "LinearSusceptibilityModel",
    molecule: Molecule,
    experiment: Experiment,
    verbose: bool = True,
) -> None:
    """Fit a linear susceptibility model to assigned experimental shifts.

    Args:
        model: Linear susceptibility model to mutate with fitted values and metrics.
        molecule: Molecule providing nuclei and geometric information.
        experiment: Experimental data object.
        verbose: If ``False``, suppresses terminal warnings.

    Returns:
        None.
    """

    bounds = np.array([model.BOUNDS[name] for name in model.fit_vars.keys()]).T
    curr_fit = lsq_linear(
        A=model.design_matrix(molecule.nuclei, model.fix_vars),
        b=model.target_vector(molecule.nuclei, experiment, model.fix_vars),
        bounds=bounds,
    )

    model.temperature = experiment.temperature
    fit_var_names = [name for name in model.VARNAMES if name in model.fit_vars.keys()]
    curr_fit_dict = {name: value for name, value in zip(fit_var_names, curr_fit.x)}

    if curr_fit.status == 0:
        if verbose:
            logger.warning(
                "Fit at %s K failed - Too many iterations", model.temperature
            )
        model.final_var_values = copy.deepcopy(curr_fit_dict)
        model.fit_stdev = {label: np.nan for label in model.fit_vars.keys()}
        model.fit_status = False
        model.rmse = np.nan
        model.r2 = np.nan
        model.adj_r2 = np.nan
        return

    curr_fit.jac = model.design_matrix(molecule.nuclei, model.fix_vars)
    stdev, _ = svd_stdev(curr_fit)
    model.fit_stdev = {
        label: val for label, val in zip(model.fit_vars.keys(), stdev)
    }
    model.fit_status = True
    model.final_var_values = copy.deepcopy(curr_fit_dict)
    for key, val in model.fix_vars.items():
        model.final_var_values[key] = val

    ss_res = np.sum(curr_fit.fun**2)
    model.rmse = np.sqrt(ss_res / len(curr_fit.fun))
    ecs = [experiment[nuc.chem_label] for nuc in molecule.nuclei]
    ss_tot = np.sum((ecs - np.mean(ecs)) ** 2)
    model.r2 = 1 - (ss_res / ss_tot)
    model.adj_r2 = 1 - (1 - model.r2) * (len(ecs) - 1) / (
        len(ecs) - len(model.fit_vars) - 1
    )
