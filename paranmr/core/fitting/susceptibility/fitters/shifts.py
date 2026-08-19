# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Shift-based susceptibility fitting workflows."""

import copy
import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares, lsq_linear

from paranmr.core.domain.exp import Experiment
from paranmr.core.domain.mol import Molecule
from paranmr.core.fitting.susceptibility.objectives.shifts.residuals import (
    shift_residual_from_float_list,
)
from paranmr.core.fitting.susceptibility.stats import svd_stdev

if TYPE_CHECKING:
    from paranmr.core.fitting.susceptibility.models.base import (
        LinearSusceptibilityModel,
        SusceptibilityModel,
    )

logger = logging.getLogger(__name__)


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
    try:
        al_to_para_shift = {
            nuc.label: experiment[nuc.signal_label].shift - nuc.shift.dia
            for nuc in molecule.nuclei
        }
    except KeyError as err:
        missing_label = err.args[0]
        raise ValueError(
            "Assignment-based susceptibility fitting could not match a molecular "
            f"signal label to the experiment: {missing_label!r}. "
            "Check that 'signal_labels:file' is provided for fixed/permute/"
            "hungarian assignment methods and that the resulting signal labels "
            "match the experimental CSV 'signal_label' entries."
        ) from err

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
    ecs = [experiment[nuc.signal_label] for nuc in molecule.nuclei]
    ss_tot = np.sum((ecs - np.mean(ecs)) ** 2)
    model.r2 = 1 - (ss_res / ss_tot)
    model.adj_r2 = 1 - (1 - model.r2) * (len(ecs) - 1) / (
        len(ecs) - len(model.fit_vars) - 1
    )
