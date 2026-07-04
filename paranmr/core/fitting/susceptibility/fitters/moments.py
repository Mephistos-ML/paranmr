# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Gaussian-mixture helpers and fit engine for susceptibility moments."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.optimize._optimize import OptimizeResult

from paranmr.core.domain.mol import Nucleus
from paranmr.core.fitting.susceptibility.moments.descriptors import (
    normalize_gaussian_mixture_moment_vectors,
)
from paranmr.core.fitting.susceptibility.moments.forward import (
    calculated_moments_from_parameters,
)
from paranmr.core.fitting.susceptibility.linewidths import (
    SusceptibilityLinewidthInputs,
    predict_r6_widths_by_atom_label,
)
from paranmr.core.fitting.susceptibility.models.base import SusceptibilityModel
from paranmr.core.fitting.susceptibility.objectives.moments.api import (
    active_moment_objective_mask,
    apply_moment_objective,
    count_active_moment_objective_residuals,
)
from paranmr.core.fitting.susceptibility.stats import svd_stdev

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MomentFitResult:
    """Structured result for a completed moment fit."""

    temperature: float
    objective_type: str
    observed_moments: dict[str, float]
    calculated_moments: dict[str, float]
    linewidth_method: str
    linewidth_vars_by_name: dict[str, float]
    calculated_linewidths_by_label: dict[str, float]
    weighted_score: float


@dataclass(frozen=True)
class MomentFitInputs:
    """Numeric inputs required to fit a susceptibility model by moments."""

    model: SusceptibilityModel
    nuclei: tuple[Nucleus, ...]
    temperature: float
    observed_moments: dict[str, float]
    moment_objective_state: dict
    linewidth_inputs: SusceptibilityLinewidthInputs
    linewidth_fit_names: tuple[str, ...]
    linewidth_fix_vars: dict[str, float]
    fit_var_names: tuple[str, ...]
    fit_guess: list[float]
    fit_bounds: NDArray[np.float64]
    use_diamagnetic: bool


def fit_moment_model(
    inputs: MomentFitInputs,
    verbose: bool = True,
) -> MomentFitResult | None:
    """Optimize a susceptibility model against moment constraints."""

    # Solve the constrained least-squares problem over susceptibility and linewidth variables.
    curr_fit = least_squares(
        fun=_moment_residual_from_float_list,
        args=(inputs,),
        x0=inputs.fit_guess,
        bounds=inputs.fit_bounds,
        jac="3-point",
    )

    # Persist the fitted temperature on the mutable model object.
    model = inputs.model
    model.temperature = float(inputs.temperature)
    n_susc_params = len(inputs.fit_var_names)

    # Capture only the susceptibility parameters from the optimizer vector.
    curr_fit_dict = {
        name: value
        for name, value in zip(inputs.fit_var_names, curr_fit.x[:n_susc_params])
    }

    # Return a failed fit with NaN diagnostics when the optimizer exhausts iterations.
    if curr_fit.status == 0:
        if verbose:
            logger.warning(
                "Moment fit at %s K failed - Too many iterations",
                model.temperature,
            )
        model.final_var_values = copy.deepcopy(curr_fit_dict)
        model.fit_stdev = {label: np.nan for label in inputs.fit_var_names}
        model.fit_status = False
        model.mae = np.nan
        model.rmse = np.nan
        model.r2 = np.nan
        model.adj_r2 = np.nan
        return None

    # Estimate parameter uncertainty only when there are enough active residuals.
    effective_residual_count = count_active_moment_objective_residuals(
        inputs.moment_objective_state
    )
    if effective_residual_count <= curr_fit.x.size:
        model.fit_stdev = {label: np.nan for label in inputs.fit_var_names}
    else:
        # Build the active Jacobian subset for SVD-based uncertainty estimation.
        active_mask = active_moment_objective_mask(inputs.moment_objective_state)
        active_fit = OptimizeResult(
            fun=np.asarray(curr_fit.fun, dtype=float)[active_mask],
            jac=np.asarray(curr_fit.jac, dtype=float)[active_mask, :],
            x=curr_fit.x,
        )
        stdev, _ = svd_stdev(active_fit)
        model.fit_stdev = {
            label: val
            for label, val in zip(inputs.fit_var_names, stdev[:n_susc_params])
        }

    # Write fitted values back onto the model and finalize derived state.
    model.fit_status = True
    model.final_var_values = copy.deepcopy(curr_fit_dict)
    for key, val in model.fix_vars.items():
        model.final_var_values[key] = val
    model._post_fit()

    # Compute fit quality metrics directly from the final residual vector.
    residual_values = np.asarray(curr_fit.fun, dtype=float)
    model.mae = float(np.sum(np.abs(residual_values)) / len(residual_values))
    ss_res = float(np.sum(residual_values**2))
    model.rmse = float(np.sqrt(ss_res / len(residual_values)))
    model.r2 = np.nan
    model.adj_r2 = np.nan

    # Reconstruct final linewidth values from the fitted optimizer vector.
    final_linewidth_vars = {**inputs.linewidth_fix_vars}
    final_linewidth_vars.update(
        {
            name: value
            for name, value in zip(
                inputs.linewidth_fit_names,
                curr_fit.x[n_susc_params:],
            )
        }
    )

    # Recompute calculated moments using the fitted model parameters.
    final_linewidths_by_atom_label = predict_r6_widths_by_atom_label(
        linewidth_inputs=inputs.linewidth_inputs,
        linewidth_vars_by_name=final_linewidth_vars,
    )
    calculated_moments = calculated_moments_from_parameters(
        model=model,
        parameters=model.final_var_values,
        nuclei=inputs.nuclei,
        linewidths_by_label=final_linewidths_by_atom_label,
        include_diamagnetic=inputs.use_diamagnetic,
    )

    # Package the final comparison between experimental and calculated moments.
    weighted_score = _weighted_moment_score(
        observed_moments=inputs.observed_moments,
        calculated_moments=calculated_moments,
        moment_objective_state=inputs.moment_objective_state,
    )
    return MomentFitResult(
        temperature=float(inputs.temperature),
        objective_type=str(inputs.moment_objective_state["type"]),
        observed_moments={
            k: float(v) for k, v in inputs.observed_moments.items()
        },
        calculated_moments={k: float(v) for k, v in calculated_moments.items()},
        linewidth_method="r6",
        linewidth_vars_by_name={
            k: float(v) for k, v in final_linewidth_vars.items()
        },
        calculated_linewidths_by_label={
            k: float(v) for k, v in final_linewidths_by_atom_label.items()
        },
        weighted_score=weighted_score,
    )


def _moment_residual_from_float_list(
    new_vals: list[float],
    inputs: MomentFitInputs,
) -> list[float]:
    optimizer_values = np.asarray(new_vals, dtype=float)
    n_susc_params = len(inputs.fit_var_names)
    susc_vals = optimizer_values[:n_susc_params]
    linewidth_vals = optimizer_values[n_susc_params:]
    new_fit_vars = {
        name: value for name, value in zip(inputs.fit_var_names, susc_vals)
    }
    all_vars = {**inputs.model.fix_vars, **new_fit_vars}
    linewidth_vars = {
        **inputs.linewidth_fix_vars,
        **{
            name: value
            for name, value in zip(inputs.linewidth_fit_names, linewidth_vals)
        },
    }
    calculated_widths_by_atom_label = predict_r6_widths_by_atom_label(
        linewidth_inputs=inputs.linewidth_inputs,
        linewidth_vars_by_name=linewidth_vars,
    )
    calculated_moments = calculated_moments_from_parameters(
        model=inputs.model,
        parameters=all_vars,
        nuclei=inputs.nuclei,
        linewidths_by_label=calculated_widths_by_atom_label,
        include_diamagnetic=inputs.use_diamagnetic,
    )
    return list(
        _weighted_moment_residuals(
            observed_moments=inputs.observed_moments,
            calculated_moments=calculated_moments,
            moment_objective_state=inputs.moment_objective_state,
        ).values()
    )


def _weighted_moment_residuals(
    *,
    observed_moments: dict[str, float],
    calculated_moments: dict[str, float],
    moment_objective_state: dict,
) -> dict[str, float]:
    objective_observed_moments, objective_calculated_moments = (
        normalize_gaussian_mixture_moment_vectors(
            observed=observed_moments,
            calculated=calculated_moments,
        )
    )
    residuals = {
        moment_name: objective_calculated_moments[moment_name]
        - objective_observed_moments[moment_name]
        for moment_name in objective_observed_moments.keys()
    }
    return apply_moment_objective(residuals, moment_objective_state)


def _weighted_moment_score(
    *,
    observed_moments: dict[str, float],
    calculated_moments: dict[str, float],
    moment_objective_state: dict,
) -> float:
    weighted_residuals = _weighted_moment_residuals(
        observed_moments=observed_moments,
        calculated_moments=calculated_moments,
        moment_objective_state=moment_objective_state,
    )
    weighted_values = np.asarray(list(weighted_residuals.values()), dtype=float)
    return float(np.sqrt(np.sum(weighted_values**2)))
