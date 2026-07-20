# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Gaussian-mixture helpers and fit engine for susceptibility moments."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.optimize._optimize import OptimizeResult

from paranmr.core.domain.mol import Nucleus
from paranmr.core.fitting.susceptibility.jacobian.assembly import (
    build_moment_jacobian,
)
from paranmr.core.fitting.susceptibility.moments.descriptors import (
    build_normalized_moment_vectors,
)
from paranmr.core.fitting.susceptibility.moments.forward import (
    calculated_moments_from_parameters,
)
from paranmr.core.fitting.susceptibility.objectives.moments.conditions import (
    build_moment_condition_vector,
)
from paranmr.core.fitting.susceptibility.linewidths import (
    SusceptibilityLinewidthInputs,
    predict_r6_widths_by_atom_label,
)
from paranmr.core.fitting.susceptibility.models.base import SusceptibilityModel
from paranmr.core.fitting.susceptibility.objectives.moments.gmm.objective import (
    GMMMomentObjective,
)
from paranmr.core.fitting.susceptibility.objectives.moments.gmm.weighting import (
    build_gmm_weighting_matrix,
    estimate_gmm_covariance_from_jacobian,
)
from paranmr.core.fitting.susceptibility.objectives.moments.api import (
    MomentObjective,
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
    score: float


@dataclass(frozen=True)
class MomentFitInputs:
    """Numeric inputs required to fit a susceptibility model by moments."""

    model: SusceptibilityModel
    nuclei: tuple[Nucleus, ...]
    temperature: float
    observed_moments: dict[str, float]
    moment_objective: MomentObjective
    linewidth_inputs: SusceptibilityLinewidthInputs
    linewidth_fit_names: tuple[str, ...]
    linewidth_fix_vars: dict[str, float]
    fit_var_names: tuple[str, ...]
    fit_guess: list[float]
    fit_bounds: NDArray[np.float64]
    use_diamagnetic: bool
    average_labels: tuple[tuple[str, ...], ...]


def fit_moment_model(
    inputs: MomentFitInputs,
    verbose: bool = True,
) -> MomentFitResult | None:
    """Optimize a susceptibility model against moment constraints."""

    curr_fit = _run_moment_least_squares(inputs)
    if inputs.moment_objective.objective_type == "gmm":
        curr_fit = _run_two_step_gmm_fit(
            stage1_fit=curr_fit,
            inputs=inputs,
            verbose=verbose,
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
    active_mask = inputs.moment_objective.active_mask
    effective_residual_count = int(np.sum(active_mask))
    if effective_residual_count <= curr_fit.x.size:
        model.fit_stdev = {label: np.nan for label in inputs.fit_var_names}
    else:
        # Build the active Jacobian subset for SVD-based uncertainty estimation.
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
        average_labels=inputs.average_labels,
    )

    # Package the final comparison between experimental and calculated moments.
    normalized_moments = build_normalized_moment_vectors(
        observed=inputs.observed_moments,
        calculated=calculated_moments,
    )
    condition_vector = build_moment_condition_vector(
        observed_moments=normalized_moments.observed,
        calculated_moments=normalized_moments.calculated,
        moment_names=tuple(inputs.observed_moments.keys()),
    )
    model.mae = float(np.mean(np.abs(condition_vector)))
    model.rmse = float(np.sqrt(np.mean(condition_vector**2)))
    model.r2 = np.nan
    model.adj_r2 = np.nan
    score = _moment_score(condition_vector)
    return MomentFitResult(
        temperature=float(inputs.temperature),
        objective_type=inputs.moment_objective.objective_type,
        observed_moments={
            f"{k}_norm": float(v)
            for k, v in normalized_moments.observed.items()
        },
        calculated_moments={
            f"{k}_norm": float(v)
            for k, v in normalized_moments.calculated.items()
        },
        linewidth_method="r6",
        linewidth_vars_by_name={
            k: float(v) for k, v in final_linewidth_vars.items()
        },
        calculated_linewidths_by_label={
            k: float(v) for k, v in final_linewidths_by_atom_label.items()
        },
        score=score,
    )


def _run_moment_least_squares(inputs: MomentFitInputs) -> OptimizeResult:
    """Run a single least-squares stage for a moment objective."""

    return least_squares(
        fun=_moment_residual_from_float_list,
        args=(inputs,),
        x0=inputs.fit_guess,
        bounds=inputs.fit_bounds,
        jac="3-point",
    )


def _run_two_step_gmm_fit(
    *,
    stage1_fit: OptimizeResult,
    inputs: MomentFitInputs,
    verbose: bool,
) -> OptimizeResult:
    """Upgrade a stage-1 GMM fit from ``W = I`` to ``W = S^{-1}``."""

    if stage1_fit.status == 0:
        return stage1_fit
    if not isinstance(inputs.moment_objective, GMMMomentObjective):
        return stage1_fit

    stage1_parameters = _resolved_susceptibility_parameters(
        fit_vector=stage1_fit.x,
        inputs=inputs,
    )
    stage1_linewidth_vars = _resolved_linewidth_parameters(
        fit_vector=stage1_fit.x,
        inputs=inputs,
    )
    stage1_jacobian = build_moment_jacobian(
        temperature=float(inputs.temperature),
        parameters=stage1_parameters,
        nuclei=list(inputs.nuclei),
        linewidth_inputs=inputs.linewidth_inputs,
        linewidth_vars_by_name=stage1_linewidth_vars,
        observed_moments=inputs.observed_moments,
        parameter_names=tuple(inputs.fit_var_names) + tuple(inputs.linewidth_fit_names),
        average_labels=inputs.average_labels,
    )
    covariance = estimate_gmm_covariance_from_jacobian(stage1_jacobian)
    weighting = build_gmm_weighting_matrix(covariance)
    stage2_objective = GMMMomentObjective.with_weighting_matrix(
        moment_names=tuple(inputs.observed_moments.keys()),
        weighting_matrix=weighting,
    )
    stage2_inputs = replace(
        inputs,
        moment_objective=stage2_objective,
        fit_guess=list(np.asarray(stage1_fit.x, dtype=float)),
    )
    stage2_fit = _run_moment_least_squares(stage2_inputs)
    if stage2_fit.status == 0:
        if verbose:
            logger.warning(
                "Second-stage GMM fit at %s K failed; falling back to the stage-1 solution",
                inputs.temperature,
            )
        return stage1_fit
    return stage2_fit


def _resolved_susceptibility_parameters(
    *,
    fit_vector: NDArray[np.float64],
    inputs: MomentFitInputs,
) -> dict[str, float]:
    """Merge fixed and fitted susceptibility parameters for a fit vector."""

    n_susc_params = len(inputs.fit_var_names)
    fitted = {
        name: float(value)
        for name, value in zip(inputs.fit_var_names, fit_vector[:n_susc_params])
    }
    return {**inputs.model.fix_vars, **fitted}


def _resolved_linewidth_parameters(
    *,
    fit_vector: NDArray[np.float64],
    inputs: MomentFitInputs,
) -> dict[str, float]:
    """Merge fixed and fitted linewidth parameters for a fit vector."""

    n_susc_params = len(inputs.fit_var_names)
    linewidth_values = fit_vector[n_susc_params:]
    fitted = {
        name: float(value)
        for name, value in zip(inputs.linewidth_fit_names, linewidth_values)
    }
    return {**inputs.linewidth_fix_vars, **fitted}


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
        average_labels=inputs.average_labels,
    )
    normalized_moments = build_normalized_moment_vectors(
        observed=inputs.observed_moments,
        calculated=calculated_moments,
    )
    return list(
        _weighted_moment_residuals(
            observed_moments=normalized_moments.observed,
            calculated_moments=normalized_moments.calculated,
            moment_objective=inputs.moment_objective,
        )
    )


def _weighted_moment_residuals(
    *,
    observed_moments: dict[str, float],
    calculated_moments: dict[str, float],
    moment_objective: MomentObjective,
) -> NDArray[np.float64]:
    return moment_objective.residuals(
        observed_moments=observed_moments,
        calculated_moments=calculated_moments,
    )


def _moment_score(condition_vector: NDArray[np.float64]) -> float:
    """Return the standardized score in normalized moment space."""

    return float(np.sqrt(np.sum(np.asarray(condition_vector, dtype=float) ** 2)))
