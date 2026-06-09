# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Moment-based fitting objective for susceptibility models."""

import copy
import logging

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.optimize._optimize import OptimizeResult

from paranmr.core.domain.exp import Experiment
from paranmr.core.domain.mol import Molecule, Nucleus
from paranmr.core.fitting.moments import (
    active_moment_residual_mask,
    apply_moment_weights,
    count_active_moment_residuals,
    gaussian_mixture_moment_residuals,
    gaussian_mixture_moments,
    gaussian_peak_representation,
    moment_residual_norm,
)
from paranmr.core.fitting.stats import svd_stdev

logger = logging.getLogger(__name__)

_PINK_LOG = "\033[95m"
_RESET_LOG = "\033[0m"


def _pink_log(message: str) -> str:
    return f"{_PINK_LOG}{message}{_RESET_LOG}"


def _format_moment_values(values: dict[str, float]) -> str:
    return ", ".join(
        f"{moment_name}={value:.6g}" for moment_name, value in values.items()
    )


def calculated_centers_from_parameters(
    model,
    parameters: dict[str, float],
    nuclei: list[Nucleus],
    average_labels: list[list[str]] = [],
) -> NDArray:
    """Compute sorted total calculated shifts from supplied model parameters."""

    trial_shifts = model.model(parameters, nuclei)
    label_to_total_shift = {
        nuc.label: trial_shifts[nuc.label] + nuc.shift.dia for nuc in nuclei
    }

    used_labels = set()
    centers = []
    for group in average_labels:
        group_values = [label_to_total_shift[label] for label in group]
        centers.append(float(np.mean(group_values)))
        used_labels.update(group)

    centers.extend(
        total_shift
        for label, total_shift in label_to_total_shift.items()
        if label not in used_labels
    )

    centers_arr = np.asarray(centers, dtype=float)
    return np.sort(centers_arr)


def calculated_centers(
    model,
    nuclei: list[Nucleus],
    average_labels: list[list[str]] = [],
) -> NDArray:
    """Compute sorted total calculated shifts from fitted moment parameters."""

    return calculated_centers_from_parameters(
        model=model,
        parameters=model.final_var_values,
        nuclei=nuclei,
        average_labels=average_labels,
    )


def moment_residual_from_float_list(
    new_vals: list[float],
    model,
    fit_vars: dict[str, float],
    fix_vars: dict[str, float],
    nuclei: list[Nucleus],
    average_labels: list[list[str]],
    observed_moments: dict[str, float],
    widths_ppm: NDArray,
    areas: NDArray,
    moment_weights: dict[str, float] | None,
) -> list[float]:
    """Compute moment residuals for optimizer-supplied model parameters."""

    new_fit_vars = {name: guess for guess, name in zip(new_vals, fit_vars.keys())}
    all_vars = {**fix_vars, **new_fit_vars}

    centers = calculated_centers_from_parameters(
        model=model,
        parameters=all_vars,
        nuclei=nuclei,
        average_labels=average_labels,
    )

    calculated_peaks = gaussian_peak_representation(
        centers=centers,
        fwhm=widths_ppm,
        areas=areas,
    )
    calculated_moments = gaussian_mixture_moments(
        centers=calculated_peaks["center"],
        sigmas=calculated_peaks["sigma"],
        area_norm=calculated_peaks["area_norm"],
    )
    residuals = gaussian_mixture_moment_residuals(
        calculated=calculated_moments,
        observed=observed_moments,
        normalize=True,
    )
    weighted_residuals = apply_moment_weights(residuals, moment_weights)

    return list(weighted_residuals.values())


def fit_model_to_moments(
    model,
    molecule: Molecule,
    experiment: Experiment,
    widths_ppm: NDArray,
    areas: NDArray,
    average_labels: list[list[str]] = [],
    moment_weights: dict[str, float] | None = None,
    verbose: bool = True,
) -> None:
    """Fit a susceptibility model by matching Gaussian mixture moments."""

    initial_parameters = {**model.fix_vars, **model.fit_vars}
    initial_centers = calculated_centers_from_parameters(
        model=model,
        parameters=initial_parameters,
        nuclei=molecule.nuclei,
        average_labels=average_labels,
    )

    if len(experiment.signals) != len(initial_centers):
        raise ValueError(
            "Moment fitting currently requires the number of observed peaks "
            "to match the number of calculated peak packages"
        )

    observed_centers = np.asarray(
        [signal.shift for signal in experiment.signals], dtype=float
    )
    widths_arr = np.asarray(widths_ppm, dtype=float)
    areas_arr = np.asarray(areas, dtype=float)

    sort_idx = np.argsort(observed_centers)
    observed_centers = observed_centers[sort_idx]
    widths_arr = widths_arr[sort_idx]
    areas_arr = areas_arr[sort_idx]

    observed_peaks = gaussian_peak_representation(
        centers=observed_centers,
        fwhm=widths_arr,
        areas=areas_arr,
    )
    observed_moments = gaussian_mixture_moments(
        centers=observed_peaks["center"],
        sigmas=observed_peaks["sigma"],
        area_norm=observed_peaks["area_norm"],
    )

    guess = [val for val in model.fit_vars.values()]
    bounds = np.array([model.BOUNDS[name] for name in model.fit_vars.keys()]).T

    curr_fit = least_squares(
        fun=moment_residual_from_float_list,
        args=(
            model,
            model.fit_vars,
            model.fix_vars,
            molecule.nuclei,
            average_labels,
            observed_moments,
            widths_arr,
            areas_arr,
            moment_weights,
        ),
        x0=guess,
        bounds=bounds,
        jac="3-point",
    )

    model.temperature = experiment.temperature
    logger.info(
        _pink_log(
            "Moment fit optimizer at %.4f K: nfev=%d, njev=%s, "
            "status=%d, message=%s"
        ),
        experiment.temperature,
        curr_fit.nfev,
        curr_fit.njev,
        curr_fit.status,
        curr_fit.message,
    )
    curr_fit_dict = {
        name: value for name, value in zip(model.fit_vars.keys(), curr_fit.x)
    }

    if curr_fit.status == 0:
        if verbose:
            logger.warning(
                _pink_log("Moment fit at %s K failed - Too many iterations"),
                model.temperature,
            )
        model.final_var_values = copy.deepcopy(curr_fit_dict)
        model.fit_stdev = {label: np.nan for label in model.fit_vars.keys()}
        model.fit_status = False
        model.mae = np.nan
        model.rmse = np.nan
        model.r2 = np.nan
        model.adj_r2 = np.nan
        return

    effective_residual_count = count_active_moment_residuals(
        observed_moments,
        moment_weights,
    )
    if effective_residual_count <= curr_fit.x.size:
        model.fit_stdev = {label: np.nan for label in model.fit_vars.keys()}
    else:
        active_mask = active_moment_residual_mask(
            observed_moments,
            moment_weights,
        )
        active_fit = OptimizeResult(
            fun=np.asarray(curr_fit.fun, dtype=float)[active_mask],
            jac=np.asarray(curr_fit.jac, dtype=float)[active_mask, :],
            x=curr_fit.x,
        )
        stdev, _ = svd_stdev(active_fit)
        model.fit_stdev = {
            label: val for label, val in zip(model.fit_vars.keys(), stdev)
        }
    model.fit_status = True
    model.final_var_values = copy.deepcopy(curr_fit_dict)
    for key, val in model.fix_vars.items():
        model.final_var_values[key] = val
    model._post_fit()

    residual_values = np.asarray(curr_fit.fun, dtype=float)
    model.mae = float(np.sum(np.abs(residual_values)) / len(residual_values))
    ss_res = float(np.sum(residual_values**2))
    model.rmse = float(np.sqrt(ss_res / len(residual_values)))
    model.r2 = np.nan
    model.adj_r2 = np.nan

    calc_centers = calculated_centers(
        model,
        molecule.nuclei,
        average_labels,
    )
    calculated_peaks = gaussian_peak_representation(
        centers=calc_centers,
        fwhm=widths_arr,
        areas=areas_arr,
    )
    calculated_moments = gaussian_mixture_moments(
        centers=calculated_peaks["center"],
        sigmas=calculated_peaks["sigma"],
        area_norm=calculated_peaks["area_norm"],
    )
    residuals = gaussian_mixture_moment_residuals(
        calculated=calculated_moments,
        observed=observed_moments,
        normalize=True,
    )
    weighted_residuals = apply_moment_weights(residuals, moment_weights)
    score = moment_residual_norm(weighted_residuals)
    unweighted_score = moment_residual_norm(residuals)

    logger.info(
        _pink_log("Moment fit score at %.4f K = %.6g"),
        experiment.temperature,
        score,
    )
    logger.info(
        _pink_log("Observed Gaussian mixture moment vector at %.4f K: %s"),
        experiment.temperature,
        _format_moment_values(observed_moments),
    )
    logger.info(
        _pink_log("Calculated Gaussian mixture moment vector at %.4f K: %s"),
        experiment.temperature,
        _format_moment_values(calculated_moments),
    )
    logger.info(
        _pink_log("Normalized Gaussian mixture moment residuals at %.4f K: %s"),
        experiment.temperature,
        _format_moment_values(residuals),
    )
    logger.info(
        _pink_log("Weighted Gaussian mixture moment residuals at %.4f K: %s"),
        experiment.temperature,
        _format_moment_values(weighted_residuals),
    )
    logger.info(
        _pink_log("Unweighted moment fit score at %.4f K = %.6g"),
        experiment.temperature,
        unweighted_score,
    )
