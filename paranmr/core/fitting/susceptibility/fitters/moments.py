# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Moment-based susceptibility fitting workflow."""

import copy
import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.optimize._optimize import OptimizeResult

from paranmr.core.domain.exp import Experiment
from paranmr.core.domain.mol import Molecule, Nucleus
from paranmr.core.fitting.susceptibility.objectives.moments.api import (
    active_moment_objective_mask,
    apply_moment_objective,
    count_active_moment_objective_residuals,
    prepare_moment_objective,
)
from paranmr.core.fitting.susceptibility.moments.descriptors import (
    gaussian_mixture_moment_residuals,
    gaussian_mixture_moments,
)
from paranmr.core.fitting.susceptibility.moments.gaussian import (
    gaussian_peak_representation,
)
from paranmr.core.fitting.susceptibility.stats import svd_stdev
from paranmr.core.fitting.linewidth import predict_r6_linewidths

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalculatedSignalPackage:
    """Calculated independent signal package.

    Args:
        label: Stable package label.
        atom_labels: Atom labels contributing to the package.
        center: Calculated package center in ppm.
    """

    label: str
    atom_labels: tuple[str, ...]
    center: float


@dataclass(frozen=True)
class MomentFitDiagnostics:
    """Publication-grade diagnostics for a completed moment fit."""

    temperature: float
    objective_type: str
    observed_moments: dict[str, float]
    calculated_moments: dict[str, float]
    weighted_score: float


def calculated_signal_packages_from_parameters(
    model,
    parameters: dict[str, float],
    nuclei: list[Nucleus],
    average_labels: list[list[str]] = [],
    include_diamagnetic: bool = True,
) -> list[CalculatedSignalPackage]:
    """Compute calculated independent signal packages from model parameters.

    Packages preserve the calculated-side identity first, then callers may sort
    them by center for assignment-free Gaussian mixture comparisons.
    """

    trial_shifts = model.model(parameters, nuclei)
    label_to_total_shift = {
        nuc.label: trial_shifts[nuc.label]
        + (nuc.shift.dia if include_diamagnetic else 0.0)
        for nuc in nuclei
    }
    nucleus_by_label = {nuc.label: nuc for nuc in nuclei}

    used_labels = set()
    packages = []
    for group in average_labels:
        group_labels = tuple(group)
        group_values = [label_to_total_shift[label] for label in group]
        packages.append(
            CalculatedSignalPackage(
                label=_package_label(group_labels),
                atom_labels=group_labels,
                center=float(np.mean(group_values)),
            )
        )
        used_labels.update(group)

    packages.extend(
        CalculatedSignalPackage(
            label=nucleus_by_label[label].label,
            atom_labels=(label,),
            center=float(total_shift),
        )
        for label, total_shift in label_to_total_shift.items()
        if label not in used_labels
    )

    return packages


def package_centers_sorted_by_center(
    packages: list[CalculatedSignalPackage],
) -> NDArray:
    """Return calculated package centers sorted in ppm space."""

    return package_centers(sort_packages_by_center(packages))


def sort_packages_by_center(
    packages: list[CalculatedSignalPackage],
) -> list[CalculatedSignalPackage]:
    """Return calculated signal packages sorted in ppm space."""

    return sorted(packages, key=lambda package: package.center)


def package_centers(packages: list[CalculatedSignalPackage]) -> NDArray:
    """Return package centers in the existing package order."""

    return np.asarray([package.center for package in packages], dtype=float)


def package_linewidths(
    packages: list[CalculatedSignalPackage],
    linewidths_by_label: dict[str, float],
) -> NDArray:
    """Return package linewidths in the existing package order.

    Args:
        packages: Calculated signal packages.
        linewidths_by_label: Linewidths keyed by package label.

    Returns:
        Linewidths in ppm ordered like `packages`.

    Raises:
        ValueError: If a package has no linewidth.
    """

    missing = [
        package.label
        for package in packages
        if package.label not in linewidths_by_label
        and not all(label in linewidths_by_label for label in package.atom_labels)
    ]
    if missing:
        raise ValueError(
            "Calculated linewidths are missing for package label(s): "
            + ", ".join(missing)
        )
    return np.asarray(
        [
            _package_linewidth(package, linewidths_by_label)
            for package in packages
        ],
        dtype=float,
    )


def _package_label(
    atom_labels: tuple[str, ...],
) -> str:
    if len(atom_labels) == 1:
        return atom_labels[0]
    return "+".join(atom_labels)


def _package_linewidth(
    package: CalculatedSignalPackage,
    linewidths_by_label: dict[str, float],
) -> float:
    if package.label in linewidths_by_label:
        return linewidths_by_label[package.label]
    return float(np.mean([linewidths_by_label[label] for label in package.atom_labels]))


def _split_linewidth_variables(
    variables: dict[str, list[object]] | None,
) -> tuple[dict[str, float], dict[str, float], dict[str, list[float]]]:
    fit_vars: dict[str, float] = {}
    fix_vars: dict[str, float] = {}
    bounds: dict[str, list[float]] = {}
    if variables is None:
        return fit_vars, fix_vars, bounds

    for name, entry in variables.items():
        mode = entry[0]
        value = float(entry[1])
        if mode == "fix":
            fix_vars[name] = value
        elif mode == "fit":
            fit_vars[name] = value
            bounds[name] = [float(entry[2][0]), float(entry[2][1])]
        else:
            raise ValueError(f"Unknown linewidth variable mode {mode!r}")
    return fit_vars, fix_vars, bounds


def _observed_widths_for_calculated_packages(
    observed_widths_ppm: NDArray,
    n_packages: int,
) -> NDArray:
    widths = np.asarray(observed_widths_ppm, dtype=float)
    if widths.size == n_packages:
        return widths
    raise ValueError(
        "Moment fitting cannot reuse experimental linewidths for calculated "
        f"packages when their counts differ: observed={widths.size}, "
        f"calculated={n_packages}. Use a calculated linewidth source such as "
        "'linewidth:method: r6'."
    )


def _calculated_moments_from_parameters(
    *,
    model,
    parameters: dict[str, float],
    nuclei: list[Nucleus],
    average_labels: list[list[str]],
    widths_ppm: NDArray,
    calculated_widths_by_label: dict[str, float] | None,
    linewidth_mean_inv_r6_by_label: dict[str, float] | None,
    linewidth_variables: dict[str, float],
    include_diamagnetic: bool,
) -> dict[str, float]:
    packages = calculated_signal_packages_from_parameters(
        model=model,
        parameters=parameters,
        nuclei=nuclei,
        average_labels=average_labels,
        include_diamagnetic=include_diamagnetic,
    )
    sorted_packages = sort_packages_by_center(packages)
    centers = package_centers(sorted_packages)
    linewidths_by_label = calculated_widths_by_label
    if linewidth_mean_inv_r6_by_label is not None and linewidth_variables:
        linewidths_by_label = predict_r6_linewidths(
            linewidth_mean_inv_r6_by_label,
            linewidth_variables["p1"],
            linewidth_variables["p2"],
        )
    if linewidths_by_label is not None:
        calculated_widths_ppm = package_linewidths(
            sorted_packages,
            linewidths_by_label,
        )
    else:
        calculated_widths_ppm = _observed_widths_for_calculated_packages(
            widths_ppm,
            len(sorted_packages),
        )

    calculated_peaks = gaussian_peak_representation(
        centers=centers,
        fwhm=calculated_widths_ppm,
        areas=np.ones(len(sorted_packages), dtype=float),
    )
    return gaussian_mixture_moments(
        centers=calculated_peaks["center"],
        sigmas=calculated_peaks["sigma"],
        area_norm=calculated_peaks["area_norm"],
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
    calculated_widths_by_label: dict[str, float] | None,
    linewidth_mean_inv_r6_by_label: dict[str, float] | None,
    linewidth_fit_names: list[str],
    linewidth_fix_vars: dict[str, float],
    moment_objective_state: dict,
    include_diamagnetic: bool,
) -> list[float]:
    """Compute moment residuals for optimizer-supplied model parameters."""

    n_susc_params = len(fit_vars)
    susc_vals = new_vals[:n_susc_params]
    linewidth_vals = new_vals[n_susc_params:]
    new_fit_vars = {name: guess for guess, name in zip(susc_vals, fit_vars.keys())}
    all_vars = {**fix_vars, **new_fit_vars}
    linewidth_vars = {
        **linewidth_fix_vars,
        **{
            name: value
            for name, value in zip(linewidth_fit_names, linewidth_vals)
        },
    }

    calculated_moments = _calculated_moments_from_parameters(
        model=model,
        parameters=all_vars,
        nuclei=nuclei,
        average_labels=average_labels,
        widths_ppm=widths_ppm,
        calculated_widths_by_label=calculated_widths_by_label,
        linewidth_mean_inv_r6_by_label=linewidth_mean_inv_r6_by_label,
        linewidth_variables=linewidth_vars,
        include_diamagnetic=include_diamagnetic,
    )
    residuals = gaussian_mixture_moment_residuals(
        calculated=calculated_moments,
        observed=observed_moments,
        normalize=True,
    )
    weighted_residuals = apply_moment_objective(residuals, moment_objective_state)

    return list(weighted_residuals.values())


def fit_model_to_moments(
    model,
    molecule: Molecule,
    experiment: Experiment,
    widths_ppm: NDArray,
    areas: NDArray,
    average_labels: list[list[str]] = [],
    moment_objective: dict | None = None,
    calculated_widths_by_label: dict[str, float] | None = None,
    linewidth_mean_inv_r6_by_label: dict[str, float] | None = None,
    linewidth_variables: dict[str, list[object]] | None = None,
    observed_centers_ppm: NDArray | None = None,
    include_diamagnetic: bool = True,
    verbose: bool = True,
) -> MomentFitDiagnostics | None:
    """Fit a susceptibility model by matching Gaussian mixture moments."""

    if observed_centers_ppm is None:
        observed_centers = np.asarray(
            [signal.shift for signal in experiment.signals], dtype=float
        )
    else:
        observed_centers = np.asarray(observed_centers_ppm, dtype=float)
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
    moment_objective_state = prepare_moment_objective(
        observed_centers=observed_centers,
        widths_ppm=widths_arr,
        areas=areas_arr,
        observed_moments=observed_moments,
        objective_config=moment_objective,
    )

    linewidth_fit_vars, linewidth_fix_vars, linewidth_bounds = (
        _split_linewidth_variables(linewidth_variables)
    )
    guess = [val for val in model.fit_vars.values()]
    guess.extend(linewidth_fit_vars.values())
    susc_bounds = [model.BOUNDS[name] for name in model.fit_vars.keys()]
    bounds_list = susc_bounds + [linewidth_bounds[name] for name in linewidth_fit_vars]
    bounds = np.asarray(bounds_list, dtype=float).T

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
            calculated_widths_by_label,
            linewidth_mean_inv_r6_by_label,
            list(linewidth_fit_vars.keys()),
            linewidth_fix_vars,
            moment_objective_state,
            include_diamagnetic,
        ),
        x0=guess,
        bounds=bounds,
        jac="3-point",
    )

    model.temperature = experiment.temperature
    n_susc_params = len(model.fit_vars)
    curr_fit_dict = {
        name: value
        for name, value in zip(model.fit_vars.keys(), curr_fit.x[:n_susc_params])
    }
    if curr_fit.status == 0:
        if verbose:
            logger.warning(
                "Moment fit at %s K failed - Too many iterations",
                model.temperature,
            )
        model.final_var_values = copy.deepcopy(curr_fit_dict)
        model.fit_stdev = {label: np.nan for label in model.fit_vars.keys()}
        model.fit_status = False
        model.mae = np.nan
        model.rmse = np.nan
        model.r2 = np.nan
        model.adj_r2 = np.nan
        return None

    effective_residual_count = count_active_moment_objective_residuals(
        moment_objective_state
    )
    if effective_residual_count <= curr_fit.x.size:
        model.fit_stdev = {label: np.nan for label in model.fit_vars.keys()}
    else:
        active_mask = active_moment_objective_mask(moment_objective_state)
        active_fit = OptimizeResult(
            fun=np.asarray(curr_fit.fun, dtype=float)[active_mask],
            jac=np.asarray(curr_fit.jac, dtype=float)[active_mask, :],
            x=curr_fit.x,
        )
        stdev, _ = svd_stdev(active_fit)
        model.fit_stdev = {
            label: val
            for label, val in zip(model.fit_vars.keys(), stdev[:n_susc_params])
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
    linewidth_fit_dict = {
        name: value
        for name, value in zip(
            linewidth_fit_vars.keys(),
            curr_fit.x[n_susc_params:],
        )
    }
    final_linewidth_vars = {**linewidth_fix_vars, **linewidth_fit_dict}
    calculated_moments = _calculated_moments_from_parameters(
        model=model,
        parameters=model.final_var_values,
        nuclei=molecule.nuclei,
        average_labels=average_labels,
        widths_ppm=widths_arr,
        calculated_widths_by_label=calculated_widths_by_label,
        linewidth_mean_inv_r6_by_label=linewidth_mean_inv_r6_by_label,
        linewidth_variables=final_linewidth_vars,
        include_diamagnetic=include_diamagnetic,
    )
    residuals = gaussian_mixture_moment_residuals(
        calculated=calculated_moments,
        observed=observed_moments,
        normalize=True,
    )
    weighted_residuals = apply_moment_objective(residuals, moment_objective_state)
    weighted_values = np.asarray(list(weighted_residuals.values()), dtype=float)
    weighted_score = float(np.sqrt(np.sum(weighted_values**2)))
    return MomentFitDiagnostics(
        temperature=float(experiment.temperature),
        objective_type=str(moment_objective_state["type"]),
        observed_moments={k: float(v) for k, v in observed_moments.items()},
        calculated_moments={k: float(v) for k, v in calculated_moments.items()},
        weighted_score=weighted_score,
    )
