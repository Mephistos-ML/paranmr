# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Moment-based assignment-free fitting branch for susceptibility workflows."""

from __future__ import annotations

import logging
import os

import numpy as np

from paranmr.app.policies.linewidth_r6 import resolve_r6_linewidth_inputs
from paranmr.core.conv.freq_to_ppm import signal_widths_hz_to_ppm
from paranmr.core.domain.exp import Experiment
from paranmr.core.domain.mol import Molecule
from paranmr.core.fitting.susceptibility.fitters.moments import (
    MomentFitInputs,
    MomentFitResult,
    evaluate_moment_fit_vector,
    fit_moment_model,
)
from paranmr.core.fitting.susceptibility.objective_map import (
    ObjectiveMapConfig,
    build_objective_map,
)
from paranmr.core.fitting.susceptibility.jacobian.assembly import (
    build_moment_jacobian,
)
from paranmr.core.fitting.susceptibility.moments.descriptors import (
    compute_gaussian_mixture_moments,
)
from paranmr.core.fitting.susceptibility.moments.gaussian import (
    gaussian_peak_representation,
)
from paranmr.core.fitting.susceptibility.models.base import SusceptibilityModel
from paranmr.core.fitting.susceptibility.objectives.moments.gmm import (
    GMMMomentObjective,
    MonteCarloMomentCovarianceConfig,
    build_gmm_weighting_matrix,
    estimate_moment_covariance_from_monte_carlo,
)
from paranmr.core.fitting.susceptibility.objectives.moments.ls.objective import (
    WeightedLSMomentObjective,
)
from paranmr.io.csv.fit import (
    save_moment_fit_diagnostics,
    save_fit_linewidth_model,
    save_moment_covariance,
    save_moment_jacobian,
    save_objective_map,
)
from paranmr.viz.plots.covariance import plot_moment_covariance_heatmap
from paranmr.viz.plots.jacobian import plot_moment_jacobian_heatmap
from paranmr.viz.plots.objective_map import plot_objective_map
from paranmr.viz.style.theme import PlotSpec

logger = logging.getLogger(__name__)


def fit_moment_assignment(
    *,
    model: SusceptibilityModel,
    molecule: Molecule,
    experiment: Experiment,
    spec: PlotSpec,
    show_plots: bool,
    project_name: str,
    assignment_moment_objective: dict | None,
    susc_fit_objective_map: dict | None,
    linewidth_variables: dict | None,
    average_labels: list[list[str]] | None = None,
) -> MomentFitResult | None:
    """Run moment-based susceptibility fitting for one experiment."""
    if average_labels is None:
        average_labels = []

    # Convert experiment widths from Hz to ppm for moment construction.
    observed_widths_hz = [signal.width for signal in experiment.signals]
    observed_widths_ppm = signal_widths_hz_to_ppm(
        observed_widths_hz,
        experiment.isotope,
        experiment.magnetic_field,
    )

    # Resolve the R6 linewidth model inputs for the current molecule/isotope set.
    linewidth_inputs = resolve_r6_linewidth_inputs(
        molecule=molecule,
        isotope_filter=experiment.isotope,
        label_kind="atom_label",
    )

    moment_labels = build_moment_labels_up_to(
        int(assignment_moment_objective["number_of_moments"])
    )

    # Build experimental moments directly from the measured peaks.
    observed_peaks = _observed_peak_representation_from_experiment(
        experiment=experiment,
        widths_ppm=observed_widths_ppm,
    )
    experimental_moments = compute_gaussian_mixture_moments(
        centers=observed_peaks["center"],
        sigmas=observed_peaks["sigma"],
        area_norm=observed_peaks["area_norm"],
        moment_labels=moment_labels,
    )

    moment_covariance = None
    gmm_weighting_matrix = None
    if (
        assignment_moment_objective is not None
        and assignment_moment_objective.get("type") == "gmm"
    ):
        covariance_config = assignment_moment_objective["covariance"]
        moment_covariance = estimate_moment_covariance_from_monte_carlo(
            observed_peaks=observed_peaks,
            moment_names=moment_labels,
            config=MonteCarloMomentCovarianceConfig(
                n_samples=int(covariance_config["n_samples"]),
                shift_sigma_abs=float(
                    covariance_config["perturbation"]["shift_sigma_abs"]
                ),
                width_sigma_rel=float(
                    covariance_config["perturbation"]["width_sigma_rel"]
                ),
                random_seed=(
                    None
                    if covariance_config.get("random_seed") is None
                    else int(covariance_config["random_seed"])
                ),
            ),
        )
        gmm_weighting_matrix = build_gmm_weighting_matrix(
            moment_covariance.covariance
        )

    # Build the configured moment objective before assembling optimizer inputs.
    objective_type = str(assignment_moment_objective["type"]).lower()
    if objective_type == "ls":
        moment_objective = WeightedLSMomentObjective.from_config(
            moment_names=moment_labels,
            weights=assignment_moment_objective.get("moment_weights", {}),
        )
    elif objective_type == "gmm":
        if gmm_weighting_matrix is None:
            raise ValueError(
                "GMM moment objective requires an explicit covariance-derived "
                "weighting matrix"
            )
        moment_objective = GMMMomentObjective.with_weighting_matrix(
            moment_names=moment_labels,
            weighting_matrix=gmm_weighting_matrix,
        )
    else:
        raise ValueError(
            "Unknown moment objective type "
            f"{objective_type!r}. Supported values are 'ls' and 'gmm'."
        )

    # Split linewidth variables into fit, fixed, and bounded subsets.
    linewidth_fit_vars, linewidth_fix_vars, linewidth_bounds = (
        _split_linewidth_variables(linewidth_variables)
    )

    # Collect susceptibility parameter names and initial guesses.
    fit_var_names = tuple(model.fit_vars.keys())
    linewidth_fit_names = tuple(linewidth_fit_vars.keys())
    fit_guess = [value for value in model.fit_vars.values()]
    fit_guess.extend(linewidth_fit_vars.values())

    # Assemble optimizer bounds in the same order as the fit vector.
    bounds_list = [
        [float(model.BOUNDS[name][0]), float(model.BOUNDS[name][1])]
        for name in fit_var_names
    ] + [linewidth_bounds[name] for name in linewidth_fit_names]
    fit_bounds = np.asarray(bounds_list, dtype=float).T

    # Track whether diamagnetic shifts must be included in the forward model.
    use_diamagnetic = any(
        getattr(nuc.shift, "dia", 0.0) != 0.0 for nuc in molecule.nuclei
    )

    # Package the numeric inputs for the core optimizer.
    fit_inputs = MomentFitInputs(
        model=model,
        nuclei=tuple(molecule.nuclei),
        temperature=float(experiment.temperature),
        moment_labels=moment_labels,
        observed_moments=experimental_moments,
        moment_objective=moment_objective,
        linewidth_inputs=linewidth_inputs,
        linewidth_fit_names=linewidth_fit_names,
        linewidth_fix_vars=linewidth_fix_vars,
        fit_var_names=fit_var_names,
        fit_guess=fit_guess,
        fit_bounds=fit_bounds,
        use_diamagnetic=use_diamagnetic,
        average_labels=tuple(tuple(group) for group in average_labels),
    )

    # Run the core moment fit with the configured LS or GMM residual model.
    moment_fit_result = fit_moment_model(fit_inputs)

    # Persist fit diagnostics next to the project outputs.
    if moment_fit_result is not None:
        for nucleus in molecule.nuclei:
            linewidth = moment_fit_result.calculated_linewidths_by_label.get(
                nucleus.label
            )
            if linewidth is not None:
                nucleus.shift.lw = float(linewidth)
        save_moment_fit_diagnostics(
            diagnostics=moment_fit_result,
            file_name=os.path.join(
                project_name,
                f"moment_fit_diagnostics_{experiment.temperature:.2f}_K.csv",
            ),
        )
        save_fit_linewidth_model(
            diagnostics=moment_fit_result,
            file_name=os.path.join(
                project_name,
                f"linewidth_model_{experiment.temperature:.2f}_K.csv",
            ),
        )
        if moment_covariance is not None:
            save_moment_covariance(
                estimate=moment_covariance,
                file_name=os.path.join(
                    project_name,
                    f"moment_covariance_{experiment.temperature:.2f}_K.csv",
                ),
                temperature=float(experiment.temperature),
            )
            with spec.context():
                plot_moment_covariance_heatmap(
                    covariance=moment_covariance,
                    spec=spec,
                    save=True,
                    show=show_plots,
                    save_name=os.path.join(
                        project_name,
                        f"moment_covariance_heatmap_{experiment.temperature:.2f}_K",
                    ),
                )
        moment_jacobian = build_moment_jacobian(
            temperature=float(experiment.temperature),
            parameters=model.final_var_values,
            nuclei=list(molecule.nuclei),
            linewidth_inputs=linewidth_inputs,
            linewidth_vars_by_name=moment_fit_result.linewidth_vars_by_name,
            observed_moments=experimental_moments,
            parameter_names=fit_var_names + linewidth_fit_names,
            average_labels=tuple(tuple(group) for group in average_labels),
        )
        save_moment_jacobian(
            jacobian=moment_jacobian,
            file_name=os.path.join(
                project_name,
                f"moment_jacobian_{experiment.temperature:.2f}_K.csv",
            ),
        )
        with spec.context():
            plot_moment_jacobian_heatmap(
                jacobian=moment_jacobian,
                spec=spec,
                save=True,
                show=show_plots,
                save_name=os.path.join(
                    project_name,
                    f"moment_jacobian_heatmap_{experiment.temperature:.2f}_K",
                ),
            )
        objective_map_config = susc_fit_objective_map or {}
        if objective_map_config:
            fitted_vector = [
                float(model.final_var_values[name]) for name in fit_var_names
            ] + [
                float(moment_fit_result.linewidth_vars_by_name[name])
                for name in linewidth_fit_names
            ]
            def moment_score(point: np.ndarray) -> float:
                evaluation = evaluate_moment_fit_vector(point, fit_inputs)
                return fit_inputs.moment_objective.score(
                    observed_moments=evaluation.normalized_observed_moments,
                    calculated_moments=evaluation.normalized_calculated_moments,
                )

            objective_map = build_objective_map(
                temperature=float(experiment.temperature),
                objective_type=fit_inputs.moment_objective.objective_type,
                parameter_names=fit_var_names + linewidth_fit_names,
                fit_vector=fitted_vector,
                fit_bounds=fit_inputs.fit_bounds,
                config=ObjectiveMapConfig(
                    parameters=tuple(objective_map_config["parameters"]),
                    window_rel=float(objective_map_config["window_rel"]),
                    n_grid=int(objective_map_config["n_grid"]),
                    gradient=bool(objective_map_config["gradient"]),
                ),
                score_evaluator=moment_score,
            )
            file_stub = (
                "objective_map_"
                f"{objective_map.parameter_names[0]}_"
                f"{objective_map.parameter_names[1]}_"
                f"{experiment.temperature:.2f}_K"
            )
            save_objective_map(
                objective_map=objective_map,
                file_name=os.path.join(project_name, f"{file_stub}.csv"),
            )
            with spec.context():
                plot_objective_map(
                    objective_map,
                    spec=spec,
                    save=True,
                    show=show_plots,
                    save_name=os.path.join(project_name, file_stub),
                )
    return moment_fit_result


def build_moment_labels_up_to(number_of_moments: int) -> tuple[str, ...]:
    if number_of_moments <= 0:
        raise ValueError("number_of_moments must be positive")
    return tuple(f"m{index}" for index in range(1, number_of_moments + 1))


def _observed_peak_representation_from_experiment(
    *,
    experiment: Experiment,
    widths_ppm: np.ndarray,
) -> dict[str, np.ndarray]:
    centers_ppm = np.asarray(
        [signal.shift for signal in experiment.signals],
        dtype=float,
    )
    areas = np.asarray(
        [signal.area for signal in experiment.signals],
        dtype=float,
    )
    sort_idx = np.argsort(centers_ppm)
    observed_peaks = gaussian_peak_representation(
        centers=centers_ppm[sort_idx],
        fwhm=widths_ppm[sort_idx],
        areas=areas[sort_idx],
    )
    return observed_peaks


def _split_linewidth_variables(
    variables: dict[str, list[object]] | None,
) -> tuple[dict[str, float], dict[str, float], dict[str, list[float]]]:
    """Split linewidth variables into fit, fixed, and bounds maps."""

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
