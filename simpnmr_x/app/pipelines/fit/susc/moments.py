# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Moment-based assignment-free fitting branch for susceptibility workflows."""

from __future__ import annotations

import logging
import os

import numpy as np

from simpnmr_x.app.policies.linewidth_r6 import resolve_r6_linewidth_inputs
from simpnmr_x.core.conv.freq_to_ppm import signal_widths_hz_to_ppm
from simpnmr_x.core.domain.exp import Experiment
from simpnmr_x.core.domain.mol import Molecule
from simpnmr_x.core.fitting.susceptibility.fitters.moments import (
    MomentFitInputs,
    MomentFitResult,
    evaluate_moment_fit_vector,
    fit_moment_model,
)
from simpnmr_x.core.fitting.susceptibility.jacobian.assembly import (
    build_moment_jacobian,
)
from simpnmr_x.core.fitting.susceptibility.models.base import SusceptibilityModel
from simpnmr_x.core.fitting.susceptibility.moments.forward import (
    calculated_signal_packages_from_parameters,
    integral_scale_from_calculated_packages,
    observed_moment_data_from_peaks,
)
from simpnmr_x.core.fitting.susceptibility.objective_map import (
    ObjectiveMapConfig,
    build_objective_map,
)
from simpnmr_x.core.fitting.susceptibility.objectives.gmm.objective import (
    GMMMomentObjective,
)
from simpnmr_x.io.csv.fit import (
    save_fit_linewidth_model,
    save_moment_fit_diagnostics,
    save_moment_jacobian,
)
from simpnmr_x.viz.plots.jacobian import plot_moment_jacobian_heatmap
from simpnmr_x.viz.plots.objective_map import plot_objective_map
from simpnmr_x.viz.style.theme import PlotSpec

logger = logging.getLogger(__name__)


def fit_moment_assignment(
    *,
    model: SusceptibilityModel,
    molecule: Molecule,
    experiment: Experiment,
    spec: PlotSpec,
    show_plots: bool,
    project_name: str,
    max_moment_order: int,
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

    # ``m0`` is integral-only; GMM conditions are ``m1`` through ``mN``.
    moment_labels = tuple(f"m{order}" for order in range(1, max_moment_order + 1))

    observed_moment_data = observed_moment_data_from_peaks(
        centers=np.asarray(
            [signal.shift for signal in experiment.signals], dtype=float
        ),
        fwhm=observed_widths_ppm,
        areas=np.asarray([signal.area for signal in experiment.signals], dtype=float),
        moment_labels=moment_labels,
    )
    experimental_moments = observed_moment_data.moments
    theoretical_packages = calculated_signal_packages_from_parameters(
        model=model,
        parameters={**model.fix_vars, **model.fit_vars},
        nuclei=list(molecule.nuclei),
        include_diamagnetic=any(
            getattr(nucleus.shift, "dia", 0.0) != 0.0 for nucleus in molecule.nuclei
        ),
        average_labels=tuple(tuple(group) for group in average_labels),
    )
    integral_scale = integral_scale_from_calculated_packages(
        observed_integral=observed_moment_data.integral,
        packages=theoretical_packages,
    )

    gmm_objective = GMMMomentObjective(
        moment_names=moment_labels,
        observed_moments=experimental_moments,
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
        gmm_objective=gmm_objective,
        linewidth_inputs=linewidth_inputs,
        linewidth_fit_names=linewidth_fit_names,
        linewidth_fix_vars=linewidth_fix_vars,
        fit_var_names=fit_var_names,
        fit_guess=fit_guess,
        fit_bounds=fit_bounds,
        use_diamagnetic=use_diamagnetic,
        integral_scale=integral_scale,
        average_labels=tuple(tuple(group) for group in average_labels),
    )

    # Run the core moment fit with raw generalized-moment residuals.
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
        moment_jacobian = build_moment_jacobian(
            temperature=float(experiment.temperature),
            parameters=model.final_var_values,
            nuclei=list(molecule.nuclei),
            linewidth_inputs=linewidth_inputs,
            linewidth_vars_by_name=moment_fit_result.linewidth_vars_by_name,
            moment_names=moment_labels,
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
                return fit_inputs.gmm_objective.score(
                    calculated_moments=evaluation.calculated_moments,
                )

            objective_map = build_objective_map(
                temperature=float(experiment.temperature),
                objective_type="gmm",
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
            with spec.context():
                plot_objective_map(
                    objective_map,
                    spec=spec,
                    save=True,
                    show=show_plots,
                    save_name=os.path.join(project_name, file_stub),
                )
    return moment_fit_result


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
