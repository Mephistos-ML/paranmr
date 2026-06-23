# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Moment-based assignment-free fitting branch for susceptibility workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from paranmr.app.pipelines.fit.linewidth_r6 import resolve_r6_linewidth_inputs
from paranmr.core.conv.freq_to_ppm import signal_widths_hz_to_ppm
from paranmr.core.domain.exp import Experiment
from paranmr.core.domain.mol import Molecule
from paranmr.core.fitting.susceptibility.fitters.moments import (
    MomentFitInputs,
    MomentFitResult,
    fit_moment_model,
)
from paranmr.core.fitting.susceptibility.moments.descriptors import (
    gaussian_mixture_moments,
)
from paranmr.core.fitting.susceptibility.moments.gaussian import (
    gaussian_peak_representation,
)
from paranmr.core.fitting.susceptibility.models.base import SusceptibilityModel
from paranmr.core.fitting.susceptibility.objectives.moments.api import (
    prepare_moment_objective,
)
from paranmr.io.csv.fit import save_moment_fit_diagnostics


@dataclass(frozen=True)
class MomentBranchContext:
    """Prepared workflow context for moment-based fitting."""

    model: SusceptibilityModel
    molecule: Molecule
    experiment: Experiment
    project_name: str
    assignment_moment_objective: dict | None
    linewidth_variables: dict | None


def fit_moment_assignment(
    *,
    model: SusceptibilityModel,
    molecule: Molecule,
    experiment: Experiment,
    project_name: str,
    assignment_moment_objective: dict | None,
    linewidth_variables: dict | None,
) -> MomentFitResult | None:
    """Run moment-based susceptibility fitting for one experiment."""
    context = MomentBranchContext(
        model=model,
        molecule=molecule,
        experiment=experiment,
        project_name=project_name,
        assignment_moment_objective=assignment_moment_objective,
        linewidth_variables=linewidth_variables,
    )
    # Convert experiment widths from Hz to ppm for moment construction.
    observed_widths_ppm = signal_widths_hz_to_ppm(context.experiment)

    # Resolve the R6 linewidth model inputs for the current molecule/isotope set.
    linewidth_inputs = resolve_r6_linewidth_inputs(
        molecule=context.molecule,
        isotope_filter=context.experiment.isotope,
        variables=context.linewidth_variables,
        label_kind="atom_label",
    )

    # Build experimental moments directly from the measured peaks.
    experimental_moments = _experimental_moments_from_experiment(
        experiment=context.experiment,
        widths_ppm=observed_widths_ppm,
    )

    # Prepare the residual transform and weighting scheme for the objective.
    moment_objective_state = prepare_moment_objective(
        observed_moments=experimental_moments,
        objective_config=context.assignment_moment_objective,
    )

    # Split linewidth variables into fit, fixed, and bounded subsets.
    linewidth_fit_vars, linewidth_fix_vars, linewidth_bounds = (
        _split_linewidth_variables(context.linewidth_variables)
    )

    # Collect susceptibility parameter names and initial guesses.
    fit_var_names = tuple(context.model.fit_vars.keys())
    linewidth_fit_names = tuple(linewidth_fit_vars.keys())
    fit_guess = [value for value in context.model.fit_vars.values()]
    fit_guess.extend(linewidth_fit_vars.values())

    # Assemble optimizer bounds in the same order as the fit vector.
    bounds_list = [
        [float(context.model.BOUNDS[name][0]), float(context.model.BOUNDS[name][1])]
        for name in fit_var_names
    ] + [linewidth_bounds[name] for name in linewidth_fit_names]
    fit_bounds = np.asarray(bounds_list, dtype=float).T

    # Track whether diamagnetic shifts must be included in the forward model.
    use_diamagnetic = any(
        getattr(nuc.shift, "dia", 0.0) != 0.0 for nuc in context.molecule.nuclei
    )

    # Package the numeric inputs for the core optimizer.
    fit_inputs = MomentFitInputs(
        model=context.model,
        nuclei=tuple(context.molecule.nuclei),
        temperature=float(context.experiment.temperature),
        observed_moments=experimental_moments,
        moment_objective_state=moment_objective_state,
        linewidth_mean_inv_r6_by_label=linewidth_inputs.mean_inv_r6_by_label,
        linewidth_fit_names=linewidth_fit_names,
        linewidth_fix_vars=linewidth_fix_vars,
        fit_var_names=fit_var_names,
        fit_guess=fit_guess,
        fit_bounds=fit_bounds,
        use_diamagnetic=use_diamagnetic,
    )

    # Run the core least-squares fit.
    moment_fit_result = fit_moment_model(fit_inputs)
    
    # Persist fit diagnostics next to the project outputs.
    if moment_fit_result is not None:
        save_moment_fit_diagnostics(
            diagnostics=moment_fit_result,
            file_name=os.path.join(
                project_name,
                f"moment_fit_diagnostics_{experiment.temperature:.2f}_K.csv",
            ),
    )
    return moment_fit_result


def _experimental_moments_from_experiment(
    *,
    experiment: Experiment,
    widths_ppm: np.ndarray,
) -> dict[str, float]:
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
    return gaussian_mixture_moments(
        centers=observed_peaks["center"],
        sigmas=observed_peaks["sigma"],
        area_norm=observed_peaks["area_norm"],
    )


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
