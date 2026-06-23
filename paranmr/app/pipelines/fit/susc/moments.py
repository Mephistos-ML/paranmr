# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Moment-based assignment-free fitting branch for susceptibility workflows."""

from __future__ import annotations

import os

import numpy as np

from paranmr.app.pipelines.fit.linewidth_r6 import resolve_r6_linewidth_inputs
from paranmr.core.domain.exp import Experiment
from paranmr.core.domain.mol import Molecule
from paranmr.core.conv.freq_to_ppm import signal_widths_hz_to_ppm
from paranmr.core.fitting.susceptibility.fitters.moments import fit_model_to_moments
from paranmr.core.fitting.susceptibility.models.base import SusceptibilityModel
from paranmr.io.csv.fit import save_moment_fit_diagnostics


def fit_moment_assignment(
    *,
    model: SusceptibilityModel,
    molecule: Molecule,
    experiment: Experiment,
    project_name: str,
    assignment_moment_objective: dict | None,
    linewidth_variables: dict | None,
) -> None:
    """Run moment-based susceptibility fitting for one experiment."""

    observed_widths_ppm = signal_widths_hz_to_ppm(experiment)
    linewidth_inputs = resolve_r6_linewidth_inputs(
        molecule=molecule,
        isotope_filter=experiment.isotope,
        variables=linewidth_variables,
        label_kind="atom_label",
    )
    moment_fit_result = fit_model_to_moments(
        model=model,
        molecule=molecule,
        centers_ppm=np.asarray(
            [signal.shift for signal in experiment.signals],
            dtype=float,
        ),
        widths_ppm=observed_widths_ppm,
        areas=np.asarray([signal.area for signal in experiment.signals], dtype=float),
        temperature=experiment.temperature,
        moment_objective=assignment_moment_objective,
        linewidth_mean_inv_r6_by_label=linewidth_inputs.mean_inv_r6_by_label,
        linewidth_variables=linewidth_variables,
    )
    if moment_fit_result is not None:
        save_moment_fit_diagnostics(
            diagnostics=moment_fit_result,
            file_name=os.path.join(
                project_name,
                f"moment_fit_diagnostics_{experiment.temperature:.2f}_K.csv",
            ),
        )
