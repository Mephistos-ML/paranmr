# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Linewidth-parameter estimation helpers for fixed-assignment susceptibility workflows."""

from __future__ import annotations

import os

from paranmr.core.conv.freq_to_ppm import signal_widths_hz_to_ppm
from paranmr.core.domain.exp import Experiment
from paranmr.core.domain.mol import Molecule
from paranmr.core.fitting.linewidth import (
    R6LinewidthParameterEstimate,
    estimate_r6_linewidth_parameters,
    mean_inv_r6_by_label,
)
from paranmr.io.csv.fit import save_linewidth_parameter_estimate


def run_fixed_assignment_linewidth_estimation(
    *,
    molecule: Molecule,
    experiment: Experiment,
    project_name: str,
) -> R6LinewidthParameterEstimate:
    """Estimate ``r^-6`` linewidth-model parameters for a fixed assignment.

    Args:
        molecule: Molecule carrying coordinates, signal labels, and a
            paramagnetic centre.
        experiment: Experimental peak table with assigned signal labels and
            linewidths in Hz.
        project_name: Output directory used for linewidth-estimate CSV export.

    Returns:
        Structured estimate with fitted ``p1`` and ``p2`` values.

    Raises:
        ValueError: If the estimation inputs are incomplete or inconsistent.
    """

    if molecule.paramagnetic_centre is None:
        raise ValueError(
            "linewidth:estimate requires hyperfine:paramagnetic_centre"
        )

    mean_inv_r6 = mean_inv_r6_by_label(
        nuclei=molecule.nuclei,
        paramagnetic_centre=molecule.paramagnetic_centre,
        isotope_filter=experiment.isotope,
        label_kind="signal_label",
    )
    widths_ppm = signal_widths_hz_to_ppm(
        [signal.width for signal in experiment.signals],
        experiment.isotope,
        experiment.magnetic_field,
    )

    observed_widths_by_label: dict[str, float] = {}
    for signal, width_ppm in zip(experiment.signals, widths_ppm):
        label = signal.signal_label
        if label in observed_widths_by_label:
            raise ValueError(
                "linewidth:estimate requires unique experimental signal "
                f"labels; duplicate label {label!r} was found"
            )
        observed_widths_by_label[label] = float(width_ppm)

    estimate = estimate_r6_linewidth_parameters(
        mean_inv_r6_by_label=mean_inv_r6,
        observed_widths_by_label=observed_widths_by_label,
        fit_offset=True,
    )
    save_linewidth_parameter_estimate(
        estimate=estimate,
        temperature=float(experiment.temperature),
        file_name=os.path.join(
            project_name,
            f"linewidth_estimate_{experiment.temperature:.2f}_K.csv",
        ),
    )
    return estimate
