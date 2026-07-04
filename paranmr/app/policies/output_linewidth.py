# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Resolve linewidth values for prediction and fitting outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from paranmr.core.conv.freq_to_ppm import signal_widths_hz_to_ppm
from paranmr.core.domain.exp import Experiment
from paranmr.core.domain.mol import Molecule

AUTO_LINEWIDTH_FRACTION = 0.005
LinewidthMode = Literal["auto", "relax", "fit", "exp"]


@dataclass(frozen=True)
class LinewidthOutput:
    """Resolved linewidth values for plots and peak CSV output.

    Args:
        mode: Linewidth resolution mode.
        column_name: CSV column name for the resolved linewidth values.
        values_by_label: Per-nucleus linewidth values in ppm.
    """

    mode: LinewidthMode
    column_name: str
    values_by_label: dict[str, float]


def resolve_output_linewidths(
    molecule: Molecule,
    shift_range: Sequence[float],
    *,
    experiment: Experiment | None = None,
    explicit_linewidth_by_label: Mapping[str, float] | None = None,
    explicit_column_name: str | None = None,
) -> LinewidthOutput:
    """Resolve linewidths for prediction plots and peak CSV output.

    Args:
        molecule: Molecule with shifts and optional relaxation-derived linewidths.
        shift_range: Two-value ppm range used for the predicted spectrum window.
        experiment: Optional experiment providing assigned linewidth values in Hz.
        explicit_linewidth_by_label: Optional precomputed per-nucleus linewidths
            to use directly for fit outputs.
        explicit_column_name: Column name for explicit linewidth values.

    Returns:
        Resolved linewidth values and their CSV column contract.
    """
    if explicit_linewidth_by_label is not None:
        if explicit_column_name is None:
            raise ValueError(
                "explicit_column_name is required when explicit linewidths are provided"
            )
        return LinewidthOutput(
            mode="fit",
            column_name=explicit_column_name,
            values_by_label=dict(explicit_linewidth_by_label),
        )

    experimental_linewidths = _experimental_linewidths_by_label(
        molecule=molecule,
        experiment=experiment,
    )
    if experimental_linewidths is not None:
        return LinewidthOutput(
            mode="exp",
            column_name="linewidth_exp (ppm)",
            values_by_label=experimental_linewidths,
        )

    relaxation = getattr(molecule, "relaxation", None)
    if relaxation is not None:
        missing = [nuc.label for nuc in molecule.nuclei if nuc.shift.lw is None]
        if missing:
            raise ValueError("Relaxation linewidths are incomplete")
        return LinewidthOutput(
            mode="relax",
            column_name="linewidth_avg_relax (ppm)",
            values_by_label={
                nuc.label: nuc.shift.lw
                for nuc in molecule.nuclei
                if nuc.shift.lw is not None
            },
        )

    linewidth = _auto_display_linewidth_ppm(shift_range)
    return LinewidthOutput(
        mode="auto",
        column_name="linewidth_avg_auto (ppm)",
        values_by_label={nuc.label: linewidth for nuc in molecule.nuclei},
    )


def _auto_display_linewidth_ppm(shift_range: Sequence[float]) -> float:
    span = abs(max(shift_range) - min(shift_range))
    return AUTO_LINEWIDTH_FRACTION * span


def _experimental_linewidths_by_label(
    *,
    molecule: Molecule,
    experiment: Experiment | None,
) -> dict[str, float] | None:
    if experiment is None:
        return None

    widths_ppm = signal_widths_hz_to_ppm(
        [signal.width for signal in experiment.signals],
        experiment.isotope,
        experiment.magnetic_field,
    )
    width_by_signal_label = {
        signal.signal_label: float(width)
        for signal, width in zip(experiment.signals, widths_ppm)
    }
    if not width_by_signal_label:
        return None

    values_by_label = {}
    for nucleus in molecule.nuclei:
        if nucleus.signal_label in width_by_signal_label:
            values_by_label[nucleus.label] = width_by_signal_label[nucleus.signal_label]
    if not values_by_label:
        return None
    return values_by_label
