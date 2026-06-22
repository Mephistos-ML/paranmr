# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Resolve linewidth sources for susceptibility fitting workflows."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from paranmr.app.policies.peak_projection import resolve_gaussian_peak_inputs
from paranmr.core.domain.exp import Experiment
from paranmr.core.domain.mol import Molecule
from paranmr.core.fitting.linewidth import (
    mean_inv_r6_by_label,
    predict_r6_linewidths,
)


@dataclass(frozen=True)
class FittingLinewidths:
    """Resolved linewidths for fitting workflows.

    Args:
        observed_widths_ppm: Observed linewidths in ppm ordered like
            ``experiment.signals``.
        calculated_widths_by_label: Optional calculated linewidths in ppm keyed
            by calculated signal-package label.
        mean_inv_r6_by_label: Optional ``mean(1/r^6)`` values keyed by the
            calculated package label for ``r6`` linewidth models.
    """

    observed_widths_ppm: NDArray
    calculated_widths_by_label: dict[str, float] | None = None
    mean_inv_r6_by_label: dict[str, float] | None = None


def resolve_fitting_linewidths(
    *,
    method: str,
    experiment: Experiment,
    molecule: Molecule | None = None,
    variables: dict[str, list[object]] | None = None,
    experimental_widths_ppm: NDArray | None = None,
    label_kind: str = "signal_label",
) -> FittingLinewidths:
    """Return linewidths used by susceptibility fitting workflows.

    Args:
        method: Linewidth source selector. Currently only ``"experimental"``
            and ``"r6"`` are supported.
        experiment: Experiment containing peak linewidths.
        molecule: Optional molecule used by model-derived linewidth sources.
        variables: Optional linewidth model variables.
        experimental_widths_ppm: Optional precomputed experimental linewidths
            in ppm, ordered like ``experiment.signals``.
        label_kind: Label kind used by calculated linewidth models.

    Returns:
        Resolved observed linewidths and optional calculated linewidths.

    Raises:
        ValueError: If `method` is unsupported.
    """

    observed_widths_ppm = _experimental_widths_ppm(
        experiment=experiment,
        experimental_widths_ppm=experimental_widths_ppm,
    )

    if method == "experimental":
        return FittingLinewidths(observed_widths_ppm=observed_widths_ppm)

    if method == "r6":
        if molecule is None:
            raise ValueError("linewidth:method 'r6' requires a molecule")
        if molecule.paramagnetic_centre is None:
            raise ValueError("linewidth:method 'r6' requires a paramagnetic centre")
        if variables is None:
            raise ValueError("linewidth:method 'r6' requires linewidth variables")
        mean_inv_r6 = mean_inv_r6_by_label(
            nuclei=molecule.nuclei,
            paramagnetic_centre=molecule.paramagnetic_centre,
            isotope_filter=experiment.isotope,
            label_kind=label_kind,
        )
        fixed_values = _fixed_r6_values(variables)
        calculated_widths_by_label = None
        if fixed_values is not None:
            p1, p2 = fixed_values
            calculated_widths_by_label = predict_r6_linewidths(mean_inv_r6, p1, p2)
        return FittingLinewidths(
            observed_widths_ppm=observed_widths_ppm,
            calculated_widths_by_label=calculated_widths_by_label,
            mean_inv_r6_by_label=mean_inv_r6,
        )

    raise ValueError(
        "Unsupported fitting linewidth method "
        f"{method!r}. Supported methods: 'experimental', 'r6'."
    )


def _experimental_widths_ppm(
    *,
    experiment: Experiment,
    experimental_widths_ppm: NDArray | None,
) -> NDArray:
    if experimental_widths_ppm is not None:
        return np.asarray(experimental_widths_ppm, dtype=float)
    _, widths_ppm, _ = resolve_gaussian_peak_inputs(experiment)
    return widths_ppm


def _fixed_r6_values(
    variables: dict[str, list[object]],
) -> tuple[float, float] | None:
    values = []
    for name in ["p1", "p2"]:
        try:
            entry = variables[name]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"linewidth:variables must define {name}") from exc
        if entry[0] != "fix":
            return None
        values.append(float(entry[1]))
    return values[0], values[1]
