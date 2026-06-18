# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Resolve linewidth sources for susceptibility fitting workflows."""

import numpy as np
from numpy.typing import NDArray

from paranmr.app.policies.peak_projection import resolve_gaussian_peak_inputs
from paranmr.core.domain.exp import Experiment


def resolve_fitting_linewidths(
    *,
    method: str,
    experiment: Experiment,
    experimental_widths_ppm: NDArray | None = None,
) -> NDArray:
    """Return linewidths used by susceptibility fitting workflows.

    Args:
        method: Linewidth source selector. Currently only ``"experimental"``
            is supported.
        experiment: Experiment containing peak linewidths.
        experimental_widths_ppm: Optional precomputed experimental linewidths
            in ppm, ordered like ``experiment.signals``.

    Returns:
        Linewidth values in ppm ordered like ``experiment.signals``.

    Raises:
        ValueError: If `method` is unsupported.
    """

    if method == "experimental":
        if experimental_widths_ppm is not None:
            return np.asarray(experimental_widths_ppm, dtype=float)
        _, widths_ppm, _ = resolve_gaussian_peak_inputs(experiment)
        return widths_ppm

    raise ValueError(
        "Unsupported fitting linewidth method "
        f"{method!r}. Supported methods: 'experimental'."
    )
