# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Adapt experimental peak descriptors for spectral projection workflows."""

import numpy as np

from paranmr.core.const.gammas import NUCLEAR_GAMMAS
from paranmr.core.domain.exp import Experiment
from paranmr.core.util.strings import remove_numbers


def resolve_gaussian_peak_inputs(
    experiment: Experiment,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return experimental peak centers, FWHM values, and areas for Gaussians.

    Args:
        experiment: Experiment containing signal shifts, widths in Hz, and areas.

    Returns:
        Tuple of ``(centers_ppm, widths_ppm, areas)`` ordered like
        ``experiment.signals``.

    Raises:
        ValueError: If the experiment isotope has no usable gyromagnetic ratio.
    """

    isotope_key = remove_numbers(experiment.isotope)
    gamma = NUCLEAR_GAMMAS.get(isotope_key)
    if gamma is None or gamma == 0.0:
        raise ValueError(
            "Gaussian peak projection requires a nonzero gyromagnetic ratio "
            f"for isotope {experiment.isotope}"
        )

    centers_ppm = np.asarray(
        [signal.shift for signal in experiment.signals], dtype=float
    )
    widths_hz = np.asarray(
        [signal.width for signal in experiment.signals], dtype=float
    )
    areas = np.asarray([signal.area for signal in experiment.signals], dtype=float)

    return centers_ppm, widths_hz / (gamma * experiment.magnetic_field), areas
