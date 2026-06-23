# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Convert experimental linewidths from Hz to ppm."""

import numpy as np
from numpy.typing import NDArray

from paranmr.core.const.gammas import NUCLEAR_GAMMAS
from paranmr.core.domain.exp import Experiment
from paranmr.core.util.strings import remove_numbers


def signal_widths_hz_to_ppm(experiment: Experiment) -> NDArray:
    """Convert experimental signal widths from Hz to ppm.

    Args:
        experiment: Experiment containing signal widths in Hz.

    Returns:
        Array of widths in ppm ordered like ``experiment.signals``.

    Raises:
        ValueError: If the experiment isotope has no usable gyromagnetic ratio.
    """

    isotope_key = remove_numbers(experiment.isotope)
    gamma = NUCLEAR_GAMMAS.get(isotope_key)
    if gamma is None or gamma == 0.0:
        raise ValueError(
            "Signal width conversion requires a nonzero gyromagnetic ratio "
            f"for isotope {experiment.isotope}"
        )

    widths_hz = np.asarray([signal.width for signal in experiment.signals], dtype=float)
    return widths_hz / (gamma * experiment.magnetic_field)
