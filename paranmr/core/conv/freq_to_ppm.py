# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Convert signal linewidths from Hz to ppm."""

import numpy as np
from numpy.typing import NDArray

from paranmr.core.const.gammas import NUCLEAR_GAMMAS
from paranmr.core.util.strings import remove_numbers


def signal_widths_hz_to_ppm(
    widths_hz: list[float] | NDArray,
    isotope: str,
    magnetic_field: float,
) -> NDArray:
    """Convert signal widths from Hz to ppm.

    Args:
        widths_hz: Signal widths in Hz.
        isotope: Isotope label used to resolve the gyromagnetic ratio.
        magnetic_field: Spectrometer magnetic field in Tesla.

    Returns:
        Array of widths in ppm ordered like ``widths_hz``.

    Raises:
        ValueError: If the isotope has no usable gyromagnetic ratio.
    """

    isotope_key = remove_numbers(isotope)
    gamma = NUCLEAR_GAMMAS.get(isotope_key)
    if gamma is None or gamma == 0.0:
        raise ValueError(
            "Signal width conversion requires a nonzero gyromagnetic ratio "
            f"for isotope {isotope}"
        )

    widths_hz_arr = np.asarray(widths_hz, dtype=float)
    return widths_hz_arr / (gamma * float(magnetic_field))
