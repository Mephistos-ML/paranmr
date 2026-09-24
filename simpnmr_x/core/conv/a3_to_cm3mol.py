# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Convert susceptibility values from Å³ to cm³ mol⁻¹."""

import numpy as np
from numpy.typing import NDArray

from simpnmr_x.core.const.physics import NA

A3_TO_CM3MOL = 1.0e-24 * NA / (4.0 * np.pi)


def a3_to_cm3mol(values: NDArray | float) -> NDArray:
    """Convert susceptibility or χT values from Å³ to cm³ mol⁻¹.

    Args:
        values: Scalar or array in the internal Å³ representation.

    Returns:
        Values in cm³ mol⁻¹, with the same numerical temperature factor if
        ``values`` represents χT.
    """

    return np.asarray(values, dtype=float) * A3_TO_CM3MOL
