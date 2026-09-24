# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Convert susceptibility values from cm³ mol⁻¹ to Å³."""

import numpy as np
from numpy.typing import NDArray

from simpnmr_x.core.conv.a3_to_cm3mol import A3_TO_CM3MOL

CM3MOL_TO_A3 = 1.0 / A3_TO_CM3MOL


def cm3mol_to_a3(values: NDArray | float) -> NDArray:
    """Convert susceptibility or χT values from cm³ mol⁻¹ to Å³.

    Args:
        values: Scalar or array in cm³ mol⁻¹.

    Returns:
        Values in the internal Å³ representation, with the same numerical
        temperature factor if ``values`` represents χT.
    """

    return np.asarray(values, dtype=float) * CM3MOL_TO_A3
