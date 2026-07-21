# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Shared moment-condition helpers for susceptibility fitting objectives."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def build_moment_difference_vector(
    *,
    observed_moments: dict[str, float],
    calculated_moments: dict[str, float],
    moment_names: tuple[str, ...],
) -> NDArray[np.float64]:
    """Return the ordered moment-condition vector ``m_calc - m_exp``.

    Args:
        observed_moments: Experimental moment descriptors.
        calculated_moments: Calculated moment descriptors.
        moment_names: Canonical residual order to publish.

    Returns:
        A floating-point vector ordered as ``moment_names``.

    Raises:
        ValueError: If required moment keys are missing from either mapping.
    """

    missing_observed = [name for name in moment_names if name not in observed_moments]
    if missing_observed:
        raise ValueError(
            'Observed moments are missing required keys: ' + ', '.join(missing_observed)
        )

    missing_calculated = [name for name in moment_names if name not in calculated_moments]
    if missing_calculated:
        raise ValueError(
            'Calculated moments are missing required keys: ' + ', '.join(missing_calculated)
        )

    return np.asarray(
        [
            float(calculated_moments[name]) - float(observed_moments[name])
            for name in moment_names
        ],
        dtype=float,
    )
