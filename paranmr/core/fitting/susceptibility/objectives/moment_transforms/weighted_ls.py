# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Manual weighted least-squares moment objective transform."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from paranmr.core.fitting.susceptibility.objectives.moment_transforms.state import (
    weights_from_transform,
)


def weighted_ls_transform(
    moment_names: tuple[str, ...],
    weights: dict[str, float] | None,
) -> tuple[NDArray[np.float64], dict]:
    """Build a diagonal residual transform from user-provided moment weights."""
    if weights is None:
        weights = {}

    unknown = set(weights) - set(moment_names)
    if unknown:
        raise ValueError(
            "Moment weights contain unknown moment(s): "
            + ", ".join(sorted(unknown))
        )

    weight_values = []
    for moment_name in moment_names:
        weight = float(weights.get(moment_name, 1.0))
        if weight < 0.0:
            raise ValueError("Moment weights must be non-negative")
        weight_values.append(weight)

    transform = np.diag(np.asarray(weight_values, dtype=float))
    diagnostics = {"weights": weights_from_transform(moment_names, transform)}
    return transform, diagnostics
