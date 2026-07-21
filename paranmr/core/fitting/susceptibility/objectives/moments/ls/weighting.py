# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Weight construction helpers for least-squares moment objectives."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def build_ls_weights_by_name(
    *,
    moment_names: tuple[str, ...],
    weights: dict[str, float] | None,
) -> dict[str, float]:
    """Validate and materialize per-moment least-squares weights."""
    if weights is None:
        weights = {}

    unknown = set(weights) - set(moment_names)
    if unknown:
        raise ValueError(
            "Moment weights contain unknown moment(s): "
            + ", ".join(sorted(unknown))
        )

    weights_by_name: dict[str, float] = {}
    for moment_name in moment_names:
        weight = float(weights.get(moment_name, 1.0))
        if weight < 0.0:
            raise ValueError("Moment weights must be non-negative")
        weights_by_name[moment_name] = weight
    return weights_by_name


def build_ls_weight_vector(
    *,
    moment_names: tuple[str, ...],
    weights_by_name: dict[str, float],
) -> NDArray[np.float64]:
    """Return the ordered least-squares weight vector."""
    return np.asarray(
        [weights_by_name[moment_name] for moment_name in moment_names],
        dtype=float,
    )
