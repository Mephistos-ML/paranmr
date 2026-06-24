# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Public API for preparing and applying moment objective transforms."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from paranmr.core.fitting.susceptibility.objectives.moments.state import (
    build_moment_objective_state,
)
from paranmr.core.fitting.susceptibility.objectives.moments.weighted_ls import (
    weighted_ls_transform,
)


def prepare_moment_objective(
    *,
    observed_moments: dict[str, float],
    objective_config: dict | None = None,
) -> dict:
    """Prepare a transform-based moment objective state."""
    if objective_config is None:
        objective_config = {"type": "weighted_ls", "weights": {}}

    objective_type = str(objective_config.get("type", "weighted_ls")).lower()
    moment_names = tuple(observed_moments.keys())

    if objective_type == "weighted_ls":
        transform, diagnostics = weighted_ls_transform(
            moment_names,
            objective_config.get("weights", {}),
        )
        return build_moment_objective_state(
            objective_type=objective_type,
            moment_names=moment_names,
            transform=transform,
            diagnostics=diagnostics,
        )

    raise ValueError(
        "Unknown moment objective type "
        f"{objective_type!r}. Supported value is 'weighted_ls'."
    )


def apply_moment_objective(
    residuals: dict[str, float],
    objective_state: dict,
) -> dict[str, float]:
    """Apply a prepared moment objective transform to residuals."""
    moment_names = tuple(objective_state["moment_names"])
    if tuple(residuals.keys()) != moment_names:
        raise ValueError("Moment residual order does not match objective state")

    residual_vec = np.asarray([residuals[name] for name in moment_names], dtype=float)
    transform = np.asarray(objective_state["transform"], dtype=float)
    transformed = transform @ residual_vec
    component_names = objective_state["component_names"]
    return {
        name: float(value) for name, value in zip(component_names, transformed)
    }


def active_moment_objective_mask(objective_state: dict) -> NDArray[np.bool_]:
    """Return active transformed residual rows for a prepared objective."""
    return np.asarray(objective_state["active_mask"], dtype=bool)


def count_active_moment_objective_residuals(objective_state: dict) -> int:
    """Count active transformed residual components."""
    return int(np.sum(active_moment_objective_mask(objective_state)))
