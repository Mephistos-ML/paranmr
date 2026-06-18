# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Public API for preparing and applying moment objective transforms."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from paranmr.core.fitting.susceptibility.moments.bootstrap import (
    bootstrap_moment_residual_covariance,
)
from paranmr.core.fitting.susceptibility.objectives.moment_transforms.diagonal_gmm import (
    diagonal_gmm_transform,
)
from paranmr.core.fitting.susceptibility.objectives.moment_transforms.full_gmm import (
    full_gmm_transform,
)
from paranmr.core.fitting.susceptibility.objectives.moment_transforms.state import (
    bootstrap_float,
    build_moment_objective_state,
)
from paranmr.core.fitting.susceptibility.objectives.moment_transforms.weighted_ls import (
    weighted_ls_transform,
)


def prepare_moment_objective(
    *,
    observed_centers: ArrayLike,
    widths_ppm: ArrayLike,
    areas: ArrayLike,
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

    if objective_type in {"diagonal_gmm", "full_gmm"}:
        uncertainty = objective_config.get("uncertainty", {"method": "bootstrap"})
        uncertainty_method = str(uncertainty.get("method", "bootstrap")).lower()
        if uncertainty_method != "bootstrap":
            raise ValueError("Only bootstrap moment uncertainty is currently supported")
        bootstrap_config = {
            key: value for key, value in uncertainty.items() if key != "method"
        }
        covariance, variances = bootstrap_moment_residual_covariance(
            observed_centers=observed_centers,
            widths_ppm=widths_ppm,
            areas=areas,
            observed_moments=observed_moments,
            bootstrap_config=bootstrap_config,
        )
        variance_floor = bootstrap_float(
            bootstrap_config, "variance_floor", 1.0e-12
        )
        if objective_type == "diagonal_gmm":
            transform, diagnostics = diagonal_gmm_transform(
                moment_names=moment_names,
                covariance=covariance,
                variances=variances,
                variance_floor=variance_floor,
            )
            component_names = moment_names
        else:
            transform, diagnostics = full_gmm_transform(
                moment_names=moment_names,
                covariance=covariance,
                variances=variances,
                variance_floor=variance_floor,
                bootstrap_config=bootstrap_config,
            )
            component_names = tuple(
                f"gmm_component_{idx + 1}" for idx in range(transform.shape[0])
            )

        return build_moment_objective_state(
            objective_type=objective_type,
            moment_names=moment_names,
            transform=transform,
            diagnostics=diagnostics,
            component_names=component_names,
        )

    raise ValueError(
        "Unknown moment objective type "
        f"{objective_type!r}. Supported values are 'weighted_ls', "
        "'diagonal_gmm', and 'full_gmm'."
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
