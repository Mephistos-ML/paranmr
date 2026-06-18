# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Residual transforms for moment-matching susceptibility objectives."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from paranmr.core.fitting.susceptibility.moments.bootstrap import (
    bootstrap_moment_residual_covariance,
)


def prepare_moment_objective(
    *,
    observed_centers: ArrayLike,
    widths_ppm: ArrayLike,
    areas: ArrayLike,
    observed_moments: dict[str, float],
    objective_config: dict | None = None,
) -> dict:
    """Prepare a transform-based moment objective state.

    The state maps the raw normalized moment residual vector to the residual
    vector passed to ``least_squares``. This keeps the fitting code independent
    of whether the transform comes from manual weights or an estimated
    uncertainty model.
    """

    if objective_config is None:
        objective_config = {"type": "weighted_ls", "weights": {}}

    objective_type = str(objective_config.get("type", "weighted_ls")).lower()
    moment_names = tuple(observed_moments.keys())

    if objective_type == "weighted_ls":
        weights = objective_config.get("weights", {})
        transform = _manual_weight_transform(moment_names, weights)
        return _build_moment_objective_state(
            objective_type=objective_type,
            moment_names=moment_names,
            transform=transform,
            diagnostics={"weights": _weights_from_transform(moment_names, transform)},
        )

    if objective_type in {"bootstrap_diagonal_gmm", "bootstrap_full_gmm"}:
        bootstrap = objective_config.get("bootstrap", {})
        covariance, variances = bootstrap_moment_residual_covariance(
            observed_centers=observed_centers,
            widths_ppm=widths_ppm,
            areas=areas,
            observed_moments=observed_moments,
            bootstrap_config=bootstrap,
        )
        variance_floor = _bootstrap_float(bootstrap, "variance_floor", 1.0e-12)
        if objective_type == "bootstrap_diagonal_gmm":
            safe_variances = np.maximum(variances, variance_floor)
            transform = np.diag(1.0 / np.sqrt(safe_variances))
            diagnostics = {
                "weights": _weights_from_transform(moment_names, transform),
                "variances": {
                    name: float(value)
                    for name, value in zip(moment_names, variances)
                },
                "covariance": covariance,
                "variance_floor": variance_floor,
            }
            component_names = moment_names
        else:
            transform, diagnostics = _full_gmm_transform(
                moment_names=moment_names,
                covariance=covariance,
                variances=variances,
                variance_floor=variance_floor,
                bootstrap_config=bootstrap,
            )
            component_names = tuple(
                f"gmm_component_{idx + 1}" for idx in range(transform.shape[0])
            )

        return _build_moment_objective_state(
            objective_type=objective_type,
            moment_names=moment_names,
            transform=transform,
            diagnostics=diagnostics,
            component_names=component_names,
        )

    raise ValueError(
        "Unknown moment objective type "
        f"{objective_type!r}. Supported values are 'weighted_ls', "
        "'bootstrap_diagonal_gmm', and 'bootstrap_full_gmm'."
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


def _manual_weight_transform(
    moment_names: tuple[str, ...],
    weights: dict[str, float] | None,
) -> NDArray[np.float64]:
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
    return np.diag(np.asarray(weight_values, dtype=float))


def _build_moment_objective_state(
    *,
    objective_type: str,
    moment_names: tuple[str, ...],
    transform: NDArray[np.float64],
    diagnostics: dict,
    component_names: tuple[str, ...] | None = None,
) -> dict:
    if component_names is None:
        component_names = tuple(moment_names)
    active_mask = np.linalg.norm(transform, axis=1) != 0.0
    return {
        "type": objective_type,
        "moment_names": moment_names,
        "component_names": component_names,
        "transform": np.asarray(transform, dtype=float),
        "active_mask": active_mask,
        "diagnostics": diagnostics,
    }


def _weights_from_transform(
    moment_names: tuple[str, ...],
    transform: NDArray[np.float64],
) -> dict[str, float]:
    diagonal = np.diag(np.asarray(transform, dtype=float))
    return {name: float(value) for name, value in zip(moment_names, diagonal)}


def _full_gmm_transform(
    *,
    moment_names: tuple[str, ...],
    covariance: NDArray[np.float64],
    variances: NDArray[np.float64],
    variance_floor: float,
    bootstrap_config: dict,
) -> tuple[NDArray[np.float64], dict]:
    shrinkage = _bootstrap_float(
        bootstrap_config,
        "covariance_regularization",
        _bootstrap_float(bootstrap_config, "shrinkage", 0.1),
    )
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("covariance_regularization must be between 0 and 1")

    covariance = np.asarray(covariance, dtype=float)
    safe_variances = np.maximum(np.asarray(variances, dtype=float), variance_floor)
    diagonal_covariance = np.diag(safe_variances)
    floored_covariance = covariance.copy()
    np.fill_diagonal(floored_covariance, safe_variances)
    regularized_covariance = (
        (1.0 - shrinkage) * floored_covariance + shrinkage * diagonal_covariance
    )
    regularized_covariance = 0.5 * (
        regularized_covariance + regularized_covariance.T
    )

    eigenvalues, eigenvectors = np.linalg.eigh(regularized_covariance)
    safe_eigenvalues = np.maximum(eigenvalues, variance_floor)
    transform = np.diag(1.0 / np.sqrt(safe_eigenvalues)) @ eigenvectors.T
    precision = eigenvectors @ np.diag(1.0 / safe_eigenvalues) @ eigenvectors.T
    effective_weights = np.sqrt(np.maximum(np.diag(precision), 0.0))

    diagnostics = {
        "weights": {
            name: float(value)
            for name, value in zip(moment_names, effective_weights)
        },
        "variances": {
            name: float(value) for name, value in zip(moment_names, variances)
        },
        "covariance": covariance,
        "regularized_covariance": regularized_covariance,
        "precision": precision,
        "covariance_regularization": shrinkage,
        "variance_floor": variance_floor,
        "covariance_condition_number": float(
            np.max(safe_eigenvalues) / np.min(safe_eigenvalues)
        ),
    }
    return transform, diagnostics


def _bootstrap_float(config: dict, key: str, default: float) -> float:
    value = config.get(key, default)
    if value in (None, ""):
        value = default
    return float(value)
