# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Objective-map diagnostics for susceptibility fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

@dataclass(frozen=True)
class ObjectiveMapConfig:
    """Configuration for a two-parameter objective map."""

    parameters: tuple[str, str]
    window_rel: float = 0.25
    n_grid: int = 60
    gradient: bool = True


@dataclass(frozen=True)
class ObjectiveMapResult:
    """Structured score grid for a two-parameter objective map."""

    temperature: float
    objective_type: str
    parameter_names: tuple[str, str]
    center_values: tuple[float, float]
    x_values: NDArray[np.float64]
    y_values: NDArray[np.float64]
    score_grid: NDArray[np.float64]
    gradient_x: NDArray[np.float64] | None = None
    gradient_y: NDArray[np.float64] | None = None


def build_objective_map(
    *,
    temperature: float,
    objective_type: str,
    parameter_names: tuple[str, ...],
    fit_vector: list[float] | NDArray[np.float64],
    fit_bounds: NDArray[np.float64],
    config: ObjectiveMapConfig,
    score_evaluator,
) -> ObjectiveMapResult:
    """Evaluate the configured objective on a two-parameter grid."""

    missing = [name for name in config.parameters if name not in parameter_names]
    if missing:
        raise ValueError(
            "Objective-map parameters must be active fitted parameters: "
            + ", ".join(missing)
        )

    fit_vector_arr = np.asarray(fit_vector, dtype=float)
    if fit_vector_arr.shape != (len(parameter_names),):
        raise ValueError("Objective-map fit vector does not match active fit dimension")
    fit_bounds_arr = np.asarray(fit_bounds, dtype=float)
    if fit_bounds_arr.shape != (2, len(parameter_names)):
        raise ValueError("Objective-map bounds must have shape (2, n_parameters)")

    x_name, y_name = config.parameters
    x_index = parameter_names.index(x_name)
    y_index = parameter_names.index(y_name)

    x_values = _objective_axis_values(
        center_value=float(fit_vector_arr[x_index]),
        lower_bound=float(fit_bounds_arr[0, x_index]),
        upper_bound=float(fit_bounds_arr[1, x_index]),
        window_rel=float(config.window_rel),
        n_grid=int(config.n_grid),
    )
    y_values = _objective_axis_values(
        center_value=float(fit_vector_arr[y_index]),
        lower_bound=float(fit_bounds_arr[0, y_index]),
        upper_bound=float(fit_bounds_arr[1, y_index]),
        window_rel=float(config.window_rel),
        n_grid=int(config.n_grid),
    )

    score_grid = np.empty((len(y_values), len(x_values)), dtype=float)
    for row_index, y_value in enumerate(y_values):
        for col_index, x_value in enumerate(x_values):
            point = fit_vector_arr.copy()
            point[x_index] = x_value
            point[y_index] = y_value
            score_grid[row_index, col_index] = float(score_evaluator(point))

    gradient_x = None
    gradient_y = None
    if config.gradient:
        gradient_y, gradient_x = np.gradient(score_grid, y_values, x_values)

    return ObjectiveMapResult(
        temperature=float(temperature),
        objective_type=str(objective_type),
        parameter_names=(x_name, y_name),
        center_values=(float(fit_vector_arr[x_index]), float(fit_vector_arr[y_index])),
        x_values=x_values,
        y_values=y_values,
        score_grid=score_grid,
        gradient_x=gradient_x,
        gradient_y=gradient_y,
    )


def _objective_axis_values(
    *,
    center_value: float,
    lower_bound: float,
    upper_bound: float,
    window_rel: float,
    n_grid: int,
) -> NDArray[np.float64]:
    eps = 1.0e-12
    if abs(center_value) >= eps:
        half_width = abs(center_value) * window_rel
    else:
        if np.isfinite(lower_bound) and np.isfinite(upper_bound):
            half_width = window_rel * (upper_bound - lower_bound)
        elif np.isfinite(lower_bound) or np.isfinite(upper_bound):
            finite_edge = upper_bound if np.isfinite(upper_bound) else lower_bound
            half_width = window_rel * abs(finite_edge)
        else:
            half_width = max(window_rel, 1.0e-3)
    half_width = max(float(half_width), eps)
    low = center_value - half_width
    high = center_value + half_width
    if np.isfinite(lower_bound):
        low = max(low, lower_bound)
    if np.isfinite(upper_bound):
        high = min(high, upper_bound)
    if not high > low:
        low = center_value - half_width
        high = center_value + half_width
    return np.linspace(low, high, int(n_grid), dtype=float)
