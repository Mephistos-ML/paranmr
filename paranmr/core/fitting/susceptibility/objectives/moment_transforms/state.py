# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Shared state helpers for moment objective transforms."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def build_moment_objective_state(
    *,
    objective_type: str,
    moment_names: tuple[str, ...],
    transform: NDArray[np.float64],
    diagnostics: dict,
    component_names: tuple[str, ...] | None = None,
) -> dict:
    """Build the state consumed by the moment-fitting least-squares callback."""
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


def weights_from_transform(
    moment_names: tuple[str, ...],
    transform: NDArray[np.float64],
) -> dict[str, float]:
    """Return diagonal effective weights from a residual transform matrix."""
    diagonal = np.diag(np.asarray(transform, dtype=float))
    return {name: float(value) for name, value in zip(moment_names, diagonal)}


def bootstrap_float(config: dict, key: str, default: float) -> float:
    """Read a float bootstrap option with support for blank defaults."""
    value = config.get(key, default)
    if value in (None, ""):
        value = default
    return float(value)
