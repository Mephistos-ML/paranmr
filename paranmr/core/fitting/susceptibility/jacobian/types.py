# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Types for moment-Jacobian results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MomentJacobianResult:
    """Structured Jacobian matrix for a fitted moment-based model.

    Attributes:
        temperature: Temperature at which the Jacobian was evaluated, in Kelvin.
        moment_names: Public row order of the moment descriptors.
        parameter_names: Public column order of the fitted parameters.
        values: Jacobian matrix with shape
            ``(len(moment_names), len(parameter_names))``.
    """

    temperature: float
    moment_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate the published Jacobian contract."""
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 2:
            raise ValueError("Moment Jacobian values must be a two-dimensional array")
        expected_shape = (len(self.moment_names), len(self.parameter_names))
        if values.shape != expected_shape:
            raise ValueError(
                "Moment Jacobian shape does not match the declared row/column "
                f"labels: expected {expected_shape}, got {values.shape}"
            )
        if len(set(self.moment_names)) != len(self.moment_names):
            raise ValueError(
                "Moment Jacobian row labels must be unique"
            )
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError(
                "Moment Jacobian column labels must be unique"
            )
        object.__setattr__(self, "values", values)
