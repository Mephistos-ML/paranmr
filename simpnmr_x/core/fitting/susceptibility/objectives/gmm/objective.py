# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Fixed relative-moment objective for susceptibility fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GMMMomentObjective:
    """Observed-scaled moment objective and its analytical Jacobian contract."""

    moment_names: tuple[str, ...]
    observed_moments: Mapping[str, float]

    def __post_init__(self) -> None:
        """Validate and freeze the observed moments defining this objective."""
        observed = {
            name: float(self.observed_moments[name]) for name in self.moment_names
        }
        if not np.all(np.isfinite(list(observed.values()))):
            raise ValueError("Observed moments must be finite")
        object.__setattr__(self, "observed_moments", observed)

    def subset(self, moment_names: tuple[str, ...]) -> "GMMMomentObjective":
        """Return the objective restricted to a leading continuation stage."""

        if self.moment_names[: len(moment_names)] != moment_names:
            raise ValueError("Continuation moment names must be a leading subset")
        return GMMMomentObjective(
            moment_names=moment_names,
            observed_moments=self.observed_moments,
        )

    def conditions(
        self,
        *,
        calculated_moments: dict[str, float],
    ) -> NDArray[np.float64]:
        """Return the raw moment-condition vector ``m_calc - m_exp``."""
        return np.asarray(
            [
                float(calculated_moments[name]) - self.observed_moments[name]
                for name in self.moment_names
            ],
            dtype=float,
        )

    def residuals(
        self,
        *,
        calculated_moments: dict[str, float],
    ) -> NDArray[np.float64]:
        """Return residuals scaled by the fixed configured moment scales."""
        conditions = self.conditions(
            calculated_moments=calculated_moments,
        )
        return conditions / _relative_observed_scales(
            self.moment_names, self.observed_moments
        )

    def residual_jacobian(
        self,
        *,
        moment_jacobian: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the analytical Jacobian of the configured residual vector."""
        jacobian = _validate_moment_jacobian(moment_jacobian, len(self.moment_names))
        return (
            jacobian
            / _relative_observed_scales(self.moment_names, self.observed_moments)[
                :, np.newaxis
            ]
        )

    def weighting_matrix(self) -> NDArray[np.float64]:
        """Return the fixed quadratic-form matrix for the configured scales."""

        scales = _relative_observed_scales(self.moment_names, self.observed_moments)
        return np.diag(scales**-2)

    def score(
        self,
        *,
        calculated_moments: dict[str, float],
    ) -> float:
        """Return the Euclidean norm of the scaled moment residual vector."""
        residuals = self.residuals(
            calculated_moments=calculated_moments,
        )
        return float(np.sqrt(np.sum(residuals**2)))


def _relative_observed_scales(
    moment_names: tuple[str, ...], observed_moments: Mapping[str, float]
) -> NDArray[np.float64]:
    """Return the fixed relative scale for each configured moment."""
    return np.asarray(
        [max(1.0, abs(observed_moments[name])) for name in moment_names], dtype=float
    )


def _validate_moment_jacobian(
    moment_jacobian: NDArray[np.float64], n_moments: int
) -> NDArray[np.float64]:
    jacobian = np.asarray(moment_jacobian, dtype=float)
    if jacobian.ndim != 2 or jacobian.shape[0] != n_moments:
        raise ValueError(
            "Moment Jacobian shape does not match the configured moment conditions"
        )
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("Moment Jacobian must be finite")
    return jacobian
