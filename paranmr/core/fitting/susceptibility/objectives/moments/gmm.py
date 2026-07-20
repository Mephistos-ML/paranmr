# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Generalized-method-of-moments objectives for susceptibility fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from paranmr.core.fitting.susceptibility.objectives.moments.conditions import (
    build_moment_condition_vector,
)


@dataclass(frozen=True)
class GMMMomentObjective:
    """Moment objective for the two-step generalized method of moments workflow."""

    moment_names: tuple[str, ...]
    weighting_matrix: NDArray[np.float64]
    residual_transform: NDArray[np.float64]

    @classmethod
    def from_config(
        cls,
        *,
        moment_names: tuple[str, ...],
        objective_config: dict | None = None,
    ) -> "GMMMomentObjective":
        """Build the public GMM objective.

        The public YAML contract exposes only ``type: gmm``. Internally the
        workflow starts from the identity weighting matrix and will later be
        extended to the second-step covariance-weighted stage.
        """
        del objective_config
        return cls.with_weighting_matrix(
            moment_names=moment_names,
            weighting_matrix=np.eye(len(moment_names), dtype=float),
        )

    @classmethod
    def with_weighting_matrix(
        cls,
        *,
        moment_names: tuple[str, ...],
        weighting_matrix: NDArray[np.float64],
    ) -> "GMMMomentObjective":
        """Build a GMM objective from a symmetric positive-definite weighting matrix."""
        weighting_matrix = np.asarray(weighting_matrix, dtype=float)
        n_moments = len(moment_names)
        if weighting_matrix.shape != (n_moments, n_moments):
            raise ValueError(
                'GMM weighting matrix shape does not match the configured moment count'
            )
        if not np.allclose(weighting_matrix, weighting_matrix.T):
            raise ValueError('GMM weighting matrix must be symmetric')
        residual_transform = np.linalg.cholesky(weighting_matrix)
        return cls(
            moment_names=moment_names,
            weighting_matrix=weighting_matrix,
            residual_transform=residual_transform,
        )

    @property
    def objective_type(self) -> str:
        """Return the public objective type name."""
        return 'gmm'

    @property
    def active_mask(self) -> NDArray[np.bool_]:
        """Return active residual components for uncertainty estimation."""
        return np.ones(len(self.moment_names), dtype=bool)

    def conditions(
        self,
        *,
        observed_moments: dict[str, float],
        calculated_moments: dict[str, float],
    ) -> NDArray[np.float64]:
        """Return the shared raw moment-condition vector ``m_calc - m_exp``."""
        return build_moment_condition_vector(
            observed_moments=observed_moments,
            calculated_moments=calculated_moments,
            moment_names=self.moment_names,
        )

    def residuals(
        self,
        *,
        observed_moments: dict[str, float],
        calculated_moments: dict[str, float],
    ) -> NDArray[np.float64]:
        """Return the least-squares residual vector associated with the current GMM weighting."""
        return self.residual_transform @ self.conditions(
            observed_moments=observed_moments,
            calculated_moments=calculated_moments,
        )

    def score(
        self,
        *,
        observed_moments: dict[str, float],
        calculated_moments: dict[str, float],
    ) -> float:
        """Return the quadratic-form norm implied by the current GMM weighting."""
        residuals = self.residuals(
            observed_moments=observed_moments,
            calculated_moments=calculated_moments,
        )
        return float(np.sqrt(np.sum(residuals**2)))
