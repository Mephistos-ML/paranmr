# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Generalized-method-of-moments objectives for susceptibility fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular
from numpy.typing import NDArray

from paranmr.core.fitting.susceptibility.objectives.moments.differences import (
    build_moment_difference_vector,
)


@dataclass(frozen=True)
class GMMMomentObjective:
    """Moment objective for the two-step generalized method of moments workflow."""

    moment_names: tuple[str, ...]
    weighting_matrix: NDArray[np.float64]
    covariance_factor: NDArray[np.float64]

    @classmethod
    def from_config(
        cls,
        *,
        moment_names: tuple[str, ...],
        objective_config: dict | None = None,
    ) -> "GMMMomentObjective":
        """Reject direct construction without an explicit weighting matrix."""
        del moment_names, objective_config
        raise NotImplementedError(
            "GMM objective construction requires an explicit covariance-derived "
            "weighting matrix."
        )

    @classmethod
    def with_covariance(
        cls,
        *,
        moment_names: tuple[str, ...],
        covariance: NDArray[np.float64],
    ) -> "GMMMomentObjective":
        """Build a GMM objective directly from its covariance matrix."""
        from .weighting import build_gmm_weighting_matrix, build_gmm_whitening_factor

        covariance_factor = build_gmm_whitening_factor(covariance)
        weighting_matrix = build_gmm_weighting_matrix(covariance)
        n_moments = len(moment_names)
        if weighting_matrix.shape != (n_moments, n_moments):
            raise ValueError(
                'GMM weighting matrix shape does not match the configured moment count'
            )
        if not np.allclose(weighting_matrix, weighting_matrix.T):
            raise ValueError('GMM weighting matrix must be symmetric')
        return cls(
            moment_names=moment_names,
            weighting_matrix=weighting_matrix,
            covariance_factor=covariance_factor,
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
        """Return the shared normalized moment-condition vector ``m_calc - m_exp``."""
        return build_moment_difference_vector(
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
        """Return the transformed residual vector implied by the current GMM weighting."""
        return solve_triangular(self.covariance_factor, self.conditions(
            observed_moments=observed_moments,
            calculated_moments=calculated_moments,
        ), lower=True)

    def residual_jacobian(
        self,
        *,
        moment_jacobian: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the residual Jacobian implied by the current GMM weighting."""
        return solve_triangular(
            self.covariance_factor,
            np.asarray(moment_jacobian, dtype=float),
            lower=True,
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
