# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Generalized-method-of-moments objectives for susceptibility fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular


@dataclass(frozen=True)
class GMMMomentObjective:
    """Moment objective for the two-step generalized method of moments workflow."""

    moment_names: tuple[str, ...]
    covariance_factor: NDArray[np.float64]

    @classmethod
    def with_covariance(
        cls,
        *,
        moment_names: tuple[str, ...],
        covariance: NDArray[np.float64],
    ) -> "GMMMomentObjective":
        """Build a GMM objective directly from its covariance matrix."""
        from .weighting import build_gmm_whitening_factor

        covariance_factor = build_gmm_whitening_factor(covariance)
        n_moments = len(moment_names)
        if covariance_factor.shape != (n_moments, n_moments):
            raise ValueError(
                "GMM covariance factor shape does not match the configured moment count"
            )
        return cls(
            moment_names=moment_names,
            covariance_factor=covariance_factor,
        )

    def conditions(
        self,
        *,
        observed_moments: dict[str, float],
        calculated_moments: dict[str, float],
    ) -> NDArray[np.float64]:
        """Return the raw moment-condition vector ``m_calc - m_exp``."""
        return np.asarray(
            [
                float(calculated_moments[name]) - float(observed_moments[name])
                for name in self.moment_names
            ],
            dtype=float,
        )

    def residuals(
        self,
        *,
        observed_moments: dict[str, float],
        calculated_moments: dict[str, float],
    ) -> NDArray[np.float64]:
        """Return residuals transformed by the current GMM weighting."""
        return solve_triangular(
            self.covariance_factor,
            self.conditions(
                observed_moments=observed_moments,
                calculated_moments=calculated_moments,
            ),
            lower=True,
        )

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
