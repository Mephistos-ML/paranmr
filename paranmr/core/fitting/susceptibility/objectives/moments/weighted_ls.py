# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Weighted least-squares objective for moment-based susceptibility fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from paranmr.core.fitting.susceptibility.moments.descriptors import (
    normalize_gaussian_mixture_moment_vectors,
)
from paranmr.core.fitting.susceptibility.objectives.moments.conditions import (
    build_moment_condition_vector,
)


@dataclass(frozen=True)
class WeightedLSMomentObjective:
    """Weighted relative-residual objective for Gaussian-mixture moments."""

    moment_names: tuple[str, ...]
    weights_by_name: dict[str, float]

    @classmethod
    def from_config(
        cls,
        *,
        moment_names: tuple[str, ...],
        weights: dict[str, float] | None,
    ) -> "WeightedLSMomentObjective":
        """Build a weighted least-squares moment objective from user weights."""
        if weights is None:
            weights = {}

        unknown = set(weights) - set(moment_names)
        if unknown:
            raise ValueError(
                "Moment weights contain unknown moment(s): "
                + ", ".join(sorted(unknown))
            )

        weights_by_name = {}
        for moment_name in moment_names:
            weight = float(weights.get(moment_name, 1.0))
            if weight < 0.0:
                raise ValueError("Moment weights must be non-negative")
            weights_by_name[moment_name] = weight

        return cls(moment_names=moment_names, weights_by_name=weights_by_name)

    @property
    def objective_type(self) -> str:
        """Return the public objective type name."""
        return "ls"

    @property
    def active_mask(self) -> NDArray[np.bool_]:
        """Return active moment components for uncertainty estimation."""
        return np.asarray(
            [self.weights_by_name[name] != 0.0 for name in self.moment_names],
            dtype=bool,
        )

    @property
    def diagnostics(self) -> dict[str, dict[str, float]]:
        """Return serializable objective diagnostics."""
        return {"weights": dict(self.weights_by_name)}

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
        """Return weighted residuals for the configured moment order.

        The residual for moment ``n`` is
        ``weight_n * (calculated_n / observed_n - 1)``.
        """
        normalized_observed, normalized_calculated = (
            normalize_gaussian_mixture_moment_vectors(
                observed=observed_moments,
                calculated=calculated_moments,
            )
        )
        condition_vector = self.conditions(
            observed_moments=normalized_observed,
            calculated_moments=normalized_calculated,
        )
        weights = np.asarray(
            [self.weights_by_name[moment_name] for moment_name in self.moment_names],
            dtype=float,
        )
        return weights * condition_vector

    def score(
        self,
        *,
        observed_moments: dict[str, float],
        calculated_moments: dict[str, float],
    ) -> float:
        """Return the weighted residual norm used in diagnostics."""
        residuals = self.residuals(
            observed_moments=observed_moments,
            calculated_moments=calculated_moments,
        )
        return float(np.sqrt(np.sum(residuals**2)))
