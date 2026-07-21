# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Weighted least-squares objective for moment-based susceptibility fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from paranmr.core.fitting.susceptibility.objectives.moments.differences import (
    build_moment_difference_vector,
)
from paranmr.core.fitting.susceptibility.objectives.moments.ls.weighting import (
    build_ls_weight_vector,
    build_ls_weights_by_name,
)


@dataclass(frozen=True)
class WeightedLSMomentObjective:
    """Weighted least-squares objective for normalized moment differences."""

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
        weights_by_name = build_ls_weights_by_name(
            moment_names=moment_names,
            weights=weights,
        )
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
        return {"moment_weights": dict(self.weights_by_name)}

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
        """Return weighted residuals for normalized moment differences."""
        condition_vector = self.conditions(
            observed_moments=observed_moments,
            calculated_moments=calculated_moments,
        )
        weights = build_ls_weight_vector(
            moment_names=self.moment_names,
            weights_by_name=self.weights_by_name,
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
