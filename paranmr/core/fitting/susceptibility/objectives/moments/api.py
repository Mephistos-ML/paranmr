# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Build moment objectives for susceptibility fitting."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from paranmr.core.fitting.susceptibility.objectives.moments.gmm.objective import (
    GMMMomentObjective,
)
from paranmr.core.fitting.susceptibility.objectives.moments.ls.objective import (
    WeightedLSMomentObjective,
)


class MomentObjective(Protocol):
    """Protocol implemented by moment objective functions."""

    @property
    def objective_type(self) -> str:
        """Return the public objective type name."""
        ...

    @property
    def active_mask(self) -> NDArray[np.bool_]:
        """Return active residual components for uncertainty estimation."""
        ...

    def conditions(
        self,
        *,
        observed_moments: dict[str, float],
        calculated_moments: dict[str, float],
    ) -> NDArray[np.float64]:
        """Return the shared moment-condition vector ``m_calc - m_exp``."""
        ...

    def residuals(
        self,
        *,
        observed_moments: dict[str, float],
        calculated_moments: dict[str, float],
    ) -> NDArray[np.float64]:
        """Return the residual vector consumed by least-squares optimizers."""
        ...

    def score(
        self,
        *,
        observed_moments: dict[str, float],
        calculated_moments: dict[str, float],
    ) -> float:
        """Return the scalar diagnostic score for a fitted model."""
        ...


def prepare_moment_objective(
    *,
    observed_moments: dict[str, float],
    objective_config: dict | None = None,
) -> MomentObjective:
    """Build the configured moment objective.

    Args:
        observed_moments: Experimental moment descriptors. Their key order
            defines the residual order used by the optimizer.
        objective_config: Parsed ``assignment:moment_objective`` mapping.

    Returns:
        A moment objective instance.

    Raises:
        NotImplementedError: If the requested objective is recognized but not
            implemented.
        ValueError: If the objective type is unknown.
    """
    if objective_config is None:
        objective_config = {"type": "ls", "weights": {}}

    objective_type = str(objective_config.get("type", "ls")).lower()
    moment_names = tuple(observed_moments.keys())

    if objective_type == "ls":
        return WeightedLSMomentObjective.from_config(
            moment_names=moment_names,
            weights=objective_config.get("weights", {}),
        )

    if objective_type == "gmm":
        return GMMMomentObjective.from_config(
            moment_names=moment_names,
            objective_config=objective_config,
        )

    raise ValueError(
        "Unknown moment objective type "
        f"{objective_type!r}. Supported values are 'ls' and 'gmm'."
    )
