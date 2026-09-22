# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Tests for deterministic moment residual objectives."""

import numpy as np
import pytest

from simpnmr_x.core.fitting.susceptibility.objectives.gmm.objective import (
    GMMMomentObjective,
)


@pytest.mark.unit
def test_objective_scales_residuals_and_jacobian_by_observed_moments() -> None:
    observed = {"m1": 0.5, "m2": 100.0}
    objective = GMMMomentObjective(
        moment_names=("m1", "m2"),
        observed_moments=observed,
    )
    calculated = {"m1": 1.5, "m2": 300.0}
    raw_jacobian = np.asarray([[2.0, 4.0], [30.0, 50.0]])

    assert objective.residuals(
        calculated_moments=calculated,
    ) == pytest.approx(np.asarray([1.0, 2.0]))
    assert objective.residual_jacobian(
        moment_jacobian=raw_jacobian,
    ) == pytest.approx(np.asarray([[2.0, 4.0], [0.3, 0.5]]))
    assert objective.weighting_matrix() == pytest.approx(np.diag([1.0, 1.0e-4]))


@pytest.mark.unit
def test_objective_subsets_fixed_scales_for_continuation() -> None:
    observed = {"m1": 2.0, "m2": 10.0, "m3": 100.0}
    objective = GMMMomentObjective(
        moment_names=("m1", "m2", "m3"),
        observed_moments=observed,
    )

    stage = objective.subset(("m1", "m2"))

    assert stage.weighting_matrix() == pytest.approx(np.diag([0.25, 0.01]))
