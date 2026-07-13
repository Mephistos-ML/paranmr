# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

import numpy as np
import pytest

from paranmr.core.fitting.susceptibility.objectives.moments.api import (
    prepare_moment_objective,
)
from paranmr.core.fitting.susceptibility.objectives.moments.weighted_ls import (
    WeightedLSMomentObjective,
)


@pytest.mark.unit
def test_weighted_ls_moment_objective_matches_relative_residual_formula():
    objective = WeightedLSMomentObjective.from_config(
        moment_names=("m1", "m2", "m3", "m4", "m5", "m6"),
        weights={"m1": 1.0, "m2": 5.0, "m3": 0.5},
    )
    observed = {
        "m1": 2.0,
        "m2": 4.0,
        "m3": 8.0,
        "m4": 16.0,
        "m5": 32.0,
        "m6": 64.0,
    }
    calculated = {
        "m1": 3.0,
        "m2": 2.0,
        "m3": 12.0,
        "m4": 16.0,
        "m5": 16.0,
        "m6": 128.0,
    }

    residuals = objective.residuals(
        observed_moments=observed,
        calculated_moments=calculated,
    )

    expected = np.asarray(
        [
            1.0 * (3.0 / 2.0 - 1.0),
            5.0 * (2.0 / 4.0 - 1.0),
            0.5 * (12.0 / 8.0 - 1.0),
            1.0 * (16.0 / 16.0 - 1.0),
            1.0 * (16.0 / 32.0 - 1.0),
            1.0 * (128.0 / 64.0 - 1.0),
        ],
        dtype=float,
    )
    assert residuals == pytest.approx(expected)
    assert objective.score(
        observed_moments=observed,
        calculated_moments=calculated,
    ) == pytest.approx(float(np.sqrt(np.sum(expected**2))))


@pytest.mark.unit
def test_prepare_moment_objective_reports_gmm_placeholder():
    observed = {
        "m1": 1.0,
        "m2": 2.0,
        "m3": 3.0,
        "m4": 4.0,
        "m5": 5.0,
        "m6": 6.0,
    }

    with pytest.raises(NotImplementedError, match="ls"):
        prepare_moment_objective(
            observed_moments=observed,
            objective_config={"type": "gmm"},
        )
