# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

import numpy as np
import pytest

from paranmr.core.fitting.susceptibility.moments.descriptors import (
    build_normalized_moment_vectors,
)
from paranmr.core.fitting.susceptibility.objectives.moments.differences import (
    build_moment_difference_vector,
)
from paranmr.core.fitting.susceptibility.objectives.moments.gmm.objective import (
    GMMMomentObjective,
)
from paranmr.core.fitting.susceptibility.objectives.moments.ls.objective import (
    WeightedLSMomentObjective,
)

MOMENT_LABELS = ("m1", "m2", "m3", "m4", "m5", "m6")


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
        observed_moments=build_normalized_moment_vectors(
            observed=observed,
            calculated=calculated,
            moment_names=MOMENT_LABELS,
        ).observed,
        calculated_moments=build_normalized_moment_vectors(
            observed=observed,
            calculated=calculated,
            moment_names=MOMENT_LABELS,
        ).calculated,
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
        observed_moments=build_normalized_moment_vectors(
            observed=observed,
            calculated=calculated,
            moment_names=MOMENT_LABELS,
        ).observed,
        calculated_moments=build_normalized_moment_vectors(
            observed=observed,
            calculated=calculated,
            moment_names=MOMENT_LABELS,
        ).calculated,
    ) == pytest.approx(float(np.sqrt(np.sum(expected**2))))


@pytest.mark.unit
def test_weighted_ls_moment_objective_transforms_residual_jacobian_rowwise():
    objective = WeightedLSMomentObjective.from_config(
        moment_names=("m1", "m2"),
        weights={"m1": 2.0, "m2": 3.0},
    )

    residual_jacobian = objective.residual_jacobian(
        moment_jacobian=np.asarray([[1.0, 4.0], [2.0, 5.0]], dtype=float)
    )

    assert residual_jacobian == pytest.approx(
        np.asarray([[2.0, 8.0], [6.0, 15.0]], dtype=float)
    )


@pytest.mark.unit
def test_weighted_ls_moment_objective_exposes_raw_condition_vector():
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

    conditions = objective.conditions(
        observed_moments=observed,
        calculated_moments=calculated,
    )

    assert conditions == pytest.approx(
        np.asarray([1.0, -2.0, 4.0, 0.0, -16.0, 64.0], dtype=float)
    )

@pytest.mark.unit
def test_prepare_moment_objective_builds_gmm_objective_from_weighting_matrix():
    objective = GMMMomentObjective.with_weighting_matrix(
        moment_names=MOMENT_LABELS,
        weighting_matrix=np.eye(6, dtype=float),
    )

    assert objective.objective_type == "gmm"
    assert np.array_equal(objective.active_mask, np.ones(6, dtype=bool))


@pytest.mark.unit
def test_prepare_moment_objective_rejects_gmm_without_weighting_matrix():
    with pytest.raises(TypeError):
        GMMMomentObjective.with_weighting_matrix(  # type: ignore[call-arg]
            moment_names=MOMENT_LABELS,
        )


@pytest.mark.unit
def test_gmm_moment_objective_returns_raw_condition_residuals_for_identity_weighting():
    objective = GMMMomentObjective.with_weighting_matrix(
        moment_names=MOMENT_LABELS,
        weighting_matrix=np.eye(6, dtype=float),
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

    expected = np.asarray([1.0, -2.0, 4.0, 0.0, -16.0, 64.0], dtype=float)
    assert residuals == pytest.approx(expected)
    assert objective.score(
        observed_moments=observed,
        calculated_moments=calculated,
    ) == pytest.approx(float(np.sqrt(np.sum(expected**2))))


@pytest.mark.unit
def test_gmm_moment_objective_applies_general_weighting_matrix():
    objective = GMMMomentObjective.with_weighting_matrix(
        moment_names=("m1", "m2"),
        weighting_matrix=np.asarray([[4.0, 0.0], [0.0, 9.0]], dtype=float),
    )
    observed = {"m1": 1.0, "m2": 2.0}
    calculated = {"m1": 3.0, "m2": 5.0}

    residuals = objective.residuals(
        observed_moments=observed,
        calculated_moments=calculated,
    )

    assert residuals == pytest.approx(np.asarray([4.0, 9.0], dtype=float))


@pytest.mark.unit
def test_gmm_moment_objective_score_matches_weighting_quadratic_form():
    weighting_matrix = np.asarray([[5.0, 2.0], [2.0, 3.0]], dtype=float)
    objective = GMMMomentObjective.with_weighting_matrix(
        moment_names=("m1", "m2"),
        weighting_matrix=weighting_matrix,
    )
    observed = {"m1": 1.0, "m2": 2.0}
    calculated = {"m1": 3.0, "m2": 5.0}

    conditions = objective.conditions(
        observed_moments=observed,
        calculated_moments=calculated,
    )
    score = objective.score(
        observed_moments=observed,
        calculated_moments=calculated,
    )

    expected = float(np.sqrt(conditions @ weighting_matrix @ conditions))
    assert score == pytest.approx(expected)


@pytest.mark.unit
def test_build_moment_difference_vector_returns_calculated_minus_observed_in_order():
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

    vector = build_moment_difference_vector(
        observed_moments=observed,
        calculated_moments=calculated,
        moment_names=("m1", "m2", "m3", "m4", "m5", "m6"),
    )

    assert vector == pytest.approx(np.asarray([1.0, -2.0, 4.0, 0.0, -16.0, 64.0]))


@pytest.mark.unit
def test_gmm_moment_objective_transforms_residual_jacobian_by_cholesky_factor():
    objective = GMMMomentObjective.with_weighting_matrix(
        moment_names=("m1", "m2"),
        weighting_matrix=np.asarray([[4.0, 0.0], [0.0, 9.0]], dtype=float),
    )

    residual_jacobian = objective.residual_jacobian(
        moment_jacobian=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    )

    assert residual_jacobian == pytest.approx(
        np.asarray([[2.0, 4.0], [9.0, 12.0]], dtype=float)
    )


@pytest.mark.unit
def test_gmm_moment_objective_uses_transposed_cholesky_factor():
    weighting_matrix = np.asarray([[5.0, 2.0], [2.0, 3.0]], dtype=float)
    objective = GMMMomentObjective.with_weighting_matrix(
        moment_names=("m1", "m2"),
        weighting_matrix=weighting_matrix,
    )
    raw_jacobian = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)

    residual_jacobian = objective.residual_jacobian(
        moment_jacobian=raw_jacobian,
    )
    expected = np.linalg.cholesky(weighting_matrix).T @ raw_jacobian

    assert residual_jacobian == pytest.approx(expected)
