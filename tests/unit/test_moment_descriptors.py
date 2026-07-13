# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

import pytest

from paranmr.core.fitting.susceptibility.moments.descriptors import (
    MAX_MOMENT_ORDER,
    MOMENT_NAMES,
    compute_gaussian_mixture_moments,
    moment_n,
    normalize_gaussian_mixture_moment_vectors,
)


def _manual_gaussian_mixture_moments_3_to_6():
    centers = [-1.2, 0.7, 2.4]
    sigmas = [0.3, 0.5, 0.8]
    weights = [0.2, 0.5, 0.3]

    mean = sum(weight * center for weight, center in zip(weights, centers))
    delta = [center - mean for center in centers]

    m3 = sum(
        weight * (d**3 + 3.0 * d * sigma**2)
        for weight, d, sigma in zip(weights, delta, sigmas)
    )
    m4 = sum(
        weight * (d**4 + 6.0 * d**2 * sigma**2 + 3.0 * sigma**4)
        for weight, d, sigma in zip(weights, delta, sigmas)
    )
    m5 = sum(
        weight * (d**5 + 10.0 * d**3 * sigma**2 + 15.0 * d * sigma**4)
        for weight, d, sigma in zip(weights, delta, sigmas)
    )
    m6 = sum(
        weight
        * (d**6 + 15.0 * d**4 * sigma**2 + 45.0 * d**2 * sigma**4 + 15.0 * sigma**6)
        for weight, d, sigma in zip(weights, delta, sigmas)
    )
    return centers, sigmas, weights, m3, m4, m5, m6


@pytest.mark.unit
def test_compute_gaussian_mixture_moments_returns_raw_central_moments():
    moments = compute_gaussian_mixture_moments(
        centers=[-1.0, 1.0],
        sigmas=[0.5, 0.5],
        area_norm=[0.5, 0.5],
    )

    assert tuple(moments) == MOMENT_NAMES
    assert moments["m1"] == pytest.approx(0.0)
    assert moments["m2"] > 0.0
    assert moments["m3"] == pytest.approx(0.0)
    assert moments["m5"] == pytest.approx(0.0)


@pytest.mark.unit
def test_compute_gaussian_mixture_moments_general_formula_matches_manual_3_to_6():
    centers, sigmas, weights, m3, m4, m5, m6 = _manual_gaussian_mixture_moments_3_to_6()

    moments = compute_gaussian_mixture_moments(
        centers=centers,
        sigmas=sigmas,
        area_norm=weights,
    )

    assert moments["m3"] == pytest.approx(m3)
    assert moments["m4"] == pytest.approx(m4)
    assert moments["m5"] == pytest.approx(m5)
    assert moments["m6"] == pytest.approx(m6)


@pytest.mark.unit
def test_moment_metadata_is_generated_from_max_order():
    assert MAX_MOMENT_ORDER == 6
    assert MOMENT_NAMES == tuple(f"m{order}" for order in range(1, 7))
    assert [moment_n(order) for order in range(1, 7)] == list(MOMENT_NAMES)


@pytest.mark.unit
def test_normalize_gaussian_mixture_moment_vectors_scales_all_orders():
    observed = {
        "m1": 4.0,
        "m2": 4.0,
        "m3": 16.0,
        "m4": 32.0,
        "m5": 64.0,
        "m6": 128.0,
    }
    calculated = {
        "m1": 2.0,
        "m2": 1.0,
        "m3": 8.0,
        "m4": 16.0,
        "m5": 32.0,
        "m6": 64.0,
    }

    norm_observed, norm_calculated = normalize_gaussian_mixture_moment_vectors(
        observed=observed,
        calculated=calculated,
    )

    assert norm_observed == pytest.approx(
        {
            "m1": 1.0,
            "m2": 1.0,
            "m3": 1.0,
            "m4": 1.0,
            "m5": 1.0,
            "m6": 1.0,
        }
    )
    assert norm_calculated == pytest.approx(
        {
            "m1": 0.5,
            "m2": 0.25,
            "m3": 0.5,
            "m4": 0.5,
            "m5": 0.5,
            "m6": 0.5,
        }
    )


@pytest.mark.unit
def test_normalize_gaussian_mixture_moment_vectors_fails_loudly_on_zero_observed():
    observed = {
        "m1": 4.0,
        "m2": 4.0,
        "m3": 0.0,
        "m4": 32.0,
        "m5": 64.0,
        "m6": 128.0,
    }
    calculated = {
        "m1": 2.0,
        "m2": 1.0,
        "m3": 8.0,
        "m4": 16.0,
        "m5": 32.0,
        "m6": 64.0,
    }

    with pytest.raises(ValueError, match="zero or too close to zero: m3"):
        normalize_gaussian_mixture_moment_vectors(
            observed=observed,
            calculated=calculated,
        )
