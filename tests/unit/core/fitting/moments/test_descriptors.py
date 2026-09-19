# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

import pytest

from paranmr.core.fitting.susceptibility.moments.descriptors import (
    compute_gaussian_mixture_moments,
    compute_single_gaussian_mixture_raw_moment,
    moment_n,
)

MOMENT_LABELS = tuple(f"m{order}" for order in range(1, 7))


def _manual_gaussian_mixture_raw_moments_1_to_6():
    centers = [-1.2, 0.7, 2.4]
    sigmas = [0.3, 0.5, 0.8]
    weights = [0.2, 0.5, 0.3]

    def raw(order: int) -> float:
        total = 0.0
        for weight, center, sigma in zip(weights, centers, sigmas):
            if order == 1:
                total += weight * center
            elif order == 2:
                total += weight * (center**2 + sigma**2)
            elif order == 3:
                total += weight * (center**3 + 3.0 * center * sigma**2)
            elif order == 4:
                total += weight * (
                    center**4 + 6.0 * center**2 * sigma**2 + 3.0 * sigma**4
                )
            elif order == 5:
                total += weight * (
                    center**5 + 10.0 * center**3 * sigma**2 + 15.0 * center * sigma**4
                )
            elif order == 6:
                total += weight * (
                    center**6
                    + 15.0 * center**4 * sigma**2
                    + 45.0 * center**2 * sigma**4
                    + 15.0 * sigma**6
                )
            else:
                raise AssertionError("unsupported manual raw moment order")
        return total

    return centers, sigmas, weights, {f"m{order}": raw(order) for order in range(1, 7)}


@pytest.mark.unit
def test_compute_gaussian_mixture_moments_returns_raw_moments():
    moments = compute_gaussian_mixture_moments(
        centers=[-1.0, 1.0],
        sigmas=[0.5, 0.5],
        area_norm=[0.5, 0.5],
        moment_labels=MOMENT_LABELS,
    )

    assert tuple(moments) == MOMENT_LABELS
    assert moments["m1"] == pytest.approx(0.0)
    assert moments["m2"] == pytest.approx(1.25)
    assert moments["m3"] == pytest.approx(0.0)
    assert moments["m4"] == pytest.approx(2.6875)
    assert moments["m5"] == pytest.approx(0.0)


@pytest.mark.unit
def test_compute_gaussian_mixture_moments_includes_normalized_zeroth_moment():
    moments = compute_gaussian_mixture_moments(
        centers=[-1.0, 1.0],
        sigmas=[0.5, 0.5],
        area_norm=[0.5, 0.5],
        moment_labels=("m0", "m1"),
    )

    assert moments == pytest.approx({"m0": 1.0, "m1": 0.0})


@pytest.mark.unit
def test_compute_gaussian_mixture_moments_matches_manual_raw_formula_1_to_6():
    centers, sigmas, weights, expected = _manual_gaussian_mixture_raw_moments_1_to_6()

    moments = compute_gaussian_mixture_moments(
        centers=centers,
        sigmas=sigmas,
        area_norm=weights,
        moment_labels=MOMENT_LABELS,
    )

    assert moments == pytest.approx(expected)


@pytest.mark.unit
def test_compute_gaussian_mixture_moments_preserves_sparse_requested_order():
    centers, sigmas, weights, expected = _manual_gaussian_mixture_raw_moments_1_to_6()
    labels = ("m6", "m1", "m0")

    moments = compute_gaussian_mixture_moments(
        centers=centers,
        sigmas=sigmas,
        area_norm=weights,
        moment_labels=labels,
    )

    assert tuple(moments) == labels
    assert moments == pytest.approx(
        {"m6": expected["m6"], "m1": expected["m1"], "m0": sum(weights)}
    )


@pytest.mark.unit
def test_compute_single_gaussian_mixture_raw_moment_matches_wrapper_component():
    moments = compute_gaussian_mixture_moments(
        centers=[-1.2, 0.7, 2.4],
        sigmas=[0.3, 0.5, 0.8],
        area_norm=[0.2, 0.5, 0.3],
        moment_labels=MOMENT_LABELS,
    )

    for order in range(1, 7):
        assert compute_single_gaussian_mixture_raw_moment(
            centers=[-1.2, 0.7, 2.4],
            sigmas=[0.3, 0.5, 0.8],
            area_norm=[0.2, 0.5, 0.3],
            order=order,
        ) == pytest.approx(moments[f"m{order}"])


@pytest.mark.unit
def test_moment_metadata_is_generated_from_max_order():
    assert MOMENT_LABELS == tuple(f"m{order}" for order in range(1, 7))
    assert [moment_n(order) for order in range(1, 7)] == list(MOMENT_LABELS)
