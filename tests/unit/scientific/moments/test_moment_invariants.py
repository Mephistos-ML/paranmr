import numpy as np
import pytest

from paranmr.core.fitting.susceptibility.moments.descriptors import (
    compute_gaussian_mixture_moments,
)


def test_gaussian_mixture_moments_are_invariant_to_component_permutation():
    centers = np.array([-2.0, 1.5, 4.0])
    sigmas = np.array([0.3, 0.7, 1.1])
    weights = np.array([0.2, 0.5, 0.3])
    labels = ("m1", "m2", "m3", "m4", "m5")
    baseline = compute_gaussian_mixture_moments(
        centers=centers, sigmas=sigmas, area_norm=weights, moment_labels=labels
    )
    permutation = np.array([2, 0, 1])
    reordered = compute_gaussian_mixture_moments(
        centers=centers[permutation],
        sigmas=sigmas[permutation],
        area_norm=weights[permutation],
        moment_labels=labels,
    )

    assert reordered == pytest.approx(baseline)


def test_first_moment_translates_by_the_coordinate_offset():
    centers = np.array([-1.0, 2.0])
    sigmas = np.array([0.5, 0.8])
    weights = np.array([0.25, 0.75])
    offset = 3.25
    baseline = compute_gaussian_mixture_moments(
        centers=centers, sigmas=sigmas, area_norm=weights, moment_labels=("m1",)
    )["m1"]
    translated = compute_gaussian_mixture_moments(
        centers=centers + offset,
        sigmas=sigmas,
        area_norm=weights,
        moment_labels=("m1",),
    )["m1"]

    assert translated == pytest.approx(baseline + offset)
