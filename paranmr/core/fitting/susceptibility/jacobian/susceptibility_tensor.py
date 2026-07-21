# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Analytical susceptibility-tensor derivatives for moment Jacobians."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from paranmr.core.fitting.susceptibility.models.isoaxrho_euler import (
    IsoAxRhoEulerFitter,
)


def differentiate_tensor_by_susc_ax(
    parameters: dict[str, float],
) -> NDArray[np.float64]:
    """Return ``dχ/dax`` in the working frame."""

    rho_over_ax = float(parameters["rho_over_ax"])
    tensor_paf = np.array(
        [
            [-1.0 / 3.0 + rho_over_ax, 0.0, 0.0],
            [0.0, -1.0 / 3.0 - rho_over_ax, 0.0],
            [0.0, 0.0, 2.0 / 3.0],
        ],
        dtype=float,
    )

    if all(name in parameters for name in ("alpha", "beta", "gamma")):
        rotation = IsoAxRhoEulerFitter._zyz_rotation(
            float(parameters["alpha"]),
            float(parameters["beta"]),
            float(parameters["gamma"]),
        )
        return rotation @ tensor_paf @ rotation.T
    return tensor_paf


def differentiate_tensor_by_susc_iso(
    parameters: dict[str, float],
) -> NDArray[np.float64]:
    """Return ``dχ/diso`` in the working frame."""

    return np.eye(3, dtype=float)


def differentiate_tensor_by_susc_rho_over_ax(
    parameters: dict[str, float],
) -> NDArray[np.float64]:
    """Return ``dχ/d(rho_over_ax)`` in the working frame."""

    ax = float(parameters["ax"])
    tensor_paf = np.array(
        [
            [ax, 0.0, 0.0],
            [0.0, -ax, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    if all(name in parameters for name in ("alpha", "beta", "gamma")):
        rotation = IsoAxRhoEulerFitter._zyz_rotation(
            float(parameters["alpha"]),
            float(parameters["beta"]),
            float(parameters["gamma"]),
        )
        return rotation @ tensor_paf @ rotation.T
    return tensor_paf


def differentiate_tensor_by_alpha(parameters: dict[str, float]) -> NDArray[np.float64]:
    """Return ``dχ/dalpha`` in the working frame."""

    tensor_paf = tensor_paf_from_parameters(parameters)
    rotation, d_rotation_by_alpha, _, _ = rotation_and_derivatives(parameters)
    return (
        d_rotation_by_alpha @ tensor_paf @ rotation.T
        + rotation @ tensor_paf @ d_rotation_by_alpha.T
    )


def differentiate_tensor_by_beta(parameters: dict[str, float]) -> NDArray[np.float64]:
    """Return ``dχ/dbeta`` in the working frame."""

    tensor_paf = tensor_paf_from_parameters(parameters)
    rotation, _, d_rotation_by_beta, _ = rotation_and_derivatives(parameters)
    return (
        d_rotation_by_beta @ tensor_paf @ rotation.T
        + rotation @ tensor_paf @ d_rotation_by_beta.T
    )


def differentiate_tensor_by_gamma(
    parameters: dict[str, float],
) -> NDArray[np.float64]:
    """Return ``dχ/dgamma`` in the working frame."""

    tensor_paf = tensor_paf_from_parameters(parameters)
    rotation, _, _, d_rotation_by_gamma = rotation_and_derivatives(parameters)
    return (
        d_rotation_by_gamma @ tensor_paf @ rotation.T
        + rotation @ tensor_paf @ d_rotation_by_gamma.T
    )


def tensor_paf_from_parameters(parameters: dict[str, float]) -> NDArray[np.float64]:
    """Return the principal-axis-frame susceptibility tensor."""

    ax = float(parameters["ax"])
    rho_over_ax = float(parameters["rho_over_ax"])
    tensor_paf = np.array(
        [
            [-ax / 3.0 + rho_over_ax * ax, 0.0, 0.0],
            [0.0, -ax / 3.0 - rho_over_ax * ax, 0.0],
            [0.0, 0.0, 2.0 / 3.0 * ax],
        ],
        dtype=float,
    )
    tensor_paf += np.eye(3) * float(parameters["iso"])
    return tensor_paf


def rotation_and_derivatives(
    parameters: dict[str, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return the ZYZ rotation and its derivatives by Euler angles."""

    alpha = np.deg2rad(float(parameters.get("alpha", 0.0)))
    beta = np.deg2rad(float(parameters.get("beta", 0.0)))
    gamma = np.deg2rad(float(parameters.get("gamma", 0.0)))
    deg_to_rad = np.pi / 180.0

    cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
    cos_beta, sin_beta = np.cos(beta), np.sin(beta)
    cos_gamma, sin_gamma = np.cos(gamma), np.sin(gamma)

    rz_alpha = np.array(
        [
            [cos_alpha, -sin_alpha, 0.0],
            [sin_alpha, cos_alpha, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    ry_beta = np.array(
        [
            [cos_beta, 0.0, sin_beta],
            [0.0, 1.0, 0.0],
            [-sin_beta, 0.0, cos_beta],
        ],
        dtype=float,
    )
    rz_gamma = np.array(
        [
            [cos_gamma, -sin_gamma, 0.0],
            [sin_gamma, cos_gamma, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    d_rz_alpha = deg_to_rad * np.array(
        [
            [-sin_alpha, -cos_alpha, 0.0],
            [cos_alpha, -sin_alpha, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    d_ry_beta = deg_to_rad * np.array(
        [
            [-sin_beta, 0.0, cos_beta],
            [0.0, 0.0, 0.0],
            [-cos_beta, 0.0, -sin_beta],
        ],
        dtype=float,
    )
    d_rz_gamma = deg_to_rad * np.array(
        [
            [-sin_gamma, -cos_gamma, 0.0],
            [cos_gamma, -sin_gamma, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    rotation = rz_alpha @ ry_beta @ rz_gamma
    d_rotation_by_alpha = d_rz_alpha @ ry_beta @ rz_gamma
    d_rotation_by_beta = rz_alpha @ d_ry_beta @ rz_gamma
    d_rotation_by_gamma = rz_alpha @ ry_beta @ d_rz_gamma
    return rotation, d_rotation_by_alpha, d_rotation_by_beta, d_rotation_by_gamma
