# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Five-coordinate Cartesian representation of symmetric traceless tensors."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from paranmr.core.fitting.susceptibility.models.isoaxrho_euler import (
    IsoAxRhoEulerFitter,
)


def cartesian_parameters_from_euler(
    parameters: dict[str, float],
) -> tuple[float, NDArray[np.float64]]:
    """Convert the public iso/ax/rho/ZYZ representation into Cartesian values.

    Args:
        parameters: ``iso``, ``ax``, ``rho_over_ax``, and ZYZ Euler angles in
            degrees, using the established susceptibility-fit convention.

    Returns:
        Isotropic susceptibility and five Cartesian traceless coordinates.
    """
    tensor = IsoAxRhoEulerFitter.totensor(parameters)
    isotropic = float(np.trace(tensor) / 3.0)
    traceless = tensor - isotropic * np.eye(3)
    coordinates = np.asarray(
        [
            0.5 * (traceless[0, 0] - traceless[1, 1]),
            traceless[2, 2],
            traceless[0, 1],
            traceless[0, 2],
            traceless[1, 2],
        ],
        dtype=float,
    )
    return isotropic, coordinates


def tensor_from_cartesian_parameters(
    *,
    isotropic: float,
    coordinates: ArrayLike,
) -> NDArray[np.float64]:
    """Build a symmetric susceptibility tensor from five traceless coordinates.

    Args:
        isotropic: Isotropic susceptibility component.
        coordinates: Five values ordered as half ``d_xx_minus_yy``, ``d_zz``,
            ``d_xy``, ``d_xz``, and ``d_yz``.

    Returns:
        Symmetric ``(3, 3)`` susceptibility tensor with the requested trace.

    Raises:
        ValueError: If ``coordinates`` does not have shape ``(5,)``.
    """
    values = np.asarray(coordinates, dtype=float)
    if values.shape != (5,):
        raise ValueError("Cartesian traceless coordinates must have shape (5,)")
    d_xx_minus_yy, d_zz, d_xy, d_xz, d_yz = values
    d_xx = d_xx_minus_yy - 0.5 * d_zz
    d_yy = -d_xx_minus_yy - 0.5 * d_zz
    tensor = np.asarray(
        [
            [d_xx, d_xy, d_xz],
            [d_xy, d_yy, d_yz],
            [d_xz, d_yz, d_zz],
        ],
        dtype=float,
    )
    return tensor + float(isotropic) * np.eye(3)


def euler_parameters_from_cartesian(
    *,
    isotropic: float,
    coordinates: ArrayLike,
    axiality_tolerance: float = 1.0e-12,
) -> dict[str, float]:
    """Convert Cartesian traceless coordinates into canonical public parameters.

    The dominant deviatoric eigenvalue defines the reported Z axis. Remaining
    principal axes are ordered so ``rho_over_ax`` belongs to ``[0, 1/3]``.
    For axial tensors, gamma is conventionally reported as zero.

    Args:
        isotropic: Isotropic susceptibility component.
        coordinates: Five Cartesian traceless coordinates.
        axiality_tolerance: Absolute threshold for an isotropic or axial tensor.

    Returns:
        Canonical ``iso``, ``ax``, ``rho_over_ax``, and ZYZ Euler parameters.

    Raises:
        ValueError: If ``axiality_tolerance`` is not positive.
    """
    if axiality_tolerance <= 0.0:
        raise ValueError("Axiality tolerance must be positive")

    tensor = tensor_from_cartesian_parameters(
        isotropic=isotropic,
        coordinates=coordinates,
    )
    deviatoric = tensor - float(isotropic) * np.eye(3)
    eigenvalues, eigenvectors = np.linalg.eigh(deviatoric)
    maximum_magnitude = float(np.max(np.abs(eigenvalues)))
    if maximum_magnitude < axiality_tolerance:
        return {
            "iso": float(isotropic),
            "ax": 0.0,
            "rho_over_ax": 0.0,
            "alpha": 0.0,
            "beta": 0.0,
            "gamma": 0.0,
        }

    dominant = np.flatnonzero(
        np.isclose(np.abs(eigenvalues), maximum_magnitude, atol=axiality_tolerance)
    )
    z_index = int(dominant[np.argmax(eigenvalues[dominant])])
    remaining = [index for index in range(3) if index != z_index]
    axiality = 1.5 * float(eigenvalues[z_index])
    if axiality > 0.0:
        x_index, y_index = sorted(
            remaining,
            key=lambda index: float(eigenvalues[index]),
            reverse=True,
        )
    else:
        x_index, y_index = sorted(
            remaining,
            key=lambda index: float(eigenvalues[index]),
        )

    rho_over_ax = float(
        (eigenvalues[x_index] - eigenvalues[y_index]) / (2.0 * axiality)
    )
    rho_over_ax = float(np.clip(rho_over_ax, 0.0, 1.0 / 3.0))
    rotation = eigenvectors[:, [x_index, y_index, z_index]].copy()
    for column in range(3):
        dominant_component = int(np.argmax(np.abs(rotation[:, column])))
        if rotation[dominant_component, column] < 0.0:
            rotation[:, column] *= -1.0
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0

    alpha = float(np.rad2deg(np.arctan2(rotation[1, 2], rotation[0, 2])) % 360.0)
    beta = float(np.rad2deg(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))))
    gamma = 0.0
    if rho_over_ax >= axiality_tolerance:
        gamma = float(np.rad2deg(np.arctan2(rotation[2, 1], -rotation[2, 0])) % 360.0)
    return {
        "iso": float(isotropic),
        "ax": axiality,
        "rho_over_ax": rho_over_ax,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
    }
