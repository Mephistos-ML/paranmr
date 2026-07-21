# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Analytical center derivatives for susceptibility moment Jacobians."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from paranmr.core.domain.mol import Nucleus
from paranmr.core.fitting.susceptibility.jacobian.susceptibility_tensor import (
    differentiate_tensor_by_alpha,
    differentiate_tensor_by_beta,
    differentiate_tensor_by_gamma,
    differentiate_tensor_by_susc_ax,
    differentiate_tensor_by_susc_iso,
    differentiate_tensor_by_susc_rho_over_ax,
)
from paranmr.core.fitting.susceptibility.moments.forward import (
    calculated_signal_packages_from_parameters,
    sort_packages_by_center,
)
from paranmr.core.fitting.susceptibility.models.isoaxrho_euler import (
    IsoAxRhoEulerFitter,
)


def differentiate_centers_by_susc_ax(
    *,
    parameters: dict[str, float],
    nuclei: list[Nucleus],
    average_labels: tuple[tuple[str, ...], ...] = (),
) -> NDArray[np.float64]:
    """Return analytical center derivatives with respect to susceptibility axiality."""

    return differentiate_centers_by_tensor_derivative(
        parameters=parameters,
        nuclei=nuclei,
        average_labels=average_labels,
        d_tensor=differentiate_tensor_by_susc_ax(parameters),
    )


def differentiate_centers_by_susc_iso(
    *,
    parameters: dict[str, float],
    nuclei: list[Nucleus],
    average_labels: tuple[tuple[str, ...], ...] = (),
) -> NDArray[np.float64]:
    """Return analytical center derivatives with respect to susceptibility isotropy."""

    return differentiate_centers_by_tensor_derivative(
        parameters=parameters,
        nuclei=nuclei,
        average_labels=average_labels,
        d_tensor=differentiate_tensor_by_susc_iso(parameters),
    )


def differentiate_centers_by_susc_rho_over_ax(
    *,
    parameters: dict[str, float],
    nuclei: list[Nucleus],
    average_labels: tuple[tuple[str, ...], ...] = (),
) -> NDArray[np.float64]:
    """Return analytical center derivatives with respect to rhombicity ratio."""

    return differentiate_centers_by_tensor_derivative(
        parameters=parameters,
        nuclei=nuclei,
        average_labels=average_labels,
        d_tensor=differentiate_tensor_by_susc_rho_over_ax(parameters),
    )


def differentiate_centers_by_alpha(
    *,
    parameters: dict[str, float],
    nuclei: list[Nucleus],
    average_labels: tuple[tuple[str, ...], ...] = (),
) -> NDArray[np.float64]:
    """Return analytical center derivatives with respect to ``alpha``."""

    return differentiate_centers_by_tensor_derivative(
        parameters=parameters,
        nuclei=nuclei,
        average_labels=average_labels,
        d_tensor=differentiate_tensor_by_alpha(parameters),
    )


def differentiate_centers_by_beta(
    *,
    parameters: dict[str, float],
    nuclei: list[Nucleus],
    average_labels: tuple[tuple[str, ...], ...] = (),
) -> NDArray[np.float64]:
    """Return analytical center derivatives with respect to ``beta``."""

    return differentiate_centers_by_tensor_derivative(
        parameters=parameters,
        nuclei=nuclei,
        average_labels=average_labels,
        d_tensor=differentiate_tensor_by_beta(parameters),
    )


def differentiate_centers_by_gamma(
    *,
    parameters: dict[str, float],
    nuclei: list[Nucleus],
    average_labels: tuple[tuple[str, ...], ...] = (),
) -> NDArray[np.float64]:
    """Return analytical center derivatives with respect to ``gamma``."""

    return differentiate_centers_by_tensor_derivative(
        parameters=parameters,
        nuclei=nuclei,
        average_labels=average_labels,
        d_tensor=differentiate_tensor_by_gamma(parameters),
    )


class ShiftOnlyIsoAxRhoModel:
    """Minimal shift forward wrapper used to reuse package construction logic."""

    @staticmethod
    def model(parameters: dict[str, float], nuclei: list[Nucleus]) -> dict[str, float]:
        tensor = IsoAxRhoEulerFitter.totensor(
            {
                "iso": float(parameters["iso"]),
                "ax": float(parameters["ax"]),
                "rho_over_ax": float(parameters["rho_over_ax"]),
                "alpha": float(parameters.get("alpha", 0.0)),
                "beta": float(parameters.get("beta", 0.0)),
                "gamma": float(parameters.get("gamma", 0.0)),
            }
        )
        return {
            nucleus.label: float(np.trace(tensor @ nucleus.A.tensor_full) / 3.0)
            for nucleus in nuclei
        }


def differentiate_centers_by_tensor_derivative(
    *,
    parameters: dict[str, float],
    nuclei: list[Nucleus],
    average_labels: tuple[tuple[str, ...], ...],
    d_tensor: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return center derivatives for a supplied tensor derivative."""

    packages = sort_packages_by_center(
        calculated_signal_packages_from_parameters(
            model=ShiftOnlyIsoAxRhoModel(),
            parameters=parameters,
            nuclei=nuclei,
            include_diamagnetic=True,
            average_labels=average_labels,
        )
    )
    derivative_by_atom_label = {
        nucleus.label: float(np.trace(d_tensor @ nucleus.A.tensor_full) / 3.0)
        for nucleus in nuclei
    }
    return np.asarray(
        [
            float(
                np.mean(
                    [derivative_by_atom_label[atom_label] for atom_label in package.atom_labels]
                )
            )
            for package in packages
        ],
        dtype=float,
    )
