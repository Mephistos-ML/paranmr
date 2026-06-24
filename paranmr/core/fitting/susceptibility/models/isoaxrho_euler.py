# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Euler-oriented iso/ax/rho-over-ax susceptibility fitting model."""

import numpy as np
from numpy.typing import NDArray

from paranmr.core.domain.mol import Nucleus
from paranmr.core.fitting.susceptibility.models.base import SusceptibilityModel


class IsoAxRhoEulerFitter(SusceptibilityModel):
    """Iso/ax/rho susceptibility model with fitted ZYZ Euler orientation."""

    NAME = "Euler-Oriented Isotropic, Axial, and Rhombic Susceptibility"

    VARNAMES = [
        "iso",
        "ax",
        "rho_over_ax",
        "alpha",
        "beta",
        "gamma",
    ]

    VARNAMES_MM = {
        "iso": r"$\chi_\mathregular{iso}$",
        "ax": r"$\Delta\chi_\mathregular{ax}$",
        "rho_over_ax": r"$\chi_\mathregular{rho} / \Delta\chi_\mathregular{ax}$",
        "alpha": r"$\alpha$",
        "beta": r"$\beta$",
        "gamma": r"$\gamma$",
    }

    UNITS_MM = {
        "iso": r"Å$^3$",
        "ax": r"Å$^3$",
        "rho_over_ax": "",
        "alpha": r"°",
        "beta": r"°",
        "gamma": r"°",
    }

    BOUNDS = {
        "iso": [0.0, np.inf],
        "ax": [-np.inf, np.inf],
        "rho_over_ax": [0.0, 1 / 3],
        "alpha": [0.0, 360.0],
        "beta": [0.0, 180.0],
        "gamma": [0.0, 360.0],
    }

    @staticmethod
    def _zyz_rotation(alpha_deg: float, beta_deg: float, gamma_deg: float) -> NDArray:
        """Build the ZYZ rotation matrix ``Rz(alpha) @ Ry(beta) @ Rz(gamma)``.

        Args:
            alpha_deg: First Z-axis rotation angle in degrees.
            beta_deg: Y-axis rotation angle in degrees.
            gamma_deg: Second Z-axis rotation angle in degrees.

        Returns:
            A ``(3, 3)`` rotation matrix.
        """

        alpha = np.deg2rad(alpha_deg)
        beta = np.deg2rad(beta_deg)
        gamma = np.deg2rad(gamma_deg)

        cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
        cos_beta, sin_beta = np.cos(beta), np.sin(beta)
        cos_gamma, sin_gamma = np.cos(gamma), np.sin(gamma)

        rz_alpha = np.array(
            [
                [cos_alpha, -sin_alpha, 0.0],
                [sin_alpha, cos_alpha, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        ry_beta = np.array(
            [
                [cos_beta, 0.0, sin_beta],
                [0.0, 1.0, 0.0],
                [-sin_beta, 0.0, cos_beta],
            ]
        )
        rz_gamma = np.array(
            [
                [cos_gamma, -sin_gamma, 0.0],
                [sin_gamma, cos_gamma, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        return rz_alpha @ ry_beta @ rz_gamma

    @staticmethod
    def model(parameters: dict[str, float], nuclei: list[Nucleus]) -> dict[str, float]:
        """Computes predicted paramagnetic shifts for the Euler iso/ax/rho model.

        Args:
            parameters: Model parameters. Keys are `VARNAMES`.
            nuclei: Nuclei for which shifts will be computed.

        Returns:
            Mapping from nucleus labels to predicted paramagnetic shifts.
        """

        tensor = IsoAxRhoEulerFitter.totensor(parameters)
        return {
            nuc.label: 1.0 / 3.0 * np.trace(tensor @ nuc.A.tensor_full)
            for nuc in nuclei
        }

    @staticmethod
    def totensor(params: dict[str, float]) -> NDArray:
        """Convert Euler iso/ax/rho parameters to a susceptibility tensor.

        The principal-axis-frame tensor is parameterized by ``iso``, ``ax``,
        and ``rho_over_ax``. It is then rotated into the input frame with the
        fitted ZYZ Euler angles.

        Args:
            params: Model parameters. Keys are `VARNAMES`.

        Returns:
            Susceptibility tensor as a ``(3, 3)`` NumPy array.
        """

        ax = params["ax"]
        rho_over_ax = params["rho_over_ax"]
        tensor_paf = np.array(
            [
                [-ax / 3 + rho_over_ax * ax, 0.0, 0.0],
                [0.0, -ax / 3 - rho_over_ax * ax, 0.0],
                [0.0, 0.0, 2 / 3 * ax],
            ]
        )
        tensor_paf += np.eye(3) * params["iso"]

        rotation = IsoAxRhoEulerFitter._zyz_rotation(
            params["alpha"], params["beta"], params["gamma"]
        )
        return rotation @ tensor_paf @ rotation.T

    def _post_fit(self) -> None:
        """Adds derived uncertainty for chi_rho."""

        ax = self.final_var_values.get("ax")
        rho_over_ax = self.final_var_values.get("rho_over_ax")
        ax_st_dev = self.fit_stdev.get("ax")
        rho_over_ax_st_dev = self.fit_stdev.get("rho_over_ax")

        if ax is None or rho_over_ax is None:
            self.fit_stdev.pop("rho", None)
            return

        if ax_st_dev is None:
            self.fit_stdev.pop("rho", None)
            return

        if rho_over_ax_st_dev is None:
            if "rho_over_ax" in self.fix_vars:
                self.fit_stdev["rho"] = float(np.abs(rho_over_ax) * ax_st_dev)
                return
            self.fit_stdev.pop("rho", None)
            return

        self.fit_stdev["rho"] = float(
            np.hypot(rho_over_ax * ax_st_dev, ax * rho_over_ax_st_dev)
        )
        return
