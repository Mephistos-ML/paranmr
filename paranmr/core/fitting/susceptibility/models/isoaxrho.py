# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Iso/ax/rho-over-ax susceptibility fitting model."""

import copy

import numpy as np
from numpy.typing import NDArray

from paranmr.core.domain.mol import Nucleus
from paranmr.core.fitting.susceptibility.models.base import SusceptibilityModel


class IsoAxRhoFitter(SusceptibilityModel):
    NAME = "Isotropic, Axial, and Rhombic over Axial Components of Susceptibility"

    VARNAMES = [
        "iso",
        "ax",
        "rho_over_ax",
    ]

    VARNAMES_MM = {
        "iso": r"$\chi_\mathregular{iso}$",
        "ax": r"$\Delta\chi_\mathregular{ax}$",
        "rho_over_ax": r"$\chi_\mathregular{rho} / \Delta\chi_\mathregular{ax}$",
    }

    UNITS_MM = {
        "iso": r"Å$^3$",
        "ax": r"Å$^3$",
        "rho_over_ax": "",
    }

    BOUNDS = {
        "iso": [0.0, np.inf],
        "ax": [-np.inf, np.inf],
        "rho_over_ax": [0.0, 1 / 3],
    }

    @staticmethod
    def model(parameters: dict[str, float], nuclei: list[Nucleus]) -> dict[str, float]:
        """Computes predicted paramagnetic shifts for the iso/ax/rho model.

        The anisotropic part is parameterized by axiality and a rhombicity ratio
        ``rho_over_ax``.

        Args:
            parameters: Model parameters. Keys are `VARNAMES`.
            nuclei: Nuclei for which shifts will be computed.

        Returns:
            Mapping from nucleus labels to predicted paramagnetic shifts.
        """

        delta_params = copy.deepcopy(parameters)
        tnsr = IsoAxRhoFitter.totensor(delta_params)

        shifts = {
            nuc.label: 1.0 / 3.0 * np.trace(tnsr @ nuc.A.tensor_full) for nuc in nuclei
        }

        return shifts

    @staticmethod
    def totensor(params: dict[str, float]) -> NDArray:
        """Converts iso/ax/rho parameters to a susceptibility tensor.

        Args:
            params: Model parameters. Keys are `VARNAMES`.

        Returns:
            Susceptibility tensor as a ``(3, 3)`` NumPy array.
        """

        tensor = np.array(
            [
                [-params["ax"] / 3 + params["rho_over_ax"] * params["ax"], 0.0, 0.0],
                [0.0, -params["ax"] / 3 - params["rho_over_ax"] * params["ax"], 0.0],
                [0.0, 0.0, 2 / 3 * params["ax"]],
            ]
        )
        tensor += np.eye(3) * params["iso"]

        return tensor

    def _post_fit(self) -> None:
        """Adds derived uncertainty for chi_rho.

        The fit uses `rho_over_ax`, but reporting prefers `rho = ax * rho_over_ax`.

        Notes:
            - If `rho_over_ax` is fixed, treat its uncertainty as zero and propagate
              only the `ax` uncertainty.
            - If `ax` is fixed (no `ax` stdev available), `rho` uncertainty cannot be
              propagated reliably and is omitted.
        """
        ax = self.final_var_values.get("ax")
        rho_over_ax = self.final_var_values.get("rho_over_ax")
        ax_st_dev = self.fit_stdev.get("ax")
        rho_over_ax_st_dev = self.fit_stdev.get("rho_over_ax")

        # Require values for rho computation
        if ax is None or rho_over_ax is None:
            self.fit_stdev.pop("rho", None)
            return

        # If ax is fixed, we cannot propagate an uncertainty for rho.
        if ax_st_dev is None:
            self.fit_stdev.pop("rho", None)
            return

        # If rho_over_ax is fixed, assume sigma_rho_over_ax = 0.
        if rho_over_ax_st_dev is None:
            if "rho_over_ax" in self.fix_vars:
                self.fit_stdev["rho"] = float(np.abs(rho_over_ax) * ax_st_dev)
                return
            self.fit_stdev.pop("rho", None)
            return

        # General case: first-order propagation under independence.
        self.fit_stdev["rho"] = float(
            np.hypot(rho_over_ax * ax_st_dev, ax * rho_over_ax_st_dev)
        )
        return
