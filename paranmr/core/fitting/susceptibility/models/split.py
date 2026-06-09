# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Split isotropic/anisotropic susceptibility fitting model."""

import copy

import numpy as np
from numpy.typing import NDArray

from paranmr.core.domain.mol import Nucleus
from paranmr.core.domain.tensor import Susceptibility
from paranmr.core.fitting.susceptibility.models.base import SusceptibilityModel
from paranmr.core.util.uncertainty import delta_method_sigma


class SplitFitter(SusceptibilityModel):
    NAME = "Split Isotropic and Anisotropic Components of Susceptibility"

    VARNAMES = ["iso", "dxx", "dyy", "dxy", "dxz", "dyz"]

    VARNAMES_MM = {
        "iso": r"$\chi_\mathregular{iso}$",
        "dxx": r"$\Delta\chi_{xx}$",
        "dyy": r"$\Delta\chi_{yy}$",
        "dxy": r"$\Delta\chi_{xy}$",
        "dxz": r"$\Delta\chi_{xz}$",
        "dyz": r"$\Delta\chi_{yz}$",
    }

    UNITS_MM = {
        "iso": r"Å$^3$",
        "dxx": r"Å$^3$",
        "dyy": r"Å$^3$",
        "dxy": r"Å$^3$",
        "dxz": r"Å$^3$",
        "dyz": r"Å$^3$",
    }

    BOUNDS = {
        "iso": [0.0, np.inf],
        "dxx": [-np.inf, np.inf],
        "dyy": [-np.inf, np.inf],
        "dxy": [-np.inf, np.inf],
        "dxz": [-np.inf, np.inf],
        "dyz": [-np.inf, np.inf],
    }

    @staticmethod
    def model(parameters: dict[str, float], nuclei: list[Nucleus]) -> dict[str, float]:
        """Computes predicted paramagnetic shifts for the split-tensor model.

        The model uses an isotropic term and a traceless anisotropic tensor written
        in Cartesian components.

        Args:
            parameters: Model parameters. Keys are `VARNAMES`.
            nuclei: Nuclei for which shifts will be computed.

        Returns:
            Mapping from nucleus labels to predicted paramagnetic shifts.
        """

        delta_params = copy.deepcopy(parameters)
        tnsr = SplitFitter.totensor(delta_params)

        shifts = {
            nuc.label: 1.0 / 3.0 * np.trace(tnsr @ nuc.A.tensor_full) for nuc in nuclei
        }

        return shifts

    @staticmethod
    def totensor(params: dict[str, float]) -> NDArray:
        """Converts split-model parameters to a susceptibility tensor.

        Args:
            params: Model parameters. Keys are `VARNAMES`.

        Returns:
            Susceptibility tensor as a ``(3, 3)`` NumPy array.
        """

        tensor = np.array(
            [
                [params["dxx"], params["dxy"], params["dxz"]],
                [params["dxy"], params["dyy"], params["dyz"]],
                [
                    params["dxz"],
                    params["dyz"],
                    -params["dxx"] - params["dyy"],
                ],
            ]
        )
        tensor += np.eye(3) * params["iso"]

        return tensor

    def _post_fit(self) -> None:
        """Adds derived uncertainties for susceptibility invariants.

        Computes 1σ uncertainties for `Susceptibility.axiality` and
        `Susceptibility.rhombicity` via a finite-difference Jacobian with respect to
        the split parameters (iso, dxx, dyy, dxy, dxz, dyz) and propagates using the
        package delta-method helper.

        Notes:
            - Uses the independence assumption (diagonal covariance) consistent with
              `delta_method_sigma`.
            - Fixed parameters are treated as having zero uncertainty.
        """

        # Build input sigma vector in VARNAMES order; fixed params => sigma = 0.
        sig_in = []
        for name in self.VARNAMES:
            if name in self.fit_vars:
                sig = self.fit_stdev.get(name)
                sig_in.append(float(sig) if sig is not None else np.nan)
            else:
                sig_in.append(0.0)
        sig_in = np.asarray(sig_in, dtype=float)

        # Nothing to propagate if all inputs are fixed.
        if np.all(sig_in == 0.0):
            return

        base = {k: float(v) for k, v in self.final_var_values.items()}

        def _invariants(params: dict[str, float]) -> NDArray:
            tensor = SplitFitter.totensor(params)
            susc = Susceptibility(tensor, self.temperature)
            return np.asarray(
                [float(susc.axiality), float(susc.rhombicity)], dtype=float
            )

        jac = np.zeros((2, len(self.VARNAMES)), dtype=float)

        # Central finite differences for d(axiality, rhombicity)/d(params).
        for i, name in enumerate(self.VARNAMES):
            x0 = base[name]
            step = 1e-6 * max(1.0, abs(x0))

            p_plus = base.copy()
            p_minus = base.copy()
            p_plus[name] = x0 + step
            p_minus[name] = x0 - step

            y_plus = _invariants(p_plus)
            y_minus = _invariants(p_minus)

            jac[:, i] = (y_plus - y_minus) / (2.0 * step)

        sig_out = delta_method_sigma(jac, sig_in)

        # Store derived uncertainties alongside fitted ones.
        self.fit_stdev["ax"] = float(sig_out[0])
        self.fit_stdev["rho"] = float(sig_out[1])
        return
