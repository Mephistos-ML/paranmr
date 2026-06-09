# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Base classes for susceptibility fitting models."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from paranmr.core.domain.exp import Experiment
from paranmr.core.domain.mol import Nucleus
from paranmr.core.domain.tensor import Susceptibility


class SusceptibilityModel(ABC):
    """Base class for susceptibility fitting models.

    Concrete subclasses define a parameterization of the magnetic susceptibility
    tensor and a corresponding chemical-shift model.
    """

    def __init__(
        self, fit_vars: dict[str, float | str], fix_vars: dict[str, float | str]
    ):
        """Initializes a susceptibility model.

        Args:
            fit_vars: Parameters to be fitted. Keys must be in `VARNAMES`.
            fix_vars: Parameters to be held fixed. Keys must be in `VARNAMES`.

        Raises:
            ValueError: If required model variables are missing from `fit_vars` and
                `fix_vars`.
        """

        self.fit_vars = fit_vars
        self.fix_vars = fix_vars

        # Check all VARNAMES are provided in fit+fix
        input_names = [name for name in {**self.fit_vars, **self.fix_vars}.keys()]

        if any([req_name not in input_names for req_name in self.VARNAMES]):
            raise ValueError(f"Missing fit/fix parameters in {self.NAME} Model")

        # Final model parameter values
        self._final_var_values = {var: None for var in self.VARNAMES}
        # Standard deviation of each parameter
        self._fit_stdev = {var: None for var in self.fit_vars.keys()}

        # Fit status and temperature
        self._fit_status = False
        self._temperature = None

        # r2 and adjusted r2
        self._r2 = None
        self._adj_r2 = None

        # Residual
        self._mae = None
        # RMSE
        self._rmse = None

        return

    @property
    def fit_status(self) -> bool:
        """Whether the last fit was successful."""
        return self._fit_status

    @fit_status.setter
    def fit_status(self, value: bool):
        if isinstance(value, bool):
            self._fit_status = value
        else:
            raise TypeError
        return

    @property
    def temperature(self) -> float:
        """Temperature of the fit (K)."""
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        if isinstance(value, (np.floating, float)):
            self._temperature = value
        else:
            raise TypeError
        return

    @property
    def final_var_values(self) -> float:
        """Final values of all model parameters (fitted + fixed)."""
        return self._final_var_values

    @final_var_values.setter
    def final_var_values(self, value: dict):
        if isinstance(value, dict):
            self._final_var_values = value
        else:
            raise TypeError
        return

    @property
    def fit_stdev(self) -> float:
        """Standard deviation of fitted parameters from the fitting routine."""
        return self._fit_stdev

    @fit_stdev.setter
    def fit_stdev(self, value: dict):
        if isinstance(value, dict):
            self._fit_stdev = value
        else:
            raise TypeError
        return

    @property
    def fix_vars(self) -> dict[str, float]:
        """Fixed model parameters.

        Returns:
            Mapping from parameter name (in `VARNAMES`) to fixed value.
        """
        return self._fix_vars

    @fix_vars.setter
    def fix_vars(self, value: dict):
        if isinstance(value, dict):
            unknown = [key for key in value.keys() if key not in self.VARNAMES]
            if any(unknown):
                raise KeyError(f"Unknown variable names {unknown} provided to fix")
            self._fix_vars = value
        else:
            raise TypeError("fix must be dictionary")
        return

    @property
    def fit_vars(self) -> dict[str, float]:
        """Fitted model parameters.

        Returns:
            Mapping from parameter name (in `VARNAMES`) to initial guess.
        """
        return self._fit_vars

    @fit_vars.setter
    def fit_vars(self, value: dict):
        if isinstance(value, dict):
            unknown = [key for key in value.keys() if key not in self.VARNAMES]
            if any(unknown):
                raise KeyError(f"Unknown variable names {unknown} provided to fix")
            self._fit_vars = value
        else:
            raise TypeError("Fit must be dictionary")

        # Reset final model parameter values
        self._final_var_values = {var: None for var in self.VARNAMES}
        # Reset standard deviation of each parameter
        self._fit_stdev = {var: None for var in self.fit_vars.keys()}
        return

    @property
    def r2(self) -> float:
        """Coefficient of determination (R²) of the fit."""
        return self._r2

    @r2.setter
    def r2(self, value):
        if isinstance(value, (np.floating, float)):
            self._r2 = value
        else:
            raise TypeError
        return

    @property
    def adj_r2(self) -> float:
        """Adjusted coefficient of determination (adjusted R²)."""
        return self._adj_r2

    @adj_r2.setter
    def adj_r2(self, value):
        if isinstance(value, (np.floating, float)):
            self._adj_r2 = value
        else:
            raise TypeError
        return

    @property
    def mae(self) -> float:
        """Mean absolute error (MAE) of the fit."""
        return self._mae

    @mae.setter
    def mae(self, value):
        if isinstance(value, (np.floating, float)):
            self._mae = value
        else:
            raise TypeError
        return

    @property
    def rmse(self) -> float:
        """Root mean square error (RMSE) of the fit."""
        return self._rmse

    @rmse.setter
    def rmse(self, value):
        if isinstance(value, (np.floating, float)) or np.isnan(value):
            self._rmse = value
        else:
            raise TypeError
        return

    @property
    @abstractmethod
    def NAME() -> str:
        """Human-readable name of the model."""
        raise NotImplementedError

    @property
    @abstractmethod
    def VARNAMES() -> list[str]:
        """Names of parameters that can be fitted or fixed."""
        raise NotImplementedError

    @property
    @abstractmethod
    def VARNAMES_MM() -> dict[str, str]:
        """Math-mode (LaTeX) labels for model parameters.

        Returns:
            Mapping from parameter names in `VARNAMES` to LaTeX strings.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def UNITS_MM() -> dict[str, str]:
        """Math-mode (LaTeX) units for model parameters.

        Returns:
            Mapping from parameter names in `VARNAMES` to LaTeX unit strings.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def BOUNDS() -> dict[str, list[float]]:
        """Bounds for each model parameter.

        Returns:
            Mapping from parameter name to ``[lower, upper]`` bounds.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def model(parameters: dict[str, float], nuclei: list[Nucleus]) -> dict[str, float]:
        """Evaluates the model prediction for paramagnetic shifts.

        Args:
            parameters: Model parameters. Keys are `VARNAMES`.
            nuclei: Nuclei for which shifts will be computed.

        Returns:
            Mapping from nucleus atom labels to predicted paramagnetic shifts.
        """
        raise NotImplementedError

    def tosusceptibility(self) -> Susceptibility:
        """Converts the fitted model into a `Susceptibility` instance.

        Returns:
            A `Susceptibility` object at `self.temperature` with canonical
            ``chi.iso`` assigned from the fitted model.
        """
        tensor = self.totensor(self.final_var_values)
        susc = Susceptibility(tensor, self.temperature)

        fitted_iso = self.final_var_values.get("iso")
        if fitted_iso is None:
            fitted_iso = float(np.trace(tensor) / 3.0)

        susc.iso = float(fitted_iso)
        return susc

    @staticmethod
    @abstractmethod
    def totensor(params: dict[str, float]) -> NDArray:
        """Converts model parameters to a susceptibility tensor.

        Args:
            params: Model parameters. Keys are `VARNAMES`.

        Returns:
            Susceptibility tensor as a ``(3, 3)`` NumPy array.
        """
        raise NotImplementedError

    def _post_fit(self) -> None:
        """Hook for model-specific post-processing after a successful fit.

        Called after fitting once `final_var_values` and `fit_stdev` are set.
        Subclasses may override to compute derived quantities.

        Returns:
            None.
        """
        return


class LinearSusceptibilityModel(SusceptibilityModel):
    """Base class for linear susceptibility fitting models."""

    @staticmethod
    @abstractmethod
    def design_matrix(nuclei: list[Nucleus], fix_vars: dict[str, float]):
        """Builds the design matrix for the linear model.

        Args:
            nuclei: Nuclei for which the model is evaluated.
            fix_vars: Fixed model parameters.

        Returns:
            The design matrix ``A`` in the linear system ``A x = b``.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def target_vector(
        nuclei: list[Nucleus],
        experiment: Experiment,
        fix_vars: dict[str, float],
    ):
        """Builds the target vector for the linear model.

        Args:
            nuclei: Nuclei for which the model is evaluated.
            experiment: Experimental data object.
            fix_vars: Fixed model parameters.

        Returns:
            The target vector ``b`` in the linear system ``A x = b``.
        """
        raise NotImplementedError
