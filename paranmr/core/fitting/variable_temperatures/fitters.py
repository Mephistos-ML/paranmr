# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Fit Curie-normalised susceptibility chiT(T) data."""

import numpy as np
from scipy.optimize import curve_fit

from paranmr.core.fitting.variable_temperatures.components import compute_curie_prefactor


def fit_chit_linear_model(
    spin: float,
    fit_temps: np.ndarray,
    chi_vals: np.ndarray,
    chi_errors: np.ndarray,
    susc_vt_variables: dict,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | None | np.ndarray]]:
    """
    Fit a linear chiT(T) = A + B / T + tip * T model to
    Curie-normalised susceptibility data

    If valid error estimates are provided, a weighted least-squares fit is performed

    Args:
        spin (float): Total spin quantum number S.
        fit_temps (np.ndarray): Temperature values.
        chi_vals (np.ndarray): Susceptibility values for a single chi_component.
        chi_errors (np.ndarray): Uncertainties associated with `chi_vals` as an array
        of the same shape.
        susc_vt_variables (dict): Variables controlling fit modes and initial values.
            Must include keys `intercept` and `slope`. The optional key `tip` may be
            provided as `["fit", <guess>]` or `["fix", <value>]`.

    Returns:
        tuple[np.ndarray, np.ndarray, dict[str, float | None | np.ndarray]]:
            - chiT_reduced (np.ndarray): Curie-normalised chiT values (dimensionless).
            - chi_errT_reduced (np.ndarray): Uncertainties for `chiT_reduced`.
            - fit_results (dict[str, float | None | np.ndarray]):
            Fit parameters and statistics.
              Includes `intercept`, `slope`, optional `tip`, and their uncertainties,
              along with `adj_r2`.
              The dict may also include precomputed plotting arrays:
              `fit_y`, `fit_y_low`, `fit_y_high` arrays evaluated on `fit_temps`
              for downstream visualization.
    """

    def _model(T, Intercept, Slope, tip):
        # TIP contributes a temperature-independent term in chi(T), which becomes
        # a linear-in-T term in chiT(T).

        return Intercept + Slope / T + tip * T

    norm_factor = compute_curie_prefactor(spin)

    fit_param_names: list[str] = []
    x0: list[float] = []
    fixed_intercept: float | None = None
    fixed_slope: float | None = None
    fixed_tip: float | None = 0.0

    mode_i, val_i = susc_vt_variables["intercept"]
    mode_i = str(mode_i).strip().lower()
    if mode_i == "fit":
        fit_param_names.append("intercept")
        x0.append(float(val_i))
    elif mode_i == "fix":
        fixed_intercept = float(val_i)
    else:
        raise ValueError(
            f"Invalid mode {mode_i!r} for 'intercept'. Expected 'fit' or 'fix'."
        )

    mode_s, val_s = susc_vt_variables["slope"]
    mode_s = str(mode_s).strip().lower()
    if mode_s == "fit":
        fit_param_names.append("slope")
        x0.append(float(val_s))
    elif mode_s == "fix":
        fixed_slope = float(val_s)
    else:
        raise ValueError(
            f"Invalid mode {mode_s!r} for 'slope'. Expected 'fit' or 'fix'."
        )

    # TIP is optional; if not provided, it is treated as fixed to 0.0.
    if "tip" in susc_vt_variables:
        mode_t, val_t = susc_vt_variables["tip"]
        mode_t = str(mode_t).strip().lower()
        if mode_t == "fit":
            fit_param_names.append("tip")
            x0.append(float(val_t))
            fixed_tip = None
        elif mode_t == "fix":
            fixed_tip = float(val_t)
        else:
            raise ValueError(
                f"Invalid mode {mode_t!r} for 'tip'. Expected 'fit' or 'fix'."
            )

    def _model_free(T, *theta):
        params = {
            "intercept": (fixed_intercept if fixed_intercept is not None else 0.0),
            "slope": (fixed_slope if fixed_slope is not None else 0.0),
            "tip": (fixed_tip if fixed_tip is not None else 0.0),
        }
        for name, val in zip(fit_param_names, theta, strict=False):
            params[name] = float(val)

        return _model(T, params["intercept"], params["slope"], params["tip"])

    # Compute chiT values and chiT errors at the given temperatures
    chiT = chi_vals * fit_temps
    chi_errT = chi_errors * fit_temps

    # Curie-normalised (dimensionless) values
    chiT_reduced = chiT / norm_factor
    chi_errT_reduced = chi_errT / norm_factor

    # Detect the degenerate case of identically zero chiT
    if np.allclose(chiT_reduced, 0.0):
        fit_results: dict[str, float | None | np.ndarray] = {
            "intercept": 0.0,
            "slope": 0.0,
            "tip": 0.0,
            "intercept_err": 0.0,
            "slope_err": 0.0,
            "tip_err": 0.0,
            "adj_r2": None,
            "fit_y": chiT_reduced.copy(),
            "fit_y_low": None,
            "fit_y_high": None,
        }

        return chiT_reduced, chi_errT_reduced, fit_results

    # Prepare sigma only if there are positive errors; or perform an unweighted fit
    sigma = None
    abs_sigma = False
    _errs = np.asarray(chi_errT_reduced, dtype=float)
    if np.any(_errs > 0):
        sigma = _errs
        abs_sigma = True

    if not fit_param_names:
        yhat = _model(fit_temps, fixed_intercept, fixed_slope, fixed_tip)

        ss_res = np.sum((chiT_reduced - yhat) ** 2)
        ss_tot = np.sum((chiT_reduced - np.mean(chiT_reduced)) ** 2)

        adj_r2 = None
        if ss_tot != 0 and len(chiT_reduced) > 3:
            r2 = 1 - (ss_res / ss_tot)
            adj_r2 = 1 - (1 - r2) * (len(chiT_reduced) - 1) / (
                len(chiT_reduced) - 2 - 1
            )

        fit_results: dict[str, float | None | np.ndarray] = {
            "intercept": float(fixed_intercept),
            "slope": float(fixed_slope),
            "tip": float(fixed_tip if fixed_tip is not None else 0.0),
            "intercept_err": 0.0,
            "slope_err": 0.0,
            "tip_err": 0.0,
            "adj_r2": (None if adj_r2 is None else float(adj_r2)),
            "fit_y": np.asarray(yhat, dtype=float),
            "fit_y_low": None,
            "fit_y_high": None,
        }

        return chiT_reduced, chi_errT_reduced, fit_results

    popt, pcov = curve_fit(
        _model_free,
        fit_temps,
        chiT_reduced,
        p0=x0,
        sigma=sigma,
        absolute_sigma=abs_sigma,
    )
    perr = np.sqrt(np.diag(pcov))

    # Reconstruct full parameter set
    params = {
        "intercept": (fixed_intercept if fixed_intercept is not None else 0.0),
        "slope": (fixed_slope if fixed_slope is not None else 0.0),
        "tip": (fixed_tip if fixed_tip is not None else 0.0),
    }
    for name, val in zip(fit_param_names, popt, strict=False):
        params[name] = float(val)

    # Precompute fit curve and 1-sigma band for downstream visualization.
    fit_y = np.asarray(
        _model(fit_temps, params["intercept"], params["slope"], params["tip"]),
        dtype=float,
    )

    # Build a full 3x3 covariance matrix in (intercept, slope, tip) order.
    pcov_full = np.zeros((3, 3), dtype=float)
    name_to_idx = {"intercept": 0, "slope": 1, "tip": 2}
    for i_name, i in name_to_idx.items():
        if i_name not in fit_param_names:
            continue
        ii = fit_param_names.index(i_name)
        for j_name, j in name_to_idx.items():
            if j_name not in fit_param_names:
                continue
            jj = fit_param_names.index(j_name)
            pcov_full[i, j] = float(pcov[ii, jj])

    # Jacobian of y wrt (intercept, slope, tip): [1, 1/T, T]
    T = np.asarray(fit_temps, dtype=float)
    J = np.column_stack([np.ones_like(T), 1.0 / T, T])  # shape (n, 3)

    # Var(y) ≈ J * pcov_full * J^T (parameter uncertainty only)
    var_y = np.einsum("ni,ij,nj->n", J, pcov_full, J)
    std_y = np.sqrt(np.maximum(var_y, 0.0))

    fit_y_low = fit_y - std_y
    fit_y_high = fit_y + std_y

    # R^2 metrics (guard against zero variance)
    yhat = _model(fit_temps, params["intercept"], params["slope"], params["tip"])
    ss_res = np.sum((chiT_reduced - yhat) ** 2)
    ss_tot = np.sum((chiT_reduced - np.mean(chiT_reduced)) ** 2)

    adj_r2 = None
    if ss_tot != 0 and len(chiT_reduced) > 3:
        r2 = 1 - (ss_res / ss_tot)
        adj_r2 = 1 - (1 - r2) * (len(chiT_reduced) - 1) / (len(chiT_reduced) - 2 - 1)

    intercept_err = (
        float(perr[fit_param_names.index("intercept")])
        if "intercept" in fit_param_names
        else 0.0
    )

    slope_err = (
        float(perr[fit_param_names.index("slope")])
        if "slope" in fit_param_names
        else 0.0
    )

    tip_err = (
        float(perr[fit_param_names.index("tip")]) if "tip" in fit_param_names else 0.0
    )

    fit_results: dict[str, float | None | np.ndarray] = {
        "intercept": float(params["intercept"]),
        "slope": float(params["slope"]),
        "tip": float(params["tip"]),
        "intercept_err": intercept_err,
        "slope_err": slope_err,
        "tip_err": tip_err,
        "adj_r2": (None if adj_r2 is None else float(adj_r2)),
        "fit_y": fit_y,
        "fit_y_low": fit_y_low,
        "fit_y_high": fit_y_high,
    }

    return chiT_reduced, chi_errT_reduced, fit_results


def compute_chit_high_t_limit(
    spin: float,
    fit_temps: np.ndarray,
    chi_vals: np.ndarray,
    chi_errors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | None | np.ndarray]]:
    """
    Evaluate the high-temperature chiT limit assuming a zero slope

    This handler assumes B = 0 and takes the intercept from the
    highest-temperature data point. TIP is not estimated in this mode and is
    returned fixed at 0.0.

    Args:
        temperature (array_like): Temperature values
        chi_value (array_like): Susceptibility values for a single chi_component
        chi_errors: Uncertainties associated with `chi_vals` as an array
        of the same shape

    Returns:
        tuple[np.ndarray, np.ndarray, dict[str, float | None | np.ndarray]]:
            - chiT_reduced (np.ndarray): Curie-normalised chiT values (dimensionless)
            - errT_reduced (np.ndarray): Uncertainties for `chiT_reduced`
            - fit_results (dict[str, float | None | np.ndarray]): Fit parameters for the
            fixed-slope branch (intercept from max(T) point,
            slope=0, tip=0, uncertainties, adj_r2=1)
            The dict may also include precomputed plotting arrays:
            `fit_y`, `fit_y_low`, `fit_y_high` arrays evaluated on `fit_temps`
            for downstream visualization.
    """
    norm_factor = compute_curie_prefactor(spin)

    # chiT in internal units -> Curie-normalised (dimensionless)
    chiT_reduced = (chi_vals * fit_temps) / norm_factor

    chi_errors = np.asarray(chi_errors, dtype=float)
    errT_reduced = (chi_errors * fit_temps) / norm_factor

    idx = int(np.nanargmax(fit_temps))

    intercept = float(chiT_reduced[idx])

    fit_results: dict[str, float | None | np.ndarray] = {
        "intercept": intercept,
        "slope": 0.0,
        "tip": 0.0,
        "intercept_err": float(errT_reduced[idx] if errT_reduced is not None else 0.0),
        "slope_err": 0.0,
        "tip_err": 0.0,
        "adj_r2": 1.0,
        "fit_y": np.asarray(chiT_reduced, dtype=float),
        "fit_y_low": None,
        "fit_y_high": None,
    }

    return chiT_reduced, errT_reduced, fit_results

