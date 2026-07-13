# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Read and write chiT regression fit results as CSV.

Provides helpers to serialize fitted slope/intercept parameters and to parse
flattened regression results from CSV files.
"""

import logging

import numpy as np
import pandas as pd

from paranmr.core.fitting.linewidth import R6LinewidthParameterEstimate
from paranmr.core.fitting.susceptibility.fitters.moments import MomentFitResult
from paranmr.io.csv.csv_util import read_csv_safe, write_csv_safe

logger = logging.getLogger(__name__)


def save_moment_fit_diagnostics(
    diagnostics: MomentFitResult,
    file_name: str,
    verbose: bool = True,
) -> None:
    """Write publication-grade moment fit diagnostics to CSV.

    Args:
        diagnostics: Moment fit diagnostics returned by the moments fitter.
        file_name: Output CSV path.
        verbose: If ``True``, log the output path.

    Returns:
        None.
    """

    moment_names = list(diagnostics.observed_moments)
    rows = [
        {
            "quantity": "observed",
            **{name: diagnostics.observed_moments[name] for name in moment_names},
        },
        {
            "quantity": "calculated",
            **{name: diagnostics.calculated_moments[name] for name in moment_names},
        },
    ]
    comment = [
        f"T = {diagnostics.temperature:.2f} K",
        f"objective = {diagnostics.objective_type}",
        f"weighted_score = {diagnostics.weighted_score:.6g}",
    ]
    write_csv_safe(pd.DataFrame(rows), file_name, comment)

    if verbose:
        logger.info("Moment fit diagnostics written to %s", file_name)

    return


def save_fit_linewidth_model(
    diagnostics: MomentFitResult,
    file_name: str,
    verbose: bool = True,
) -> None:
    """Write fitted linewidth-model parameters for a fit to CSV.

    Args:
        diagnostics: Fit diagnostics returned by the moments fitter.
        file_name: Output CSV path.
        verbose: If ``True``, log the output path.

    Returns:
        None.
    """

    out = {
        "linewidth_method": [diagnostics.linewidth_method],
        "p1": [diagnostics.linewidth_vars_by_name["p1"]],
        "p2": [diagnostics.linewidth_vars_by_name["p2"]],
    }
    comment = [
        f"T = {diagnostics.temperature:.2f} K",
        f"objective = {diagnostics.objective_type}",
    ]
    write_csv_safe(pd.DataFrame(data=out), file_name, comment)

    if verbose:
        logger.info("Fit linewidth model written to %s", file_name)

    return


def save_linewidth_parameter_estimate(
    estimate: R6LinewidthParameterEstimate,
    file_name: str,
    *,
    temperature: float,
    verbose: bool = True,
) -> None:
    """Write estimated linewidth-model parameters for fixed-assignment fits.

    Args:
        estimate: Structured linewidth parameter estimate.
        file_name: Output CSV path.
        temperature: Experiment temperature in Kelvin.
        verbose: If ``True``, log the output path.

    Returns:
        None.
    """

    out = {
        "linewidth_method": [estimate.linewidth_method],
        "estimate_mode": [estimate.estimate_mode],
        "p1": [estimate.p1],
        "p2": [estimate.p2],
        "rmse (ppm)": [estimate.rmse],
    }
    comment = [
        f"T = {temperature:.2f} K",
        "Experimental linewidths were converted to ppm before parameter estimation.",
    ]
    write_csv_safe(pd.DataFrame(data=out), file_name, comment)

    if verbose:
        logger.info("Linewidth parameter estimate written to %s", file_name)

    return


def save_slope_intercept(
    fits,
    spin: float | None = None,
    file_name: str = "isoaxrho_fit.csv",
    verbose: bool = True,
) -> None:
    """Writes slope/intercept results for chiT fits to a CSV file.

    Args:
        fits: Fit results for each component. Each entry is expected to be a mapping
            containing values such as ``slope``, ``intercept``, and associated errors.
        spin: Spin value from the domain model to serialize in the CSV comments.
        file_name: Output CSV file path.
        verbose: If ``True``, prints the output file path.

    Returns:
        None.

    Notes:
        The output includes fitted parameters and adjusted R² values for each
        component type.
    """

    labels = ["iso", "ax", "rho"]

    types = []
    intercepts = []
    slopes = []
    intercept_errs = []
    slope_errs = []
    adj_r2s = []

    for i in range(len(fits)):
        types.append(labels[i] if i < len(labels) else "c{}".format(i))

        fit = fits[i] if fits[i] is not None else {}

        intercepts.append(fit.get("intercept", np.nan))
        slopes.append(fit.get("slope", np.nan))
        intercept_errs.append(fit.get("intercept_err", np.nan))
        slope_errs.append(fit.get("slope_err", np.nan))
        adj_r2s.append(fit.get("adj_r2", np.nan))

    out = {
        "type": types,
        "intercept": intercepts,
        "slope": slopes,
        "intercept_err": intercept_errs,
        "slope_err": slope_errs,
        "adj_r2": adj_r2s,
    }
    comment = [
        "Data reported in Curie-normalised chiT (dimensionless) Units",
    ]
    if spin is not None:
        comment.append(f"spin {spin}")

    df = pd.DataFrame(data=out)

    write_csv_safe(df, file_name, comment)

    if verbose:
        logger.info("Temperature dependence data is written to %s", file_name)

    return


def read_chiT_regression_csv(filename: str) -> dict[str, float]:
    """
    Read a Curie-normalised chiT regression CSV file and return fit parameters.

    The CSV is expected to contain columns: `type`, `intercept`, and `slope`. Each row
    is flattened into keys of the form `{type}_intercept` and `{type}_slope`.

    Args:
        filename (str): Path to the regression CSV file.

    Returns:
        dict[str, float]: Flattened fit parameters keyed by `{type}_{intercept|slope}`.

    Raises:
        ValueError: If required columns are missing from the CSV.
    """

    # Read CSV, skipping comment lines
    df = read_csv_safe(filename)

    # Sanity check
    required_cols = {"type", "intercept", "slope", "intercept_err", "slope_err"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain {required_cols}")

    params = {}

    for _, row in df.iterrows():
        label = row["type"]
        params[f"{label}_intercept"] = float(row["intercept"])
        params[f"{label}_slope"] = float(row["slope"])
        params[f"{label}_intercept_err"] = float(row["intercept_err"])
        params[f"{label}_slope_err"] = float(row["slope_err"])

    return params
