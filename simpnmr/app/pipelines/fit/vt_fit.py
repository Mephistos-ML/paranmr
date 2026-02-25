# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Fit temperature-dependent (VT) susceptibility components to experimental data.

Fits iso/ax/rho susceptibility components using the VT model to extract
temperature dependence, optionally fixing the temperature-independent
paramagnetism (TIP) from ab initio susceptibilities.
"""

import copy
import logging
import os

import numpy as np

# Application layer
from simpnmr.app.loaders.susc_load import load_susceptibilities

# Core / domain
from simpnmr.core.build.susc import build_ab_initio_chit_series
from simpnmr.core.fitting import vt

# IO layer
from simpnmr.io.csv.fit import save_slope_intercept
from simpnmr.io.qc import gateway as rdrs

# Visualisation
from simpnmr.viz.plots.susc import plot_exp_vs_ab_initio, plot_isoaxrho
from simpnmr.viz.style.theme import apply_profile

logger = logging.getLogger(__name__)


def fit_vt(
    config,
    molecules,
    spin,
    susc_models,
    plot_profile: str,
    show_plots: bool,
) -> None:
    # Define the components to fit
    fit_component = ["iso", "ax", "rho"]

    # Default to high-temperature limit unless the user explicitly requests vt_2nd_order
    method = config.susc_vt_method or "ht_limit"

    # Read VT variables (may be None if not provided)
    susc_vt_variables = config.susc_vt_variables

    if method == "vt_2nd_order":
        assert susc_vt_variables is not None

    # Load the optional susceptibility-model input used for TIP extraction
    tip_type = config.susc_vt_tip_type

    # Load temperatures from the fitted susceptibility tensors
    temps_fit = np.array([mol.susc.temperature for mol in molecules])

    # Load the fitted Iso/Ax/Rho components
    chi_vals = {
        "iso": np.array([mol.susc.iso for mol in molecules]),
        "ax": np.array([abs(mol.susc.axiality) for mol in molecules]),
        "rho": np.array([abs(mol.susc.rhombicity) for mol in molecules]),
    }

    # Optional ab initio series and analytic chi(T) curves
    # (used only for vt_2nd_order + fixed TIP)
    ab_series: dict[str, np.ndarray] | None = None
    analytic_chi_vt: dict[str, np.ndarray] | None = None

    if tip_type == "fix_tip_from_ab_initio" and method == "vt_2nd_order":
        if (
            config.susc_vt_ab_initio_format is None
            or "orca" not in config.susc_vt_ab_initio_format
        ):
            raise ValueError("Only Orca is currently supported")
        # TODO: use new functionality for an auto method detection here
        section = config.susc_vt_ab_initio_format.split("orca_", 1)[1]

        g_tensor = rdrs.read_orca_g_tensor(
            config.susc_vt_ab_initio_file,
            section=section,
        )

        suscs_ab_initio = load_susceptibilities(
            config.susc_vt_ab_initio_file,
            config.susc_vt_ab_initio_format,
            electronic=molecules[0].electronic,
            g_tensor=g_tensor,
        )
        eff_H = rdrs.read_eff_hamiltonian_tensor(
            config.susc_vt_ab_initio_file,
            section=section,
        )

        # Find the ab initio susceptibility temperature closest to the fit temperatures
        ab_temps = np.array([s.temperature for s in suscs_ab_initio], dtype=float)
        idx = int(np.argmin(np.abs(ab_temps - np.max(temps_fit))))

        # Use only the reference susceptibility for the VT/TIP pipeline
        susc_ab_initio = copy.deepcopy(suscs_ab_initio[idx])

        # Rotate the effective Hamiltonian tensor into the chi eigenframe
        eff_H_rot = susc_ab_initio.eigvecs.T @ eff_H @ susc_ab_initio.eigvecs

        # Construct a diagonal g-tensor in the chi eigenframe
        g_rot_diag = np.diag(
            np.diag(susc_ab_initio.eigvecs.T @ g_tensor @ susc_ab_initio.eigvecs)
        )

        # Precompute g^2 invariants in the chi eigenframe for analytic chi(T) evaluation
        g_sq = vt.compute_g_sq_components(g_rot_diag)

        # Compute the axial and rhombic parts of the effective Hamiltonian tensor (J)
        D_J, E_J = vt.calculate_E_D_components(eff_H_rot)

        # Map VT component identifiers to Susceptibility attribute names
        comp_to_attr = {"iso": "iso", "ax": "axiality", "rho": "rhombicity"}

        # Build the full ab initio chiT series
        ab_series_full = build_ab_initio_chit_series(
            suscs_ab_initio,
            g_corr_iso=True,
            spin=spin,
            orbit=molecules[0].electronic.orbit_L,
            total_momentum_J=molecules[0].electronic.total_J,
            g_tensor=g_tensor,
        )

        exp_t = np.asarray(temps_fit, dtype=float)
        ab_inv_full = np.asarray(ab_series_full["inv_t"], dtype=float)
        ab_t_full = 1.0 / ab_inv_full

        exp_t_min = float(np.min(exp_t))
        exp_t_max = float(np.max(exp_t))
        ab_mask = (ab_t_full >= exp_t_min) & (ab_t_full <= exp_t_max)

        # Native ab initio temperature grid within the fit window.
        t_grid = np.asarray(ab_t_full[ab_mask], dtype=float)
        if t_grid.size == 0:
            raise ValueError(
                "No ab initio temperatures fall within the experimental fit window. "
                "Cannot fix TIP from ab initio data."
            )

        # Compute analytic chi(T) curves on the in-window ab initio temperature grid
        analytic_chi_vt = {}
        for comp in fit_component:
            analytic_chi_vt[comp] = np.asarray(
                vt.compute_analytic_component(comp, t_grid, g_sq, D_J, E_J, spin),
                dtype=float,
            )

        # Fix TIP using the analytic chi(T) value associated with the chosen
        # ab initio reference susceptibility temperature
        t_ref = float(susc_ab_initio.temperature)
        idx_ref = int(np.argmin(np.abs(t_grid - t_ref)))

        for comp in fit_component:
            analytic_val_ref = float(analytic_chi_vt[comp][idx_ref])
            tip_ref = vt.compute_tip_correction(
                getattr(susc_ab_initio, comp_to_attr[comp]),
                analytic_val_ref,
                spin,
            )
            susc_vt_variables[comp]["tip"] = ["fix", float(tip_ref)]

        # Prepare ab initio series for plotting (clipped to the same temperature window)
        ab_series = {"inv_t": ab_inv_full[ab_mask]}
        for comp in fit_component:
            ab_series[comp] = np.asarray(ab_series_full[comp], dtype=float)[ab_mask]

        # Normalise ab initio chiT series by the Curie prefactor for consistency
        curie_prefactor = vt.compute_curie_prefactor(spin)
        for comp in fit_component:
            ab_series[comp] = ab_series[comp] / curie_prefactor

    # Initialize fitted chi errors to zero (if not available)
    chi_errors = {comp: np.zeros(len(temps_fit)) for comp in fit_component}

    # Fixed vars => sigma = 0.0. Missing stdev => sigma = 0.0.
    if susc_models:
        fix = getattr(susc_models[0], "fix_vars", {}) or {}

        if "iso" not in fix:
            chi_errors["iso"] = np.asarray(
                [float(m.fit_stdev.get("iso") or 0.0) for m in susc_models], dtype=float
            )

        if "ax" not in fix:
            chi_errors["ax"] = np.asarray(
                [float(m.fit_stdev.get("ax") or 0.0) for m in susc_models], dtype=float
            )

        if "rho" not in fix:
            chi_errors["rho"] = np.asarray(
                [float(m.fit_stdev.get("rho") or 0.0) for m in susc_models], dtype=float
            )

    # Create dictionaries to store fitted chiT values, errors, and fit parameters
    chiT_reduced = {}
    chiT_err_reduced = {}
    chiT_fit_params = {}

    # Fit the VT model parameters for each susceptibility component
    for comp in fit_component:
        if method == "vt_2nd_order":
            vals, errs, params = vt.fit_chit_linear_model(
                spin=spin,
                fit_temps=temps_fit,
                chi_vals=chi_vals[comp],
                chi_errors=chi_errors[comp],
                susc_vt_variables=susc_vt_variables[comp],
            )

        if temps_fit.size == 1 or method == "ht_limit":
            vals, errs, params = vt.compute_chit_high_t_limit(
                spin=spin,
                fit_temps=temps_fit,
                chi_vals=chi_vals[comp],
                chi_errors=chi_errors[comp],
            )

        # Store results
        chiT_reduced[comp] = vals
        chiT_err_reduced[comp] = errs
        chiT_fit_params[comp] = params

    # Precompute inverse temperature for plotting
    inv_temps_fit = 1.0 / temps_fit

    # Build the resolved plotting contract for this run.
    spec = apply_profile(plot_profile)

    if tip_type == "fix_tip_from_ab_initio" and method == "vt_2nd_order":
        with spec.context():
            plot_exp_vs_ab_initio(
                params=chiT_fit_params,
                g_sq=g_sq,
                inv_t=inv_temps_fit,
                ab_series=ab_series,
                analytic_chi_vt=analytic_chi_vt,
                spec=spec,
                show=show_plots,
                save=True,
                save_name=os.path.join(config.project_name, "exp_vs_ab_initio_susc"),
                verbose=True,
            )

    # Plot chiT temperature dependence
    with spec.context():
        plot_isoaxrho(
            vals=chiT_reduced,
            errs=chiT_err_reduced,
            params=chiT_fit_params,
            inv_t=inv_temps_fit,
            spec=spec,
            show=show_plots,
            save=True,
            save_name=os.path.join(
                config.project_name, "susceptibility_components_chiT"
            ),
            verbose=True,
        )

    # Write iso/ax/rho fit parameters to CSV
    out_file = os.path.join(config.project_name, "isoaxrho_fit.csv")
    fits_list = [
        chiT_fit_params.get("iso"),
        chiT_fit_params.get("ax"),
        chiT_fit_params.get("rho"),
    ]
    save_slope_intercept(fits_list, out_file)

    return
