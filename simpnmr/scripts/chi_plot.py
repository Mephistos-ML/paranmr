'''
            SimpNMR

        Copyright (C) 2025

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''

"""
Module for plotting temperature-dependent magnetic susceptibility tensors.

This script reads susceptibility tensors for CASSCF or NEVPT2 sections from an ORCA output file,
calculates isotropic, axial, and rhombic components, normalizes them, and plots these components
against inverse temperature with an optional secondary axis showing temperature in K.
It can also perform linear regression on provided experimental standard deviations (where present) and display predicted uncertainty bands.
"""

# Library imports
import math
import argparse
import csv
import logging
from sympy import symbols, nsolve
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import mu_0, physical_constants, k, Avogadro, h, c
from simpnmr import utils as ut
from simpnmr import readers as rdrs

# Physical constants from scipy.constants:
#   mu_0: Vacuum permeability [H/m]
#   k: Boltzmann constant [J/K]
#   Avogadro: Avogadro constant [1/mol]
#   h: Planck constant [J·s]
#   c: Speed of light [m/s]

# Import g-factor of the free electron
g_e = abs(physical_constants['electron g factor'][0])

# --- Short label mapping and plot helpers ---
SHORT_LABELS = {
    'NEVPT2': 'NEVPT2',
    'Analytical': 'Analytical',
    'Fitted': 'Exp.',
    'Fitted (LR)': 'LR.',
    'Fitted TIP (LR)': 'LR. + TIP.',
    'iso': 'iso',
    'ax': 'ax',
    'rho': 'rho',
    r'$(g^2)_{\mathrm{iso}}$': r'$(g^2)_{\mathrm{iso}}$',
    r'$(g^2)_{\mathrm{ax}}$': r'$(g^2)_{\mathrm{ax}}$',
    r'$(g^2)_{\rho}$': r'$(g^2)_{\rho}$'
}

def _finalize_axes(ax, inv_t, inv_t_csv, ylabel_suffix: str = '', y_pad_frac: float = 0.2, legend_ncol: int | None = None):
    """
    Finalize styling for χT vs 1/T plots with a secondary temperature axis on top.
    """
    ax.set_xlabel(r'$1/\mathrm{Temperature}\ (1/\mathrm{K})$', fontsize=22, labelpad=10)
    base_ylabel = r'$\chi\,T$ (dimensionless)'
    if ylabel_suffix:
        ax.set_ylabel(fr'{base_ylabel} — {ylabel_suffix}', fontsize=22, labelpad=10)
    else:
        ax.set_ylabel(base_ylabel, fontsize=22, labelpad=10)

    sec_ax = ax.secondary_xaxis(
        'top',
        functions=(
            lambda inv: np.divide(1, inv, out=np.full_like(inv, np.nan), where=inv != 0),
            lambda T: np.divide(1, T, out=np.full_like(T, np.nan), where=T != 0),
        ),
    )
    sec_ax.set_xlabel('Temperature (K)', fontsize=22, labelpad=12)

    if legend_ncol is not None:
        legend = ax.legend(loc='upper left', fontsize=12, ncol=legend_ncol)
    else:
        if len(inv_t_csv) > 0:
            legend = ax.legend(loc='upper left', fontsize=12, ncol=5)
        else:
            legend = ax.legend(loc='upper left', fontsize=12, ncol=2)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_alpha(1)

    ax.grid(True)

    try:
        y_min, y_max = ax.get_ylim()
        y_range = (y_max - y_min)
        if np.isfinite(y_range) and y_range > 0:
            pad = y_range * float(y_pad_frac)
            ax.set_ylim(y_min - pad, y_max + pad)
    except Exception:
        pass

    ax.tick_params(axis='both', which='major', labelsize=18)
    sec_ax.tick_params(axis='x', which='major', labelsize=18)
    return legend


def _weighted_r2(x_tr, y_tr, sigma_tr, a, b) -> float | None:
    """
    Compute weighted R^2 for y = a*x + b using optional sigmas as 1/sigma^2 weights.
    Returns None if inputs are insufficient.
    """
    try:
        if (x_tr is None) or (y_tr is None) or (a is None) or (b is None):
            return None
        if len(x_tr) != len(y_tr) or len(x_tr) < 2:
            return None

        x = np.asarray(x_tr, dtype=float)
        y = np.asarray(y_tr, dtype=float)

        if sigma_tr is not None and len(sigma_tr) == len(x_tr):
            s = np.array(
                [
                    (np.nan if (v is None or (isinstance(v, (int, float)) and v <= 0)) else float(v))
                    for v in sigma_tr
                ],
                dtype=float,
            )
            valid = np.isfinite(s) & (s > 0)
            if np.any(valid):
                med = float(np.median(s[valid]))
                s[~valid] = med
                w = 1.0 / (s**2)
            else:
                w = np.ones_like(x)
        else:
            w = np.ones_like(x)

        yhat = a * x + b
        wy_sum = float((w * y).sum())
        w_sum = float(w.sum()) if float(w.sum()) != 0 else 1.0
        y_bar_w = wy_sum / w_sum

        ss_res_w = float((w * (y - yhat) ** 2).sum())
        ss_tot_w = float((w * (y - y_bar_w) ** 2).sum())
        if ss_tot_w == 0:
            return None
        return 1.0 - ss_res_w / ss_tot_w
    except Exception:
        return None


def _annotate_fit(r2, a, b, sa, sb) -> str:
    """
    Build a compact text caption with R^2, slope and intercept (with errors)
    for placement on a Matplotlib axes.
    """
    def _fmt(v):
        if v is None:
            return '—'
        try:
            if not np.isfinite(v):
                return '—'
            return f"{v:.3g}"
        except Exception:
            return '—'

    return (
        rf"$\mathbf{{R^2:}}$ {_fmt(r2)}   |   "
        rf"$\mathbf{{Slope\ (a):}}$ {_fmt(a)} ± {_fmt(sa)}   |   "
        rf"$\mathbf{{Intercept\ (b):}}$ {_fmt(b)} ± {_fmt(sb)}"
    )


def _add_caption(fig: plt.Figure, text: str, ax: plt.Axes | None = None) -> None:
    """
    Add a formatted text caption with fit statistics to the provided axes.
    """
    if not text:
        return
    target_ax = ax or fig.axes[0]
    target_ax.text(
        0.985,
        0.03,
        text,
        ha='right',
        va='bottom',
        transform=target_ax.transAxes,
        fontsize=16,
        color='#333333',
        bbox=dict(
            facecolor='white',
            edgecolor='black',
            boxstyle='round,pad=0.4',
            linewidth=0.8,
        ),
    )


def _plot_component(
    color_name: str,
    inv_t: list[float],
    nevpt2_series: list[float],
    analytic_series: list[float],
    g_sq_series: list[float],
    inv_t_csv: list[float],
    fit_csv: list[float],
    sdev_csv: list[float] | None,
    fit_pred: list[float] | None,
    sdev_pred: list[float] | None,
    a: float | None,
    b: float | None,
    a_se: float | None,
    b_se: float | None,
    suffix: str,
    outfile: str,
    fit_pred_TIP: list[float] | None = None,
    sdev_pred_TIP: list[float] | None = None,
    a_TIP: float | None = None,
    b_TIP: float | None = None,
    a_TIP_se: float | None = None,
    b_TIP_se: float | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a single χT component (iso, ax, or rho) together with analytic,
    (g^2) and experimental / LR curves.
    """
    fig_c, ax_c = plt.subplots(figsize=(10, 6), constrained_layout=True)

    ax_c.plot(
        inv_t,
        nevpt2_series,
        label=SHORT_LABELS["NEVPT2"],
        color=color_name,
    )
    ax_c.plot(
        inv_t,
        analytic_series,
        label=SHORT_LABELS["Analytical"],
        color=color_name,
        linestyle='--',
    )

    g_label_dict = {
        'iso': r'$(g^2)_{\mathrm{iso}}$',
        'ax': r'$(g^2)_{\mathrm{ax}}$',
        'rho': r'$(g^2)_{\rho}$',
    }
    ax_c.plot(
        inv_t,
        g_sq_series,
        label=g_label_dict[suffix],
        color=color_name,
        linestyle=':',
    )

    if len(inv_t_csv) == len(fit_csv) and len(inv_t_csv) > 0:
        ax_c.plot(
            inv_t_csv,
            fit_csv,
            label=SHORT_LABELS["Fitted"],
            color=color_name,
            marker='o',
            linestyle='',
            markersize=5,
        )
        if sdev_csv is not None and len(sdev_csv) == len(inv_t_csv) and any(
            v is not None for v in sdev_csv
        ):
            yerr = [v if (v is not None) else 0.0 for v in sdev_csv]
            ax_c.errorbar(
                inv_t_csv,
                fit_csv,
                yerr=yerr,
                fmt='none',
                ecolor=color_name,
                alpha=0.5,
                capsize=2,
            )

    if fit_pred is not None:
        ax_c.plot(
            inv_t,
            fit_pred,
            label=SHORT_LABELS["Fitted (LR)"],
            linestyle='-.',
            linewidth=1.5,
            color=color_name,
        )
    if fit_pred is not None and sdev_pred is not None:
        center = np.array(fit_pred, dtype=float)
        sdev = np.array(sdev_pred, dtype=float)
        ax_c.fill_between(
            inv_t,
            center - sdev,
            center + sdev,
            alpha=0.15,
            facecolor=color_name,
        )

    if fit_pred_TIP is not None:
        ax_c.plot(
            inv_t,
            fit_pred_TIP,
            label=SHORT_LABELS["Fitted TIP (LR)"],
            linestyle=(0, (5, 2, 1, 2)),
            linewidth=2.0,
            color='purple',
            alpha=0.8,
        )
    if fit_pred_TIP is not None and sdev_pred_TIP is not None:
        center_tip = np.array(fit_pred_TIP, dtype=float)
        sdev_tip = np.array(sdev_pred_TIP, dtype=float)
        ax_c.fill_between(
            inv_t,
            center_tip - sdev_tip,
            center_tip + sdev_tip,
            alpha=0.10,
            facecolor=color_name,
        )

    r2_val = _weighted_r2(inv_t_csv, fit_csv, sdev_csv, a, b)
    caption = _annotate_fit(r2_val, a, b, a_se, b_se)

    label_map = {'iso': 'Isotropic', 'ax': 'Axiality', 'rho': 'Rhombicity'}
    _finalize_axes(ax_c, inv_t, inv_t_csv, label_map.get(suffix, suffix), legend_ncol=10)
    _add_caption(fig_c, caption, ax_c)

    plt.savefig(outfile, dpi=600)
    return fig_c, ax_c


# --- CSV-only plot helper ---
def _plot_component_csv_only(
    color_name: str,
    inv_t_csv: list[float],
    fit_csv: list[float],
    sdev_csv: list[float] | None,
    fit_pred: list[float] | None,
    sdev_pred: list[float] | None,
    a: float | None,
    b: float | None,
    a_se: float | None,
    b_se: float | None,
    suffix: str,
    outfile: str,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a single χT component (iso, ax, or rho) for CSV-only mode, i.e. without
    NEVPT2/analytic/g^2 reference curves.
    """
    fig_c, ax_c = plt.subplots(figsize=(10, 6), constrained_layout=True)

    # Plot fitted CSV points
    if len(inv_t_csv) == len(fit_csv) and len(inv_t_csv) > 0:
        ax_c.plot(
            inv_t_csv,
            fit_csv,
            label=SHORT_LABELS["Fitted"],
            color=color_name,
            marker='o',
            linestyle='',
            markersize=5,
        )
        if sdev_csv is not None and len(sdev_csv) == len(inv_t_csv) and any(
            v is not None for v in sdev_csv
        ):
            yerr = [v if (v is not None) else 0.0 for v in sdev_csv]
            ax_c.errorbar(
                inv_t_csv,
                fit_csv,
                yerr=yerr,
                fmt='none',
                ecolor=color_name,
                alpha=0.5,
                capsize=2,
            )

    # Plot LR prediction and its band (if available)
    if fit_pred is not None:
        ax_c.plot(
            inv_t_csv,
            fit_pred,
            label=SHORT_LABELS["Fitted (LR)"],
            linestyle='-.',
            linewidth=1.5,
            color=color_name,
        )
    if fit_pred is not None and sdev_pred is not None:
        center = np.array(fit_pred, dtype=float)
        sdev = np.array(sdev_pred, dtype=float)
        ax_c.fill_between(
            inv_t_csv,
            center - sdev,
            center + sdev,
            alpha=0.15,
            facecolor=color_name,
        )

    r2_val = _weighted_r2(inv_t_csv, fit_csv, sdev_csv, a, b)
    caption = _annotate_fit(r2_val, a, b, a_se, b_se)

    label_map = {'iso': 'Isotropic', 'ax': 'Axiality', 'rho': 'Rhombicity'}
    _finalize_axes(ax_c, inv_t_csv, inv_t_csv, label_map.get(suffix, suffix), legend_ncol=5)
    _add_caption(fig_c, caption, ax_c)

    plt.savefig(outfile, dpi=600)
    return fig_c, ax_c

def calculate_chi_components_nevpt2(tensors: dict[float, np.ndarray], g_matrix: np.ndarray) -> tuple[list[float], list[float], list[float], list[np.ndarray]]:
    """
    Calculate inverse temperature list and isotropic, axial, rhombic chi components.

    Args:
        tensors (dict of float->ndarray): Mapping from temperature to 3×3 susceptibility tensor.

    Returns:
        chi_iso (list of float): Isotropic susceptibility components.
        chi_ax (list of float): Axial susceptibility components.
        chi_rho (list of float): Rhombic susceptibility components.
    """

    # Initialize containers for susceptibility components
    chi_iso, chi_ax, chi_rho = [], [], []
    chi_eigenvectors_list = []

    # Convert
    conv = 1e-6 * 4 * math.pi

    for T in tensors.keys():
        chiT = tensors[T]

        # Use trace/3 only to get the traceless part for eigen-decomposition (axial/rhombic)
        iso_trace = np.trace(chiT) / 3.0
        traceless = chiT - np.eye(3) * iso_trace

        # Principal values/vectors of the traceless tensor
        eigs, eigvls = np.linalg.eig(traceless)
        idx = np.argsort(np.abs(eigs))
        chi_diag_traceless = np.diag(eigs.real[idx])
        eigvls_sorted = eigvls[:, idx]
        chi_eigenvectors_list.append(eigvls_sorted)

        # Axial and rhombic components from the traceless eigenvalues
        chi_ax.append(1.5 * chi_diag_traceless[2, 2] / Avogadro * conv)
        chi_rho.append((chi_diag_traceless[0, 0] - chi_diag_traceless[1, 1]) / 2.0 / Avogadro * conv)

        # Work in the χ eigenframe, but use the FULL χ tensor principal values
        V = eigvls_sorted
        chi_diag_full = np.diag(V.T @ chiT @ V)
        g_diag = np.diag(V.T @ g_matrix @ V)

        # Scale χ principal components to per-particle SI and then form the weighted average
        chi_diag_scaled = (chi_diag_full / Avogadro) * conv
        term = np.divide(chi_diag_scaled, g_diag, out=np.zeros_like(chi_diag_scaled, dtype=float), where=g_diag != 0)
        chi_iso.append((g_e/3) * float(np.sum(term)))

    return chi_iso, chi_ax, chi_rho, chi_eigenvectors_list

def calculate_g_components(rotated_g_tensors: list[np.ndarray]) -> tuple[list[float], list[float], list[float]]:
    """
    Calculate isotropic, axial, and rhombic g-matrix components
    for each rotated g-tensor in χ bases.

    Args:
        rotated_g_tensors (list of ndarray): List of 3×3 g-matrices
        already rotated into each χ eigenframe.

    Returns:
        g_sq_iso (list of float): Isotropic components squared at each temperature.
        g_sq_ax (list of float): Axial components squared at each temperature.
        g_sq_rh (list of float): Rhombic components squared at each temperature.
    """

    # Calculate g-components directly from the diagonal of each rotated tensor
    g_sq_iso = []
    g_sq_ax  = []
    g_sq_rh  = []

    for g_mat in rotated_g_tensors:
        iso_sq = (g_mat[0, 0]**2 + g_mat[1, 1]**2 + g_mat[2, 2]**2) / 3.0
        ax_sq  = 1.5 * (g_mat[2, 2]**2 - iso_sq)
        rh_sq  = (g_mat[0, 0]**2 - g_mat[1, 1]**2) / 2.0

        g_sq_iso.append(iso_sq)
        g_sq_ax.append(ax_sq)
        g_sq_rh.append(rh_sq)

    return g_sq_iso, g_sq_ax, g_sq_rh

def calculate_E_D_components(rotated_eff_H_tensors: list[np.ndarray]) -> tuple[list[float], list[float]]:
    """
    Calculate E and D Effective Hamiltonian matrix components
    for each rotated Effective Hamiltonian tensor in χ bases.

    Args:
        rotated_eff_H_tensors (list of ndarray): List of 3×3 Effective Hamiltonian matrices
        already rotated into each χ eigenframe.

    Returns:
        D (list of float): Axial components at each temperature.
        E (list of float): Rhombic components at each temperature.
    """

    # Calculate Effective Hamiltonian directly from the diagonal of each rotated tensor
    D = []
    E = []

    for eff_H_mat in rotated_eff_H_tensors:
        iso = np.trace(eff_H_mat) / 3.0
        D.append(1.5 * (eff_H_mat[2, 2] - iso))
        # D.append(1.5 * (eff_H_mat[2, 2]))
        E.append((eff_H_mat[0, 0] - eff_H_mat[1, 1]) / 2.0)

    return D, E


# --- Helper: Analytic (g^2)_iso from LR intercepts for D estimation ---
from typing import Optional

def calculate_g_sq_iso_analytic(
    b_iso_intercept: float,
    b_ax_intercept: float,
    K: float = 1.0,
) -> float:
    """
    Compute (g^2)_iso analytically from the **linear-regression intercepts** of
    the scalar (isotropic) and axial chi·T vs 1/T fits, under the axial-only
    (gx ≈ gy) approximation.

    This is a thin wrapper around `_solve_g_principals_axial_only` so that the
    algebra for recovering principal g-values is implemented in a single place.

    Parameters
    ----------
    b_iso_intercept : float
        Intercept of the isotropic fit (chi_iso·T) at 1/T → 0 (T → ∞).
        According to the working convention, this equals g_e * g_iso * K.
    b_ax_intercept : float
        Intercept of the axial fit (chi_ax·T) at 1/T → 0 (T → ∞), i.e. (g^2)_ax * K.
    K : float, optional
        Curie-like factor used before normalization. In this workflow, chi values have already
        been normalized by the Curie constant (norm_factor), so use K=1.0 (default).

    Returns
    -------
    float
        (g^2)_iso computed under a weak-rhombicity model (g_x ≈ g_y), using the LR intercepts.
    """
    # Undo any external scaling so that the effective intercepts match the
    # expectations of the axial-only solver.
    b_iso_eff = b_iso_intercept / K
    b_ax_eff = b_ax_intercept / K

    gx, gy, gz = _solve_g_principals_axial_only(
        b_iso=b_iso_eff,
        b_ax=b_ax_eff,
    )
    g_sq_iso_analytic = (gx**2 + gy**2 + gz**2) / 3.0
    return float(g_sq_iso_analytic)


# --- Helper to write all computed series to a CSV file ---
def write_results_csv(
    output_csv_path: str,
    temps: list[float],
    inv_t: list[float],
    chi_iso_nevpt2_si: list[float],
    chi_ax_nevpt2_si: list[float],
    chi_rho_nevpt2_si: list[float],
    chi_iso_analytic: list[float],
    chi_ax_analytic: list[float],
    chi_rho_analytic: list[float],
    g_sq_iso: list[float],
    g_sq_ax: list[float],
    g_sq_rh: list[float],
    D_cm_inv: list[float],
    E_cm_inv: list[float],
    D_J: list[float],
    E_J: list[float],
    chi_iso_fit_pred: list[float] | None,
    chi_ax_fit_pred: list[float] | None,
    chi_rho_fit_pred: list[float] | None,
    chi_iso_sdev_pred: list[float] | None,
    chi_ax_sdev_pred: list[float] | None,
    chi_rho_sdev_pred: list[float] | None,
    temps_csv: list[float] | None,
    inv_t_csv: list[float] | None,
    chi_iso_fit_csv: list[float] | None,
    chi_ax_fit_csv: list[float] | None,
    chi_rho_fit_csv: list[float] | None,
    chi_iso_sdev_csv: list[float] | None,
    chi_ax_sdev_csv: list[float] | None,
    chi_rho_sdev_csv: list[float] | None,
    a_iso: float | None,
    b_iso: float | None,
    a_ax: float | None,
    b_ax: float | None,
    a_rho: float | None,
    b_rho: float | None,
    a_iso_se: float | None,
    b_iso_se: float | None,
    a_ax_se: float | None,
    b_ax_se: float | None,
    a_rho_se: float | None,
    b_rho_se: float | None,
) -> None:
    """
    Write all calculated NEVPT2/analytic quantities and (if available) linear-regression
    fitted series to a CSV that shares the same folder as the plot image.

    Also writes a second block with the original CSV fitted points (on their own temperature grid), if provided.
    A third block stores the linear regression coefficients (slope a, intercept b) for each fitted target.

    The CSV is written on the NEVPT2 temperature grid `temps`.
    """
    # Guard against length mismatches by truncating everything to the common min length
    series = [
        temps, inv_t,
        chi_iso_nevpt2_si, chi_ax_nevpt2_si, chi_rho_nevpt2_si,
        chi_iso_analytic, chi_ax_analytic, chi_rho_analytic,
        g_sq_iso, g_sq_ax, g_sq_rh,
        D_cm_inv, E_cm_inv, D_J, E_J,
    ]

    # Optional series
    opt_series = [
        chi_iso_fit_pred, chi_ax_fit_pred, chi_rho_fit_pred,
        chi_iso_sdev_pred, chi_ax_sdev_pred, chi_rho_sdev_pred
    ]

    # Compute minimum valid length across mandatory series
    min_len = min(len(s) for s in series)

    # Helper to safely fetch a value or return '' when optional is missing
    def _get_opt(seq, i):
        if seq is None:
            return ''
        if i < len(seq):
            return seq[i]
        return ''

    headers = [
        'Temperature (K)', '1/T (1/K)',
        'chi_iso_NEVPT2_norm', 'chi_ax_NEVPT2_norm', 'chi_rho_NEVPT2_norm',
        'chi_iso_analytic', 'chi_ax_analytic', 'chi_rho_analytic',
        '(g^2)_iso', '(g^2)_ax', '(g^2)_rho',
        'D (cm^-1)', 'E (cm^-1)', 'D (J)', 'E (J)',
        'chi_iso_fit_LR', 'chi_ax_fit_LR', 'chi_rho_fit_LR',
        'chi_iso_sdev_LR', 'chi_ax_sdev_LR', 'chi_rho_sdev_LR'
    ]

    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(min_len):
            row = [
                temps[i], inv_t[i],
                chi_iso_nevpt2_si[i], chi_ax_nevpt2_si[i], chi_rho_nevpt2_si[i],
                chi_iso_analytic[i],  chi_ax_analytic[i],  chi_rho_analytic[i],
                g_sq_iso[i], g_sq_ax[i], g_sq_rh[i],
                D_cm_inv[i], E_cm_inv[i], D_J[i], E_J[i],
                _get_opt(chi_iso_fit_pred, i), _get_opt(chi_ax_fit_pred, i), _get_opt(chi_rho_fit_pred, i),
                _get_opt(chi_iso_sdev_pred, i), _get_opt(chi_ax_sdev_pred, i), _get_opt(chi_rho_sdev_pred, i)
            ]
            writer.writerow(row)

        # If we have raw CSV-fitted points, append them as a second block
        if temps_csv is not None and inv_t_csv is not None and chi_iso_fit_csv is not None and chi_ax_fit_csv is not None and chi_rho_fit_csv is not None:
            # Blank separator row
            writer.writerow([])
            # Header for raw CSV data
            raw_headers = [
                'CSV Temperature (K)', 'CSV 1/T (1/K)',
                'chi_iso_fitted_raw', 'chi_ax_fitted_raw', 'chi_rho_fitted_raw',
                'chi_iso_sdev_raw', 'chi_ax_sdev_raw', 'chi_rho_sdev_raw'
            ]
            writer.writerow(raw_headers)
            # Determine length safely
            n_raw = min(
                len(temps_csv), len(inv_t_csv),
                len(chi_iso_fit_csv), len(chi_ax_fit_csv), len(chi_rho_fit_csv)
            )
            # sdev arrays may be None or contain None entries
            for j in range(n_raw):
                iso_sd = '' if (chi_iso_sdev_csv is None or j >= len(chi_iso_sdev_csv) or chi_iso_sdev_csv[j] is None) else chi_iso_sdev_csv[j]
                ax_sd  = '' if (chi_ax_sdev_csv  is None or j >= len(chi_ax_sdev_csv)  or chi_ax_sdev_csv[j]  is None) else chi_ax_sdev_csv[j]
                rho_sd  = '' if (chi_rho_sdev_csv  is None or j >= len(chi_rho_sdev_csv)  or chi_rho_sdev_csv[j]  is None) else chi_rho_sdev_csv[j]
                writer.writerow([
                    temps_csv[j], inv_t_csv[j],
                    chi_iso_fit_csv[j], chi_ax_fit_csv[j], chi_rho_fit_csv[j],
                    iso_sd, ax_sd, rho_sd
                ])

        # Append LR coefficients as a summary block
        writer.writerow([])
        lr_headers = ['target', 'slope_a', 'intercept_b', 'slope_err', 'intercept_err']
        writer.writerow(lr_headers)
        def _w(name, a, b, a_err, b_err):
            writer.writerow([
                name,
                '' if a is None else a,
                '' if b is None else b,
                '' if a_err is None else a_err,
                '' if b_err is None else b_err,
            ])
        _w('chi_iso (fit)', a_iso, b_iso, a_iso_se, b_iso_se)
        _w('chi_ax (fit)',  a_ax,  b_ax,  a_ax_se,  b_ax_se)
        _w('chi_rho (fit)', a_rho, b_rho, a_rho_se, b_rho_se)

# Weighted linear regression with optional known pointwise standard deviations.
def weighted_linreg_predict(x_train: list[float],
                            y_train: list[float],
                            x_pred: list[float],
                            sigma: list[float] | None = None,
                            fixed_intercept: float | None = None
                            ) -> tuple[list[float] | None, float | None, float | None, list[float] | None, float | None, float | None]:
    """
    Perform weighted least squares for the linear model y = a*x + b.

    If `sigma` is provided (pointwise standard deviations for y_train), uses
    weights w_i = 1/sigma_i^2 for those entries where sigma_i is not None and > 0.
    Missing or non-positive sigmas fall back to the median of available sigmas.

    Parameters
    ----------
    x_train : list[float]
        Training x values.
    y_train : list[float]
        Training y values.
    x_pred : list[float]
        X values to predict at.
    sigma : list[float] | None
        Optional standard deviations for y_train.
    fixed_intercept : float | None
        If provided, holds b fixed at this value and fits only the slope a.

    Returns
    -------
    y_pred : list[float] | None
        Predicted y values at x_pred.
    a, b : float | None
        Fitted slope and intercept.
    y_pred_std : list[float] | None
        Standard deviation of the predicted MEAN at each x_pred based on the
        parameter covariance matrix (i.e., sqrt(f^T Cov f), with f=[x,1]).
    sigma_a, sigma_b : float | None
        Standard errors of the fitted slope and intercept.
    """
    if x_train is None or y_train is None:
        return None, None, None, None, None, None
    if len(x_train) < 2 or len(y_train) < 2:
        return None, None, None, None, None, None
    try:
        x = np.asarray(x_train, dtype=float)
        y = np.asarray(y_train, dtype=float)
        if fixed_intercept is not None:
            # Fit y - b_fixed = a * x using weighted least squares
            y_centered = y - float(fixed_intercept)
            # Build design matrix with a single column [x]
            A = x.reshape(-1, 1)

            # Prepare weights (reuse logic from existing branch)
            if sigma is not None and any(s is not None for s in sigma):
                s_arr = np.array([np.nan if (s is None or (isinstance(s, (int, float)) and s <= 0)) else float(s) for s in sigma], dtype=float)
                valid = np.isfinite(s_arr) & (s_arr > 0)
                if np.any(valid):
                    med = float(np.median(s_arr[valid]))
                    s_arr[~valid] = med
                    w = 1.0 / (s_arr ** 2)
                else:
                    w = np.ones_like(x)
            else:
                w = np.ones_like(x)
            W_sqrt = np.sqrt(w)
            A_w = A * W_sqrt[:, None]
            y_w = y_centered * W_sqrt

            # Solve for slope a
            theta, *_ = np.linalg.lstsq(A_w, y_w, rcond=None)
            a = float(theta[0])
            b = float(fixed_intercept)

            # Parameter covariance for a
            ATA = A_w.T @ A_w
            ATA_inv = np.linalg.inv(ATA)
            resid = y_centered - a * x
            dof = max(0, len(x) - 1)  # one parameter (a)
            RSS_w = float((resid**2 * w).sum())
            if np.allclose(w, w[0]):
                s2 = RSS_w / w[0] / dof if dof > 0 else 0.0
                Cov = ATA_inv * s2
            else:
                Cov = ATA_inv
                if dof > 0 and not (sigma is not None and any(s is not None for s in sigma)):
                    s2 = RSS_w / dof
                    Cov = Cov * s2

            # Predictions and their std (of the mean)
            xp = np.asarray(x_pred, dtype=float)
            y_pred = (a * xp + b).astype(float)
            y_var = (xp**2) * Cov[0, 0]
            y_var = np.maximum(y_var, 0.0)
            y_pred_std = np.sqrt(y_var)

            sigma_a = math.sqrt(Cov[0, 0]) if np.isfinite(Cov[0, 0]) else None
            sigma_b = 0.0  # fixed intercept
            return y_pred.tolist(), a, b, y_pred_std.tolist(), sigma_a, sigma_b
        else:
            # Build design matrix A = [x, 1]
            A = np.column_stack([x, np.ones_like(x)])

            # Prepare weights
            if sigma is not None and any(s is not None for s in sigma):
                # Convert sigma list to array, replacing None/<=0 with median of valid sigmas
                s_arr = np.array([np.nan if (s is None or (isinstance(s, (int, float)) and s <= 0)) else float(s) for s in sigma], dtype=float)
                valid = np.isfinite(s_arr) & (s_arr > 0)
                if np.any(valid):
                    med = float(np.median(s_arr[valid]))
                    s_arr[~valid] = med
                    w = 1.0 / (s_arr ** 2)
                else:
                    w = np.ones_like(x)
            else:
                w = np.ones_like(x)
            W_sqrt = np.sqrt(w)
            A_w = A * W_sqrt[:, None]
            y_w = y * W_sqrt

            # Solve (A^T W A) theta = (A^T W y)
            theta, *_ = np.linalg.lstsq(A_w, y_w, rcond=None)
            a = float(theta[0]); b = float(theta[1])

            # Parameter covariance
            # If weights are absolute (known variances), use Cov = (A^T W A)^{-1}
            # Otherwise scale by the residual variance s2.
            ATA = A_w.T @ A_w
            ATA_inv = np.linalg.inv(ATA)
            resid = y - (a * x + b)
            dof = max(0, len(x) - 2)
            # Weighted residual sum of squares
            RSS_w = float((resid**2 * w).sum())
            if np.allclose(w, w[0]):  # unweighted or constant weights
                s2 = RSS_w / w[0] / dof if dof > 0 else 0.0
                Cov = ATA_inv * s2
            else:
                # For known sigmas, best is Cov = (A^T W A)^{-1}
                Cov = ATA_inv
                # If degrees of freedom exist and weights are heuristic, inflate by RSS/(n-2)
                if dof > 0 and not (sigma is not None and any(s is not None for s in sigma)):
                    s2 = RSS_w / dof
                    Cov = Cov * s2

            # Predictions and their standard deviations (of the mean)
            xp = np.asarray(x_pred, dtype=float)
            F = np.column_stack([xp, np.ones_like(xp)])
            y_pred = (F @ theta).astype(float)
            # Var(y_pred) = diag(F Cov F^T)
            y_var = np.einsum('ij,jk,ik->i', F, Cov, F)
            y_var = np.maximum(y_var, 0.0)
            y_pred_std = np.sqrt(y_var)
            sigma_a = math.sqrt(Cov[0, 0]) if np.isfinite(Cov[0, 0]) else None
            sigma_b = math.sqrt(Cov[1, 1]) if np.isfinite(Cov[1, 1]) else None
            return y_pred.tolist(), a, b, y_pred_std.tolist(), sigma_a, sigma_b
    except Exception as e:
        logging.warning(f"Weighted linear regression failed: {e}")
        return None, None, None, None, None, None

def compute_D(
    S: float,
    b_ax: float,
    g_sq_iso_analytic: float,
    slope_ax: float,
) -> float:
    """
    Estimate the axial ZFS parameter D (in cm^-1) from the weighted linear regression
    of χ·T vs 1/T using:
        a_ax  = −(f_S/(30k)) * D_J * ( (g^2)_ax + 3 (g^2)_iso )

    Here (g^2)_ax and (g^2)_iso are taken from the **intercepts** of the LR fits:
        b_ax  ≈ (g^2)_ax,   b_iso ≈ (g^2)_iso   (limit T → ∞).

    Parameters
    ----------
    S : float
        Spin quantum number.
    b_ax : float
        Intercept of χ_ax·T vs 1/T (≈ (g^2)_ax).
    b_iso : float
        Intercept of χ_iso·T vs 1/T (≈ (g^2)_iso).
    slope_ax : float
        Slope a_ax from LR of χ_ax·T vs 1/T.

    Returns
    -------
    float
        D in cm^-1.
    """

    # Coef. to convert D from K to cm^-1
    conv_K_to_cm_inv = 0.695 

    # Use only the axial slope (assume E ≈ 0) and LR intercepts for (g^2)
    f_S = (2.0 * S - 1.0) * (2.0 * S + 3.0)
    if f_S == 0.0:
        raise ValueError("Invalid spin S leading to f_S = 0.")

    numerator = 30 * slope_ax
    denominator = - f_S * (b_ax + 3 * g_sq_iso_analytic)
    if denominator == 0.0:
        raise ZeroDivisionError("Cannot determine D.")

    D_cm_inv = (numerator / denominator) * conv_K_to_cm_inv

    return D_cm_inv


# --- Helpers to recover principal g-values and (D, E) from LR coefficients ---

def _solve_g_principals_axial_only(
    b_iso: float,
    b_ax: float,
) -> tuple[float, float, float]:
    """
    Solve for principal g-values gx, gy, gz under the assumption of zero rhombicity
    (E ≈ 0 and gx ≈ gy). This reproduces the previous analytic branch.

    Parameters
    ----------
    b_iso : float
        Intercept of the isotropic fit (chi_iso·T) at 1/T → 0.
    b_ax : float
        Intercept of the axial fit (chi_ax·T) at 1/T → 0.

    Returns
    -------
    gx, gy, gz : float
        Principal g-values with gx = gy in this approximation.
    """
    g_iso = b_iso / g_e
    g_sq_ax_val = b_ax

    try:
        gx = -math.sqrt(g_sq_ax_val / 3.0 + g_iso**2) + 2.0 * g_iso
    except ValueError:
        gx = float('nan')
    gz = 3.0 * g_iso - 2.0 * gx
    gy = gx
    return gx, gy, gz


def _solve_g_principals_full(
    b_iso: float,
    b_ax: float,
    b_rh: float,
) -> tuple[float, float, float]:
    """
    Solve for principal g-values gx, gy, gz from LR intercepts without assuming
    gx = gy, i.e. for non-zero rhombicity.

    Uses the definitions
        (g^2)_iso = (gx^2 + gy^2 + gz^2)/3
        (g^2)_ax  = 1.5 * (gz^2 - (g^2)_iso)
        (g^2)_rh  = 0.5 * (gx^2 - gy^2)

    together with
        g_iso = (gx + gy + gz)/3 = b_iso / g_e
        (g^2)_ax = b_ax
        (g^2)_rh = b_rh
    and solves the resulting system of 3 equations with sympy.nsolve.

    Returns sorted principal values (gx ≤ gy ≤ gz) for reproducibility.
    """
    gx, gy, gz = symbols('gx gy gz', real=True)

    g_iso_fit = b_iso / g_e

    g2_iso = (gx**2 + gy**2 + gz**2) / 3.0
    g2_ax = 1.5 * (gz**2 - g2_iso)
    g2_rh = 0.5 * (gx**2 - gy**2)

    eqs = [
        (gx + gy + gz) / 3.0 - g_iso_fit,
        g2_ax - b_ax,
        g2_rh - b_rh,
    ]

    # Initial guess: nearly isotropic solution around g_iso
    g0 = float(g_iso_fit)
    gx_val, gy_val, gz_val = nsolve(eqs, (gx, gy, gz), (g0 - 0.05, g0 + 0.05, g0 + 0.10))
    # gx_val, gy_val, gz_val = map(float, sol)
    g_sorted = sorted([gx_val, gy_val, gz_val])
    return g_sorted[0], g_sorted[1], g_sorted[2]


def solve_g_principals_from_LR(
    b_iso: float,
    b_ax: float,
    b_rh: float | None,
    rhombicity_tol: float = 1e-6,
) -> tuple[float, float, float]:
    """
    High-level helper that chooses between the axial-only (gx = gy) closed-form
    solution and the full nonlinear nsolve branch depending on |b_rh|.

    If b_rh is None or |b_rh| < rhombicity_tol, falls back to the old gx = gy logic.
    Otherwise uses nsolve to determine gx, gy, gz without symmetry assumptions.
    """
    if b_rh is None or abs(b_rh) < rhombicity_tol:
        return _solve_g_principals_axial_only(b_iso=b_iso, b_ax=b_ax)
    return _solve_g_principals_full(b_iso=b_iso, b_ax=b_ax, b_rh=b_rh)


def solve_D_E_from_LR(
    a_ax: float | None,
    a_rh: float | None,
    S: float,
    gx: float,
    gy: float,
    gz: float,
) -> tuple[float | None, float | None]:
    """
    Solve for axial (D) and rhombic (E) ZFS parameters from the slopes a_ax and a_rh
    of chi_ax·T and chi_rh·T vs 1/T using equations analogous to (46) and (47):

        a_ax = -(f(S)/(30 k_B)) * [ D((g^2)_ax + 3(g^2)_iso) - 3E (g^2)_rh ]
        a_rh =  (f(S)/(30 k_B)) * [ E((g^2)_ax - 3(g^2)_iso) + D (g^2)_rh     ]

    Here a_ax and a_rh are the slopes in the same normalized units used elsewhere
    in this script (i.e. after division by the Curie-like factor). D and E are
    returned in Joules; they can be converted to cm^-1 outside via D_cm = D_J / (h * c * 100).

    If a_rh is None, returns (None, None).
    """
    if a_ax is None or a_rh is None:
        return None, None

    g2_iso = (gx**2 + gy**2 + gz**2) / 3.0
    g2_ax = 1.5 * (gz**2 - g2_iso)
    g2_rh = 0.5 * (gx**2 - gy**2)

    f_S = (2.0 * S - 1.0) * (2.0 * S + 3.0)
    if f_S == 0.0:
        return None, None

    K = f_S / (30.0 * k)

    rhs1 = -a_ax / K
    rhs2 =  a_rh / K

    A = np.array(
        [
            [g2_ax + 3.0 * g2_iso, -3.0 * g2_rh],
            [g2_rh,                g2_ax - 3.0 * g2_iso],
        ],
        dtype=float,
    )
    rhs = np.array([rhs1, rhs2], dtype=float)

    try:
        D_J, E_J = np.linalg.solve(A, rhs)
        return float(D_J), float(E_J)
    except Exception as exc:
        logging.warning(f"Solving for D and E from LR failed: {exc}")
        return None, None

def rotate_tensor_to_chi_basis(
    tensor: np.ndarray, chi_eigenvectors_list: list[np.ndarray], temps: list[float]
) -> list[np.ndarray]:
    """
    Rotate a single 3×3 tensor into each χ eigenframe.

    Args:
        tensor (ndarray): 3×3 tensor to rotate (e.g., g_matrix or D_tensor).
        chi_eigenvectors_list (list of ndarray): List of 3×3 eigenvector matrices for each temperature.
        temps (list of float): List of temperatures corresponding to each eigenvector matrix.

    Returns:
        list[ndarray]: Rotated tensor for each χ basis, in the order of temps.
    """
    rotated = []
    for T, V in zip(temps, chi_eigenvectors_list):
        # transform tensor into the χ basis at temperature T
        rotated.append(V.T @ tensor @ V)
    return rotated

# Helper function to read susceptibility data from a CSV/TSV/whitespace file
def read_susceptibility_csv(csv_path: str) -> tuple[list[dict[str, float | None]], dict[str, str]]:
    """
    Read susceptibility data from a CSV/TSV/whitespace file and return:
      - rows: list of dicts with keys 'Temperature (K)', 'chi_iso', 'chi_ax', 'chi_rho'
      - units: dict with units for the chi columns, taken from headers
    The parser tolerates:
      * comma/semicolon/tab separated text
      * columns separated by variable runs of spaces (fixed-width-like tables)
      * optional preamble or comment lines beginning with '#'
    """
    import re
    with open(csv_path, 'r', newline='') as f:
        raw_lines = f.readlines()

    # Drop comment/blank lines for the header search
    candidate_lines = []
    for raw in raw_lines:
        s = raw.strip()
        if not s or s.startswith('#'):
            continue
        candidate_lines.append(s)

    # Find the header line including Temperature and chi_* bases
    header_idx_in_candidates = None
    for i, line in enumerate(candidate_lines):
        lc = line.lower()
        if 'temperature' in lc and 'chi_iso' in lc and 'chi_ax' in lc and 'chi_rho' in lc:
            header_idx_in_candidates = i
            break
    if header_idx_in_candidates is None:
        raise KeyError("Column 'Temperature (K)' not found")

    # Normalize header+data from that point forward into CSV by collapsing:
    #   - tabs or runs of 2+ spaces -> ','
    # Keep single spaces (e.g., inside 'Temperature (K)').
    def normalize(s: str) -> str:
        return re.sub(r'(?:\t+|\s{2,}|;)', ',', s.strip())

    normalized = []
    # Rebuild a slice starting at the detected header using the ORIGINAL raw lines,
    # but skip comments/blank lines.
    header_found = False
    for raw in raw_lines:
        s = raw.strip()
        if not s or s.startswith('#'):
            continue
        if not header_found:
            # Seek the exact header string we saw in candidate_lines
            if s != candidate_lines[header_idx_in_candidates]:
                continue
            header_found = True
        normalized.append(normalize(s))

    reader = csv.DictReader(normalized, delimiter=',')
    headers = [h.strip() if h else h for h in (reader.fieldnames or [])]

    # Debug: keep a human-readable snapshot of headers we detected
    detected_header = ', '.join(headers)

    def find_col(base_name):
        # Look for headers like "chi_iso (cm^3 mol^-1)" or "chi_iso (Å^3)"
        for h in headers:
            if h is None:
                continue
            hs = h.strip()
            if hs.lower().startswith(base_name) and '(' in hs and hs.endswith(')'):
                unit = hs[hs.rfind('(')+1:-1]
                return hs, unit
        # Try exact base name (unitless)
        for h in headers:
            if h and h.strip().lower() == base_name:
                return h.strip(), ''
        raise KeyError(f"Column for '{base_name}' not found. Headers: {headers}")

    def find_optional_col(base_name):
        for h in headers:
            if h is None:
                continue
            hs = h.strip()
            if hs.lower().startswith(base_name) and '(' in hs and hs.endswith(')'):
                unit = hs[hs.rfind('(')+1:-1]
                return hs, unit
        for h in headers:
            if h and h.strip().lower() == base_name:
                return h.strip(), ''
        return None, ''

    # Temperature header tolerance:
    # accept "Temperature (K)", "Temperature(K)", or any header starting with "Temperature"
    temp_col = None
    for h in headers:
        if not h:
            continue
        hs = h.strip()
        hsl = hs.lower().replace(' ', '')  # remove spaces to allow Temperature(K)
        if hsl == 'temperature(k)' or hs.lower().startswith('temperature'):
            temp_col = hs
            break
    if temp_col is None:
        raise KeyError(f"Column 'Temperature (K)' not found. Detected headers: {detected_header}")

    iso_col, iso_unit = find_col('chi_iso')
    ax_col,  ax_unit  = find_col('chi_ax')
    rho_col, rho_unit = find_col('chi_rho')
    iso_sdev_col, iso_sdev_unit = find_optional_col('chi_iso-s-dev')
    ax_sdev_col, ax_sdev_unit = find_optional_col('chi_ax-s-dev')
    rho_sdev_col, rho_sdev_unit = find_optional_col('chi_rho-s-dev')

    rows = []
    for row in reader:
        if not row:
            continue
        # Skip rows where temperature cell is empty or non-numeric
        t_val = str(row.get(temp_col, '')).strip()
        try:
            T = float(t_val)
        except ValueError:
            continue
        try:
            # Determine a multiplier to convert input units to cm^3 mol^-1
            unit_iso = (iso_unit or '').strip()  # header unit for chi_iso
            # This tool only supports two exact units coming from the paired program:
            #   'cm^3 mol^-1'  (already molar)
            #   'Å^3'          (Angstrom^3 per particle)
            # For 'Å^3', convert to cm^3·mol^-1 using: 1 Å^3 = 1e-24 cm^3, per-particle → per-mol: × N_A,
            # and retain the legacy normalization factor 1/(4π) as required by this workflow.
            if unit_iso == 'cm^3 mol^-1':
                to_cm3_per_mol = 1.0
            elif unit_iso == 'Å^3':
                to_cm3_per_mol = 1e-24 * Avogadro / (4 * np.pi)
            else:
                logging.warning(f"Unrecognized susceptibility unit '{unit_iso}'. Assuming cm^3 mol^-1.")
                to_cm3_per_mol = 1.0

            chi_iso_val = float(str(row.get(iso_col, '')).strip()) * to_cm3_per_mol
            chi_ax_val  = float(str(row.get(ax_col,  '')).strip()) * to_cm3_per_mol
            chi_rho_val = float(str(row.get(rho_col, '')).strip()) * to_cm3_per_mol
        except ValueError:
            continue

        # Scale susceptibilities to desired units and multiply by temperature T from CSV:
        # Convert per-mole to per-particle (divide by Avogadro), multiply by T (to get chi*T),
        # then apply SI scaling factor: 1e-6 * 4 * pi
        conv = 1e-6 * 4 * math.pi
        chi_iso_fit_csv = ((chi_iso_val * T / Avogadro) * conv)
        chi_ax_fit_csv  = (chi_ax_val  * T / Avogadro) * conv
        chi_rho_fit_csv = (chi_rho_val * T / Avogadro) * conv

        chi_iso_sdev_val = None
        if iso_sdev_col is not None:
            try:
                sdev_raw = float(str(row.get(iso_sdev_col, '')).strip()) * to_cm3_per_mol
                # Convert the std dev of chi to std dev of chi*T with the same scaling
                chi_iso_sdev_val = ((sdev_raw * T / Avogadro) * conv)
            except ValueError:
                chi_iso_sdev_val = None

        chi_ax_sdev_val = None
        if ax_sdev_col is not None:
            try:
                sdev_raw_ax = float(str(row.get(ax_sdev_col, '')).strip()) * to_cm3_per_mol
                # Convert the std dev of chi to std dev of chi*T with the same scaling
                chi_ax_sdev_val = ((sdev_raw_ax * T / Avogadro) * conv)
            except ValueError:
                chi_ax_sdev_val = None

        chi_rho_sdev_val = None
        if rho_sdev_col is not None:
            try:
                sdev_raw_rho = float(str(row.get(rho_sdev_col, '')).strip()) * to_cm3_per_mol
                chi_rho_sdev_val = ((sdev_raw_rho * T / Avogadro) * conv)
            except ValueError:
                chi_rho_sdev_val = None

        rows.append({
            'Temperature (K)': T,
            'chi_iso': chi_iso_fit_csv,
            'chi_ax':  chi_ax_fit_csv,
            'chi_rho': chi_rho_fit_csv,
            'chi_iso_sdev': chi_iso_sdev_val,
            'chi_ax_sdev': chi_ax_sdev_val,
            'chi_rho_sdev': chi_rho_sdev_val,
        })

    units = {
        'chi_iso': iso_unit,
        'chi_ax':  ax_unit,
        'chi_rho': rho_unit,
    }
    if iso_sdev_col is not None:
        units['chi_iso_sdev'] = iso_sdev_unit

    if ax_sdev_col is not None:
        units['chi_ax_sdev'] = ax_sdev_unit

    if rho_sdev_col is not None:
        units['chi_rho_sdev'] = rho_sdev_unit

    return rows, units

# Script to parse ORCA output and plot magnetic susceptibility components vs inverse temperature


# --- Data container for all temperature-dependent χ analysis results ---
@dataclass
class ChiTemperatureAnalysis:
    temps: list[float]
    inv_t: list[float]
    temps_csv: list[float]
    inv_t_csv: list[float]
    chi_iso_nevpt2_si: list[float]
    chi_ax_nevpt2_si: list[float]
    chi_rho_nevpt2_si: list[float]
    chi_iso_analytic: list[float]
    chi_ax_analytic: list[float]
    chi_rho_analytic: list[float]
    g_sq_iso: list[float]
    g_sq_ax: list[float]
    g_sq_rh: list[float]
    D_list: list[float]
    E_list: list[float]
    D_J: list[float]
    E_J: list[float]
    chi_iso_fit_csv: list[float]
    chi_ax_fit_csv: list[float]
    chi_rho_fit_csv: list[float]
    chi_iso_sdev_csv: list[float | None]
    chi_ax_sdev_csv: list[float | None]
    chi_rho_sdev_csv: list[float | None]
    chi_iso_fit_pred: list[float] | None
    chi_ax_fit_pred: list[float] | None
    chi_rho_fit_pred: list[float] | None
    chi_iso_sdev_pred: list[float] | None
    chi_ax_sdev_pred: list[float] | None
    chi_rho_sdev_pred: list[float] | None
    chi_iso_fit_pred_TIP: list[float] | None
    chi_ax_fit_pred_TIP: list[float] | None
    chi_iso_sdev_pred_TIP: list[float] | None
    chi_ax_sdev_pred_TIP: list[float] | None
    a_iso: float | None
    b_iso: float | None
    a_ax: float | None
    b_ax: float | None
    a_rho: float | None
    b_rho: float | None
    a_iso_se: float | None
    b_iso_se: float | None
    a_ax_se: float | None
    b_ax_se: float | None
    a_rho_se: float | None
    b_rho_se: float | None
    a_iso_TIP: float | None
    b_iso_TIP: float | None
    a_ax_TIP: float | None
    b_ax_TIP: float | None
    a_iso_TIP_se: float | None
    b_iso_TIP_se: float | None
    a_ax_TIP_se: float | None
    b_ax_TIP_se: float | None


def run_csv_only_pipeline(
    csv_path: str,
    spin_S: float,
    fix_intercept_highT: bool = False,
) -> None:
    """
    CSV-only workflow: read susceptibility data, perform weighted LR fits, estimate
    g and D from the fit, and generate χT vs 1/T plots without any NEVPT2/analytic
    reference curves.

    Parameters
    ----------
    csv_path : str
        Path to the susceptibility CSV/TSV/whitespace file.
    spin_S : float
        Spin quantum number S (required for estimating D).
    fix_intercept_highT : bool
        If True, fix LR intercepts to the chi·T value at the highest CSV temperature.
    """
    t_limit = 248.0

    try:
        csv_rows, _ = read_susceptibility_csv(csv_path)
    except Exception as e:
        msg = f"Error reading susceptibility CSV '{csv_path}': {e}"
        logging.error(msg)
        raise RuntimeError(msg)

    temps_csv: list[float] = []
    chi_iso_fit_csv: list[float] = []
    chi_ax_fit_csv: list[float] = []
    chi_rho_fit_csv: list[float] = []
    chi_iso_sdev_csv: list[float | None] = []
    chi_ax_sdev_csv: list[float | None] = []
    chi_rho_sdev_csv: list[float | None] = []

    for r in csv_rows:
        T = r['Temperature (K)']
        if T is None or T < t_limit:
            continue
        temps_csv.append(T)
        chi_iso_fit_csv.append(r['chi_iso'])
        chi_ax_fit_csv.append(r['chi_ax'])
        chi_rho_fit_csv.append(r['chi_rho'])
        chi_iso_sdev_csv.append(r.get('chi_iso_sdev'))
        chi_ax_sdev_csv.append(r.get('chi_ax_sdev'))
        chi_rho_sdev_csv.append(r.get('chi_rho_sdev'))

    if len(temps_csv) < 2:
        msg = "Not enough CSV data points (after temperature filtering) for linear regression."
        logging.error(msg)
        raise RuntimeError(msg)

    inv_t_csv = [1.0 / T for T in temps_csv]

    # Normalization factor using the supplied spin
    mu_B = physical_constants['Bohr magneton'][0]
    S = float(spin_S)
    norm_factor = (mu_0 * mu_B**2 * S * (S + 1)) / (3 * k)

    for i in range(len(chi_iso_fit_csv)):
        chi_iso_fit_csv[i] /= norm_factor
        chi_ax_fit_csv[i] /= norm_factor
        chi_rho_fit_csv[i] /= norm_factor

    for i in range(len(chi_iso_sdev_csv)):
        if chi_iso_sdev_csv[i] is not None:
            chi_iso_sdev_csv[i] /= norm_factor
    for i in range(len(chi_ax_sdev_csv)):
        if chi_ax_sdev_csv[i] is not None:
            chi_ax_sdev_csv[i] /= norm_factor
    for i in range(len(chi_rho_sdev_csv)):
        if chi_rho_sdev_csv[i] is not None:
            chi_rho_sdev_csv[i] /= norm_factor

    fixed_b_iso = fixed_b_ax = fixed_b_rho = None
    if fix_intercept_highT and len(temps_csv) >= 1:
        idx_maxT = max(range(len(temps_csv)), key=lambda i: temps_csv[i])
        fixed_b_iso = chi_iso_fit_csv[idx_maxT]
        fixed_b_ax = chi_ax_fit_csv[idx_maxT]
        fixed_b_rho = chi_rho_fit_csv[idx_maxT]
        logging.debug(
            "CSV-only fixed intercepts at T_max=%.2f K: b_iso=%.6g, b_ax=%.6g, b_rho=%.6g",
            temps_csv[idx_maxT],
            fixed_b_iso,
            fixed_b_ax,
            fixed_b_rho,
        )

    chi_iso_fit_pred, a_iso, b_iso, chi_iso_sdev_pred, a_iso_se, b_iso_se = weighted_linreg_predict(
        inv_t_csv,
        chi_iso_fit_csv,
        inv_t_csv,
        sigma=chi_iso_sdev_csv,
        fixed_intercept=fixed_b_iso,
    )
    chi_ax_fit_pred, a_ax, b_ax, chi_ax_sdev_pred, a_ax_se, b_ax_se = weighted_linreg_predict(
        inv_t_csv,
        chi_ax_fit_csv,
        inv_t_csv,
        sigma=chi_ax_sdev_csv,
        fixed_intercept=fixed_b_ax,
    )
    chi_rho_fit_pred, a_rho, b_rho, chi_rho_sdev_pred, a_rho_se, b_rho_se = weighted_linreg_predict(
        inv_t_csv,
        chi_rho_fit_csv,
        inv_t_csv,
        sigma=chi_rho_sdev_csv,
        fixed_intercept=fixed_b_rho,
    )

    # Estimate g and D (and optionally E) from LR intercepts/slopes.
    if (a_ax is not None) and (b_ax is not None) and (b_iso is not None):
        try:
            # Decide whether rhombicity is effectively zero.
            use_axial_only = (b_rho is None) or (abs(b_rho) < 1e-6) or (a_rho is None)

            if use_axial_only:
                gx, gy, gz = solve_g_principals_from_LR(
                    b_iso=b_iso,
                    b_ax=b_ax,
                    b_rh=None,
                )
                g_sq_iso_analytic_val = calculate_g_sq_iso_analytic(
                    b_iso_intercept=b_iso,
                    b_ax_intercept=b_ax,
                    K=1.0,
                )
                D_from_LR = compute_D(
                    S=S,
                    b_ax=b_ax,
                    g_sq_iso_analytic=g_sq_iso_analytic_val,
                    slope_ax=a_ax,
                )
                ut.cprint(
                    f"CSV-only fit (axial-only): gx = {gx:.4f}, gy = {gy:.4f}, gz = {gz:.4f}, "
                    f"D = {D_from_LR:.4f} cm^-1",
                    'cyan',
                )
            else:
                gx, gy, gz = solve_g_principals_from_LR(
                    b_iso=b_iso,
                    b_ax=b_ax,
                    b_rh=b_rho,
                )
                D_J_LR, E_J_LR = solve_D_E_from_LR(
                    a_ax=a_ax,
                    a_rh=a_rho,
                    S=S,
                    gx=gx,
                    gy=gy,
                    gz=gz,
                )
                D_from_LR = None if D_J_LR is None else D_J_LR / (h * c * 100)
                E_from_LR = None if E_J_LR is None else E_J_LR / (h * c * 100)

                if (D_from_LR is not None) and (E_from_LR is not None):
                    ut.cprint(
                        f"CSV-only fit (full rhombic): gx = {gx:.4f}, gy = {gy:.4f}, gz = {gz:.4f}, "
                        f"D = {D_from_LR:.4f} cm^-1, E = {E_from_LR:.4f} cm^-1",
                        'cyan',
                    )
                else:
                    ut.cprint(
                        f"CSV-only fit (full rhombic, D/E solve incomplete): "
                        f"gx = {gx:.4f}, gy = {gy:.4f}, gz = {gz:.4f}",
                        'cyan',
                    )
        except Exception as e:
            logging.warning(f"CSV-only g/D/E estimation failed: {e}")

    # Generate per-component plots (CSV-only)
    _plot_component_csv_only(
        'blue',
        inv_t_csv,
        chi_iso_fit_csv,
        chi_iso_sdev_csv,
        chi_iso_fit_pred,
        chi_iso_sdev_pred,
        a_iso,
        b_iso,
        a_iso_se,
        b_iso_se,
        'iso',
        'chi_plot_iso.png',
    )
    _plot_component_csv_only(
        'green',
        inv_t_csv,
        chi_ax_fit_csv,
        chi_ax_sdev_csv,
        chi_ax_fit_pred,
        chi_ax_sdev_pred,
        a_ax,
        b_ax,
        a_ax_se,
        b_ax_se,
        'ax',
        'chi_plot_ax.png',
    )
    _plot_component_csv_only(
        'red',
        inv_t_csv,
        chi_rho_fit_csv,
        chi_rho_sdev_csv,
        chi_rho_fit_pred,
        chi_rho_sdev_pred,
        a_rho,
        b_rho,
        a_rho_se,
        b_rho_se,
        'rho',
        'chi_plot_rho.png',
    )

    # Combined CSV-only plot
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    ax.plot(
        inv_t_csv,
        chi_iso_fit_csv,
        label=rf'$\frac{{g_e}}{{3}}\,\mathrm{{Tr}}(\frac{{\chi}}{{g}})$ {SHORT_LABELS["Fitted"]}',
        color='blue',
        marker='o',
        linestyle='',
        markersize=5,
    )
    ax.plot(inv_t_csv, chi_ax_fit_csv, label=rf'$\chi_{{ax}}$ {SHORT_LABELS["Fitted"]}', color='green', marker='o', linestyle='', markersize=5)
    ax.plot(inv_t_csv, chi_rho_fit_csv, label=rf'$\chi_{{rho}}$ {SHORT_LABELS["Fitted"]}', color='red', marker='o', linestyle='', markersize=5)

    if chi_iso_fit_pred is not None:
        ax.plot(
            inv_t_csv,
            chi_iso_fit_pred,
            label=rf'$\frac{{g_e}}{{3}}\,\mathrm{{Tr}}(\frac{{\chi}}{{g}})$ {SHORT_LABELS["Fitted (LR)"]}',
            linestyle='-.',
            linewidth=1.5,
            color='blue',
        )
    if chi_ax_fit_pred is not None:
        ax.plot(inv_t_csv, chi_ax_fit_pred, label=rf'$\chi_{{ax}}$ {SHORT_LABELS["Fitted (LR)"]}', linestyle='-.', linewidth=1.5, color='green')
    if chi_rho_fit_pred is not None:
        ax.plot(inv_t_csv, chi_rho_fit_pred, label=rf'$\chi_{{rho}}$ {SHORT_LABELS["Fitted (LR)"]}', linestyle='-.', linewidth=1.5, color='red')

    # Error bars on combined plot (optional)
    if len(chi_iso_sdev_csv) == len(inv_t_csv) and any(v is not None for v in chi_iso_sdev_csv):
        yerr = [v if (v is not None) else 0.0 for v in chi_iso_sdev_csv]
        ax.errorbar(inv_t_csv, chi_iso_fit_csv, yerr=yerr, fmt='none', ecolor='blue', alpha=0.5, capsize=2)
    if len(chi_ax_sdev_csv) == len(inv_t_csv) and any(v is not None for v in chi_ax_sdev_csv):
        yerr = [v if (v is not None) else 0.0 for v in chi_ax_sdev_csv]
        ax.errorbar(inv_t_csv, chi_ax_fit_csv, yerr=yerr, fmt='none', ecolor='green', alpha=0.5, capsize=2)
    if len(chi_rho_sdev_csv) == len(inv_t_csv) and any(v is not None for v in chi_rho_sdev_csv):
        yerr = [v if (v is not None) else 0.0 for v in chi_rho_sdev_csv]
        ax.errorbar(inv_t_csv, chi_rho_fit_csv, yerr=yerr, fmt='none', ecolor='red', alpha=0.5, capsize=2)

    _finalize_axes(ax, inv_t_csv, inv_t_csv, 'All Components (CSV-only)')

    plt.savefig('chi_plot_all.png', dpi=600)
    plt.show()
    plt.close('all')


def compute_chi_temperature_dependence(
    file_name: str,
    section: str,
    csv_path: str | None = None,
    fix_intercept_highT: bool = False,
) -> ChiTemperatureAnalysis:
    """
    Run the χ(T) pipeline for a given ORCA output and (optional) CSV and return
    all intermediate series as a ChiTemperatureAnalysis instance.

    This function performs file I/O (reading ORCA/CSV) and numerical computations
    only. It does not create any plots or write result tables to disk.
    """

    # High‑level pipeline layout:
    #   1) Read χ(T), spin S, g and Effective Hamiltonian from ORCA.
    #   2) Optionally read and preprocess experimental CSV χ(T) data.
    #   3) Normalize all χ·T series by the Curie‑like factor.
    #   4) Rotate g and Effective Hamiltonian into χ eigenframes and build g, D, E.
    #   5) Build analytic χ·T components from g, D, E.
    #   6) If CSV is present, do weighted LR (with optional fixed intercepts and TIP correction).
    #   7) From LR results, recover g‑principals and D/E (TIP and no‑TIP variants).
    #   8) Propagate experimental uncertainties σ(χ) to σ(g) and σ(D/E).
    #   9) Pack everything into ChiTemperatureAnalysis.
    # === 1) Read χ(T), spin, g and Effective Hamiltonian from ORCA ===

    # Set minimal temperature considered
    t_limit = 248

    # Read susceptibility tensors via the readers module helper
    try:
        tensors = rdrs.read_orca_susceptibility(file_name, section)
    except Exception as e:
        msg = f"Error reading susceptibilities from '{file_name}' for section '{section}': {e}"
        logging.error(msg)
        raise RuntimeError(msg)

    if not tensors:
        msg = f"No susceptibility data found for section '{section}' in file '{file_name}'"
        logging.error(msg)
        raise RuntimeError(msg)

    # Sort and filter temperatures according to t_limit
    temps = sorted(tensors.keys())
    temps = [T for T in temps if T >= t_limit]
    tensors = {T: tensors[T] for T in temps}

    if not temps:
        msg = f"No susceptibility temperatures >= {t_limit} K for section '{section}' in file '{file_name}'"
        logging.error(msg)
        raise RuntimeError(msg)

    # Extract g-matrix, spin S, and Effective Hamiltonian matrix from the same QDPT section
    try:
        S = rdrs.read_orca_spin(file_name)
        g_matrix = rdrs.read_orca_g_tensor(file_name, section)
    except Exception as e:
        msg = f"Error reading spin/g-tensor from '{file_name}' for section '{section}': {e}"
        logging.error(msg)
        raise RuntimeError(msg)

    try:
        eff_H_raw = rdrs.read_eff_hamiltonian_tensor(file_name, section)

    except (IOError, StopIteration) as e:
        msg = f"Error reading Effective Hamiltonian from '{file_name}': {e}"
        logging.error(msg)
        raise RuntimeError(msg)

    if g_matrix is None or S is None or eff_H_raw is None:
        msg = (
            f"Failed to locate spin, g-matrix or Effective Hamiltonian for section '{section}' "
            f"in file '{file_name}'"
        )
        logging.error(msg)
        raise RuntimeError(msg)

    chi_iso_nevpt2_si, chi_ax_nevpt2_si, chi_rho_nevpt2_si, chi_eigenvectors = calculate_chi_components_nevpt2(
        tensors, g_matrix
    )

    # === 2) Optionally read and preprocess experimental χ(T) CSV data ===
    chi_iso_fit_csv: list[float] = []
    chi_ax_fit_csv: list[float] = []
    chi_rho_fit_csv: list[float] = []
    chi_iso_sdev_csv: list[float | None] = []
    chi_ax_sdev_csv: list[float | None] = []
    chi_rho_sdev_csv: list[float | None] = []
    temps_csv: list[float] = []

    if csv_path is not None:
        try:
            csv_rows, _ = read_susceptibility_csv(csv_path)
            # keep rows that satisfy T >= t_limit to match tensors filtering
            for r in csv_rows:
                if r['Temperature (K)'] >= t_limit:
                    temps_csv.append(r['Temperature (K)'])
                    chi_iso_fit_csv.append(r['chi_iso'])
                    chi_ax_fit_csv.append(r['chi_ax'])
                    chi_rho_fit_csv.append(r['chi_rho'])
                    chi_iso_sdev_csv.append(r.get('chi_iso_sdev'))
                    chi_ax_sdev_csv.append(r.get('chi_ax_sdev'))
                    chi_rho_sdev_csv.append(r.get('chi_rho_sdev'))
        except Exception as e:
            logging.error(
                f"There is an error in processing CSV '{csv_path}' in plot_chi_temperature_dependence: {e}"
            )

    # === 3) Normalize χ·T values by the Curie‑like factor ===
    # Upload constants
    mu_B = physical_constants['Bohr magneton'][0]
    f_S = (2 * S - 1) * (2 * S + 3)

    # Compute normalization factor: mu0 * mu_B**2 * S(S+1) / (3k)
    norm_factor = (mu_0 * mu_B**2 * S * (S + 1)) / (3 * k)

    for i in range(len(temps)):
        chi_iso_nevpt2_si[i] /= norm_factor
        chi_ax_nevpt2_si[i] /= norm_factor
        chi_rho_nevpt2_si[i] /= norm_factor

    # Normalize CSV-derived chi*T values if provided
    for i in range(len(chi_iso_fit_csv)):
        chi_iso_fit_csv[i] /= norm_factor
        chi_ax_fit_csv[i] /= norm_factor
        chi_rho_fit_csv[i] /= norm_factor

    for i in range(len(chi_iso_sdev_csv)):
        if chi_iso_sdev_csv[i] is not None:
            chi_iso_sdev_csv[i] /= norm_factor
    for i in range(len(chi_ax_sdev_csv)):
        if chi_ax_sdev_csv[i] is not None:
            chi_ax_sdev_csv[i] /= norm_factor
    for i in range(len(chi_rho_sdev_csv)):
        if chi_rho_sdev_csv[i] is not None:
            chi_rho_sdev_csv[i] /= norm_factor

    # Build χ(T) from CSV χ·T values (still in the same normalized units)
    chi_iso_fit_csv_chi = [val / T for val, T in zip(chi_iso_fit_csv, temps_csv)]
    chi_ax_fit_csv_chi = [val / T for val, T in zip(chi_ax_fit_csv, temps_csv)]

    # === 4) Rotate tensors into χ eigenframes and build g, D, E components ===
    # Rotate g-tensors into each χ eigenframe
    rotated_g_tensors = rotate_tensor_to_chi_basis(g_matrix, chi_eigenvectors, temps)

    # Now compute g-components for each rotated tensor
    g_sq_iso, g_sq_ax, g_sq_rh = calculate_g_components(rotated_g_tensors)

    # Rotate D-tensors into each χ eigenframe
    rotated_eff_H_tensors = rotate_tensor_to_chi_basis(eff_H_raw, chi_eigenvectors, temps)

    # Now compute D and E components for each rotated tensor
    D_list, E_list = calculate_E_D_components(rotated_eff_H_tensors)

    # Convert to joules for each temperature
    D_J = [d * h * c * 100 for d in D_list]
    E_J = [e * h * c * 100 for e in E_list]

    inv_t = [1.0 / T for T in temps]
    # Build a separate inverse-temperature axis for CSV data (lengths may differ from theory)
    inv_t_csv = [1.0 / T for T in temps_csv] if len(temps_csv) > 0 else []

    # === 5) Build analytic χ·T components (iso, ax, rho) from g, D, E ===
    chi_iso_analytic = [
        g_sq_iso[i]
        - (f_S / (45 * k * temps[i])) * (D_J[i] * g_sq_ax[i] + 3 * E_J[i] * g_sq_rh[i])
        for i in range(len(temps))
    ]

    chi_ax_analytic = [
        g_sq_ax[i]
        - (f_S / (30 * k * temps[i])) * ((D_J[i]) * (g_sq_ax[i] + 3 * g_sq_iso[i]) - 3 * E_J[i] * g_sq_rh[i])
        for i in range(len(temps))
    ]

    chi_rho_analytic = [
        g_sq_rh[i]
        + (f_S / (30 * k * temps[i])) * (E_J[i] * (g_sq_ax[i] - 3 * g_sq_iso[i]) + D_J[i] * g_sq_rh[i])
        for i in range(len(temps))
    ]

    # Build χ(T) series (from χ·T) for NEVPT2 and analytic components
    chi_iso_nevpt2_chi = [val / T for val, T in zip(chi_iso_nevpt2_si, temps)]
    chi_ax_nevpt2_chi = [val / T for val, T in zip(chi_ax_nevpt2_si, temps)]
    chi_iso_analytic_chi = [val / T for val, T in zip(chi_iso_analytic, temps)]
    chi_ax_analytic_chi = [val / T for val, T in zip(chi_ax_analytic, temps)]

    # === 6) Compute TIP offsets at highest NEVPT2 temperature (in χ(T)) ===
    chi_ax_red_TIP = 0.0
    chi_iso_red_TIP = 0.0
    if temps:
        try:
            t_ref = max(temps)
            idx_ref = temps.index(t_ref)
            # Define TIP as the difference NEVPT2 - analytic at the level of χ(T), not χ·T
            chi_ax_red_TIP = chi_ax_nevpt2_chi[idx_ref] - chi_ax_analytic_chi[idx_ref]
            chi_iso_red_TIP = chi_iso_nevpt2_chi[idx_ref] - chi_iso_analytic_chi[idx_ref]
            ut.cprint(
                f"TIP offsets at {t_ref:.1f} K (in χ): chi_ax_red_TIP = {chi_ax_red_TIP:.6g}, chi_iso_red_TIP = {chi_iso_red_TIP:.6g}",
                'yellow',
            )
        except Exception:
            pass

    # === 7) If CSV is present, perform weighted LR (with optional fixed intercepts and TIP correction) ===
    chi_iso_fit_pred = None
    chi_ax_fit_pred = None
    chi_rho_fit_pred = None
    chi_iso_sdev_pred = None
    chi_ax_sdev_pred = None
    chi_rho_sdev_pred = None
    a_iso = b_iso = a_ax = b_ax = a_rho = b_rho = None
    a_iso_se = b_iso_se = a_ax_se = b_ax_se = a_rho_se = b_rho_se = None

    # TIP-corrected regression outputs (iso and ax only)
    chi_iso_fit_pred_TIP = None
    chi_ax_fit_pred_TIP = None
    chi_iso_sdev_pred_TIP = None
    chi_ax_sdev_pred_TIP = None
    a_iso_TIP = b_iso_TIP = a_ax_TIP = b_ax_TIP = None
    a_iso_TIP_se = b_iso_TIP_se = a_ax_TIP_se = b_ax_TIP_se = None

    if csv_path is not None and len(inv_t_csv) >= 2:
        fixed_b_iso = fixed_b_ax = fixed_b_rho = None
        fixed_b_iso_TIP = fixed_b_ax_TIP = None
        if fix_intercept_highT and len(temps_csv) >= 1:
            idx_maxT_csv = max(range(len(temps_csv)), key=lambda i: temps_csv[i])
            fixed_b_iso = chi_iso_fit_csv[idx_maxT_csv]
            fixed_b_ax = chi_ax_fit_csv[idx_maxT_csv]
            fixed_b_rho = chi_rho_fit_csv[idx_maxT_csv]
            logging.debug(
                "Fixed intercepts at T_max=%.2f K: b_iso=%.6g, b_ax=%.6g, b_rho=%.6g",
                temps_csv[idx_maxT_csv],
                fixed_b_iso,
                fixed_b_ax,
                fixed_b_rho,
            )
            # TIP-corrected intercepts at the highest CSV temperature
            fixed_b_iso_TIP = fixed_b_iso - chi_iso_red_TIP
            fixed_b_ax_TIP = fixed_b_ax - chi_ax_red_TIP

        chi_iso_fit_pred, a_iso, b_iso, chi_iso_sdev_pred, a_iso_se, b_iso_se = weighted_linreg_predict(
            inv_t_csv,
            chi_iso_fit_csv,
            inv_t,
            sigma=chi_iso_sdev_csv,
            fixed_intercept=fixed_b_iso,
        )
        chi_ax_fit_pred, a_ax, b_ax, chi_ax_sdev_pred, a_ax_se, b_ax_se = weighted_linreg_predict(
            inv_t_csv,
            chi_ax_fit_csv,
            inv_t,
            sigma=chi_ax_sdev_csv,
            fixed_intercept=fixed_b_ax,
        )
        chi_rho_fit_pred, a_rho, b_rho, chi_rho_sdev_pred, a_rho_se, b_rho_se = weighted_linreg_predict(
            inv_t_csv,
            chi_rho_fit_csv,
            inv_t,
            sigma=chi_rho_sdev_csv,
            fixed_intercept=fixed_b_rho,
        )
        
        # --- TIP-corrected CSV series and linear regression (iso and ax only) ---
        try:
            chi_iso_fit_csv_TIPcorr = [
                (chi_val - chi_iso_red_TIP) * T_val
                for chi_val, T_val in zip(chi_iso_fit_csv_chi, temps_csv)
            ]
            chi_ax_fit_csv_TIPcorr = [
                (chi_val - chi_ax_red_TIP) * T_val
                for chi_val, T_val in zip(chi_ax_fit_csv_chi, temps_csv)
            ]
        except Exception as e:
            logging.warning(f"Failed to build TIP-corrected CSV series: {e}")
            chi_iso_fit_csv_TIPcorr = None
            chi_ax_fit_csv_TIPcorr = None

        if chi_iso_fit_csv_TIPcorr is not None and chi_ax_fit_csv_TIPcorr is not None:
            (
                chi_iso_fit_pred_TIP,
                a_iso_TIP,
                b_iso_TIP,
                chi_iso_sdev_pred_TIP,
                a_iso_TIP_se,
                b_iso_TIP_se,
            ) = weighted_linreg_predict(
                inv_t_csv,
                chi_iso_fit_csv_TIPcorr,
                inv_t,
                sigma=chi_iso_sdev_csv if len(chi_iso_sdev_csv) == len(inv_t_csv) else None,
                fixed_intercept=fixed_b_iso_TIP,
            )
            (
                chi_ax_fit_pred_TIP,
                a_ax_TIP,
                b_ax_TIP,
                chi_ax_sdev_pred_TIP,
                a_ax_TIP_se,
                b_ax_TIP_se,
            ) = weighted_linreg_predict(
                inv_t_csv,
                chi_ax_fit_csv_TIPcorr,
                inv_t,
                sigma=chi_ax_sdev_csv if len(chi_ax_sdev_csv) == len(inv_t_csv) else None,
                fixed_intercept=fixed_b_ax_TIP,
            )
    else:
        ut.cprint('Not enough CSV points for linear regression (need at least 2).', 'cyan')

    # === 8) Estimate g‑principals and (D, E) from LR results, with and without TIP ===
    D_from_LR = None
    E_from_LR = None
    g_sq_iso_analytic_val = None

    D_from_LR_TIP = None
    E_from_LR_TIP = None
    g_sq_iso_analytic_TIP_val = None

    gx_noTIP = gy_noTIP = gz_noTIP = None
    gx_TIP = gy_TIP = gz_TIP = None

    # No-TIP case: estimate gx, gy, gz and D (and optionally E) from LR.
    if (a_ax is not None) and (b_ax is not None) and (b_iso is not None):
        try:
            use_axial_only = (b_rho is None) or (abs(b_rho) < 1e-6) or (a_rho is None)

            if use_axial_only:
                gx_noTIP, gy_noTIP, gz_noTIP = solve_g_principals_from_LR(
                    b_iso=b_iso,
                    b_ax=b_ax,
                    b_rh=None,
                )
                g_sq_iso_analytic_val = calculate_g_sq_iso_analytic(
                    b_iso_intercept=b_iso,
                    b_ax_intercept=b_ax,
                    K=1.0,
                )
                D_from_LR = compute_D(
                    S=S,
                    b_ax=b_ax,
                    g_sq_iso_analytic=g_sq_iso_analytic_val,
                    slope_ax=a_ax,
                )
                E_from_LR = 0.0
            else:
                gx_noTIP, gy_noTIP, gz_noTIP = solve_g_principals_from_LR(
                    b_iso=b_iso,
                    b_ax=b_ax,
                    b_rh=b_rho,
                )
                D_J_LR, E_J_LR = solve_D_E_from_LR(
                    a_ax=a_ax,
                    a_rh=a_rho,
                    S=S,
                    gx=gx_noTIP,
                    gy=gy_noTIP,
                    gz=gz_noTIP,
                )
                D_from_LR = None if D_J_LR is None else D_J_LR / (h * c * 100)
                E_from_LR = None if E_J_LR is None else E_J_LR / (h * c * 100)
        except Exception as e:
            logging.warning(f"compute_D/E invocation failed (no TIP): {e}")

    # TIP-corrected case: estimate gx, gy, gz and D (and optionally E) from LR using TIP-corrected iso/ax.
    if (a_ax_TIP is not None) and (b_ax_TIP is not None) and (b_iso_TIP is not None):
        try:
            use_axial_only_TIP = (b_rho is None) or (abs(b_rho) < 1e-6) or (a_rho is None)

            if use_axial_only_TIP:
                # Axial-only approximation (gx = gy, E ≈ 0) with TIP-corrected iso/ax
                gx_TIP, gy_TIP, gz_TIP = solve_g_principals_from_LR(
                    b_iso=b_iso_TIP,
                    b_ax=b_ax_TIP,
                    b_rh=None,
                )
                g_sq_iso_analytic_TIP_val = calculate_g_sq_iso_analytic(
                    b_iso_intercept=b_iso_TIP,
                    b_ax_intercept=b_ax_TIP,
                    K=1.0,
                )
                D_from_LR_TIP = compute_D(
                    S=S,
                    b_ax=b_ax_TIP,
                    g_sq_iso_analytic=g_sq_iso_analytic_TIP_val,
                    slope_ax=a_ax_TIP,
                )
                E_from_LR_TIP = 0.0
            else:
                # Full rhombic branch with TIP-corrected iso/ax and original rhombic fit (TIP assumed isotropic)
                gx_TIP, gy_TIP, gz_TIP = solve_g_principals_from_LR(
                    b_iso=b_iso_TIP,
                    b_ax=b_ax_TIP,
                    b_rh=b_rho,
                )
                D_J_TIP, E_J_TIP = solve_D_E_from_LR(
                    a_ax=a_ax_TIP,
                    a_rh=a_rho,
                    S=S,
                    gx=gx_TIP,
                    gy=gy_TIP,
                    gz=gz_TIP,
                )
                D_from_LR_TIP = None if D_J_TIP is None else D_J_TIP / (h * c * 100)
                E_from_LR_TIP = None if E_J_TIP is None else E_J_TIP / (h * c * 100)
        except Exception as e:
            logging.warning(f"compute_D/E invocation failed (TIP): {e}")

    # Print compact comparison: gx, gz and D with and without TIP
    try:
        if (gx_noTIP is not None) and (gz_noTIP is not None) and (D_from_LR is not None):
            if E_from_LR is not None:
                ut.cprint(
                    f"no TIP: gx = {gx_noTIP:.4f}, gy = {gy_noTIP:.4f}, gz = {gz_noTIP:.4f}, "
                    f"D = {D_from_LR:.4f} cm^-1, E = {E_from_LR:.4f} cm^-1",
                    'cyan',
                )
            else:
                ut.cprint(
                    f"no TIP: gx = {gx_noTIP:.4f}, gy = {gy_noTIP:.4f}, gz = {gz_noTIP:.4f}, "
                    f"D = {D_from_LR:.4f} cm^-1",
                    'cyan',
                )
        if (gx_TIP is not None) and (gz_TIP is not None) and (D_from_LR_TIP is not None):
            if E_from_LR_TIP is not None:
                ut.cprint(
                    f"TIP:    gx = {gx_TIP:.4f}, gy = {gy_TIP:.4f}, gz = {gz_TIP:.4f}, "
                    f"D = {D_from_LR_TIP:.4f} cm^-1, E = {E_from_LR_TIP:.4f} cm^-1",
                    'cyan',
                )
            else:
                ut.cprint(
                    f"TIP:    gx = {gx_TIP:.4f}, gy = {gy_TIP:.4f}, gz = {gz_TIP:.4f}, "
                    f"D = {D_from_LR_TIP:.4f} cm^-1",
                    'cyan',
                )
    except Exception as e:
        logging.warning(f"Printing TIP/no-TIP comparison failed: {e}")

    # === 9) Propagate experimental χ uncertainties σ to σ(g) and σ(D/E) (no‑TIP branch) ===
    gx_err = gy_err = gz_err = D_err = E_err = None
    try:
        # Only attempt error propagation if we have CSV data with at least some std devs
        if (
            csv_path is not None
            and len(inv_t_csv) >= 2
            and (
                any(v is not None for v in chi_iso_sdev_csv)
                or any(v is not None for v in chi_ax_sdev_csv)
                or any(v is not None for v in chi_rho_sdev_csv)
            )
        ):
            # Build perturbed χ·T series: central ± σ (per-point experimental errors)
            chi_iso_minus = []
            chi_iso_plus = []
            chi_ax_minus = []
            chi_ax_plus = []
            chi_rho_minus = []
            chi_rho_plus = []

            for iso, ax, rho, s_iso, s_ax, s_rho in zip(
                chi_iso_fit_csv,
                chi_ax_fit_csv,
                chi_rho_fit_csv,
                chi_iso_sdev_csv,
                chi_ax_sdev_csv,
                chi_rho_sdev_csv,
            ):
                s_iso_val = 0.0 if s_iso is None else float(s_iso)
                s_ax_val = 0.0 if s_ax is None else float(s_ax)
                s_rho_val = 0.0 if s_rho is None else float(s_rho)

                chi_iso_minus.append(iso - s_iso_val)
                chi_iso_plus.append(iso + s_iso_val)

                chi_ax_minus.append(ax - s_ax_val)
                chi_ax_plus.append(ax + s_ax_val)

                chi_rho_minus.append(rho - s_rho_val)
                chi_rho_plus.append(rho + s_rho_val)

            # Helper to run weighted LR for one branch (minus or plus)
            def _lr_branch(chi_iso_series, chi_ax_series, chi_rho_series, use_fixed: bool):
                fixed_b_iso_loc = fixed_b_ax_loc = fixed_b_rho_loc = None
                if use_fixed and len(temps_csv) >= 1:
                    idx_maxT = max(range(len(temps_csv)), key=lambda i: temps_csv[i])
                    fixed_b_iso_loc = chi_iso_series[idx_maxT]
                    fixed_b_ax_loc = chi_ax_series[idx_maxT]
                    fixed_b_rho_loc = chi_rho_series[idx_maxT]

                _, a_iso_loc, b_iso_loc, _, _, _ = weighted_linreg_predict(
                    inv_t_csv,
                    chi_iso_series,
                    inv_t,
                    sigma=chi_iso_sdev_csv,
                    fixed_intercept=fixed_b_iso_loc,
                )
                _, a_ax_loc, b_ax_loc, _, _, _ = weighted_linreg_predict(
                    inv_t_csv,
                    chi_ax_series,
                    inv_t,
                    sigma=chi_ax_sdev_csv,
                    fixed_intercept=fixed_b_ax_loc,
                )
                _, a_rho_loc, b_rho_loc, _, _, _ = weighted_linreg_predict(
                    inv_t_csv,
                    chi_rho_series,
                    inv_t,
                    sigma=chi_rho_sdev_csv,
                    fixed_intercept=fixed_b_rho_loc,
                )
                return a_ax_loc, b_ax_loc, a_rho_loc, b_rho_loc, b_iso_loc

            use_fixed_intercept = bool(fix_intercept_highT and len(temps_csv) >= 1)

            # LR for χ - σ
            (
                a_ax_minus,
                b_ax_minus,
                a_rho_minus,
                b_rho_minus,
                b_iso_minus,
            ) = _lr_branch(chi_iso_minus, chi_ax_minus, chi_rho_minus, use_fixed_intercept)

            # LR for χ + σ
            (
                a_ax_plus,
                b_ax_plus,
                a_rho_plus,
                b_rho_plus,
                b_iso_plus,
            ) = _lr_branch(chi_iso_plus, chi_ax_plus, chi_rho_plus, use_fixed_intercept)

            # Helper to go from LR coefficients to (gx, gy, gz, D, E) for a given branch
            def _params_from_LR(a_ax_loc, b_ax_loc, a_rho_loc, b_rho_loc, b_iso_loc):
                if (a_ax_loc is None) or (b_ax_loc is None) or (b_iso_loc is None):
                    return None, None, None, None, None

                use_axial_only_loc = (b_rho_loc is None) or (abs(b_rho_loc) < 1e-6) or (a_rho_loc is None)

                if use_axial_only_loc:
                    gx_loc, gy_loc, gz_loc = solve_g_principals_from_LR(
                        b_iso=b_iso_loc,
                        b_ax=b_ax_loc,
                        b_rh=None,
                    )
                    g_sq_iso_loc = calculate_g_sq_iso_analytic(
                        b_iso_intercept=b_iso_loc,
                        b_ax_intercept=b_ax_loc,
                        K=1.0,
                    )
                    D_loc = compute_D(
                        S=S,
                        b_ax=b_ax_loc,
                        g_sq_iso_analytic=g_sq_iso_loc,
                        slope_ax=a_ax_loc,
                    )
                    E_loc = 0.0
                else:
                    gx_loc, gy_loc, gz_loc = solve_g_principals_from_LR(
                        b_iso=b_iso_loc,
                        b_ax=b_ax_loc,
                        b_rh=b_rho_loc,
                    )
                    D_J_loc, E_J_loc = solve_D_E_from_LR(
                        a_ax=a_ax_loc,
                        a_rh=a_rho_loc,
                        S=S,
                        gx=gx_loc,
                        gy=gy_loc,
                        gz=gz_loc,
                    )
                    D_loc = None if D_J_loc is None else D_J_loc / (h * c * 100)
                    E_loc = None if E_J_loc is None else E_J_loc / (h * c * 100)

                return gx_loc, gy_loc, gz_loc, D_loc, E_loc

            # Parameters for χ - σ and χ + σ (no-TIP branch)
            gx_minus, gy_minus, gz_minus, D_minus, E_minus = _params_from_LR(
                a_ax_minus, b_ax_minus, a_rho_minus, b_rho_minus, b_iso_minus
            )
            gx_plus, gy_plus, gz_plus, D_plus, E_plus = _params_from_LR(
                a_ax_plus, b_ax_plus, a_rho_plus, b_rho_plus, b_iso_plus
            )

            # Estimate symmetric errors as half-difference between ± branches
            def _half_diff(v_plus, v_minus):
                if (v_plus is None) or (v_minus is None):
                    return None
                return abs(v_plus - v_minus) / 2.0

            gx_err = _half_diff(gx_plus, gx_minus)
            gy_err = _half_diff(gy_plus, gy_minus)
            gz_err = _half_diff(gz_plus, gz_minus)
            D_err = _half_diff(D_plus, D_minus)
            E_err = _half_diff(E_plus, E_minus)

            if (gx_err is not None) and (gz_err is not None) and (D_err is not None):
                if E_err is not None:
                    ut.cprint(
                        f"no TIP uncertainties (from experimental σ): "
                        f"Δgx = {gx_err:.4f}, Δgy = {gy_err:.4f}, Δgz = {gz_err:.4f}, "
                        f"ΔD = {D_err:.4f} cm^-1, ΔE = {E_err:.4f} cm^-1",
                        'cyan',
                    )
                else:
                    ut.cprint(
                        f"no TIP uncertainties (from experimental σ): "
                        f"Δgx = {gx_err:.4f}, Δgy = {gy_err:.4f}, Δgz = {gz_err:.4f}, "
                        f"ΔD = {D_err:.4f} cm^-1",
                        'cyan',
                    )
    except Exception as e:
        logging.warning(f"Error propagation for g/D/E failed: {e}")

    return ChiTemperatureAnalysis(
        temps=temps,
        inv_t=inv_t,
        temps_csv=temps_csv,
        inv_t_csv=inv_t_csv,
        chi_iso_nevpt2_si=chi_iso_nevpt2_si,
        chi_ax_nevpt2_si=chi_ax_nevpt2_si,
        chi_rho_nevpt2_si=chi_rho_nevpt2_si,
        chi_iso_analytic=chi_iso_analytic,
        chi_ax_analytic=chi_ax_analytic,
        chi_rho_analytic=chi_rho_analytic,
        g_sq_iso=g_sq_iso,
        g_sq_ax=g_sq_ax,
        g_sq_rh=g_sq_rh,
        D_list=D_list,
        E_list=E_list,
        D_J=D_J,
        E_J=E_J,
        chi_iso_fit_csv=chi_iso_fit_csv,
        chi_ax_fit_csv=chi_ax_fit_csv,
        chi_rho_fit_csv=chi_rho_fit_csv,
        chi_iso_sdev_csv=chi_iso_sdev_csv,
        chi_ax_sdev_csv=chi_ax_sdev_csv,
        chi_rho_sdev_csv=chi_rho_sdev_csv,
        chi_iso_fit_pred=chi_iso_fit_pred,
        chi_ax_fit_pred=chi_ax_fit_pred,
        chi_rho_fit_pred=chi_rho_fit_pred,
        chi_iso_sdev_pred=chi_iso_sdev_pred,
        chi_ax_sdev_pred=chi_ax_sdev_pred,
        chi_rho_sdev_pred=chi_rho_sdev_pred,
        chi_iso_fit_pred_TIP=chi_iso_fit_pred_TIP,
        chi_ax_fit_pred_TIP=chi_ax_fit_pred_TIP,
        chi_iso_sdev_pred_TIP=chi_iso_sdev_pred_TIP,
        chi_ax_sdev_pred_TIP=chi_ax_sdev_pred_TIP,
        a_iso=a_iso,
        b_iso=b_iso,
        a_ax=a_ax,
        b_ax=b_ax,
        a_rho=a_rho,
        b_rho=b_rho,
        a_iso_se=a_iso_se,
        b_iso_se=b_iso_se,
        a_ax_se=a_ax_se,
        b_ax_se=b_ax_se,
        a_rho_se=a_rho_se,
        b_rho_se=b_rho_se,
        a_iso_TIP=a_iso_TIP,
        b_iso_TIP=b_iso_TIP,
        a_ax_TIP=a_ax_TIP,
        b_ax_TIP=b_ax_TIP,
        a_iso_TIP_se=a_iso_TIP_se,
        b_iso_TIP_se=b_iso_TIP_se,
        a_ax_TIP_se=a_ax_TIP_se,
        b_ax_TIP_se=b_ax_TIP_se,
    )


def plot_chi_temperature_dependence(
    file_name: str,
    section: str,
    csv_path: str | None = None,
    fix_intercept_highT: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Reads susceptibility tensors from an ORCA output file and plots
    isotropic, axial and rhombic components versus inverse temperature.

    Args:
        file_name (str): Path to the ORCA output file.
        section (str): Section to parse ('casscf' or 'nevpt2').

    Returns:
        fig, ax: Matplotlib figure and axes objects of the combined plot.

    If fix_intercept_highT=True, each LR intercept b is fixed to the corresponding chi·T value at
    the highest available CSV temperature.
    """

    analysis = compute_chi_temperature_dependence(
        file_name=file_name,
        section=section,
        csv_path=csv_path,
        fix_intercept_highT=fix_intercept_highT,
    )

    # --- Write results CSV next to the PNG ---
    output_png = 'chi_plot_all.png'
    output_csv = output_png.replace('.png', '.csv')

    write_results_csv(
        output_csv,
        analysis.temps,
        analysis.inv_t,
        analysis.chi_iso_nevpt2_si,
        analysis.chi_ax_nevpt2_si,
        analysis.chi_rho_nevpt2_si,
        analysis.chi_iso_analytic,
        analysis.chi_ax_analytic,
        analysis.chi_rho_analytic,
        analysis.g_sq_iso,
        analysis.g_sq_ax,
        analysis.g_sq_rh,
        analysis.D_list,
        analysis.E_list,
        analysis.D_J,
        analysis.E_J,
        analysis.chi_iso_fit_pred,
        analysis.chi_ax_fit_pred,
        analysis.chi_rho_fit_pred,
        analysis.chi_iso_sdev_pred,
        analysis.chi_ax_sdev_pred,
        analysis.chi_rho_sdev_pred,
        analysis.temps_csv,
        analysis.inv_t_csv,
        analysis.chi_iso_fit_csv,
        analysis.chi_ax_fit_csv,
        analysis.chi_rho_fit_csv,
        analysis.chi_iso_sdev_csv,
        analysis.chi_ax_sdev_csv,
        analysis.chi_rho_sdev_csv,
        analysis.a_iso,
        analysis.b_iso,
        analysis.a_ax,
        analysis.b_ax,
        analysis.a_rho,
        analysis.b_rho,
        analysis.a_iso_se,
        analysis.b_iso_se,
        analysis.a_ax_se,
        analysis.b_ax_se,
        analysis.a_rho_se,
        analysis.b_rho_se,
    )

    # --- ISO-only plot ---
    _plot_component(
        'blue',
        analysis.inv_t,
        analysis.chi_iso_nevpt2_si,
        analysis.chi_iso_analytic,
        analysis.g_sq_iso,
        analysis.inv_t_csv,
        analysis.chi_iso_fit_csv,
        analysis.chi_iso_sdev_csv,
        analysis.chi_iso_fit_pred,
        analysis.chi_iso_sdev_pred,
        analysis.a_iso,
        analysis.b_iso,
        analysis.a_iso_se,
        analysis.b_iso_se,
        'iso',
        'chi_plot_iso.png',
        fit_pred_TIP=analysis.chi_iso_fit_pred_TIP,
        sdev_pred_TIP=analysis.chi_iso_sdev_pred_TIP,
        a_TIP=analysis.a_iso_TIP,
        b_TIP=analysis.b_iso_TIP,
        a_TIP_se=analysis.a_iso_TIP_se,
        b_TIP_se=analysis.b_iso_TIP_se,
    )

    # --- AX-only plot ---
    _plot_component(
        'green',
        analysis.inv_t,
        analysis.chi_ax_nevpt2_si,
        analysis.chi_ax_analytic,
        analysis.g_sq_ax,
        analysis.inv_t_csv,
        analysis.chi_ax_fit_csv,
        analysis.chi_ax_sdev_csv,
        analysis.chi_ax_fit_pred,
        analysis.chi_ax_sdev_pred,
        analysis.a_ax,
        analysis.b_ax,
        analysis.a_ax_se,
        analysis.b_ax_se,
        'ax',
        'chi_plot_ax.png',
        fit_pred_TIP=analysis.chi_ax_fit_pred_TIP,
        sdev_pred_TIP=analysis.chi_ax_sdev_pred_TIP,
        a_TIP=analysis.a_ax_TIP,
        b_TIP=analysis.b_ax_TIP,
        a_TIP_se=analysis.a_ax_TIP_se,
        b_TIP_se=analysis.b_ax_TIP_se,
    )

    # --- RHO-only plot ---
    _plot_component(
        'red',
        analysis.inv_t,
        analysis.chi_rho_nevpt2_si,
        analysis.chi_rho_analytic,
        analysis.g_sq_rh,
        analysis.inv_t_csv,
        analysis.chi_rho_fit_csv,
        analysis.chi_rho_sdev_csv,
        analysis.chi_rho_fit_pred,
        analysis.chi_rho_sdev_pred,
        analysis.a_rho,
        analysis.b_rho,
        analysis.a_rho_se,
        analysis.b_rho_se,
        'rho',
        'chi_plot_rho.png',
        fit_pred_TIP=None,
        sdev_pred_TIP=None,
        a_TIP=None,
        b_TIP=None,
        a_TIP_se=None,
        b_TIP_se=None,
    )

    # --- Combined ALL plot ---
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    ax.plot(
        analysis.inv_t,
        analysis.chi_iso_nevpt2_si,
        label=rf'$\frac{{g_e}}{{3}}\,\mathrm{{Tr}}(\frac{{\chi}}{{g}})$ {SHORT_LABELS["NEVPT2"]}',
        color='blue',
    )
    ax.plot(
        analysis.inv_t,
        analysis.chi_ax_nevpt2_si,
        label=rf'$\chi_{{ax}}$ {SHORT_LABELS["NEVPT2"]}',
        color='green',
    )
    ax.plot(
        analysis.inv_t,
        analysis.chi_rho_nevpt2_si,
        label=rf'$\chi_{{rho}}$ {SHORT_LABELS["NEVPT2"]}',
        color='red',
    )

    ax.plot(
        analysis.inv_t,
        analysis.chi_iso_analytic,
        label=rf'$\frac{{g_e}}{{3}}\,\mathrm{{Tr}}(\frac{{\chi}}{{g}})$ {SHORT_LABELS["Analytical"]}',
        color='blue',
        linestyle='--',
    )
    ax.plot(
        analysis.inv_t,
        analysis.chi_ax_analytic,
        label=rf'$\chi_{{ax}}$ {SHORT_LABELS["Analytical"]}',
        color='green',
        linestyle='--',
    )
    ax.plot(
        analysis.inv_t,
        analysis.chi_rho_analytic,
        label=rf'$\chi_{{rho}}$ {SHORT_LABELS["Analytical"]}',
        color='red',
        linestyle='--',
    )

    ax.plot(
        analysis.inv_t,
        analysis.g_sq_iso,
        label=SHORT_LABELS[r'$(g^2)_{\mathrm{iso}}$'],
        color='blue',
        linestyle=':',
    )
    ax.plot(
        analysis.inv_t,
        analysis.g_sq_ax,
        label=SHORT_LABELS[r'$(g^2)_{\mathrm{ax}}$'],
        color='green',
        linestyle=':',
    )
    ax.plot(
        analysis.inv_t,
        analysis.g_sq_rh,
        label=SHORT_LABELS[r'$(g^2)_{\rho}$'],
        color='red',
        linestyle=':',
    )

    if (
        len(analysis.inv_t_csv)
        == len(analysis.chi_iso_fit_csv)
        == len(analysis.chi_ax_fit_csv)
        == len(analysis.chi_rho_fit_csv)
        and len(analysis.inv_t_csv) > 0
    ):
        ax.plot(
            analysis.inv_t_csv,
            analysis.chi_iso_fit_csv,
            label=rf'$\chi_{{iso}}$ {SHORT_LABELS["Fitted"]}',
            color='blue',
            marker='o',
            linestyle='',
            markersize=5,
        )
        ax.plot(
            analysis.inv_t_csv,
            analysis.chi_ax_fit_csv,
            label=rf'$\chi_{{ax}}$ {SHORT_LABELS["Fitted"]}',
            color='green',
            marker='o',
            linestyle='',
            markersize=5,
        )
        ax.plot(
            analysis.inv_t_csv,
            analysis.chi_rho_fit_csv,
            label=rf'$\chi_{{rho}}$ {SHORT_LABELS["Fitted"]}',
            color='red',
            marker='o',
            linestyle='',
            markersize=5,
        )

        if len(analysis.chi_iso_sdev_csv) == len(analysis.inv_t_csv) and any(
            v is not None for v in analysis.chi_iso_sdev_csv
        ):
            yerr = [v if (v is not None) else 0.0 for v in analysis.chi_iso_sdev_csv]
            ax.errorbar(
                analysis.inv_t_csv,
                analysis.chi_iso_fit_csv,
                yerr=yerr,
                fmt='none',
                ecolor='blue',
                alpha=0.5,
                capsize=2,
            )
        if len(analysis.chi_ax_sdev_csv) == len(analysis.inv_t_csv) and any(
            v is not None for v in analysis.chi_ax_sdev_csv
        ):
            yerr = [v if (v is not None) else 0.0 for v in analysis.chi_ax_sdev_csv]
            ax.errorbar(
                analysis.inv_t_csv,
                analysis.chi_ax_fit_csv,
                yerr=yerr,
                fmt='none',
                ecolor='green',
                alpha=0.5,
                capsize=2,
            )
        if len(analysis.chi_rho_sdev_csv) == len(analysis.inv_t_csv) and any(
            v is not None for v in analysis.chi_rho_sdev_csv
        ):
            yerr = [v if (v is not None) else 0.0 for v in analysis.chi_rho_sdev_csv]
            ax.errorbar(
                analysis.inv_t_csv,
                analysis.chi_rho_fit_csv,
                yerr=yerr,
                fmt='none',
                ecolor='red',
                alpha=0.5,
                capsize=2,
            )

    if analysis.chi_iso_fit_pred is not None:
        ax.plot(
            analysis.inv_t,
            analysis.chi_iso_fit_pred,
            label=rf'$\chi_{{iso}}$ {SHORT_LABELS["Fitted (LR)"]}',
            linestyle='-.',
            linewidth=1.5,
            color='blue',
        )
    if analysis.chi_ax_fit_pred is not None:
        ax.plot(
            analysis.inv_t,
            analysis.chi_ax_fit_pred,
            label=rf'$\chi_{{ax}}$ {SHORT_LABELS["Fitted (LR)"]}',
            linestyle='-.',
            linewidth=1.5,
            color='green',
        )
    if analysis.chi_rho_fit_pred is not None:
        ax.plot(
            analysis.inv_t,
            analysis.chi_rho_fit_pred,
            label=rf'$\chi_{{rho}}$ {SHORT_LABELS["Fitted (LR)"]}',
            linestyle='-.',
            linewidth=1.5,
            color='red',
        )

    legend = _finalize_axes(ax, analysis.inv_t, analysis.inv_t_csv, 'All Components')

    # Adjust ylim to avoid legend clipping as in original
    fig.canvas.draw()
    legend_bbox = legend.get_window_extent()
    ax_bbox = ax.get_window_extent()

    if legend_bbox.y0 < ax_bbox.y1:
        y_min, y_max = ax.get_ylim()
        y_padding = (y_max - y_min) * 0.10
        ax.set_ylim(y_min, y_max + y_padding)

    plt.savefig('chi_plot_all.png', dpi=600)
    plt.show()
    plt.close('all')
    return fig, ax

def main():
    """
    Parse command-line arguments and generate the χ(T) components plot.
    """

    # Define command-line interface for input file and section choice
    parser = argparse.ArgumentParser(
        description=(
            'Plot χ(T) components from ORCA output\n\n'
            'Example:\n'
            '  chi_plot filename.out nevpt2 [filename.csv]'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'susc_file',
        help='ORCA output file with susceptibility data'
    )
    parser.add_argument(
        'section',
        nargs='?',
        choices=['casscf', 'nevpt2'],
        default=None,
        help='Section to plot (casscf or nevpt2). If omitted, run in CSV-only mode using "susc_file" as the susceptibility CSV.'
    )
    parser.add_argument(
        'csv_file',
        nargs='?',
        default=None,
        help='Optional additional CSV file to read'
    )
    parser.add_argument(
        '--log-level',
        choices=['CRITICAL','ERROR','WARNING','INFO','DEBUG'],
        default='INFO',
        help='Logging verbosity'
    )
    parser.add_argument(
        '--spin',
        type=float,
        help='Spin quantum number S (required for CSV-only mode when no ORCA section is provided).'
    )
    parser.add_argument(
        '--fix-intercept-highT',
        action='store_true',
        help='Fix LR intercept b to the chi·T value at the highest available CSV temperature for iso/ax/rho fits.'
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format='[%(levelname)s] %(message)s')

    # Decide between full (ORCA+NEVPT2) mode and CSV-only mode
    if args.section is None:
        # CSV-only mode: susc_file is interpreted as a susceptibility CSV
        if args.spin is None:
            parser.error("CSV-only mode detected (no 'section' argument). Please provide --spin S (e.g. --spin 2.0).")
        ut.cprint("Running in CSV-only mode (no ORCA/NEVPT2 data).", 'cyan')
        run_csv_only_pipeline(
            csv_path=args.susc_file,
            spin_S=args.spin,
            fix_intercept_highT=args.fix_intercept_highT,
        )
        ut.cprint("Saved plots: chi_plot_iso.png, chi_plot_ax.png, chi_plot_rho.png, chi_plot_all.png", 'cyan')
    else:
        # Full mode: ORCA output + optional CSV
        _fig, _ax = plot_chi_temperature_dependence(
            args.susc_file,
            args.section,
            args.csv_file,
            fix_intercept_highT=args.fix_intercept_highT,
        )
        ut.cprint("Saved plots: chi_plot_iso.png, chi_plot_ax.png, chi_plot_rho.png, chi_plot_all.png", 'cyan')
        ut.cprint("Saved table: chi_plot_all.csv", 'cyan')

if __name__ == '__main__':
    main()