# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Physical helper functions for variable-temperature susceptibility models."""

import numpy as np

from paranmr.core.const.physics import GE, KB, MU0, MUB, C, H  # noqa


def compute_analytic_component(
    chi_component: str,
    temperature: np.ndarray,
    g_components_sq: dict[str, float],
    g_components: dict[str, float],
    D_J: float,
    E_J: float,
    spin: float,
) -> np.ndarray:
    g_sq_iso = float(g_components_sq["g_sq_iso"])
    g_sq_ax = float(g_components_sq["g_sq_ax"])
    g_sq_rh = float(g_components_sq["g_sq_rh"])
    g_iso = float(g_components["g_iso"])
    g_ax = float(g_components["g_ax"])
    g_rho = float(g_components["g_rho"])

    # Accept both scalar and array temperatures.
    t = np.asarray(temperature, dtype=float)

    # Compute Spin coefficient
    f_S = (2 * spin - 1) * (2 * spin + 3)

    # Calculate chi component in reduced (Curie) units
    if chi_component == "iso":
        analytic = (
            GE * g_iso
            - (f_S / (45 * KB * t)) * (D_J * GE * g_ax + 3 * E_J * GE * g_rho)
        ) / t
    elif chi_component == "ax":
        analytic = (
            g_sq_ax
            - (f_S / (30 * KB * t))
            * (D_J * (g_sq_ax + 3 * g_sq_iso) - 3 * E_J * g_sq_rh)
        ) / t
    elif chi_component == "rho":
        analytic = (
            g_sq_rh
            + (f_S / (30 * KB * t)) * (E_J * (g_sq_ax - 3 * g_sq_iso) + D_J * g_sq_rh)
        ) / t
    else:
        raise ValueError(
            f"Unknown chi_component={chi_component!r}; expected 'iso', 'ax', or 'rho'."
        )

    return analytic


def compute_g_sq_components(g_tensor: np.ndarray) -> dict[str, float]:
    """Compute g² invariants for susceptibility components.

    This helper evaluates the squared g-tensor invariants corresponding to the
    isotropic, axial, and rhombic susceptibility components. It assumes that
    the g-tensor is expressed in its working principal-axis basis, i.e. the
    diagonal elements correspond to (g_x, g_y, g_z).

    The returned quantities are defined as:
        g_sq_iso = (g_x² + g_y² + g_z²) / 3
        g_sq_ax  = 3/2 · (g_z² − g_sq_iso)
        g_sq_rh  = (g_x² − g_y²) / 2

    These invariants are used in analytic high-temperature expansions of the
    magnetic susceptibility.

    Args:
        g_tensor: 3×3 g-tensor matrix in the principal-axis representation.

    Returns:
        dict[str, float]:
            A mapping with keys `g_sq_iso`, `g_sq_ax`, `g_sq_rh`.
    """
    g_x2 = float(g_tensor[0, 0] ** 2)
    g_y2 = float(g_tensor[1, 1] ** 2)
    g_z2 = float(g_tensor[2, 2] ** 2)

    g_sq_iso = (g_x2 + g_y2 + g_z2) / 3.0
    g_sq_ax = 1.5 * (g_z2 - g_sq_iso)
    g_sq_rh = (g_x2 - g_y2) / 2.0

    return {
        "g_sq_iso": g_sq_iso,
        "g_sq_ax": g_sq_ax,
        "g_sq_rh": g_sq_rh,
    }


def calculate_E_D_components(
    eff_H: np.ndarray,
) -> tuple[float, float]:
    """
    Calculate the E and D components of the effective Hamiltonian matrix

    Args:
        rotated_eff_H_tensors (list of ndarray): 3×3 Effective Hamiltonian matrix

    Returns:
        D (float): Axial component converted to Joules
        E (float): Rhombic component converted to Joules
    """

    eff_H_iso = np.trace(eff_H) / 3.0
    eff_H_traceless = eff_H - eff_H_iso * np.eye(3)

    evals, _ = np.linalg.eigh(eff_H_traceless)
    idx = np.argsort(np.abs(evals))
    eff_H_diag = np.diag(evals.real[idx])

    D = 1.5 * eff_H_diag[2, 2]
    E = (eff_H_diag[0, 0] - eff_H_diag[1, 1]) / 2

    # Convert values to Joules
    D_J = D * H * C * 100
    E_J = E * H * C * 100

    return D_J, E_J


def compute_tip_correction(
    ab_initio_chi: float,
    analytic_chi: float,
    spin: float,
) -> float:
    norm_factor = compute_curie_prefactor(spin)

    # Convert Å^3 to reduced units
    ab_initio_chi = ab_initio_chi / norm_factor

    chi_tip = ab_initio_chi - analytic_chi

    return chi_tip


def compute_curie_prefactor(spin: float) -> float:
    """
    Compute the Curie prefactor for a given spin quantum number.

    The prefactor is used to normalise susceptibility data and is returned in
    Å^3·K (using 1 Å^3 = 1e-30 m^3).

    Args:
        spin (float): Total spin quantum number S.

    Returns:
        float: Curie prefactor in Å^3·K.
    """
    return (MU0 * MUB**2 * spin * (spin + 1)) / (3 * KB) * 1e30  # [Å^3·K]
