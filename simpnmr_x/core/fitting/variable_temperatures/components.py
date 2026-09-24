# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Physical helper functions for variable-temperature susceptibility models."""

import numpy as np

from simpnmr_x.core.const.physics import GE, KB, MU0, MUB, C, H  # noqa


def compute_analytic_component(
    chi_component: str,
    temperature: np.ndarray,
    g_components_sq: dict[str, float],
    g_components: dict[str, float],
    D_J: float,
    E_J: float,
    spin: float,
    total_J: float | None = None,
) -> np.ndarray:
    """Evaluate one analytic second-order susceptibility component.

    Args:
        chi_component: Component identifier: ``iso``, ``ax``, or ``rho``.
        temperature: Temperatures in K.
        g_components_sq: Squared g-tensor components in the working frame.
        g_components: Unsquared g-tensor components in the working frame.
        D_J: Axial ZFS parameter in J.
        E_J: Rhombic ZFS parameter in J.
        spin: Spin quantum number S.
        total_J: Effective total angular momentum J, if available.

    Returns:
        Analytic susceptibility values in reduced units.
    """
    g_sq_iso = float(g_components_sq["g_sq_iso"])
    g_sq_ax = float(g_components_sq["g_sq_ax"])
    g_sq_rh = float(g_components_sq["g_sq_rh"])
    g_iso = float(g_components["g_iso"])
    g_ax = float(g_components["g_ax"])
    g_rho = float(g_components["g_rho"])

    # Accept both scalar and array temperatures.
    t = np.asarray(temperature, dtype=float)

    # The second-order coefficient follows the effective angular momentum.
    effective_J = total_J if total_J is not None else spin
    f_S = (2 * effective_J - 1) * (2 * effective_J + 3)

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


def compute_g_components(g_tensor: np.ndarray) -> dict[str, float]:
    """Compute g-tensor components in the working principal-axis frame.

    Args:
        g_tensor: 3×3 g-tensor in the working principal-axis frame.

    Returns:
        A mapping with keys ``g_iso``, ``g_ax``, and ``g_rho``.

    Raises:
        ValueError: If ``g_tensor`` is not a 3×3 matrix.
    """
    array = np.asarray(g_tensor, dtype=float)
    if array.shape != (3, 3):
        raise ValueError("g_tensor must have shape (3, 3)")

    g_iso = float(np.trace(array) / 3.0)
    return {
        "g_iso": g_iso,
        "g_ax": float(1.5 * (array[2, 2] - g_iso)),
        "g_rho": float(0.5 * (array[0, 0] - array[1, 1])),
    }


def rotate_tensors_to_frame(
    eff_H: np.ndarray,
    g_tensor: np.ndarray,
    frame: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate ZFS and g tensors into a common right-handed frame.

    Args:
        eff_H: Effective ZFS Hamiltonian tensor in the laboratory frame.
        g_tensor: g-tensor in the laboratory frame.
        frame: Rotation matrix whose columns are the target-frame axes.

    Returns:
        A tuple ``(eff_H_rotated, g_tensor_rotated)``.

    Raises:
        ValueError: If any input is not a 3×3 matrix.
    """
    tensors = (np.asarray(eff_H, dtype=float), np.asarray(g_tensor, dtype=float))
    rotation = np.asarray(frame, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("frame must have shape (3, 3)")
    if any(tensor.shape != (3, 3) for tensor in tensors):
        raise ValueError("eff_H and g_tensor must have shape (3, 3)")

    return tuple(rotation.T @ tensor @ rotation for tensor in tensors)


def validate_common_principal_axes(
    eff_H: np.ndarray,
    g_tensor: np.ndarray,
    tolerance: float,
) -> None:
    """Validate that ZFS and g tensors share the supplied principal frame.

    Args:
        eff_H: Effective ZFS Hamiltonian tensor in the proposed frame.
        g_tensor: g-tensor in the same proposed frame.
        tolerance: Maximum allowed relative off-diagonal component.

    Raises:
        ValueError: If either tensor is not symmetric or is not sufficiently
            diagonal in the proposed frame.
    """
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    for name, tensor in (("effective Hamiltonian", eff_H), ("g-tensor", g_tensor)):
        array = np.asarray(tensor, dtype=float)
        if array.shape != (3, 3):
            raise ValueError(f"{name} must have shape (3, 3)")
        if not np.allclose(array, array.T):
            raise ValueError(f"{name} must be symmetric")

        diagonal_scale = max(float(np.max(np.abs(np.diag(array)))), 1.0e-30)
        off_diagonal = array - np.diag(np.diag(array))
        relative_off_diagonal = float(np.max(np.abs(off_diagonal))) / diagonal_scale
        if relative_off_diagonal > tolerance:
            raise ValueError(
                f"{name} is not aligned with the ZFS/chi frame: "
                f"relative off-diagonal component {relative_off_diagonal:.3g} "
                f"exceeds tolerance {tolerance:.3g}"
            )


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
    total_J: float | None = None,
) -> float:
    """Return the TIP correction between ab-initio and analytic susceptibility.

    Args:
        ab_initio_chi: Ab-initio susceptibility component in Å³.
        analytic_chi: Analytic component in reduced units.
        spin: Spin quantum number S.
        total_J: Effective total angular momentum J, if available.

    Returns:
        TIP correction in reduced units.
    """
    norm_factor = compute_curie_prefactor(spin, total_J=total_J)

    # Convert Å^3 to reduced units
    ab_initio_chi = ab_initio_chi / norm_factor

    chi_tip = ab_initio_chi - analytic_chi

    return chi_tip


def compute_curie_prefactor(spin: float, total_J: float | None = None) -> float:
    """
    Compute the Curie prefactor for a given spin quantum number.

    The prefactor is used to normalise susceptibility data and is returned in
    Å^3·K (using 1 Å^3 = 1e-30 m^3).

    Args:
        spin (float): Total spin quantum number S.
        total_J (float | None): Effective total angular momentum J. If omitted,
            the spin value is used for backwards-compatible spin-only models.

    Returns:
        float: Curie prefactor in Å^3·K.
    """
    effective_J = total_J if total_J is not None else spin
    return (MU0 * MUB**2 * effective_J * (effective_J + 1)) / (3 * KB) * 1e30
