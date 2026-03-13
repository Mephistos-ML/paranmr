# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Construct magnetic susceptibility values and tensors.

Provides helpers to build susceptibility tensors and isotropic values from
quantum-chemical outputs and spin parameters.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from simpnmr.core.const.physics import NA
from simpnmr.core.domain.tensor import Susceptibility
from simpnmr.core.phys.susc import get_g_corr_iso_susc, get_spin_only_susc


def susc_from_orca_xt(
    temperature: float,
    tensor_xt: NDArray,
    *,
    electronic: Any | None = None,
    g_tensor: np.ndarray | None = None,
) -> Susceptibility:
    """Convert an ORCA XT tensor to a Susceptibility domain object.

    ORCA reports XT in units of cm^3 mol^-1 K. This factory converts XT to a
    molar susceptibility tensor chi (Å^3) by applying the appropriate physical
    conversion and dividing by temperature.

    The builder always computes the spin-only isotropic susceptibility and
    stores it in ``susc.iso_spin_only``. If ``g_tensor`` is available, it also
    computes the g-tensor-corrected isotropic susceptibility and stores it in
    ``susc.iso_g_corr``. The canonical isotropic susceptibility used downstream
    is always assigned to ``susc.iso``: ``g_corr`` when available, otherwise
    ``spin_only``.

    Args:
        temperature: Temperature in Kelvin.
        tensor_xt: ORCA XT tensor as a (3, 3) array in cm^3 mol^-1 K.
        electronic: Electronic-state context required for spin-only isotropic
            susceptibility evaluation.
        g_tensor: Optional g-tensor used for g-corrected isotropic
            susceptibility evaluation.

    Returns:
        Susceptibility domain object.
    """

    # Conversion factor:
    # 1 cm^3 mol^-1 = 1e-6 m^3 / N_A
    # then convert m^3 -> Å^3 (1 m^3 = 1e30 Å^3)
    conv = 1e-24 * NA / (4.0 * np.pi)
    conv = 1.0 / conv

    chi_tensor = tensor_xt / temperature * conv

    susc = Susceptibility(chi_tensor, temperature=float(temperature))
    susc.calc_irred()

    if electronic is None:
        raise ValueError(
            "Susceptibility isotropic evaluation requires quantum-state "
            "information (spin/orbit/angular momentum)."
        )

    spin = getattr(electronic, "spin_S", None)
    orbit = getattr(electronic, "orbit_L", None)
    total_J = getattr(electronic, "total_J", None)

    if spin is None:
        raise ValueError(
            "Susceptibility isotropic evaluation requires quantum-state "
            "information (spin/orbit/angular momentum); "
            "spin-only isotropic susceptibility requires electronic.spin_S."
        )

    # For strict domain naming, orbit_L may legitimately be None for spin-only.
    # Treat missing orbit_L as 0.0 for the Landé-factor helper.
    orbit_val = 0.0 if orbit is None else float(orbit)

    susc.iso_spin_only = float(
        get_spin_only_susc(
            spin=float(spin),
            orbit=orbit_val,
            total_momentum_J=total_J,
            temperature=float(temperature),
        )
    )

    if g_tensor is not None:
        susc.iso_g_corr = float(
            get_g_corr_iso_susc(
                spin=float(spin),
                orbit=orbit_val,
                g_tensor=np.asarray(g_tensor, dtype=float),
                chi_tensors=chi_tensor,
                total_momentum_J=total_J,
            )
        )

    susc.iso = susc.iso_g_corr if susc.iso_g_corr is not None else susc.iso_spin_only

    return susc


def susc_from_csv(
    temperature: float,
    tensor: NDArray,
    *,
    spin: float,
    orbit: float | None = None,
    total_momentum_J: float | None = None,
) -> Susceptibility:
    """Build a susceptibility object from a CSV tensor and spin-only iso.

    The anisotropic susceptibility tensor is taken directly from the CSV input.
    The isotropic susceptibility is evaluated through the spin-only pipeline and
    stored in both ``susc.iso_spin_only`` and the canonical ``susc.iso`` field.

    Args:
        temperature: Temperature in Kelvin.
        tensor: Susceptibility tensor from CSV as a (3, 3) array.
        spin: Spin quantum number ``S``.
        orbit: Orbital angular momentum quantum number ``L``.
        total_momentum_J: Total angular momentum ``J`` or ``None``.

    Returns:
        Susceptibility domain object.
    """
    susc = Susceptibility(
        np.asarray(tensor, dtype=float), temperature=float(temperature)
    )
    susc.calc_irred()

    susc.iso_spin_only = float(
        get_spin_only_susc(
            spin=float(spin),
            orbit=0.0 if orbit is None else float(orbit),
            total_momentum_J=total_momentum_J,
            temperature=float(temperature),
        )
    )
    susc.iso = susc.iso_spin_only
    return susc


def susc_from_spin_only_iso(
    temperature: float,
    *,
    spin: float,
    orbit: float | None = None,
    total_momentum_J: float | None = None,
) -> Susceptibility:
    """Build an isotropic-only susceptibility object with zero anisotropy.

    Args:
        temperature: Temperature in Kelvin.
        spin: Spin quantum number ``S``.
        orbit: Orbital angular momentum quantum number ``L``.
        total_momentum_J: Total angular momentum ``J`` or ``None``.

    Returns:
        Susceptibility domain object with zero anisotropic contribution and
        spin-only isotropic susceptibility.
    """
    chi_tensor = np.zeros((3, 3), dtype=float)

    susc = Susceptibility(chi_tensor, temperature=float(temperature))
    susc.calc_irred()

    susc.iso_spin_only = float(
        get_spin_only_susc(
            spin=float(spin),
            orbit=0.0 if orbit is None else float(orbit),
            total_momentum_J=total_momentum_J,
            temperature=float(temperature),
        )
    )
    susc.iso = susc.iso_spin_only
    return susc


# TODO: Refactor this helper to consume stored susceptibility iso values
# from domain objects (`s.iso`, `s.iso_spin_only`, `s.iso_g_corr`) instead of
# recomputing isotropic susceptibility from builder-time inputs. The current
# `g_corr_iso` pathway reflects an obsolete pre-refactor contract.


# TODO: Consider moving this helper out of the builder layer.
# It assembles derived chiT series data rather than constructing a
# susceptibility domain object
def build_ab_initio_chit_series(
    suscs_ab_initio: list,
    *,
    g_corr_iso: bool = False,
    spin: float | None = None,
    orbit: float | None = None,
    total_momentum_J: float | None = None,
    g_tensor: NDArray | None = None,
) -> dict[str, np.ndarray]:
    """Build an ab initio chiT series from susceptibility objects.

    This factory:
    - computes irreducible representations for all ab initio tensors
    - sorts data by temperature
    - returns chiT components on the ab initio temperature grid (optionally
    using a g-corrected iso)

    Args:
        suscs_ab_initio: List of susceptibility objects.
        g_corr_iso: Whether to compute g-tensor-corrected iso values.
        spin: Spin quantum number S (required if g_corr_iso).
        orbit: Orbital angular momentum L (required if g_corr_iso).
        total_momentum_J: Total angular momentum J (required if
        g_corr_iso can be inferred).
        g_tensor: g-tensor as a (3, 3) array (required if g_corr_iso).

    Returns:
        Dictionary with:
            temps: Temperatures (K), sorted.
            inv_t: Inverse temperatures (K^-1).
            iso: chiT_iso values (A^3 K).
            ax: chiT_ax values (A^3 K).
            rho: chiT_rho values (A^3 K).
    """
    if not suscs_ab_initio:
        raise ValueError("suscs_ab_initio must be a non-empty list.")

    if g_corr_iso:
        if spin is None or g_tensor is None:
            raise ValueError(
                "g_corr_iso=True requires spin and g_tensor to be provided."
            )

    # Ensure irreducible components / eigenframes are available
    for s in suscs_ab_initio:
        s.calc_irred()

    temps = np.array([float(s.temperature) for s in suscs_ab_initio], dtype=float)

    order = np.argsort(temps)
    temps = temps[order]
    suscs_sorted = [suscs_ab_initio[i] for i in order]

    with np.errstate(divide="ignore", invalid="ignore"):
        inv_t = 1.0 / temps

    if g_corr_iso:
        iso_vals = []
        for s in suscs_sorted:
            iso_vals.append(
                float(
                    get_g_corr_iso_susc(
                        spin=float(spin),
                        orbit=0.0 if orbit is None else float(orbit),
                        g_tensor=np.asarray(g_tensor, dtype=float),
                        chi_tensors=s.tensor,
                        total_momentum_J=total_momentum_J,
                    )
                )
            )
        iso_base = np.asarray(iso_vals, dtype=float)
    else:
        iso_base = np.array([float(s.iso) for s in suscs_sorted], dtype=float)

    iso = iso_base * temps
    ax = np.array([float(s.axiality) for s in suscs_sorted], dtype=float) * temps
    rho = np.array([float(s.rhombicity) for s in suscs_sorted], dtype=float) * temps

    return {
        "temps": temps,
        "inv_t": inv_t,
        "iso": iso,
        "ax": ax,
        "rho": rho,
    }
