# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Shared relaxation-formalism evaluation helpers."""

from __future__ import annotations

from collections.abc import Mapping

from simpnmr.core.relaxation import gueron, sbm


def evaluate_relaxation_rates(
    *,
    relaxation_model: str,
    nuclei_coords: Mapping[str, object],
    electron_coords,
    gamma_I_dict: Mapping[str, float],
    omega_I_dict: Mapping[str, float],
    omega_S: float,
    spin: float,
    orbit: float,
    total_momentum_J: float,
    A_iso_dict: Mapping[str, float] | None = None,
    temperature: float | None = None,
    tau_R: float | None = None,
    tau_c1: float | None = None,
    tau_c2: float | None = None,
    tau_e1: float | None = None,
    tau_e2: float | None = None,
    compute_r1: bool = True,
    compute_r2: bool = False,
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    """Evaluate nucleus-resolved relaxation rates for a selected formalism.

    This helper centralizes the one-and-the-same formalism selection logic used
    by both correlation-time fitting and relaxation-aware prediction workflows.
    It returns raw per-nucleus rates and leaves any chemical-label averaging or
    linewidth conversion to the caller.

    Args:
        relaxation_model: Relaxation model selector. Supported values are
            ``"sbm"``, ``"curie"``, ``"sbm curie"``, and ``"curie sbm"``.
        nuclei_coords: Mapping from nucleus labels to Cartesian coordinates.
        electron_coords: Cartesian coordinate of the paramagnetic centre.
        gamma_I_dict: Mapping from nucleus labels to gyromagnetic ratios.
        omega_I_dict: Mapping from nucleus labels to nuclear Larmor frequencies.
        omega_S: Electron Larmor frequency.
        spin: Electron spin quantum number.
        orbit: Electron orbital quantum number.
        total_momentum_J: Total angular momentum quantum number.
        A_iso_dict: Optional mapping from nucleus labels to isotropic Fermi-
            contact values. Required for SBM-based contact terms.
        temperature: Optional temperature in Kelvin. Required for Curie terms.
        tau_R: Optional rotational correlation time in seconds. Required for
            Curie terms.
        tau_c1: Optional first combined correlation time in seconds. Required
            for SBM dipolar terms.
        tau_c2: Optional second combined correlation time in seconds. Required
            for SBM dipolar terms and SBM ``R2`` evaluation.
        tau_e1: Optional electronic correlation time used by SBM contact ``R2``.
        tau_e2: Optional electronic correlation time used by SBM contact ``R1``
            and ``R2``.
        compute_r1: Whether to evaluate ``R1`` rates.
        compute_r2: Whether to evaluate ``R2`` rates.

    Returns:
        Tuple ``(rates_r1, rates_r2)`` where each entry is either a
        nucleus-label-to-rate mapping or ``None`` when not requested.

    Raises:
        ValueError: If the requested formalism lacks required inputs.
    """
    labels = list(nuclei_coords.keys())
    model = relaxation_model.strip().lower()

    if not compute_r1 and not compute_r2:
        return None, None

    def _require(name: str, value) -> None:
        if value is None:
            raise ValueError(
                f"{name} is required for relaxation_model={relaxation_model!r}"
            )

    rates_r1: dict[str, float] | None = None
    rates_r2: dict[str, float] | None = None

    if model == "sbm":
        _require("A_iso_dict", A_iso_dict)
        if compute_r1:
            _require("tau_c1", tau_c1)
            _require("tau_c2", tau_c2)
            _require("tau_e2", tau_e2)
            sbm_dipolar_r1_rates = sbm.calc_r1_dipolar(
                labels,
                nuclei_coords,
                electron_coords,
                gamma_I_dict,
                omega_I_dict,
                omega_S,
                tau_c1,
                tau_c2,
                spin,
                orbit,
                total_momentum_J,
            )
            sbm_contact_r1_rates = sbm.calc_r1_contact(
                labels,
                A_iso_dict,
                omega_I_dict,
                omega_S,
                tau_e2,
                spin,
                total_momentum_J,
            )
            rates_r1 = {
                label: sbm_dipolar_r1_rates[label] + sbm_contact_r1_rates[label]
                for label in labels
            }
        if compute_r2:
            _require("tau_c1", tau_c1)
            _require("tau_c2", tau_c2)
            _require("tau_e1", tau_e1)
            _require("tau_e2", tau_e2)
            sbm_dipolar_r2_rates = sbm.calc_r2_dipolar(
                labels,
                nuclei_coords,
                electron_coords,
                gamma_I_dict,
                omega_I_dict,
                omega_S,
                tau_c1,
                tau_c2,
                spin,
                orbit,
                total_momentum_J,
            )
            sbm_contact_r2_rates = sbm.calc_r2_contact(
                labels,
                A_iso_dict,
                omega_I_dict,
                omega_S,
                tau_e1,
                tau_e2,
                spin,
                total_momentum_J,
            )
            rates_r2 = {
                label: sbm_dipolar_r2_rates[label] + sbm_contact_r2_rates[label]
                for label in labels
            }

    elif model == "curie":
        _require("temperature", temperature)
        _require("tau_R", tau_R)
        if compute_r1:
            curie_r1_rates = gueron.calc_r1_curie(
                labels,
                nuclei_coords,
                electron_coords,
                omega_I_dict,
                temperature,
                tau_R,
                spin,
                orbit,
                total_momentum_J,
            )
            rates_r1 = {label: curie_r1_rates[label] for label in labels}
        if compute_r2:
            curie_r2_rates = gueron.calc_r2_curie(
                labels,
                nuclei_coords,
                electron_coords,
                omega_I_dict,
                temperature,
                tau_R,
                spin,
                orbit,
                total_momentum_J,
            )
            rates_r2 = {label: curie_r2_rates[label] for label in labels}

    elif model in {"sbm curie", "curie sbm"}:
        _require("A_iso_dict", A_iso_dict)
        _require("temperature", temperature)
        _require("tau_R", tau_R)
        if compute_r1:
            _require("tau_c1", tau_c1)
            _require("tau_c2", tau_c2)
            _require("tau_e2", tau_e2)
            sbm_dipolar_r1_rates = sbm.calc_r1_dipolar(
                labels,
                nuclei_coords,
                electron_coords,
                gamma_I_dict,
                omega_I_dict,
                omega_S,
                tau_c1,
                tau_c2,
                spin,
                orbit,
                total_momentum_J,
            )
            sbm_contact_r1_rates = sbm.calc_r1_contact(
                labels,
                A_iso_dict,
                omega_I_dict,
                omega_S,
                tau_e2,
                spin,
                total_momentum_J,
            )
            curie_r1_rates = gueron.calc_r1_curie(
                labels,
                nuclei_coords,
                electron_coords,
                omega_I_dict,
                temperature,
                tau_R,
                spin,
                orbit,
                total_momentum_J,
            )
            rates_r1 = {
                label: sbm_dipolar_r1_rates[label]
                + sbm_contact_r1_rates[label]
                + curie_r1_rates[label]
                for label in labels
            }
        if compute_r2:
            _require("tau_c1", tau_c1)
            _require("tau_c2", tau_c2)
            _require("tau_e1", tau_e1)
            _require("tau_e2", tau_e2)
            sbm_dipolar_r2_rates = sbm.calc_r2_dipolar(
                labels,
                nuclei_coords,
                electron_coords,
                gamma_I_dict,
                omega_I_dict,
                omega_S,
                tau_c1,
                tau_c2,
                spin,
                orbit,
                total_momentum_J,
            )
            sbm_contact_r2_rates = sbm.calc_r2_contact(
                labels,
                A_iso_dict,
                omega_I_dict,
                omega_S,
                tau_e1,
                tau_e2,
                spin,
                total_momentum_J,
            )
            curie_r2_rates = gueron.calc_r2_curie(
                labels,
                nuclei_coords,
                electron_coords,
                omega_I_dict,
                temperature,
                tau_R,
                spin,
                orbit,
                total_momentum_J,
            )
            rates_r2 = {
                label: sbm_dipolar_r2_rates[label]
                + sbm_contact_r2_rates[label]
                + curie_r2_rates[label]
                for label in labels
            }

    else:
        raise ValueError(f"Unknown relaxation model: {relaxation_model!r}")

    return rates_r1, rates_r2
