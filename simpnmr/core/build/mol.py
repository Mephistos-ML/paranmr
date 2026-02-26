# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Construct Molecule domain objects from external payloads.

Provides helpers to build Molecule instances from QC-derived data or CSV payloads.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from simpnmr.core.conv.freq_to_ang import a_tensor_mhz_to_ang
from simpnmr.core.domain.mol import Molecule


def build_molecule_from_qca(
    qca: Any,
    *,
    elements: list[str] | str = "all",
    converter: str | None = "MHz_to_Ang-3",
    orbital_contribution: str = "off",
) -> Molecule:
    """Build a `Molecule` from a parsed QC hyperfine object.

    Expects `qca` to provide component hyperfine tensors keyed by atom label:
      - coords: array-like (n_atoms, 3) in Å
      - labels: list-like (n_atoms,) of indexed per-atom labels (e.g. H1, C2)
      - a_fc: mapping label -> (3, 3) A(FC) tensor
      - a_sd: mapping label -> (3, 3) A(SD) tensor
      - a_orb: mapping label -> (3, 3) A(ORB) tensor or None

    This builder derives effective hyperfine quantities for the domain:
      - a_iso_eff: isotropic hyperfine coupling derived from A(FC)+A(SD)
      - a_dtensor_eff: deviatoric tensor derived from (A(FC)+A(SD)+A(ORB) if available)
    Component tensors are not forwarded to the domain.

    Args:
        qca: Parsed QC hyperfine object.
        elements: Elements/labels to include.
        converter: Optional converter string matching the behaviour of
            `Molecule.from_QCA(..., converter=...)`. Defaults to "MHz_to_Ang-3".
        orbital_contribution: Whether to include `A(ORB)` in the effective tensor.
            Supported values are "off" and "on".

    Returns:
        A `Molecule` instance.

    Raises:
        ValueError: If required fields are missing or unknown converter specified.
    """
    if not hasattr(qca, "coords"):
        raise ValueError("QCA object is missing required attribute: coords")
    if not hasattr(qca, "a_fc"):
        raise ValueError("QCA object is missing required attribute: a_fc")
    if not hasattr(qca, "a_sd"):
        raise ValueError("QCA object is missing required attribute: a_sd")
    if not hasattr(qca, "a_orb"):
        raise ValueError("QCA object is missing required attribute: a_orb")

    if not hasattr(qca, "labels"):
        raise ValueError("QCA object is missing required attribute: labels")

    coords = np.asarray(qca.coords, dtype=float)
    labels = [str(lab) for lab in qca.labels]

    if len(labels) != coords.shape[0]:
        raise ValueError(
            "QCA labels/coords length mismatch: "
            f"{len(labels)} labels vs {coords.shape[0]} coordinate rows"
        )

    # Keep the QC geometry labels (indexed, per-atom) to preserve 1:1 mapping to coords.
    # Hyperfine tensors are expected to be keyed by the same labels; some atoms may be
    # missing (e.g. Fe/Cl/Si) depending on the QC output and user configuration.
    a_fc: dict[str, np.ndarray] = {
        str(k): np.asarray(v, dtype=float) for k, v in qca.a_fc.items()
    }
    a_sd: dict[str, np.ndarray] = {
        str(k): np.asarray(v, dtype=float) for k, v in qca.a_sd.items()
    }
    a_orb: dict[str, np.ndarray | None] = {
        str(k): (None if v is None else np.asarray(v, dtype=float))
        for k, v in qca.a_orb.items()
    }

    if orbital_contribution not in {"off", "on"}:
        raise ValueError(
            f"orbital_contribution must be 'off' or 'on', got {orbital_contribution!r}"
        )

    # A(FC)+A(SD) component tensor, independent of orbital policy.
    a_fc_sd: dict[str, np.ndarray] = {}
    for label in labels:
        if label not in a_fc or label not in a_sd:
            continue

        fc = np.asarray(a_fc[label], dtype=float)
        sd = np.asarray(a_sd[label], dtype=float)
        if fc.shape != (3, 3):
            raise ValueError(f"A(FC) tensor for {label} must be (3,3), got {fc.shape}")
        if sd.shape != (3, 3):
            raise ValueError(f"A(SD) tensor for {label} must be (3,3), got {sd.shape}")

        a_fc_sd[label] = fc + sd

    a_iso_eff: dict[str, float] = {}
    for label, fc_sd_tensor in a_fc_sd.items():
        a_iso_eff[label] = float(np.trace(fc_sd_tensor) / 3.0)

    # Effective tensor for current orbital policy, used only to derive deviatoric part.
    a_full_for_pcs: dict[str, np.ndarray] = {}
    for label, fc_sd_tensor in a_fc_sd.items():
        total = np.asarray(fc_sd_tensor, dtype=float)
        if orbital_contribution == "on":
            orb = a_orb.get(label)
            if orb is None:
                raise ValueError(
                    f"A(ORB) contribution requested but missing for label {label}"
                )
            orb_arr = np.asarray(orb, dtype=float)
            if orb_arr.shape != (3, 3):
                raise ValueError(
                    f"A(ORB) tensor for {label} must be (3,3), got {orb_arr.shape}"
                )
            total = total + orb_arr
        a_full_for_pcs[label] = total

    a_dtensor_eff: dict[str, np.ndarray] = {}
    for label, tensor in a_full_for_pcs.items():
        iso_full = float(np.trace(tensor) / 3.0)
        a_dtensor_eff[label] = tensor - np.eye(3) * iso_full

    # Convert all tensors to the requested domain units.
    if converter is None:
        pass
    elif converter == "MHz_to_Ang-3":
        a_iso_eff = {k: float(v) for k, v in a_iso_eff.items()}
        a_dtensor_eff = a_tensor_mhz_to_ang(a_dtensor_eff)
    else:
        raise ValueError(f"Unknown converter: {converter}")

    return Molecule.from_hyperfine_data(
        labels=labels,
        coords=coords,
        a_iso=a_iso_eff,
        a_dtensor=a_dtensor_eff,
        elements=elements,
    )


def build_molecule_with_pdip(
    labels: list[str],
    coords: np.ndarray,
    *,
    centres: list[int] | np.ndarray,
    elements: list[str] | str = "all",
) -> Molecule:
    """Build a `Molecule` and populate point-dipole hyperfine couplings.

    Args:
        labels: Atom labels.
        coords: Cartesian coordinates in Å.
        centres: Indices or coordinates defining PDIP centres.
        elements: Elements/labels to include.

    Returns:
        A `Molecule` instance with PDIP hyperfine couplings populated.
    """
    molecule = build_molecule_from_labels_coords(
        labels=labels,
        coords=coords,
        elements=elements,
    )
    molecule.calc_pdip(centres)
    return molecule


def build_molecule_from_labels_coords(
    labels: list[str],
    coords: np.ndarray,
    *,
    elements: list[str] | str = "all",
) -> Molecule:
    """Build a `Molecule` from explicit labels and coordinates.

    Args:
        labels: Atom labels.
        coords: Cartesian coordinates in Å.
        elements: Elements/labels to include.

    Returns:
        A `Molecule` instance.
    """
    return Molecule.from_labels_coords(
        labels=labels,
        coords=coords,
        elements=elements,
    )


def build_molecule_from_csv(
    payload: dict,
    *,
    elements: list[str] | str = "all",
) -> Molecule:
    """Build a `Molecule` from an IO CSV payload."""

    labels: list[str] = payload["labels"]
    coords = payload["coords"]

    tensors = payload.get("tensors")  # dict[label, (3,3)] | None
    chem_labels = payload.get("chem_labels")  # list[str] | None
    chem_math_labels = payload.get("chem_math_labels")  # list[str] | None

    # --- Hyperfine: tensors -> (a_iso, a_dtensor) ---
    a_iso = None
    a_dtensor = None
    if tensors is not None:
        if isinstance(tensors, dict):
            tensor_by_label = {k: np.asarray(v, float) for k, v in tensors.items()}
        else:
            tensor_by_label = {
                lab: np.asarray(t, float) for lab, t in zip(labels, tensors)
            }

        a_iso = {}
        a_dtensor = {}
        for lab in labels:
            if lab not in tensor_by_label:
                raise KeyError(f"Missing hyperfine tensor for label: {lab}")

            A = np.asarray(tensor_by_label[lab], float)
            if A.shape != (3, 3):
                raise ValueError(
                    f"Hyperfine tensor for {lab} must be (3,3), got {A.shape}"
                )

            iso = float(np.trace(A) / 3.0)
            a_iso[lab] = iso
            a_dtensor[lab] = A - np.eye(3) * iso

    # --- Chem labels ---
    al_to_cl = None
    al_to_cml = None
    if chem_labels is not None:
        if len(chem_labels) != len(labels):
            raise ValueError(
                f"chem_labels length mismatch: {len(chem_labels)} vs {len(labels)}"
            )
        al_to_cl = {lab: str(cl) for lab, cl in zip(labels, chem_labels)}

        if chem_math_labels is not None:
            if len(chem_math_labels) != len(labels):
                raise ValueError(
                    f"chem_math_labels length mismatch: "
                    f"{len(chem_math_labels)} vs {len(labels)}"
                )
            al_to_cml = {lab: str(cml) for lab, cml in zip(labels, chem_math_labels)}

    # --- Build molecule ---
    if a_iso is not None and a_dtensor is not None:
        molecule = Molecule.from_hyperfine_data(
            labels=labels,
            coords=coords,
            a_iso=a_iso,
            a_dtensor=a_dtensor,
            elements=elements,
        )
    else:
        molecule = Molecule.from_labels_coords(
            labels=labels,
            coords=coords,
            elements=elements,
        )

    if al_to_cl is not None:
        molecule.apply_chem_labels(al_to_cl, al_to_cml)

    return molecule
