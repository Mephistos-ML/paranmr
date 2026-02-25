# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Construct Molecule domain objects from external payloads.

Provides helpers to build Molecule instances from QC-derived data or CSV payloads.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from simpnmr.core.conv.freq_to_ang import a_iso_mhz_to_ang, a_tensor_mhz_to_ang
from simpnmr.core.domain.mol import Molecule


def build_molecule_from_qca(
    qca: Any,
    *,
    elements: list[str] | str = "all",
    converter: str | None = "MHz_to_Ang-3",
) -> Molecule:
    """Build a `Molecule` from a parsed QC hyperfine object.

    Expects `qca` (typically `rdrs.QCA`) to provide:
      - coords: array-like (n_atoms, 3) in Å
      - labels: list-like (n_atoms,) of indexed per-atom labels (e.g. H1, C2)
      - a_iso: mapping label -> float (keyed by the same labels as `labels`)
      - a_dtensor: mapping label -> (3, 3) array-like deviatoric (traceless) tensors (keyed by the same labels as `labels`)

    Args:
        qca: Parsed QC hyperfine object.
        elements: Elements/labels to include.
        converter: Optional converter string matching the behaviour of
            `Molecule.from_QCA(..., converter=...)`. Defaults to "MHz_to_Ang-3".

    Returns:
        A `Molecule` instance.

    Raises:
        ValueError: If required fields are missing or unknown converter specified.
    """
    if not hasattr(qca, "coords"):
        raise ValueError("QCA object is missing required attribute: coords")
    if not hasattr(qca, "a_iso"):
        raise ValueError("QCA object is missing required attribute: a_iso")
    if not hasattr(qca, "a_dtensor"):
        raise ValueError("QCA object is missing required attribute: a_dtensor")

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
    a_iso: dict[str, float] = {str(k): float(v) for k, v in qca.a_iso.items()}
    a_dtensor: dict[str, np.ndarray] = {
        str(k): np.asarray(v, dtype=float) for k, v in qca.a_dtensor.items()
    }

    # Conversion logic copied from Molecule.from_QCA to preserve previous behaviour
    if converter is None:
        pass
    elif converter == "MHz_to_Ang-3":
        # Convert isotropic hyperfine values
        a_iso = a_iso_mhz_to_ang(a_iso)
        # Convert deviatoric (traceless) hyperfine tensors
        a_dtensor = a_tensor_mhz_to_ang(a_dtensor)

    else:
        raise ValueError(f"Unknown converter: {converter}")

    return Molecule.from_hyperfine_data(
        labels=labels,
        coords=coords,
        a_iso=a_iso,
        a_dtensor=a_dtensor,
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
