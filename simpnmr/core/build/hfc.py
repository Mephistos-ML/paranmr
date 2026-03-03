# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Builders for attaching hyperfine data to an existing Molecule."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from simpnmr.core.conv.freq_to_ang import a_tensor_mhz_to_ang
from simpnmr.core.domain.mol import Molecule

logger = logging.getLogger(__name__)


def build_hfc_from_qca(
    molecule: Molecule,
    qca: Any,
    *,
    converter: str | None = "MHz_to_Ang-3",
    orbital_contribution: str = "off",
) -> Molecule:
    """Build hyperfine data from a parsed QC object and attach it to a Molecule.

    Expects `qca` to provide component hyperfine tensors keyed by atom label:
      - a_fc: mapping label -> (3, 3) A(FC) tensor
      - a_sd: mapping label -> (3, 3) A(SD) tensor
      - a_orb: mapping label -> (3, 3) A(ORB) tensor or None
      - labels: list-like (n_atoms,) of indexed per-atom labels (e.g. H1, C2)

    The derived hyperfine data matches the current Molecule hyperfine contract:
      - a_fc: physical Fermi-contact hyperfine tensor
      - a_sd: physical traceless spin-dipolar hyperfine tensor
      - a_orb: physical traceless orbital hyperfine tensor when available
      - a_tensor_full: full physical hyperfine tensor derived from
        A(FC)+A(SD)+A(ORB) when available

    Args:
        molecule: Existing Molecule to enrich with derived hyperfine data.
        qca: Parsed QC hyperfine object.
        converter: Optional converter string controlling unit conversion. When
            set to "MHz_to_Ang-3", component tensors are converted before
            deriving effective quantities. Defaults to "MHz_to_Ang-3".
        orbital_contribution: Whether to require and preserve `A(ORB)` as a
            separate orbital hyperfine contribution. Supported values are
            "off" and "on".

    Returns:
        The input Molecule enriched via per-nucleus hyperfine assignments.

    Raises:
        ValueError: If required fields are missing, tensor shapes are invalid,
            or orbital contribution policy is invalid.
    """
    if not hasattr(qca, "a_fc"):
        raise ValueError("QCA object is missing required attribute: a_fc")
    if not hasattr(qca, "a_sd"):
        raise ValueError("QCA object is missing required attribute: a_sd")
    if not hasattr(qca, "a_orb"):
        raise ValueError("QCA object is missing required attribute: a_orb")
    if not hasattr(qca, "labels"):
        raise ValueError("QCA object is missing required attribute: labels")

    labels = [str(lab) for lab in qca.labels]

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

    if converter is None:
        pass
    elif converter == "MHz_to_Ang-3":
        a_fc = a_tensor_mhz_to_ang(a_fc)
        a_sd = a_tensor_mhz_to_ang(a_sd)

        a_orb_present = {k: v for k, v in a_orb.items() if v is not None}
        a_orb_present = a_tensor_mhz_to_ang(a_orb_present)
        a_orb = {k: a_orb_present.get(k) for k in a_orb}
    else:
        raise ValueError(f"Unknown converter: {converter}")

    if orbital_contribution not in {"off", "on"}:
        raise ValueError(
            f"orbital_contribution must be 'off' or 'on', got {orbital_contribution!r}"
        )

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

    a_tensor_full: dict[str, np.ndarray] = {}
    for label, fc_sd_tensor in a_fc_sd.items():
        total = np.asarray(fc_sd_tensor, dtype=float)

        orb = a_orb.get(label)
        if orb is not None:
            orb_arr = np.asarray(orb, dtype=float)
            if orb_arr.shape != (3, 3):
                raise ValueError(
                    f"A(ORB) tensor for {label} must be (3,3), got {orb_arr.shape}"
                )
            total = total + orb_arr

        a_tensor_full[label] = total

    for nuc in molecule.nuclei:
        label = str(nuc.label)
        if label not in a_fc or label not in a_sd:
            continue

        nuc.A.fc = a_fc[label]
        nuc.A.sd = a_sd[label]

        orb_tensor = a_orb.get(label)
        if orb_tensor is None:
            if orbital_contribution == "on":
                raise ValueError(
                    f"A(ORB) contribution requested but missing for label {label}"
                )
            nuc.A.orb = np.zeros((3, 3), dtype=float)
        else:
            nuc.A.orb = orb_tensor

        nuc.A.tensor_full = a_tensor_full.get(label)

    return molecule


def build_hfc_from_pdip(
    molecule: Molecule,
    *,
    centres: list[str],
) -> Molecule:
    """Build point-dipole hyperfine data and attach it to a Molecule.

    This builder delegates the point-dipole hyperfine calculation to the
    existing domain method on `Molecule` and returns the same enriched domain
    object.

    Args:
        molecule: Existing Molecule to enrich with point-dipole hyperfine data.
        centres: Labels of the paramagnetic centres passed directly to the
            domain-level `calc_pdip` implementation.

    Returns:
        The input Molecule enriched with point-dipole hyperfine data.
    """
    molecule.calc_pdip(centres)
    return molecule


def build_hfc_from_csv(
    molecule: Molecule,
    payload: dict,
) -> Molecule:
    """Build hyperfine data from a CSV payload and attach it to a Molecule.

    This builder reads full hyperfine tensors from a CSV-derived payload,
    derives the isotropic and deviatoric effective parts, and attaches them to
    the existing per-nucleus hyperfine contract on `Molecule`. If chemical
    labels are present in the payload, they are also applied to the domain
    object.

    Args:
        molecule: Existing Molecule to enrich with CSV-derived hyperfine data.
        payload: CSV payload that may contain:
            - labels
            - tensors
            - chem_labels
            - chem_math_labels

    Returns:
        The input Molecule enriched with CSV-derived hyperfine data and, when
        available, chemical labels.

    Raises:
        KeyError: If a required tensor is missing for a labelled nucleus.
        ValueError: If tensor shapes or chemical-label lengths are invalid.
    """
    labels: list[str] = payload["labels"]

    tensors = payload.get("tensors")
    chem_labels = payload.get("chem_labels")
    chem_math_labels = payload.get("chem_math_labels")

    if tensors is not None:
        if isinstance(tensors, dict):
            tensor_by_label = {k: np.asarray(v, float) for k, v in tensors.items()}
        else:
            tensor_by_label = {
                lab: np.asarray(t, float) for lab, t in zip(labels, tensors)
            }

        for lab in labels:
            if lab not in tensor_by_label:
                raise KeyError(f"Missing hyperfine tensor for label: {lab}")

            A = np.asarray(tensor_by_label[lab], float)
            if A.shape != (3, 3):
                raise ValueError(
                    f"Hyperfine tensor for {lab} must be (3,3), got {A.shape}"
                )

            iso = float(np.trace(A) / 3.0)
            dt = A - np.eye(3) * iso

            for nuc in molecule.nuclei:
                if str(nuc.label) != lab:
                    continue
                nuc.A.fc = np.eye(3, dtype=float) * iso
                nuc.A.sd = dt
                nuc.A.tensor_full = A
                break

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

    if al_to_cl is not None:
        molecule.apply_chem_labels(al_to_cl, al_to_cml)

    return molecule
