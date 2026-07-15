# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Application-layer averaging policy helpers for susceptibility fitting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from paranmr.core.const import ptable
from paranmr.core.domain.mol import Molecule
from paranmr.tools.coords import xyz_fmt as xyzf

_BOND_TOLERANCE_ANGSTROM = 0.20


@dataclass(frozen=True)
class MethylGroup:
    carbon_label: str
    proton_labels: tuple[str, ...]


def resolve_average_shift_groups(
    *,
    molecule: Molecule,
    average_shifts: str | Sequence[str] | None,
) -> list[list[str]]:
    """Resolve user-facing averaging policy into explicit atom-label groups."""

    if average_shifts in (None, "", []):
        return []

    if isinstance(average_shifts, str):
        normalized = average_shifts.strip().lower()
        if normalized == "all":
            signal_labels = sorted({nuc.signal_label for nuc in molecule.nuclei})
            return _groups_from_signal_labels(
                molecule=molecule,
                signal_labels=signal_labels,
            )
        if normalized == "methyls":
            return [
                list(group.proton_labels)
                for group in detect_methyl_group_records(molecule)
            ]
        return _groups_from_signal_labels(
            molecule=molecule,
            signal_labels=[average_shifts],
        )

    signal_labels = [str(value) for value in average_shifts]
    if not signal_labels:
        return []
    if any(value.strip().lower() == "methyls" for value in signal_labels):
        if len(signal_labels) != 1:
            raise ValueError(
                "susc_fit:average_shifts cannot combine 'methyls' with "
                "signal-label averaging."
            )
        return [
            list(group.proton_labels)
            for group in detect_methyl_group_records(molecule)
        ]
    if any(value.strip().lower() == "all" for value in signal_labels):
        if len(signal_labels) != 1:
            raise ValueError(
                "susc_fit:average_shifts cannot combine 'all' with "
                "selected signal labels."
            )
        signal_labels = sorted({nuc.signal_label for nuc in molecule.nuclei})

    return _groups_from_signal_labels(
        molecule=molecule,
        signal_labels=signal_labels,
    )
def apply_methyl_signal_labels(molecule: Molecule) -> None:
    """Assign synthetic signal labels for detected methyl groups."""

    for group in detect_methyl_group_records(molecule):
        signal_label = f"CH3({group.carbon_label})"
        for nucleus in molecule.nuclei:
            if nucleus.label in group.proton_labels:
                nucleus.signal_label = signal_label
                nucleus.signal_math_label = signal_label


def detect_methyl_group_records(molecule: Molecule) -> list[MethylGroup]:
    """Detect methyl groups together with their anchoring carbon labels."""

    if any(nuc.label_nn != "H" for nuc in molecule.nuclei):
        raise ValueError(
            "susc_fit:average_shifts 'methyls' requires nuclei:include to "
            "select only H nuclei."
        )

    full_labels = [str(label) for label in molecule.labels]
    full_coords = np.asarray(molecule.coords, dtype=float)
    if len(full_labels) != len(full_coords):
        raise ValueError("Molecule labels and coordinates must have matching lengths")

    atom_kind_by_label = {
        label: xyzf.remove_label_indices(label)
        for label in full_labels
    }
    coord_by_label = {
        label: np.asarray(coord, dtype=float)
        for label, coord in zip(full_labels, full_coords)
    }
    selected_h_labels = {nuc.label for nuc in molecule.nuclei}

    methyl_groups: list[MethylGroup] = []
    for atom_label, atom_kind in atom_kind_by_label.items():
        if atom_kind != "C":
            continue

        carbon_label = atom_label
        hydrogen_neighbors: list[str] = []
        heavy_neighbors: list[str] = []
        carbon_coord = coord_by_label[carbon_label]
        for candidate_label, candidate_kind in atom_kind_by_label.items():
            if candidate_label == carbon_label:
                continue
            if not _atoms_are_bonded(
                atom_kind,
                carbon_coord,
                candidate_kind,
                coord_by_label[candidate_label],
            ):
                continue
            if candidate_kind == "H":
                hydrogen_neighbors.append(candidate_label)
            else:
                heavy_neighbors.append(candidate_label)

        if len(hydrogen_neighbors) != 3 or len(heavy_neighbors) != 1:
            continue
        if not set(hydrogen_neighbors).issubset(selected_h_labels):
            continue
        methyl_groups.append(
            MethylGroup(
                carbon_label=carbon_label,
                proton_labels=tuple(
                    sorted(hydrogen_neighbors, key=_natural_label_key)
                ),
            )
        )

    methyl_groups.sort(key=lambda item: _natural_label_key(item.carbon_label))
    return methyl_groups


def _groups_from_signal_labels(
    *,
    molecule: Molecule,
    signal_labels: Sequence[str],
) -> list[list[str]]:
    groups: list[list[str]] = []
    available_signal_labels = {nuc.signal_label for nuc in molecule.nuclei}
    missing = [
        signal_label
        for signal_label in signal_labels
        if signal_label not in available_signal_labels
    ]
    if missing:
        raise ValueError(
            "Unknown signal label(s) requested for susc_fit:average_shifts: "
            + ", ".join(sorted(set(missing)))
        )

    for signal_label in signal_labels:
        group = [
            nuc.label for nuc in molecule.nuclei if nuc.signal_label == signal_label
        ]
        if group:
            groups.append(group)
    return groups


def _atoms_are_bonded(
    atom_kind_a: str,
    coord_a: np.ndarray,
    atom_kind_b: str,
    coord_b: np.ndarray,
) -> bool:
    max_distance = (
        float(ptable.cov_radii[atom_kind_a])
        + float(ptable.cov_radii[atom_kind_b])
        + _BOND_TOLERANCE_ANGSTROM
    )
    return float(np.linalg.norm(coord_a - coord_b)) <= max_distance


def _natural_label_key(label: str) -> tuple[str, int]:
    atom_kind = xyzf.remove_label_indices(label)
    numeric_suffix = label[len(atom_kind) :]
    if numeric_suffix.isdigit():
        return atom_kind, int(numeric_suffix)
    return atom_kind, -1
