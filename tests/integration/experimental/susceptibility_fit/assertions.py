"""Shared assertions for experimental susceptibility-fitting workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from paranmr.app.policies.averaging import detect_methyl_group_records
from paranmr.core.domain.mol import Molecule
from paranmr.tools.coords import xyz_fmt as xyzf


def read_generated_hyperfines_table(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith("atom_label"))
    return pd.read_csv(path, skiprows=header_index, encoding="utf-8-sig")


def reference_signal_groups(labels_csv: Path) -> list[frozenset[str]]:
    labels = pd.read_csv(labels_csv, encoding="utf-8-sig")
    return [
        frozenset(group["atom_label"].astype(str).tolist())
        for _, group in labels.groupby("signal_label", sort=False)
    ]


def reference_signal_partitions(labels_csv: Path) -> set[frozenset[str]]:
    return set(reference_signal_groups(labels_csv))


def reference_signal_groups_with_methyls(
    labels_csv: Path, xyz_file: Path
) -> list[frozenset[str]]:
    groups = reference_signal_groups(labels_csv)
    labels, coords = xyzf.load_xyz(str(xyz_file), check=False)
    molecule = Molecule.from_labels_coords(labels, coords, elements="H")
    methyl_groups = [
        frozenset(group.proton_labels) for group in detect_methyl_group_records(molecule)
    ]
    refined = []
    for group in groups:
        methyls = [methyl for methyl in methyl_groups if methyl.issubset(group)]
        if methyls and frozenset().union(*methyls) == group:
            refined.extend(methyls)
        else:
            refined.append(group)
    return refined


def reference_signal_partitions_with_methyls(
    labels_csv: Path, xyz_file: Path
) -> set[frozenset[str]]:
    return set(reference_signal_groups_with_methyls(labels_csv, xyz_file))


def partition_protons_by_sorted_shift(
    hyperfines_csv: Path, group_sizes: list[int]
) -> set[frozenset[str]]:
    table = read_generated_hyperfines_table(hyperfines_csv)
    protons = table[table["atom_label ()"].astype(str).str.startswith("H")]
    atom_labels = protons.sort_values("δ_total (ppm)", kind="mergesort")["atom_label ()"].astype(str).tolist()
    assert sum(group_sizes) == len(atom_labels)
    return {
        frozenset(atom_labels[start : start + size])
        for start, size in zip(np.cumsum([0, *group_sizes[:-1]]), group_sizes)
    }


def group_centers_from_hyperfines(
    hyperfines_csv: Path, proton_groups: set[frozenset[str]] | list[frozenset[str]]
) -> dict[frozenset[str], float]:
    protons = read_generated_hyperfines_table(hyperfines_csv)
    protons = protons[protons["atom_label ()"].astype(str).str.startswith("H")]
    centers = {}
    for group in proton_groups:
        values = protons.loc[protons["atom_label ()"].isin(group), "δ_total (ppm)"].to_numpy(float)
        assert len(values) == len(group)
        centers[group] = float(np.mean(values))
    return centers


def range_based_ppm_tolerance(
    centers_by_group: dict[frozenset[str], float], fraction: float = 0.03
) -> float:
    values = np.asarray(list(centers_by_group.values()), dtype=float)
    return float(fraction * (np.max(values) - np.min(values)))
